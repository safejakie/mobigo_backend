import csv
import json
import os
import re
import threading
import unicodedata
import copy
import contextvars
from collections import Counter, defaultdict
from typing import Optional, List

import serpapi
from groq import Groq

from config import (
    DATA_DIR, GROQ_API_KEY, SERPAPI_KEY, CENTRE_ABIDJAN, MODEL,
    PRIX_LIGNES, PRIX_POUR_CLASSEMENT, RAYON_PROXIMITE_M,
    cache_get, cache_set, distance_m, log,
)

# outils texte 
def nettoyer_texte(texte: str) -> str:
    if not texte:
        return ""
    texte = unicodedata.normalize("NFD", str(texte).lower())
    return "".join(c for c in texte if unicodedata.category(c) != "Mn")


def extraire_mots(texte: str) -> list:
    mots_vides = {"de", "du", "des", "d", "l", "la", "le", "les", "a", "au", "abidjan"}
    return [m for m in re.findall(r"\w+", nettoyer_texte(texte)) if m not in mots_vides]


def _charger_trace(trace_wkt, max_points: int = 1000):
    if not trace_wkt or not isinstance(trace_wkt, str):
        return [], None
    try:
        coords = re.findall(r"(-?\d+\.?\d+)\s+(-?\d+\.?\d+)", trace_wkt)
        if not coords:
            return [], None
        points = [(float(lat), float(lon)) for lon, lat in coords]
        if not points:
            return [], None
        if len(points) > max_points:
            step = max(1, len(points) // max_points)
            points = points[::step]
        lats = [p[0] for p in points]
        lons = [p[1] for p in points]
        return points, (min(lats), min(lons), max(lats), max(lons))
    except Exception:
        return [], None


def _point_proche_du_trace(lat: float, lon: float, trace: list, bbox, seuil_m: float = 300) -> bool:
    if not trace or not bbox:
        return False
    min_lat, min_lon, max_lat, max_lon = bbox
    marge_deg = seuil_m / 100000.0 + 0.001
    if not (min_lat - marge_deg <= lat <= max_lat + marge_deg):
        return False
    if not (min_lon - marge_deg <= lon <= max_lon + marge_deg):
        return False
    for tlat, tlon in trace:
        if distance_m(lat, lon, tlat, tlon) <= seuil_m:
            return True
    return False


# parsing des champs CSV gbaka/woro 

def _parse_liste_arrets(liste_str: str) -> list:
    if not liste_str or not isinstance(liste_str, str):
        return []
    s = liste_str.strip()
    if "→" in s:
        parts = s.split("→")
    elif "->" in s:
        parts = s.split("->")
    else:
        parts = [s]
    return [p.strip() for p in parts if p.strip()]


def _parse_coords_str(coords_str: str) -> list:
    if not coords_str or not isinstance(coords_str, str):
        return []
    out = []
    for pair in coords_str.split("|"):
        pair = pair.strip()
        if not pair:
            continue
        parts = [p.strip() for p in pair.split(",")]
        if len(parts) != 2:
            continue
        try:
            lon = float(parts[0])
            lat = float(parts[1])
            out.append((lon, lat))
        except ValueError:
            continue
    return out


# gares 

NOMS_GARES = [
    "gare", "gare nord", "gare sud", "gare routiere",
    "gare adjame", "adjame gare", "gare d adjame",
    "gare abobo", "abobo gare", "gare d abobo",
    "gare cocody", "gare de cocody",
    "gare plateau", "gare de plateau", "gare du plateau",
    "gare treichville", "gare de treichville", "treichville gare",
    "gare marcory", "gare de marcory", "marcory gare",
    "gare koumassi", "gare de koumassi", "koumassi gare",
    "gare port bouet", "gare de port bouet", "port bouet gare",
    "gare yopougon", "gare de yopougon", "yopougon gare",
    "gare attecoube", "gare d attecoube", "attecoube gare",
    "gare bassam", "gare de bassam",
]


def est_gare(nom: str) -> bool:
    n = nettoyer_texte(nom or "")
    if not n:
        return False
    if "gare" in n:
        return True
    return any(g in n for g in NOMS_GARES if g != "gare")


# geocodage 

def rechercher_lieu(lieu: Optional[str]):
    if not lieu:
        return None
    cle = ("geo", lieu.strip().lower())
    deja = cache_get(cle)
    if deja:
        return deja

    try:
        res = serpapi.search({
            "api_key": SERPAPI_KEY,
            "engine": "google_maps",
            "type": "search",
            "q": f"{lieu}, Abidjan, Cote d'Ivoire",
            "ll": CENTRE_ABIDJAN,
            "hl": "fr",
            "gl": "ci",
        }).as_dict()
    except Exception as e:
        log.error("erreur serpapi : %s", e)
        return None

    candidats = []
    if res.get("place_results"):
        candidats.append(res["place_results"])
    candidats += res.get("local_results", [])

    for r in candidats:
        coords = r.get("gps_coordinates") or {}
        if coords.get("latitude") and coords.get("longitude"):
            resultat = {
                "nom": r.get("title") or lieu,
                "latitude": float(coords["latitude"]),
                "longitude": float(coords["longitude"]),
                "alternatives": [],
            }
            cache_set(cle, resultat)
            return resultat
    return None


def suggestions_lieu(lieu: str, limite: int = 8) -> list:
    if not lieu or len(lieu.strip()) < 2:
        return []

    lieu = lieu.strip()
    cle = ("suggestions", lieu.lower())
    deja = cache_get(cle)
    if deja is not None:
        return deja

    resultats = []
    vus = set()

    if not RESEAU["charge"]:
        charger_reseau_local()

    texte_nettoye = nettoyer_texte(lieu)
    for arret in RESEAU["arrets"].values():
        if arret["id"] in RESEAU["points_libres"]:
            continue
        nom_nettoye = nettoyer_texte(arret["nom"])
        if texte_nettoye in nom_nettoye:
            if arret["nom"] in vus:
                continue
            vus.add(arret["nom"])
            resultats.append({
                "nom": arret["nom"],
                "adresse": "",
                "latitude": arret["latitude"],
                "longitude": arret["longitude"],
            })
            if len(resultats) >= limite:
                break

    if len(resultats) < limite:
        try:
            res = serpapi.search({
                "api_key": SERPAPI_KEY,
                "engine": "google_maps",
                "type": "search",
                "q": f"{lieu}, Abidjan, Cote d'Ivoire",
                "ll": CENTRE_ABIDJAN,
                "hl": "fr",
                "gl": "ci",
            }).as_dict()

            for r in (res.get("local_results") or []) + ([res["place_results"]] if res.get("place_results") else []):
                coords = r.get("gps_coordinates") or {}
                if not (coords.get("latitude") and coords.get("longitude")):
                    continue
                nom = r.get("title") or "lieu"
                if nom in vus:
                    continue
                vus.add(nom)
                resultats.append({
                    "nom": nom,
                    "adresse": r.get("address") or "",
                    "latitude": float(coords["latitude"]),
                    "longitude": float(coords["longitude"]),
                })
                if len(resultats) >= limite:
                    break
        except Exception as e:
            log.warning("suggestions serpapi echouee : %s", e)

    cache_set(cle, resultats)
    return resultats


MOTS_POSITION = (
    "position", "ici", "je suis la", "je suis ici", "pres de moi", "autour de moi",
    "chez moi", "mon emplacement", "ma localisation", "localisation",
)


def determiner_depart(dep_texte: Optional[str], lat: Optional[float] = None, lon: Optional[float] = None):
    a_gps = lat is not None and lon is not None
    texte = (dep_texte or "").strip()
    texte_net = nettoyer_texte(texte)

    if a_gps and (not texte_net or texte_net in ("null", "none", "aucun")
                  or any(m in texte_net for m in MOTS_POSITION)):
        return {"nom": "ma position actuelle", "latitude": float(lat), "longitude": float(lon), "alternatives": []}

    if texte_net and texte_net not in ("null", "none", "aucun") and not any(m in texte_net for m in MOTS_POSITION):
        lieu = rechercher_lieu(texte)
        if lieu:
            return lieu

    if a_gps:
        return {"nom": "ma position actuelle", "latitude": float(lat), "longitude": float(lon), "alternatives": []}
    return None


# donnees 

CATALOGUE = []
RESEAU = {
    "arrets": {},
    "lignes": {},
    "sequences": {},
    "lignes_par_arret": defaultdict(list),
    "gares": set(),
    "points_libres": set(),
    "arrets_non_bus": set(),
    "arrets_bus": set(),
    "arrets_gbaka": set(),
    "charge": False,
}
_lock = threading.Lock()


def _trace_estimee(row) -> bool:
    origine = nettoyer_texte(row.get("origine_trace") or "")
    complet = nettoyer_texte(row.get("trace_complete") or "")
    return "approximatif" in origine or "introuvable" in origine or complet == "non"


def charger_catalogue():
    global CATALOGUE
    CATALOGUE = []

    chemin = os.path.join(DATA_DIR, "bus.csv")
    if os.path.exists(chemin):
        with open(chemin, encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f, delimiter=";")
            for row in reader:
                numero = (row.get("numero_ligne") or "").strip()
                origine = (row.get("depart") or "").strip()
                destination = (row.get("arrivee") or "").strip()
                if not origine or not destination:
                    continue
                try:
                    prix = int(float(row.get("prix_fcfa") or 200))
                except Exception:
                    prix = 200
                trace, bbox = _charger_trace(row.get("trace_wkt") or "")
                CATALOGUE.append({
                    "numero": numero,
                    "origine": origine,
                    "destination": destination,
                    "operateur": (row.get("operateur") or "SOTRA").strip(),
                    "reseau": (row.get("reseau") or "Monbus").strip(),
                    "type": "BUS",
                    "prix_fcfa": prix,
                    "trace": trace,
                    "bbox": bbox,
                    "estime": _trace_estimee(row),
                })

    chemin = os.path.join(DATA_DIR, "bateau_bus.csv")
    if os.path.exists(chemin):
        with open(chemin, encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f, delimiter=";")
            for row in reader:
                numero = (row.get("numero_ligne") or "").strip()
                origine = (row.get("depart") or "").strip()
                destination = (row.get("arrivee") or "").strip()
                if not origine or not destination:
                    continue
                try:
                    prix = int(float(row.get("prix_minimum_fcfa") or 150))
                except Exception:
                    prix = 150
                trace, bbox = _charger_trace(row.get("trace_wkt") or "")
                CATALOGUE.append({
                    "numero": numero,
                    "origine": origine,
                    "destination": destination,
                    "operateur": (row.get("operateur") or "SOTRA").strip(),
                    "reseau": (row.get("reseau") or "Monbato").strip(),
                    "type": "BATEAU",
                    "prix_fcfa": prix,
                    "trace": trace,
                    "bbox": bbox,
                    "estime": _trace_estimee(row),
                })

    chemin = os.path.join(DATA_DIR, "gbaka_woro.csv")
    if os.path.exists(chemin):
        entrees = {}
        with open(chemin, encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f, delimiter=";")
            for row in reader:
                type_raw = (row.get("type_transport") or "").strip().lower()
                origine = (row.get("depart") or "").strip()
                destination = (row.get("arrivee") or "").strip()
                if not origine or not destination:
                    continue
                type_ligne = "WORO_WORO" if "woro" in type_raw else "GBAKA"

                sa = {
                    "destination": (row.get("sens_aller_destination") or "").strip(),
                    "liste_arrets": _parse_liste_arrets(row.get("sens_aller_liste_arrets") or ""),
                    "coordonnees": _parse_coords_str(row.get("sens_aller_coordonnees_gps") or ""),
                }
                sr = {
                    "destination": (row.get("sens_retour_destination") or "").strip(),
                    "liste_arrets": _parse_liste_arrets(row.get("sens_retour_liste_arrets") or ""),
                    "coordonnees": _parse_coords_str(row.get("sens_retour_coordonnees_gps") or ""),
                }

                if not sa["liste_arrets"] and not sr["liste_arrets"]:
                    continue

                id_osm = (row.get("id_osm") or "").strip()
                origine_trace = (row.get("origine_trace") or "").strip()
                cle = id_osm or (type_ligne, origine, destination)

                nb_stops = len(sa["liste_arrets"]) + len(sr["liste_arrets"])
                if cle in entrees and nb_stops <= entrees[cle]["_nb_stops"]:
                    continue

                trace, bbox = _charger_trace(row.get("trace_wkt") or "")
                entrees[cle] = {
                    "numero": None,
                    "origine": origine,
                    "destination": destination,
                    "operateur": (row.get("operateur") or "divers").strip(),
                    "reseau": (row.get("reseau_commune") or "Autre").strip(),
                    "type": type_ligne,
                    "prix_fcfa": 200,
                    "trace": trace,
                    "bbox": bbox,
                    "estime": _trace_estimee(row),
                    "id_osm": id_osm,
                    "origine_trace": origine_trace,
                    "sens_aller": sa,
                    "sens_retour": sr,
                    "_nb_stops": nb_stops,
                }
        for e in entrees.values():
            e.pop("_nb_stops", None)
            CATALOGUE.append(e)

    log.info("catalogue charge : %d lignes", len(CATALOGUE))


# sequences d arrets 

def _dist_rapide(lat1, lon1, lat2, lon2) -> float:
    dy = (lat1 - lat2) * 111320.0
    dx = (lon1 - lon2) * 111320.0 * 0.9956
    return (dx * dx + dy * dy) ** 0.5


def _cellule(lat, lon, taille):
    return (int(lat / taille), int(lon / taille))


def _grille_arrets(taille):
    grille = defaultdict(list)
    for sid, a in RESEAU["arrets"].items():
        if sid in RESEAU["points_libres"]:
            continue
        grille[_cellule(a["latitude"], a["longitude"], taille)].append(sid)
    return grille


def _arrets_le_long_du_trace(trace, grille, taille, seuil_m, filtre=None):
    passages = defaultdict(list)
    for idx, (lat, lon) in enumerate(trace):
        ci, cj = _cellule(lat, lon, taille)
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                for sid in grille.get((ci + di, cj + dj), ()):
                    a = RESEAU["arrets"][sid]
                    d = _dist_rapide(lat, lon, a["latitude"], a["longitude"])
                    if d > seuil_m:
                        continue
                    if filtre and not filtre(sid, a["nom"]):
                        continue
                    passages[sid].append((idx, d))

    evenements = []
    for sid, hits in passages.items():
        groupe = [hits[0]]
        for h in hits[1:]:
            if h[0] - groupe[-1][0] <= 6:
                groupe.append(h)
            else:
                evenements.append((min(groupe, key=lambda x: x[1])[0], sid))
                groupe = [h]
        evenements.append((min(groupe, key=lambda x: x[1])[0], sid))
    evenements.sort()

    gardes = []
    for idx, sid in evenements:
        a = RESEAU["arrets"][sid]
        nom = nettoyer_texte(a["nom"])
        doublon = False
        for autre in gardes[-6:]:
            b = RESEAU["arrets"][autre]
            if nettoyer_texte(b["nom"]) == nom and _dist_rapide(
                    a["latitude"], a["longitude"], b["latitude"], b["longitude"]) < 250:
                doublon = True
                break
        if not doublon:
            gardes.append(sid)
    return gardes


def _points_de_chargement(trace, pas_m=300):
    if not trace:
        return []
    points = [trace[0]]
    cumul = 0.0
    for p, q in zip(trace, trace[1:]):
        cumul += _dist_rapide(p[0], p[1], q[0], q[1])
        if cumul >= pas_m:
            points.append(q)
            cumul = 0.0
    if points[-1] != trace[-1]:
        points.append(trace[-1])
    return points


def _est_arret_de_bus(sid: str, nom: str) -> bool:
    if "lagunaire" in nettoyer_texte(nom):
        return False
    if RESEAU["arrets_bus"]:
        return sid in RESEAU["arrets_bus"]
    return False


def _nom_point(lat, lon, grille_noms) -> str:
    proche, dmin = None, 400
    ci, cj = _cellule(lat, lon, 0.004)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            for sid in grille_noms.get((ci + di, cj + dj), ()):
                a = RESEAU["arrets"][sid]
                d = _dist_rapide(lat, lon, a["latitude"], a["longitude"])
                if d < dmin and len(a["nom"].strip()) >= 4 and not a["nom"].strip().isdigit():
                    proche, dmin = a["nom"], d
    return f"Bord de route près de {proche}" if proche else "Bord de route"


def _boucle_asymetrique(trace) -> bool:
    if len(trace) < 20 or _dist_rapide(trace[0][0], trace[0][1], trace[-1][0], trace[-1][1]) > 500:
        return False
    loin = max(range(len(trace)), key=lambda j: _dist_rapide(trace[0][0], trace[0][1], trace[j][0], trace[j][1]))
    aller, retour = trace[:loin + 1], trace[loin:]
    grille = defaultdict(list)
    for p in aller:
        grille[_cellule(p[0], p[1], 0.002)].append(p)
    pareil = 0
    for p in retour:
        ci, cj = _cellule(p[0], p[1], 0.002)
        if any(_dist_rapide(p[0], p[1], q[0], q[1]) < 200 for di in (-1, 0, 1) for dj in (-1, 0, 1)
               for q in grille.get((ci + di, cj + dj), ())):
            pareil += 1
    return pareil < 0.6 * len(retour)


def _sequence_gbaka(i, trace, grille_gbaka, grille_noms):
    reels = _arrets_le_long_du_trace(trace, grille_gbaka, 0.0006, 40)
    evenements = []
    for sid in reels:
        a = RESEAU["arrets"][sid]
        k = min(range(len(trace)), key=lambda j: _dist_rapide(a["latitude"], a["longitude"], trace[j][0], trace[j][1]))
        evenements.append((k, sid))
    cumul, dernier = 0.0, 0
    for k, (lat, lon) in enumerate(trace):
        if k:
            cumul += _dist_rapide(trace[k - 1][0], trace[k - 1][1], lat, lon)
        if k and cumul < 300 and k != len(trace) - 1:
            continue
        cumul = 0.0
        if any(_dist_rapide(lat, lon, RESEAU["arrets"][sid]["latitude"], RESEAU["arrets"][sid]["longitude"]) < 150
               for sid in reels):
            continue
        pid = f"g:{i}:{k}"
        RESEAU["arrets"][pid] = {"id": pid, "nom": _nom_point(lat, lon, grille_noms), "latitude": lat, "longitude": lon}
        RESEAU["points_libres"].add(pid)
        evenements.append((k, pid))
    return [sid for _, sid in sorted(evenements, key=lambda x: x[0])]


def _index_stops_par_nom() -> dict:
    idx = defaultdict(list)
    for sid, a in RESEAU["arrets"].items():
        idx[nettoyer_texte(a["nom"])].append(sid)
    return idx


def _trouver_stop_existant(nom: str, lat, lon, index: dict) -> Optional[str]:
    nom_clean = nettoyer_texte(nom)
    candidats = index.get(nom_clean, [])
    if not candidats:
        return None
    if lat is None or lon is None:
        return candidats[0]
    best, dmin = None, 120.0
    for sid in candidats:
        a = RESEAU["arrets"][sid]
        d = _dist_rapide(lat, lon, a["latitude"], a["longitude"])
        if d < dmin:
            dmin = d
            best = sid
    return best


def _resoudre_stops_csv(sens: dict, i: int, sens_label: str, index: dict) -> list:
    noms = sens.get("liste_arrets") or []
    coords = sens.get("coordonnees") or []
    if not noms:
        return []
    ids = []
    for idx, nom in enumerate(noms):
        nom_clean = (nom or "").strip()
        if not nom_clean:
            continue
        lat, lon = None, None
        if idx < len(coords):
            lon, lat = coords[idx]
        stop_id = _trouver_stop_existant(nom_clean, lat, lon, index)
        if stop_id:
            ids.append(stop_id)
            continue
        if lat is None or lon is None:
            continue
        nouveau = f"csv:{i}:{sens_label}:{idx}"
        RESEAU["arrets"][nouveau] = {
            "id": nouveau,
            "nom": nom_clean,
            "latitude": float(lat),
            "longitude": float(lon),
        }
        index.setdefault(nettoyer_texte(nom_clean), []).append(nouveau)
        ids.append(nouveau)
    return ids


def _arrets_ligne_pour_api(rid: str) -> list:
    if not isinstance(rid, str) or not rid.startswith("cat:"):
        return []
    try:
        i = int(rid[4:])
    except ValueError:
        return []
    if i < 0 or i >= len(CATALOGUE):
        return []
    c = CATALOGUE[i]
    if c["type"] not in ("GBAKA", "WORO_WORO"):
        return []

    sens_list = []
    for label, sens_data in (("aller",  c.get("sens_aller")  or {}),
                             ("retour", c.get("sens_retour") or {})):
        noms = sens_data.get("liste_arrets") or []
        coords = sens_data.get("coordonnees") or []
        if len(noms) < 2:
            continue

        arrets = []
        for idx, nom in enumerate(noms):
            lat = lon = None
            if idx < len(coords):
                lon, lat = coords[idx]
            arrets.append({
                "id": f"csv:{i}:{label}:{idx}",
                "nom": (nom or "").strip(),
                "latitude": lat,
                "longitude": lon,
            })
        if len(arrets) < 2:
            continue

        sens_list.append({
            "sens": label,
            "depart": arrets[0]["nom"],
            "arrivee": arrets[-1]["nom"],
            "nb_arrets": len(arrets),
            "arrets": arrets,
        })
    return sens_list


def _construire_sequences_depuis_traces():
    taille = 0.0006
    grille = _grille_arrets(taille)
    grille_noms = _grille_arrets(0.004)
    grille_gbaka = defaultdict(list)
    for sid in RESEAU["arrets_gbaka"]:
        a = RESEAU["arrets"][sid]
        grille_gbaka[_cellule(a["latitude"], a["longitude"], taille)].append(sid)

    index_noms = _index_stops_par_nom()
    nb = 0

    for i, c in enumerate(CATALOGUE):
        rid = f"cat:{i}"
        if rid in RESEAU["sequences"]:
            continue
        type_ligne = c["type"]
        trace = c.get("trace") or []

        if type_ligne in ("GBAKA", "WORO_WORO"):
            sa = c.get("sens_aller") or {}
            sr = c.get("sens_retour") or {}
            seq_aller = _resoudre_stops_csv(sa, i, "aller", index_noms)
            seq_retour = _resoudre_stops_csv(sr, i, "retour", index_noms)

            sequences = []
            if len(seq_aller) >= 2:
                sequences.append(seq_aller)
            if len(seq_retour) >= 2 and (not sequences or seq_retour != sequences[0]):
                sequences.append(seq_retour)

            if sequences:
                RESEAU["sequences"][rid] = sequences
                for seq in sequences:
                    for sid in seq:
                        if rid not in RESEAU["lignes_par_arret"][sid]:
                            RESEAU["lignes_par_arret"][sid].append(rid)
                nb += 1
                continue

            if len(trace) < 2 or c.get("estime"):
                continue
            sequence = _sequence_gbaka(i, trace, grille_gbaka, grille_noms)
            if len(sequence) < 2:
                continue
            boucle = _boucle_asymetrique(trace)
            RESEAU["sequences"][rid] = [sequence] if boucle else [sequence, sequence[::-1]]
            for sid in sequence:
                if rid not in RESEAU["lignes_par_arret"][sid]:
                    RESEAU["lignes_par_arret"][sid].append(rid)
            nb += 1
            continue

        if len(trace) < 2 or c.get("estime"):
            continue
        if type_ligne == "BUS":
            sequence = _arrets_le_long_du_trace(trace, grille, taille, 40, filtre=_est_arret_de_bus)
        elif type_ligne == "BATEAU":
            sequence = _arrets_le_long_du_trace(trace, grille, taille * 6, 400,
                                                filtre=lambda sid, n: "lagunaire" in nettoyer_texte(n))
        else:
            continue
        if len(sequence) < 2:
            continue
        RESEAU["sequences"][rid] = [sequence, sequence[::-1]]
        for sid in sequence:
            if rid not in RESEAU["lignes_par_arret"][sid]:
                RESEAU["lignes_par_arret"][sid].append(rid)
        nb += 1

    log.info("sequences construites depuis les traces : %d lignes", nb)


def charger_reseau_local():
    with _lock:
        if RESEAU["charge"]:
            return

        log.info("chargement du reseau local...")
        charger_catalogue()

        chemin = os.path.join(DATA_DIR, "arret.csv")
        if os.path.exists(chemin):
            with open(chemin, encoding="utf-8-sig") as f:
                for row in csv.DictReader(f):
                    try:
                        sid = row["stop_id"].strip()
                        nom = (row.get("stop_name") or "").strip()
                        lat = float(row["stop_lat"])
                        lon = float(row["stop_lon"])
                        RESEAU["arrets"][sid] = {"id": sid, "nom": nom,
                                                 "latitude": lat, "longitude": lon}
                        if est_gare(nom):
                            RESEAU["gares"].add(sid)
                    except Exception:
                        continue
        else:
            log.warning("arret.csv introuvable")

        for nom_fichier in ("stops.txt", "stops-2.txt"):
            chemin = os.path.join(DATA_DIR, nom_fichier)
            if not os.path.exists(chemin):
                continue
            with open(chemin, encoding="utf-8-sig") as f:
                for row in csv.DictReader(f):
                    try:
                        sid = row["stop_id"].strip()
                        if sid in RESEAU["arrets"]:
                            continue
                        nom = (row.get("stop_name") or "").strip()
                        lat = float(row["stop_lat"])
                        lon = float(row["stop_lon"])
                        RESEAU["arrets"][sid] = {"id": sid, "nom": nom,
                                                 "latitude": lat, "longitude": lon}
                        if est_gare(nom):
                            RESEAU["gares"].add(sid)
                    except Exception:
                        continue
            log.info("arrets ajoutes depuis %s", nom_fichier)

        chemin = os.path.join(DATA_DIR, "routes.txt")
        if os.path.exists(chemin):
            with open(chemin, encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    rid = row.get("route_id")
                    if not rid:
                        continue
                    numero = (row.get("route_short_name") or "").strip()
                    nom = (row.get("route_long_name") or "").strip()
                    agency = (row.get("agency_id") or "").lower()

                    if "gbaka" in agency or "gbaka" in nom.lower():
                        type_ligne = "GBAKA"
                    elif "woro" in agency or "woro" in nom.lower():
                        type_ligne = "WORO_WORO"
                    elif any(x in agency or x in nom.lower() for x in ("bateau", "stl", "aqua", "monbato")):
                        type_ligne = "BATEAU"
                    else:
                        type_ligne = "BUS"

                    RESEAU["lignes"][rid] = {
                        "id": rid, "numero": numero or None, "nom": nom,
                        "type": type_ligne, "operateur": row.get("agency_id") or "inconnu",
                    }

        trips = defaultdict(list)
        chemin = os.path.join(DATA_DIR, "trips.txt")
        if os.path.exists(chemin):
            with open(chemin, encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    trips[row["route_id"]].append(row["trip_id"])

        stop_times = defaultdict(list)
        chemin = os.path.join(DATA_DIR, "stop_times.txt")
        if os.path.exists(chemin):
            with open(chemin, encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    try:
                        stop_times[row["trip_id"]].append({
                            "stop_id": row["stop_id"], "seq": int(row["stop_sequence"]),
                        })
                    except Exception:
                        continue

        for rid, trip_ids in trips.items():
            if not trip_ids:
                continue
            toutes_les_sequences = []
            arrets_de_la_ligne = set()
            for tid in trip_ids:
                arrets_ordonnes = sorted(stop_times.get(tid, []), key=lambda x: x["seq"])
                sequence = [a["stop_id"] for a in arrets_ordonnes if a["stop_id"] in RESEAU["arrets"]]
                if len(sequence) >= 2:
                    if sequence not in toutes_les_sequences:
                        toutes_les_sequences.append(sequence)
                    arrets_de_la_ligne.update(sequence)
            if toutes_les_sequences:
                for seq in list(toutes_les_sequences):
                    if seq[::-1] not in toutes_les_sequences:
                        toutes_les_sequences.append(seq[::-1])
                RESEAU["sequences"][rid] = toutes_les_sequences
                for sid in arrets_de_la_ligne:
                    if rid not in RESEAU["lignes_par_arret"][sid]:
                        RESEAU["lignes_par_arret"][sid].append(rid)

        arrets_bus, arrets_autres = set(), set()
        for rid_gtfs, sequences_gtfs in RESEAU["sequences"].items():
            type_gtfs = RESEAU["lignes"].get(rid_gtfs, {}).get("type")
            cible = arrets_bus if type_gtfs == "BUS" else arrets_autres
            for seq in sequences_gtfs:
                cible.update(seq)
        RESEAU["arrets_non_bus"] = arrets_autres - arrets_bus
        RESEAU["arrets_bus"] = arrets_bus

        for rid_gtfs, sequences_gtfs in RESEAU["sequences"].items():
            if RESEAU["lignes"].get(rid_gtfs, {}).get("type") in ("GBAKA", "WORO_WORO"):
                for seq in sequences_gtfs:
                    RESEAU["arrets_gbaka"].update(seq)

        for i, c in enumerate(CATALOGUE):
            rid = f"cat:{i}"
            if rid not in RESEAU["lignes"]:
                RESEAU["lignes"][rid] = {
                    "id": rid, "numero": c.get("numero"),
                    "nom": f"{c['origine']} ↔ {c['destination']}",
                    "type": c["type"], "operateur": c.get("operateur", ""),
                }

        _construire_sequences_depuis_traces()

        RESEAU["charge"] = True
        log.info(
            "reseau pret : %d arrets, %d lignes, %d sequences, %d gares",
            len(RESEAU["arrets"]), len(RESEAU["lignes"]),
            len(RESEAU["sequences"]), len(RESEAU["gares"]),
        )


# arrets a proximite 

def arrets_proximite(lat: float, lon: float, rayon_m: int = 5000, tri_par_distance: bool = False) -> List[dict]:
    if not RESEAU["charge"]:
        charger_reseau_local()

    uniques = {}
    for arret in RESEAU["arrets"].values():
        d = distance_m(lat, lon, arret["latitude"], arret["longitude"])
        if d > rayon_m:
            continue
        types = set()
        for rid in RESEAU["lignes_par_arret"].get(arret["id"], []):
            ligne = RESEAU["lignes"].get(rid)
            if ligne:
                types.add(ligne["type"])
        # on n'affiche que les arrêts de bus, gbaka et bateau-bus (jamais les arrêts de woro-woro)
        types = types & {"BUS", "GBAKA", "BATEAU"}
        if not types:
            continue
        if "BUS" in types:
            type_principal = "BUS"
        elif "GBAKA" in types:
            type_principal = "GBAKA"
        else:
            type_principal = "BATEAU"
        nom = arret["nom"]
        if nom not in uniques or d < uniques[nom]["distance_m"]:
            uniques[nom] = {
                "id": arret["id"], "nom": nom,
                "latitude": arret["latitude"], "longitude": arret["longitude"],
                "distance_m": int(d), "type": type_principal, "types": list(types),
            }

    resultats = list(uniques.values())
    if tri_par_distance:
        resultats.sort(key=lambda x: x["distance_m"])
    else:
        ordre = {"BUS": 0, "GBAKA": 1, "BATEAU": 2}
        resultats.sort(key=lambda x: (ordre.get(x["type"], 9), x["distance_m"]))
    return resultats


def arrets_proximite_par_modes(lat: float, lon: float, rayon_m: int = 1500,
                                modes: Optional[List[str]] = None) -> List[dict]:
    if not RESEAU["charge"]:
        charger_reseau_local()
    modes_autorises = set(modes) if modes else {"BUS", "GBAKA", "BATEAU"}
    uniques = {}
    for arret in RESEAU["arrets"].values():
        d = distance_m(lat, lon, arret["latitude"], arret["longitude"])
        if d > rayon_m:
            continue
        a_des_modes = False
        for rid in RESEAU["lignes_par_arret"].get(arret["id"], []):
            ligne = RESEAU["lignes"].get(rid)
            if ligne and ligne["type"] in modes_autorises:
                a_des_modes = True
                break
        if a_des_modes:
            nom = arret["nom"]
            if nom not in uniques or d < uniques[nom]["distance_m"]:
                uniques[nom] = {
                    "id": arret["id"], "nom": nom,
                    "latitude": arret["latitude"], "longitude": arret["longitude"],
                    "distance_m": int(d),
                }
    resultats = list(uniques.values())
    resultats.sort(key=lambda x: x["distance_m"])
    return resultats


def lignes_par_arret(stop_id: str) -> Optional[dict]:
    if not RESEAU["charge"]:
        charger_reseau_local()
    arret = RESEAU["arrets"].get(stop_id)
    if not arret:
        return None
    lignes = []
    for rid in RESEAU["lignes_par_arret"].get(stop_id, []):
        ligne = RESEAU["lignes"].get(rid)
        if ligne:
            lignes.append({
                "id": ligne["id"], "numero": ligne["numero"], "nom": ligne["nom"],
                "type": ligne["type"], "operateur": ligne["operateur"],
            })
    lignes.sort(key=lambda l: (0 if l["type"] == "BUS" else 1, l["numero"] or "zzz"))
    return {"arret": arret, "lignes": lignes}


# Prix (lecture CSV uniquement) 

PRIX_CATALOGUE = {}
_prix_charges = False


def _norm_num(n) -> str:
    n = str(n or "").strip().upper()
    return str(int(n)) if n.isdigit() else n


def _charger_prix_csv():
    global _prix_charges
    _prix_charges = True
    PRIX_CATALOGUE.clear()
    for fichier, type_ligne, colonne in (("bus.csv", "BUS", "prix_fcfa"),
                                          ("bateau_bus.csv", "BATEAU", "prix_minimum_fcfa")):
        chemin = os.path.join(DATA_DIR, fichier)
        if not os.path.exists(chemin):
            continue
        with open(chemin, encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f, delimiter=";"):
                try:
                    prix = int(float(row.get(colonne)))
                except Exception:
                    continue
                PRIX_CATALOGUE.setdefault((type_ligne, _norm_num(row.get("numero_ligne"))), prix)


def _prix_csv(type_ligne, numero):
    if not _prix_charges:
        _charger_prix_csv()
    return PRIX_CATALOGUE.get((type_ligne, _norm_num(numero)))


def _rids_approximatifs() -> set:
    if "rids_approx" not in RESEAU:
        gtfs = set()
        for rid, l in RESEAU["lignes"].items():
            if not str(rid).startswith("cat:") and RESEAU["sequences"].get(rid):
                gtfs.add((l.get("type"), _norm_num(l.get("numero"))))
        RESEAU["rids_approx"] = {
            rid for rid, l in RESEAU["lignes"].items()
            if str(rid).startswith("cat:") and l.get("type") in ("BUS", "BATEAU")
            and (l.get("type"), _norm_num(l.get("numero"))) in gtfs
        }
    return RESEAU["rids_approx"]


def _finaliser_candidat(c: dict):
    connus, inconnu = 0, False
    for e in c["etapes"]:
        if e["mode"] == "MARCHE":
            continue
        p = e.get("cout_fcfa")
        if p is None:
            inconnu = True
        else:
            connus += p
    c["cout_partiel_fcfa"] = connus
    c["cout_fcfa"] = None if inconnu else connus
    c["prix_inconnu"] = inconnu
    c["nb_arrets"] = sum(e.get("nb_arrets", 0) for e in c["etapes"] if e["mode"] != "MARCHE")


def _cout_tri(c: dict) -> int:
    return c.get("cout_partiel_fcfa", 0)


_LIBELLES_MODE = {"BUS": "bus", "GBAKA": "gbaka", "WORO_WORO": "woro-woro", "BATEAU": "bateau-bus"}


def _ajouter_textes(c: dict):
    etapes = c.get("etapes", [])
    n = len(etapes)
    for i, e in enumerate(etapes):
        if e["mode"] == "MARCHE":
            d = int(e.get("distance_m", 0))
            if i == 0:
                if d < 10:
                    e["texte"] = f"Vous êtes déjà à l'arrêt {e['arrivee']} (point de montée)."
                else:
                    e["texte"] = f"Marchez {d} m jusqu'à l'arrêt {e['arrivee']} (point de montée)."
            elif i == n - 1:
                e["texte"] = f"Marchez {d} m jusqu'à votre destination."
            else:
                e["texte"] = f"Marchez {d} m de {e['depart']} à {e['arrivee']} pour la correspondance."
        else:
            lib = _LIBELLES_MODE.get(e["mode"], e["mode"].lower())
            ligne = str(e.get("numero_ligne") or e.get("nom_ligne") or "")
            for prefixe in ("gbaka :", "gbaka:", "woro-woro :", "woro :"):
                if ligne.lower().startswith(prefixe):
                    ligne = ligne[len(prefixe):].strip()
            nb = e.get("nb_arrets") or 0
            suite = f" ({nb} arrêt{'s' if nb > 1 else ''})" if nb else ""
            if e["mode"] == "WORO_WORO" and i == 0:
                depart = "votre position" if "position" in str(e.get("depart", "")).lower() else e["depart"]
                e["texte"] = (f"Prenez un woro-woro ou un gbaka depuis {depart} jusqu'à {e['arrivee']} "
                              f"(prix variable selon le chauffeur).")
            else:
                e["texte"] = f"Montez à {e['depart']} dans le {lib} {ligne}, descendez à {e['arrivee']}{suite}.".replace("  ", " ").replace(" ,", ",")
        if e.get("conseil"):
            e["texte"] += " " + e["conseil"]


_MODE_COURT = {"BUS": "bus", "GBAKA": "gbaka", "WORO_WORO": "woro-woro", "BATEAU": "bateau-bus"}


def _nom_court_ligne(e: dict) -> str:
    num = e.get("numero_ligne")
    if num:
        return str(num)
    nom = str(e.get("nom_ligne") or "").replace("gbaka :", "").strip()
    return nom or _MODE_COURT.get(e.get("mode"), "ligne")


def _etapes_transport(c: dict) -> list:
    return [e for e in c.get("etapes", []) if e.get("mode") not in ("MARCHE", "WORO_WORO")]


def _acces_m(c: dict) -> int:
    return c.get("marche_m", 0) + sum(e.get("distance_m", 0) for e in c.get("etapes", [])
                                       if e.get("mode") == "WORO_WORO")


# Trajets déjà proposés à exclure quand l'utilisateur dit « pas celui-là, autre chose ».
# (ContextVar : valable seulement pour la requête en cours, aucun changement nécessaire dans main.py)
_EXCLURE = contextvars.ContextVar("mobigo_exclure", default=None)


def _pos_proches(p, q, seuil_m: int = 500) -> bool:
    if not p or not q:
        return False
    return distance_m(p["latitude"], p["longitude"], q["latitude"], q["longitude"]) <= seuil_m


def _quasi_identiques(a: dict, b: dict) -> bool:
    """Deux trajets qui font pratiquement la même chose (mêmes modes, mêmes montées et descentes
    à moins de 500 m l'une de l'autre) = un seul trajet pour l'utilisateur."""
    ta, tb = _etapes_transport(a), _etapes_transport(b)
    if not ta or len(ta) != len(tb):
        return False
    for x, y in zip(ta, tb):
        if x.get("mode") != y.get("mode"):
            return False
        mem_dep = x.get("depart") == y.get("depart") or _pos_proches(x.get("depart_pos"), y.get("depart_pos"))
        mem_arr = x.get("arrivee") == y.get("arrivee") or _pos_proches(x.get("arrivee_pos"), y.get("arrivee_pos"))
        if not (mem_dep and mem_arr):
            return False
    return True


# choisir_propositions (Option 1 · bus X puis bus Y) 
def choisir_propositions(candidats: list, mode_filtre: str = "TOUS",
                          priorite=None, budget=None, maxi: int = 4) -> list:
    """Propositions logiques et distinctes :
    1. plus petit nombre de correspondances d'abord (jamais 3-4 bus si moins suffit) ;
    2. regroupe les trajets qui partagent la même montée et le même point de correspondance ;
    3. en mode TOUS, ajoute un autre mode UNIQUEMENT s'il reste compétitif (+50 % max)."""
    cands = [c for c in (candidats or []) if _etapes_transport(c)]
    if budget:
        cands = [c for c in cands if c.get("cout_partiel_fcfa", 0) <= budget]
    if not cands:
        return []
    # « je ne veux pas ce trajet » : on retire ce qui ressemble aux trajets déjà proposés
    deja_proposes = _EXCLURE.get()
    if deja_proposes:
        cands = [c for c in cands if not any(_quasi_identiques(c, p) for p in deja_proposes)]
        if not cands:
            return []

    def marche_finale(c):
        e = c["etapes"][-1]
        return e.get("distance_m", 0) if e.get("mode") == "MARCHE" else 0

    courtes = [c for c in cands if marche_finale(c) <= SEUIL_MARCHE_LONGUE_M]
    if courtes:
        cands = courtes

    def cle(c):
        base = c.get("nb_arrets", 0) + _acces_m(c) / 200.0
        if priorite == "prix":
            return (c.get("nb_correspondances", 0), c.get("cout_partiel_fcfa", 0), base,
                    c.get("duree_minutes", 999))
        if priorite == "duree":
            return (c.get("nb_correspondances", 0), c.get("duree_minutes", 999), base)
        return (c.get("nb_correspondances", 0), base, c.get("duree_minutes", 999))

    def famille(c):
        return frozenset(e["mode"] for e in _etapes_transport(c))

    def cle_groupe(c):
        tr = _etapes_transport(c)
        if len(tr) == 1:
            return ("D", tr[0]["depart"], tr[0]["arrivee"])
        # gbaka / woro-woro : pour l'utilisateur, deux lignes qui font les mêmes arrêts = le même trajet
        return tuple((e["mode"], "" if e["mode"] in ("GBAKA", "WORO_WORO") else _nom_court_ligne(e),
                      e["depart"], e["arrivee"]) for e in tr[:-1]) + (tr[-1]["depart"],)

    def grouper(liste):
        groupes = {}
        for c in sorted(liste, key=cle):
            groupes.setdefault(cle_groupe(c), []).append(c)
        tries = sorted(groupes.values(), key=lambda g: cle(g[0]))
        # fusionne les quasi-doublons : même trajet, juste une autre ligne ou un arrêt voisin
        fusion = []
        for g in tries:
            for k in fusion:
                if _quasi_identiques(k[0], g[0]):
                    k.extend(g)
                    break
            else:
                fusion.append(g)
        return fusion

    mini = min(c.get("nb_correspondances", 0) for c in cands)
    groupes = grouper([c for c in cands if c.get("nb_correspondances", 0) == mini])
    choisis = groupes[:maxi]

    # ── Diversité : UNIQUEMENT si le mode demandé = TOUS ──
    if mode_filtre == "TOUS" and choisis:
        deja = {m for g in choisis for m in famille(g[0])}
        dur_ref = choisis[0][0].get("duree_minutes", 999) or 999
        autres = grouper([c for c in cands if c.get("nb_correspondances", 0) <= mini + 1])
        for g in autres:
            if len(choisis) >= maxi:
                break
            if (famille(g[0]) - deja) and all(g is not x for x in choisis):
                # n'ajoute un autre mode que si le trajet n'est pas trop dégradé (max +50 %)
                if (g[0].get("duree_minutes", 999) or 999) <= dur_ref * 1.5:
                    choisis.append(g)
                    deja |= famille(g[0])
        if len(choisis) > maxi:
            choisis = choisis[:maxi]

    resultat = []
    for n, g in enumerate(choisis, 1):
        rep = copy.deepcopy(g[0])
        tr = _etapes_transport(rep)
        derniere = tr[-1]
        noms = []
        for e in tr:
            nom_l = _nom_court_ligne(e)
            if len(nom_l) > 26:
                nom_l = nom_l[:25].rstrip() + "…"
            noms.append(f"{_MODE_COURT.get(e['mode'], e['mode'].lower())} {nom_l}".strip())
        rep["libelle"] = f"Option {n} · " + " puis ".join(noms)
        nb_c = rep.get("nb_correspondances", 0)
        rep["resume"] = ("Trajet direct" if nb_c == 0 else
                         f"{nb_c} correspondance{'s' if nb_c > 1 else ''}") + \
                        f" · {rep.get('nb_arrets', 0)} arrêts"
        alternatives, vus = [], {_nom_court_ligne(derniere)}
        for autre in g[1:]:
            ta = _etapes_transport(autre)[-1]
            nom = _nom_court_ligne(ta)
            if nom in vus or ta.get("mode") != derniere.get("mode") or ta["arrivee"] != derniere["arrivee"]:
                continue
            vus.add(nom)
            alternatives.append({"ligne": nom, "mode": ta["mode"], "nb_arrets": ta.get("nb_arrets", 0)})
        alternatives.sort(key=lambda a: a["nb_arrets"])
        rep["alternatives"] = alternatives[:5]
        if alternatives:
            liste = ", ".join(f"{_MODE_COURT.get(a['mode'], a['mode'].lower())} {a['ligne']}"
                              + (f" ({a['nb_arrets']} arrêts)" if a["nb_arrets"] else "")
                              for a in alternatives[:5])
            derniere["texte"] = (derniere.get("texte") or "") + \
                f" Autres lignes possibles depuis le même arrêt : {liste}."
        resultat.append(rep)
    return resultat



def _normaliser_candidat(c: dict):
    for e in c.get("etapes", []):
        if e.get("mode") == "MARCHE":
            continue
        if e.get("nb_arrets") is None:
            e["nb_arrets"] = max(0, len(e["arrets"]) - 1) if e.get("arrets") else 0
        if e["mode"] in ("BUS", "BATEAU"):
            e["cout_fcfa"] = _prix_csv(e["mode"], e.get("numero_ligne"))
        else:
            e["cout_fcfa"] = None
    _finaliser_candidat(c)


def conseil_marche_chat(options: list) -> str:
    conseils, vus = [], set()
    for c in options or []:
        for i, e in enumerate(c.get("etapes", [])):
            if e.get("mode") != "MARCHE" or e.get("distance_m", 0) <= SEUIL_MARCHE_LONGUE_M:
                continue
            if not e.get("depart_pos") or not e.get("arrivee_pos"):
                continue
            cle = (e.get("depart"), e.get("arrivee"))
            if cle in vus:
                continue
            vus.add(cle)
            km = e["distance_m"] / 1000
            lignes = _lignes_libres_proches(e["depart_pos"], e["arrivee_pos"])
            if lignes:
                genre = "gbaka" if lignes[0][0] == "GBAKA" else "woro-woro"
                conseils.append(f"la marche de {km:.1f} km vers {e['arrivee']} est longue : "
                                f"le {genre} « {lignes[0][1]} » passe dans ce secteur "
                                f"(prix variable selon le chauffeur)")
            else:
                conseils.append(f"la marche de {km:.1f} km vers {e['arrivee']} est longue "
                                f"et je n'ai pas de gbaka/woro-woro connu sur ce tronçon : "
                                f"prévois de marcher")
            if len(conseils) >= 2:
                return " ; ".join(conseils)
    return " ; ".join(conseils)


def _signature_candidat(c: dict) -> tuple:
    return tuple((e.get("mode"), e.get("numero_ligne") or e.get("nom_ligne"),
                  e.get("depart"), e.get("arrivee"))
                 for e in c.get("etapes", []) if e.get("mode") != "MARCHE")


def candidats_multimodaux(dep: dict, dest: dict, mode_filtre: str = "TOUS", prix_max_ligne=None) -> list:
    """Mode imposé → UNIQUEMENT ce mode. Sinon : tous les modes combinés
    (bus + gbaka + bateau-bus + trajets mixtes)."""
    if mode_filtre in ("BUS", "GBAKA", "BATEAU"):
        autorises = {"BUS": {"BUS"}, "GBAKA": {"GBAKA", "WORO_WORO"}, "BATEAU": {"BATEAU"}}[mode_filtre]

        def modes_du_trajet(c):
            return {e["mode"] for e in c.get("etapes", []) if e["mode"] != "MARCHE"}

        out = construire_candidats(dep, dest, mode_filtre, prix_max_ligne)
        purs = [c for c in out if modes_du_trajet(c) <= autorises]
        if not purs:
            # aucun trajet 100 % dans ce mode : on cherche des trajets qui le COMBINENT avec un autre
            mixtes = construire_candidats(dep, dest, "TOUS", prix_max_ligne)
            purs = [c for c in mixtes if modes_du_trajet(c) & autorises and not modes_du_trajet(c) <= autorises]
            for c in purs:
                c["combo_mode"] = mode_filtre
        for c in purs:
            _ajouter_textes(c)
        return purs
    out, vus = [], set()
    groupes = [construire_candidats(dep, dest, "TOUS", prix_max_ligne)]
    par_mode = trajets_par_mode(dep, dest, prix_max_ligne)
    groupes += [par_mode.get("bus", []), par_mode.get("gbakas", []), par_mode.get("bateaux", [])]
    for groupe in groupes:
        for c in groupe:
            sig = _signature_candidat(c)
            if sig in vus:
                continue
            vus.add(sig)
            _ajouter_textes(c)
            out.append(c)
    return out


def arrets_proches_trajet(dep: dict, dest: dict, mode_filtre: str = "TOUS", rayon_m: int = 3000) -> dict:
    modes = None if mode_filtre == "TOUS" else [mode_filtre]
    arr_dep = arrets_proximite_par_modes(dep["latitude"], dep["longitude"], rayon_m, modes)
    arr_arr = arrets_proximite_par_modes(dest["latitude"], dest["longitude"], rayon_m, modes)
    return {"depart": arr_dep[:5], "arrivee": arr_arr[:5]}


def _prix_ligne(ligne) -> int:
    if ligne.get("type") not in ("BUS", "BATEAU"):
        return 0
    return _prix_csv(ligne["type"], ligne.get("numero")) or 0


def _lignes_passant_par(arret_id: str, modes_autorises: set) -> list:
    resultats = []
    approx = _rids_approximatifs()
    for rid in RESEAU["lignes_par_arret"].get(arret_id, []):
        if rid in approx:
            continue
        ligne = RESEAU["lignes"].get(rid)
        if not ligne or ligne["type"] not in modes_autorises:
            continue
        if ligne["type"] == "WORO_WORO":
            continue
        sequences = RESEAU["sequences"].get(rid, [])
        for seq in sequences:
            if arret_id in seq:
                resultats.append({"ligne": ligne, "rid": rid, "sequence": seq})
    return resultats


def _est_terminus_depart(rid: str, sid: str) -> bool:
    for seq in RESEAU["sequences"].get(rid, []):
        if seq and seq[0] == sid:
            return True
    return False


def _segment_sur_ligne(sequence: list, montee_id: str, descente_id: str):
    if montee_id not in sequence or descente_id not in sequence:
        return None
    i = sequence.index(montee_id)
    apres = [k for k, sid in enumerate(sequence) if sid == descente_id and k > i]
    if not apres:
        return None
    return sequence[i:apres[0] + 1]


# _arrets_utiles — équilibre PAR MODE 
def _arrets_utiles(arrets: list, modes: set, maxi: int = 60) -> list:
    """Garde les arrêts utiles pour le routage, en assurant une représentation
    de CHAQUE mode (sinon, entouré de bus, aucun gbaka / bateau-bus n'est retenu
    et les trajets multimodaux n'apparaissent jamais)."""
    modes_reels = {m for m in modes if m != "WORO_WORO"}
    if not modes_reels:
        return []

    par_mode = {m: [] for m in modes_reels}
    for a in arrets:
        types = set()
        for rid in RESEAU["lignes_par_arret"].get(a["id"], []):
            ligne = RESEAU["lignes"].get(rid)
            if ligne and ligne["type"] in modes_reels:
                types.add(ligne["type"])
        for t in types:
            par_mode[t].append(a)

    gardes, vus_ids, lignes_vues = [], set(), set()

    def ajouter(a):
        if a["id"] in vus_ids:
            return False
        nouvelles = {rid for rid in RESEAU["lignes_par_arret"].get(a["id"], [])
                     if (l := RESEAU["lignes"].get(rid))
                     and l["type"] in modes_reels and rid not in lignes_vues}
        if not nouvelles:
            return False
        gardes.append(a)
        vus_ids.add(a["id"])
        lignes_vues.update(nouvelles)
        return True

    quota = min(10, max(2, maxi // len(modes_reels)))
    for mode in modes_reels:
        count = 0
        for a in par_mode[mode]:
            if len(gardes) >= maxi:
                break
            if ajouter(a):
                count += 1
                if count >= quota:
                    break

    for a in arrets:
        if len(gardes) >= maxi:
            break
        ajouter(a)

    ordre = {a["id"]: i for i, a in enumerate(arrets)}
    gardes.sort(key=lambda a: ordre.get(a["id"], 10**9))
    return gardes


def _duree_sur_ligne(ids: list) -> int:
    total = 0.0
    for a, b in zip(ids, ids[1:]):
        p, q = RESEAU["arrets"].get(a), RESEAU["arrets"].get(b)
        if p and q:
            total += _dist_rapide(p["latitude"], p["longitude"], q["latitude"], q["longitude"])
    return max(5, int(total // 350))


ATTENTE_MINUTES = {"BUS": 8, "GBAKA": 3, "WORO_WORO": 3, "BATEAU": 15}
SEUIL_MARCHE_LONGUE_M = 1000


def _ajouter_attente(candidat: dict):
    total = 0
    for e in candidat["etapes"]:
        if e["mode"] != "MARCHE" and "attente_minutes" not in e:
            e["attente_minutes"] = ATTENTE_MINUTES.get(e["mode"], 5)
            total += e["attente_minutes"]
    candidat["duree_minutes"] += total


def _lignes_libres_proches(pos_a: dict, pos_b: dict) -> list:
    noms = []
    for c in CATALOGUE:
        if c["type"] not in ("GBAKA", "WORO_WORO") or c.get("estime"):
            continue
        trace, bbox = c.get("trace"), c.get("bbox")
        if not trace or not bbox:
            continue
        if _point_proche_du_trace(pos_a["latitude"], pos_a["longitude"], trace, bbox, 250) and \
           _point_proche_du_trace(pos_b["latitude"], pos_b["longitude"], trace, bbox, 500):
            noms.append((c["type"], f"{c['origine']} ↔ {c['destination']}"))
            if len(noms) >= 2:
                break
    return noms


def _ajouter_conseils_marche(candidat: dict):
    for position, e in enumerate(candidat["etapes"]):
        if e["mode"] != "MARCHE" or e.get("distance_m", 0) <= SEUIL_MARCHE_LONGUE_M:
            continue
        au_depart = position == 0
        km = e["distance_m"] / 1000
        lignes = _lignes_libres_proches(e["depart_pos"], e["arrivee_pos"])
        if lignes:
            ligne = lignes[0]
            genre = "gbaka" if ligne[0] == "GBAKA" else "woro-woro"
            if au_depart:
                e["conseil"] = (f"L'arrêt est à {km:.1f} km : prenez plutôt le {genre} « {ligne[1]} » "
                                f"qui passe près de vous pour le rejoindre.")
            else:
                e["conseil"] = (f"Il reste {km:.1f} km à parcourir : le {genre} « {ligne[1]} » "
                                f"dessert ce secteur si vous ne voulez pas marcher.")
        else:
            if au_depart:
                e["conseil"] = (f"L'arrêt est à {km:.1f} km : prenez plutôt un woro-woro, un gbaka "
                                f"ou un taxi pour le rejoindre.")
            else:
                e["conseil"] = (f"Il reste {km:.1f} km à parcourir : prenez un woro-woro, un gbaka "
                                f"ou un taxi si vous ne voulez pas marcher.")


def _variante_woro_si_longue_marche(candidat: dict):
    etapes = candidat.get("etapes", [])
    if not etapes or etapes[0].get("mode") != "MARCHE":
        return None
    marche = etapes[0]
    if marche.get("distance_m", 0) <= SEUIL_MARCHE_LONGUE_M:
        return None
    variante = copy.deepcopy(candidat)
    marche_remplacee = variante["etapes"].pop(0)
    duree_woro = max(3, marche_remplacee["distance_m"] // 400)
    cout_woro = 0
    wait_woro = ATTENTE_MINUTES.get("WORO_WORO", 3)
    etape_woro = {
        "mode": "WORO_WORO", "depart": marche_remplacee["depart"],
        "arrivee": marche_remplacee["arrivee"],
        "depart_pos": marche_remplacee["depart_pos"],
        "arrivee_pos": marche_remplacee["arrivee_pos"],
        "distance_m": marche_remplacee["distance_m"],
        "duree_minutes": duree_woro, "cout_fcfa": cout_woro,
        "attente_minutes": wait_woro,
    }
    variante["etapes"].insert(0, etape_woro)
    variante["duree_minutes"] = (variante["duree_minutes"]
                                 - marche_remplacee.get("duree_minutes", 0)
                                 + duree_woro + wait_woro)
    variante["cout_fcfa"] = variante.get("cout_fcfa", 0) + cout_woro
    variante["marche_m"] = max(0, variante.get("marche_m", 0) - marche_remplacee["distance_m"])
    variante["variante"] = "woro_remplace_marche"
    return variante


def _grille_correspondance():
    if "grille_corr" not in RESEAU:
        grille = defaultdict(list)
        for sid, a in RESEAU["arrets"].items():
            if RESEAU["lignes_par_arret"].get(sid) and sid not in RESEAU["points_libres"]:
                grille[_cellule(a["latitude"], a["longitude"], 0.003)].append(sid)
        RESEAU["grille_corr"] = grille
    return RESEAU["grille_corr"]


def _arrets_proches_de(sid: str, rayon_m: float = 300) -> list:
    a = RESEAU["arrets"][sid]
    grille = _grille_correspondance()
    ci, cj = _cellule(a["latitude"], a["longitude"], 0.003)
    res = []
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            for autre in grille.get((ci + di, cj + dj), ()):
                b = RESEAU["arrets"][autre]
                d = _dist_rapide(a["latitude"], a["longitude"], b["latitude"], b["longitude"])
                if d <= rayon_m:
                    res.append((autre, d))
    return res


def _lignes_voisines() -> dict:
    if "voisines" not in RESEAU:
        voisines = defaultdict(set)
        approx = _rids_approximatifs()
        for rid, sequences in RESEAU["sequences"].items():
            if rid in approx or RESEAU["lignes"].get(rid, {}).get("type") == "WORO_WORO":
                continue
            for sid in [x for x in sequences[0] if x not in RESEAU["points_libres"]][::2]:
                for autre, _ in _arrets_proches_de(sid, 250):
                    for rid2 in RESEAU["lignes_par_arret"].get(autre, []):
                        if (rid2 != rid and rid2 not in approx
                                and RESEAU["lignes"].get(rid2, {}).get("type") != "WORO_WORO"):
                            voisines[rid].add(rid2)
        RESEAU["voisines"] = voisines
    return RESEAU["voisines"]


def _etape_marche(depart_nom, arrivee_nom, p, q, distance):
    return {
        "mode": "MARCHE", "depart": depart_nom, "arrivee": arrivee_nom,
        "distance_m": int(distance), "duree_minutes": max(1, int(distance) // 80),
        "cout_fcfa": 0,
        "depart_pos": {"latitude": p["latitude"], "longitude": p["longitude"]},
        "arrivee_pos": {"latitude": q["latitude"], "longitude": q["longitude"]},
    }


def _etape_ligne(ligne, rid, ids):
    m, d = RESEAU["arrets"][ids[0]], RESEAU["arrets"][ids[-1]]
    return {
        "mode": ligne["type"], "numero_ligne": ligne.get("numero"),
        "nom_ligne": ligne.get("nom"),
        "depart": m["nom"], "arrivee": d["nom"],
        "depart_pos": {"latitude": m["latitude"], "longitude": m["longitude"]},
        "arrivee_pos": {"latitude": d["latitude"], "longitude": d["longitude"]},
        "distance_m": distance_m(m["latitude"], m["longitude"], d["latitude"], d["longitude"]),
        "duree_minutes": _duree_sur_ligne(ids), "cout_fcfa": _prix_ligne(ligne),
        "nb_arrets": len(ids) - 1,
    }


def _meilleur_transfert(seq_a, apres_a, seq_b, avant_b, rayon=300, ligne_b_bus=False):
    pos_b = {}
    for k, sid in enumerate(seq_b):
        if k <= avant_b and sid not in pos_b:
            pos_b[sid] = k
    rayon_max = max(rayon, 800) if ligne_b_bus else rayon
    meilleur = None
    for i in range(apres_a + 1, len(seq_a)):
        sid_a = seq_a[i]
        if sid_a in RESEAU["points_libres"]:
            continue
        for autre, d in _arrets_proches_de(sid_a, rayon_max):
            k = pos_b.get(autre)
            if k is None:
                continue
            terminus = ligne_b_bus and k == 0
            if d > rayon and not (terminus and d <= 800):
                continue
            score = (i - apres_a) + (avant_b - k) + d / 150.0 - (4 if terminus else 0)
            if not meilleur or score < meilleur[0]:
                meilleur = (score, i, k, d, sid_a, autre)
    if not meilleur:
        return None
    _, i, k, d, sid_a, autre = meilleur
    return (i, k, d, sid_a, autre)


def _candidats_deux_correspondances(dep, dest, modes, arrets_dep, arrets_arr, prix_max_ligne=None) -> list:
    voisines = _lignes_voisines()
    perm = lambda rid: RESEAU["lignes"].get(rid, {}).get("type") in modes
    lignes_dep, lignes_arr = {}, {}
    for a in arrets_dep:
        for info in _lignes_passant_par(a["id"], modes):
            lignes_dep.setdefault(info["rid"], []).append((a, info))
    for a in arrets_arr:
        for info in _lignes_passant_par(a["id"], modes):
            lignes_arr.setdefault(info["rid"], []).append((a, info))

    resultats = {}
    triples = 0
    for r1, departs in lignes_dep.items():
        for r2 in voisines.get(r1, ()):
            if r2 == r1 or not perm(r2):
                continue
            for r3 in voisines.get(r2, ()):
                if r3 in (r1, r2) or r3 not in lignes_arr:
                    continue
                triples += 1
                if triples > 400:
                    return list(resultats.values())
                l1, l2, l3 = RESEAU["lignes"][r1], RESEAU["lignes"][r2], RESEAU["lignes"][r3]
                if prix_max_ligne and max(_prix_ligne(l1), _prix_ligne(l2), _prix_ligne(l3)) > prix_max_ligne:
                    continue
                a_dep, info1 = min(departs, key=lambda x: x[0]["distance_m"])
                seq1 = info1["sequence"]
                i_mont = seq1.index(a_dep["id"])
                for seq2 in RESEAU["sequences"][r2]:
                    for a_arr, info3 in sorted(lignes_arr[r3], key=lambda x: x[0]["distance_m"])[:3]:
                        seq3 = info3["sequence"]
                        k_desc = seq3.index(a_arr["id"])
                        t1 = _meilleur_transfert(seq1, i_mont, seq2, len(seq2) - 2)
                        if not t1:
                            continue
                        i_d1, j_m2, d12, s_d1, s_m2 = t1
                        t2 = _meilleur_transfert(seq2, j_m2, seq3, k_desc - 1)
                        if not t2:
                            continue
                        j_d2, k_m3, d23, s_d2, s_m3 = t2
                        ids1, ids2, ids3 = seq1[i_mont:i_d1 + 1], seq2[j_m2:j_d2 + 1], seq3[k_m3:k_desc + 1]
                        if len(ids1) < 2 or len(ids2) < 2 or len(ids3) < 2:
                            continue
                        A = lambda sid: RESEAU["arrets"][sid]
                        trop_court = False
                        for ids in (ids1, ids2, ids3):
                            if _dist_rapide(A(ids[0])["latitude"], A(ids[0])["longitude"],
                                            A(ids[-1])["latitude"], A(ids[-1])["longitude"]) < 1500:
                                trop_court = True
                        if trop_court:
                            continue
                        etapes = [
                            _etape_marche(dep.get("nom", "depart"), A(ids1[0])["nom"],
                                          dep, A(ids1[0]), a_dep["distance_m"]),
                            _etape_ligne(l1, r1, ids1),
                            _etape_marche(A(s_d1)["nom"], A(s_m2)["nom"], A(s_d1), A(s_m2), d12),
                            _etape_ligne(l2, r2, ids2),
                            _etape_marche(A(s_d2)["nom"], A(s_m3)["nom"], A(s_d2), A(s_m3), d23),
                            _etape_ligne(l3, r3, ids3),
                            _etape_marche(A(ids3[-1])["nom"], dest.get("nom", "destination"),
                                          A(ids3[-1]), dest, a_arr["distance_m"]),
                        ]
                        for e in (etapes[2], etapes[4]):
                            e["duree_minutes"] = max(2, e["duree_minutes"])
                        cand = {
                            "duree_minutes": sum(e["duree_minutes"] for e in etapes),
                            "cout_fcfa": sum(e["cout_fcfa"] for e in etapes),
                            "marche_m": sum(e["distance_m"] for e in etapes if e["mode"] == "MARCHE"),
                            "nb_correspondances": 2,
                            "etapes": etapes, "arrets_itineraire": [],
                        }
                        cle = (r1, r2, r3)
                        if cle not in resultats or _score_trajet(cand) < _score_trajet(resultats[cle]):
                            resultats[cle] = cand
    return list(resultats.values())


def _score_trajet(c: dict) -> int:
    return c["duree_minutes"] + c.get("marche_m", 0) // 160


def construire_candidats(dep: dict, dest: dict, mode_filtre: str = "TOUS", prix_max_ligne=None) -> List[dict]:
    for rayon in (1500, 2500, 4000):
        resultats = _construire_candidats_rayon(dep, dest, mode_filtre, prix_max_ligne, rayon)
        if resultats:
            for c in resultats:
                _normaliser_candidat(c)
            return resultats
    return []


DESCENTE_MAX_GBAKA_M = 500


def _filtrer_trajets_gbaka(candidats: list) -> list:
    def marche_finale(c):
        return c["etapes"][-1].get("distance_m", 0) if c["etapes"][-1]["mode"] == "MARCHE" else 0

    def marche_entre(c):
        return max([e.get("distance_m", 0) for e in c["etapes"][1:-1] if e["mode"] == "MARCHE"] or [0])

    def detour(c):
        return any(e["mode"] == "GBAKA" and e["duree_minutes"] * 350 > 2.2 * e["distance_m"] + 1500
                   for e in c["etapes"])

    bons = [c for c in candidats if marche_entre(c) <= 250 and not detour(c)]
    if any(marche_finale(c) <= 800 for c in bons):
        bons = [c for c in bons if marche_finale(c) <= 800]
    bons.sort(key=lambda c: (c["nb_correspondances"], _score_trajet(c)))
    uniques, vus = [], set()
    for c in bons:
        dernier = [e for e in c["etapes"] if e["mode"] != "MARCHE"][-1]
        cle = tuple((e.get("mode"), e.get("numero_ligne") or e.get("nom_ligne"),
                     e.get("depart"), e.get("arrivee"))
                    for e in c["etapes"] if e.get("mode") != "MARCHE")
        if cle in vus:
            continue
        vus.add(cle)
        uniques.append(c)
    return uniques


#  (descente logique + limits)
def _construire_candidats_rayon(dep: dict, dest: dict, mode_filtre: str,
                                 prix_max_ligne, rayon: int) -> List[dict]:
    if not RESEAU["charge"]:
        charger_reseau_local()

    modes = {"BUS", "GBAKA", "BATEAU"}
    if mode_filtre == "BUS":
        modes = {"BUS"}
    elif mode_filtre == "GBAKA":
        modes = {"GBAKA"}
    elif mode_filtre == "BATEAU":
        modes = {"BATEAU"}

    arrets_dep_proximite = arrets_proximite(
        dep["latitude"], dep["longitude"], rayon, tri_par_distance=True
    )
    arrets_dep = []
    modes_depart_vus = set()
    for arret in arrets_dep_proximite:
        modes_arret = {
            RESEAU["lignes"].get(rid, {}).get("type")
            for rid in RESEAU["lignes_par_arret"].get(arret["id"], [])
        } & modes
        if not modes_arret - modes_depart_vus:
            continue
        arrets_dep.append(arret)
        modes_depart_vus.update(modes_arret)

    arret_depart_prioritaire = arrets_dep[0] if arrets_dep else None
    arrets_arr_proximite = arrets_proximite(
        dest["latitude"], dest["longitude"], rayon, tri_par_distance=True
    )
    arrets_arr = _arrets_utiles(arrets_arr_proximite, modes)

    lignes_d = {r for a in arrets_dep for r in RESEAU["lignes_par_arret"].get(a["id"], [])}
    lignes_a = {r for a in arrets_arr for r in RESEAU["lignes_par_arret"].get(a["id"], [])}
    log.info("[trajets v2] mode=%s rayon=%s arrets_dep=%d arrets_arr=%d lignes_dep=%d lignes_arr=%d en_commun=%d",
             mode_filtre, rayon, len(arrets_dep), len(arrets_arr), len(lignes_d), len(lignes_a),
             len(lignes_d & lignes_a))
    if not arrets_dep or not arrets_arr:
        return []

    candidats = []
    vus = set()
    meilleur_direct = {}

    # Gbaka / woro-woro : on descend à moins de 500 m de la destination ; si aucun arrêt de gbaka
    # n'est aussi proche, on prend le plus proche qui existe (+100 m de tolérance).
    descente_max_gbaka = DESCENTE_MAX_GBAKA_M
    if modes & {"GBAKA", "WORO_WORO"}:
        d_min = min((distance_m(dest["latitude"], dest["longitude"], ar["latitude"], ar["longitude"])
                     for ar in RESEAU["arrets"].values()
                     if any(RESEAU["lignes"].get(rid, {}).get("type") == "GBAKA"
                            for rid in RESEAU["lignes_par_arret"].get(ar["id"], []))), default=0)
        descente_max_gbaka = max(DESCENTE_MAX_GBAKA_M, int(d_min) + 100)

    # ── DIRECT : descente à l'arrêt de la ligne le plus proche de la destination ──
    # Boarding always at the stop on the line closest to the user (avoids skipping nearer stops)
    # Gbaka / woro-woro : on regarde tous les arrêts proches (pas seulement le plus proche),
    # et pour chaque ligne on monte à SON arrêt le plus proche de l'utilisateur.
    ids_dep_prioritaires = {a["id"] for a in arrets_dep}
    arrets_dep_direct = list(arrets_dep)
    if modes & {"GBAKA", "WORO_WORO"}:
        arrets_dep_direct += [a for a in _arrets_utiles(arrets_dep_proximite, modes)
                              if a["id"] not in ids_dep_prioritaires]
        arrets_dep_direct.sort(key=lambda a: a.get("distance_m", 0))
    sens_deja_pris = set()
    for a_dep in arrets_dep_direct:
        for info in _lignes_passant_par(a_dep["id"], modes):
            ligne = info["ligne"]
            seq = info["sequence"]
            if ligne["type"] in ("GBAKA", "WORO_WORO"):
                cle_sens = (info["rid"], tuple(seq))
                if cle_sens in sens_deja_pris:
                    continue
                sens_deja_pris.add(cle_sens)
            elif a_dep["id"] not in ids_dep_prioritaires:
                continue
            prix = _prix_ligne(ligne)
            if prix_max_ligne and prix > prix_max_ligne:
                continue

            montee_sid = a_dep["id"]
            if montee_sid not in seq:
                continue
            montee = RESEAU["arrets"][montee_sid]
            montee_d = distance_m(dep["latitude"], dep["longitude"],
                                  montee["latitude"], montee["longitude"])
            if montee_d > rayon:
                continue
            idx_montee = seq.index(montee_sid)

            meilleur_sid, meilleure_d = None, float("inf")
            # descente : arrêt de la ligne (après la montée) le plus proche de la destination
            if modes == {"GBAKA"}:
                for sid in seq[idx_montee + 1:]:
                    ar = RESEAU["arrets"].get(sid)
                    if not ar:
                        continue
                    d = distance_m(dest["latitude"], dest["longitude"],
                                   ar["latitude"], ar["longitude"])
                    if d < meilleure_d:
                        meilleure_d, meilleur_sid = d, sid
            for sid in seq[idx_montee + 1:]:
                if modes == {"GBAKA"} and meilleur_sid is not None:
                    break
                ar = RESEAU["arrets"].get(sid)
                if not ar:
                    continue
                d = distance_m(dest["latitude"], dest["longitude"], ar["latitude"], ar["longitude"])
                if d < meilleure_d:
                    meilleure_d, meilleur_sid = d, sid
            if not meilleur_sid or meilleure_d > rayon:
                continue

            cle = ("D", info["rid"], montee_sid, meilleur_sid)
            if cle in vus:
                continue
            vus.add(cle)

            segment = seq[idx_montee:seq.index(meilleur_sid) + 1]
            sequence_arrets = []
            for sid in segment:
                ar = RESEAU["arrets"].get(sid)
                if ar:
                    sequence_arrets.append({"id": ar["id"], "nom": ar["nom"],
                                            "latitude": ar["latitude"], "longitude": ar["longitude"]})

            descente = RESEAU["arrets"][meilleur_sid]
            duree_transport = _duree_sur_ligne(segment)

            marche_debut = {
                "mode": "MARCHE", "depart": dep.get("nom", "depart"), "arrivee": montee["nom"],
                "distance_m": int(montee_d),
                "duree_minutes": max(1, int(montee_d) // 80),
                "cout_fcfa": 0,
                "depart_pos": {"latitude": dep["latitude"], "longitude": dep["longitude"]},
                "arrivee_pos": {"latitude": montee["latitude"], "longitude": montee["longitude"]},
            }
            etape = {
                "mode": ligne["type"], "numero_ligne": ligne.get("numero"),
                "nom_ligne": ligne.get("nom"),
                "depart": montee["nom"], "arrivee": descente["nom"],
                "depart_pos": {"latitude": montee["latitude"], "longitude": montee["longitude"]},
                "arrivee_pos": {"latitude": descente["latitude"], "longitude": descente["longitude"]},
                "distance_m": distance_m(montee["latitude"], montee["longitude"],
                                          descente["latitude"], descente["longitude"]),
                "duree_minutes": duree_transport,
                "cout_fcfa": prix, "arrets": sequence_arrets,
            }
            marche_fin = {
                "mode": "MARCHE", "depart": descente["nom"],
                "arrivee": dest.get("nom", "destination"),
                "distance_m": int(meilleure_d),
                "duree_minutes": max(1, int(meilleure_d) // 80),
                "cout_fcfa": 0,
                "depart_pos": {"latitude": descente["latitude"], "longitude": descente["longitude"]},
                "arrivee_pos": {"latitude": dest["latitude"], "longitude": dest["longitude"]},
            }

            nouveau = {
                "duree_minutes": marche_debut["duree_minutes"] + duree_transport + marche_fin["duree_minutes"],
                "cout_fcfa": prix,
                "marche_m": int(montee_d) + int(meilleure_d),
                "nb_correspondances": 0,
                "etapes": [marche_debut, etape, marche_fin],
                "arrets_itineraire": sequence_arrets,
            }
            ancien = meilleur_direct.get(info["rid"])
            if not ancien or _score_trajet(nouveau) < _score_trajet(ancien):
                meilleur_direct[info["rid"]] = nouveau

    candidats.extend(meilleur_direct.values())

    # ── 1 CORRESPONDANCE : élargi à 20 arrêts de chaque côté, 25 paires max ──
    DIST_CORRESPONDANCE_M = 250 if modes == {"GBAKA"} else 500
    nb_corr = 0
    meilleur_corr = {}

    # gbaka : arrêts de descente proches de la destination (tous les identifiants, pas un seul par nom)
    arrets_arr_corr = list(arrets_arr)
    if modes & {"GBAKA"}:
        proches_gbaka = []
        for ar in RESEAU["arrets"].values():
            d = distance_m(dest["latitude"], dest["longitude"], ar["latitude"], ar["longitude"])
            if d > descente_max_gbaka:
                continue
            if any(RESEAU["lignes"].get(rid, {}).get("type") == "GBAKA"
                   for rid in RESEAU["lignes_par_arret"].get(ar["id"], [])):
                proches_gbaka.append({"id": ar["id"], "nom": ar["nom"], "latitude": ar["latitude"],
                                      "longitude": ar["longitude"], "distance_m": int(d)})
        proches_gbaka.sort(key=lambda a: a["distance_m"])
        ids_gb = {a["id"] for a in proches_gbaka}
        arrets_arr_corr = proches_gbaka + [a for a in arrets_arr if a["id"] not in ids_gb]

    sens_corr_pris = set()
    for a_dep in arrets_dep_direct[:40]:
        if nb_corr >= 25:
            break
        for info1 in _lignes_passant_par(a_dep["id"], modes):
            if nb_corr >= 25:
                break
            ligne1 = info1["ligne"]
            seq1 = info1["sequence"]
            if ligne1["type"] in ("GBAKA", "WORO_WORO"):
                cle_sens = (info1["rid"], tuple(seq1))
                if cle_sens in sens_corr_pris:
                    continue
                sens_corr_pris.add(cle_sens)
            elif a_dep["id"] not in ids_dep_prioritaires:
                continue
            prix1 = _prix_ligne(ligne1)

            montee_sid = a_dep["id"]
            if montee_sid not in seq1:
                continue
            montee1 = RESEAU["arrets"][montee_sid]
            montee_d = distance_m(dep["latitude"], dep["longitude"],
                                  montee1["latitude"], montee1["longitude"])
            if montee_d > rayon:
                continue
            idx_montee = seq1.index(montee_sid)

            for a_arr in arrets_arr_corr[:20]:
                for info2 in _lignes_passant_par(a_arr["id"], modes):
                    ligne2 = info2["ligne"]
                    if ligne2["id"] == ligne1["id"]:
                        continue
                    seq2 = info2["sequence"]
                    if a_arr["id"] not in seq2:
                        continue
                    idx_arr = seq2.index(a_arr["id"])
                    t = _meilleur_transfert(seq1, idx_montee, seq2, idx_arr - 1,
                                            rayon=DIST_CORRESPONDANCE_M,
                                            ligne_b_bus=(ligne2["type"] == "BUS"))
                    meilleur_transfert = (t[3], t[4], t[2]) if t else None
                    if not meilleur_transfert:
                        continue
                    sid_d1, sid_m2, dist_corres = meilleur_transfert
                    ar_d1 = RESEAU["arrets"][sid_d1]
                    ar_m2 = RESEAU["arrets"][sid_m2]
                    descente2 = RESEAU["arrets"][a_arr["id"]]
                    nb1 = seq1.index(sid_d1) - idx_montee
                    nb2 = idx_arr - seq2.index(sid_m2)
                    if nb1 < 1 or nb2 < 1:
                        continue
                    prix2 = _prix_ligne(ligne2)
                    duree1 = _duree_sur_ligne(seq1[idx_montee:idx_montee + nb1 + 1])
                    duree2 = _duree_sur_ligne(seq2[idx_arr - nb2:idx_arr + 1])
                    duree_corres = max(2, int(dist_corres) // 80)

                    cle = ("C", ligne1["id"], ligne2["id"], montee_sid, sid_d1, sid_m2, a_arr["id"])
                    if cle in vus:
                        continue
                    vus.add(cle)

                    marche_debut = {
                        "mode": "MARCHE", "depart": dep.get("nom", "depart"),
                        "arrivee": montee1["nom"],
                        "distance_m": int(montee_d),
                        "duree_minutes": max(1, int(montee_d) // 80),
                        "cout_fcfa": 0,
                        "depart_pos": {"latitude": dep["latitude"], "longitude": dep["longitude"]},
                        "arrivee_pos": {"latitude": montee1["latitude"], "longitude": montee1["longitude"]},
                    }
                    etape1 = {
                        "mode": ligne1["type"], "numero_ligne": ligne1.get("numero"),
                        "nom_ligne": ligne1.get("nom"),
                        "depart": montee1["nom"], "arrivee": ar_d1["nom"],
                        "depart_pos": {"latitude": montee1["latitude"], "longitude": montee1["longitude"]},
                        "arrivee_pos": {"latitude": ar_d1["latitude"], "longitude": ar_d1["longitude"]},
                        "distance_m": distance_m(montee1["latitude"], montee1["longitude"],
                                                  ar_d1["latitude"], ar_d1["longitude"]),
                        "duree_minutes": duree1, "cout_fcfa": prix1, "nb_arrets": nb1,
                    }
                    marche_corres = {
                        "mode": "MARCHE", "depart": ar_d1["nom"], "arrivee": ar_m2["nom"],
                        "distance_m": int(dist_corres), "duree_minutes": duree_corres,
                        "cout_fcfa": 0,
                        "depart_pos": {"latitude": ar_d1["latitude"], "longitude": ar_d1["longitude"]},
                        "arrivee_pos": {"latitude": ar_m2["latitude"], "longitude": ar_m2["longitude"]},
                    }
                    etape2 = {
                        "mode": ligne2["type"], "numero_ligne": ligne2.get("numero"),
                        "nom_ligne": ligne2.get("nom"),
                        "depart": ar_m2["nom"], "arrivee": descente2["nom"],
                        "depart_pos": {"latitude": ar_m2["latitude"], "longitude": ar_m2["longitude"]},
                        "arrivee_pos": {"latitude": descente2["latitude"], "longitude": descente2["longitude"]},
                        "distance_m": distance_m(ar_m2["latitude"], ar_m2["longitude"],
                                                  descente2["latitude"], descente2["longitude"]),
                        "duree_minutes": duree2, "cout_fcfa": prix2, "nb_arrets": nb2,
                    }
                    marche_fin = {
                        "mode": "MARCHE", "depart": descente2["nom"],
                        "arrivee": dest.get("nom", "destination"),
                        "distance_m": a_arr["distance_m"],
                        "duree_minutes": max(1, a_arr["distance_m"] // 80),
                        "cout_fcfa": 0,
                        "depart_pos": {"latitude": descente2["latitude"], "longitude": descente2["longitude"]},
                        "arrivee_pos": {"latitude": dest["latitude"], "longitude": dest["longitude"]},
                    }
                    nouveau = {
                        "duree_minutes": (marche_debut["duree_minutes"] + duree1
                                          + duree_corres + duree2 + marche_fin["duree_minutes"]),
                        "cout_fcfa": prix1 + prix2,
                        "marche_m": int(montee_d) + int(dist_corres) + a_arr["distance_m"],
                        "nb_correspondances": 1,
                        "etapes": [marche_debut, etape1, marche_corres, etape2, marche_fin],
                        "arrets_itineraire": [],
                    }
                    paire = (ligne1["id"], ligne2["id"])
                    ancien = meilleur_corr.get(paire)
                    if not ancien or _score_trajet(nouveau) < _score_trajet(ancien):
                        meilleur_corr[paire] = nouveau
                    nb_corr = len(meilleur_corr)
                    if nb_corr >= 25:
                        break

    candidats.extend(meilleur_corr.values())

    if not any(c["etapes"][0].get("distance_m", 0) <= SEUIL_MARCHE_LONGUE_M for c in candidats):
        candidats.extend(_candidats_deux_correspondances(dep, dest, modes, arrets_dep, arrets_arr, prix_max_ligne))

    if not candidats:
        for c in CATALOGUE:
            if c["type"] not in modes:
                continue
            ori = nettoyer_texte(c.get("origine") or "")
            des = nettoyer_texte(c.get("destination") or "")
            if not ori or not des:
                continue
            match_dep = False
            montee = None
            for a in arrets_dep:
                nom = nettoyer_texte(a["nom"])
                if any(m in nom for m in ori.split() if len(m) > 3) or \
                   any(m in ori for m in nom.split() if len(m) > 3):
                    match_dep = True
                    montee = a
                    break
            match_arr = False
            descente = None
            for a in arrets_arr:
                nom = nettoyer_texte(a["nom"])
                if any(m in nom for m in des.split() if len(m) > 3) or \
                   any(m in des for m in nom.split() if len(m) > 3):
                    match_arr = True
                    descente = a
                    break
            if not (match_dep and match_arr):
                for a in arrets_dep:
                    nom = nettoyer_texte(a["nom"])
                    if any(m in nom for m in des.split() if len(m) > 3) or \
                       any(m in des for m in nom.split() if len(m) > 3):
                        match_dep = True
                        montee = a
                        break
                for a in arrets_arr:
                    nom = nettoyer_texte(a["nom"])
                    if any(m in nom for m in ori.split() if len(m) > 3) or \
                       any(m in ori for m in nom.split() if len(m) > 3):
                        match_arr = True
                        descente = a
                        break
            if not (match_dep and match_arr and montee and descente):
                continue
            trace = c.get("trace") or []
            bbox = c.get("bbox")
            if not trace or not bbox:
                continue
            if not _point_proche_du_trace(montee["latitude"], montee["longitude"], trace, bbox):
                continue
            if not _point_proche_du_trace(descente["latitude"], descente["longitude"], trace, bbox):
                continue
            prix = c.get("prix_fcfa") or _prix_ligne({"numero": c.get("numero"), "type": c["type"]})
            duree = 40
            marche_debut = {
                "mode": "MARCHE", "depart": dep.get("nom", "depart"), "arrivee": montee["nom"],
                "distance_m": montee["distance_m"],
                "duree_minutes": max(1, montee["distance_m"] // 80),
                "cout_fcfa": 0,
                "depart_pos": {"latitude": dep["latitude"], "longitude": dep["longitude"]},
                "arrivee_pos": {"latitude": montee["latitude"], "longitude": montee["longitude"]},
            }
            etape = {
                "mode": c["type"], "numero_ligne": c.get("numero"),
                "nom_ligne": f"{c['origine']} ↔ {c['destination']}",
                "depart": montee["nom"], "arrivee": descente["nom"],
                "depart_pos": {"latitude": montee["latitude"], "longitude": montee["longitude"]},
                "arrivee_pos": {"latitude": descente["latitude"], "longitude": descente["longitude"]},
                "distance_m": distance_m(montee["latitude"], montee["longitude"],
                                          descente["latitude"], descente["longitude"]),
                "duree_minutes": duree, "cout_fcfa": prix,
            }
            marche_fin = {
                "mode": "MARCHE", "depart": descente["nom"],
                "arrivee": dest.get("nom", "destination"),
                "distance_m": descente["distance_m"],
                "duree_minutes": max(1, descente["distance_m"] // 80),
                "cout_fcfa": 0,
                "depart_pos": {"latitude": descente["latitude"], "longitude": descente["longitude"]},
                "arrivee_pos": {"latitude": dest["latitude"], "longitude": dest["longitude"]},
            }
            candidats.append({
                "duree_minutes": marche_debut["duree_minutes"] + duree + marche_fin["duree_minutes"],
                "cout_fcfa": prix,
                "marche_m": montee["distance_m"] + descente["distance_m"],
                "nb_correspondances": 0,
                "etapes": [marche_debut, etape, marche_fin],
                "arrets_itineraire": [],
            })
            if len(candidats) >= 5:
                break

    # Gbaka / woro-woro : on descend à moins de 500 m de la destination (sinon autre gbaka / correspondance)
    if modes & {"GBAKA", "WORO_WORO"}:
        def _descente_trop_loin(c):
            transports = [e for e in c["etapes"] if e["mode"] != "MARCHE"]
            fin = c["etapes"][-1]
            return bool(transports) and transports[-1]["mode"] in ("GBAKA", "WORO_WORO") \
                and fin["mode"] == "MARCHE" and fin.get("distance_m", 0) > descente_max_gbaka
        candidats = [c for c in candidats if not _descente_trop_loin(c)]

    if modes == {"GBAKA"}:
        candidats = _filtrer_trajets_gbaka(candidats)
    for c in candidats:
        _ajouter_attente(c)

    variantes_woro = []
    for c in list(candidats):
        if arret_depart_prioritaire:
            premier_transport = next(
                (e for e in c.get("etapes", []) if e.get("mode") != "MARCHE"),
                None,
            )
            if not premier_transport:
                continue
            depart_pos = premier_transport.get("depart_pos") or {}
            if depart_pos.get("latitude") is None or depart_pos.get("longitude") is None:
                continue
            if distance_m(
                arret_depart_prioritaire["latitude"],
                arret_depart_prioritaire["longitude"],
                depart_pos.get("latitude"),
                depart_pos.get("longitude"),
            ) > 100:
                continue
        v = _variante_woro_si_longue_marche(c)
        if v:
            variantes_woro.append(v)
    candidats.extend(variantes_woro)

    par_vehicules = sorted(candidats, key=lambda c: (c["nb_correspondances"], _score_trajet(c), c["cout_fcfa"]))
    par_marche = sorted(candidats, key=lambda c: (
        c["nb_correspondances"] + (1 if c["etapes"][0].get("distance_m", 0) > SEUIL_MARCHE_LONGUE_M else 0),
        _score_trajet(c), c["cout_fcfa"]))
    candidats = []
    for a, b in zip(par_vehicules, par_marche):
        candidats.append(a)
        candidats.append(b)
    uniques, vus_trajets = [], set()
    for c in candidats:
        signature = tuple((e.get("numero_ligne") or e.get("nom_ligne"), e["depart"], e["arrivee"])
                          for e in c["etapes"] if e["mode"] != "MARCHE")
        if signature in vus_trajets:
            continue
        vus_trajets.add(signature)
        uniques.append(c)
    candidats = uniques[:10]
    for c in candidats:
        c["etapes"] = [e for i, e in enumerate(c["etapes"])
                       if not (e["mode"] == "MARCHE" and 0 < i < len(c["etapes"]) - 1
                               and e.get("distance_m", 0) < 30)]
    for c in candidats:
        _ajouter_conseils_marche(c)
    return candidats


def trajets_par_mode(dep: dict, dest: dict, prix_max_ligne=None) -> dict:
    return {
        "bus":     construire_candidats(dep, dest, "BUS", prix_max_ligne)[:5],
        "gbakas":  construire_candidats(dep, dest, "GBAKA", prix_max_ligne)[:5],
        "bateaux": construire_candidats(dep, dest, "BATEAU", prix_max_ligne)[:5],
    }


def passages_gbaka_pres_de(lat: float, lon: float, rayon_m: int = 800) -> List[dict]:
    if not RESEAU["charge"]:
        charger_reseau_local()
    resultats = []
    for rid, sequences in RESEAU["sequences"].items():
        ligne = RESEAU["lignes"].get(rid)
        if not ligne or ligne["type"] not in ("GBAKA", "WORO_WORO"):
            continue
        for idx_sens, seq in enumerate(sequences):
            sens_label = "aller" if idx_sens == 0 else "retour"
            for sid in seq:
                arret = RESEAU["arrets"].get(sid)
                if not arret:
                    continue
                d = distance_m(lat, lon, arret["latitude"], arret["longitude"])
                if d <= rayon_m:
                    resultats.append({
                        "arret_id": sid, "arret_nom": arret["nom"],
                        "latitude": arret["latitude"], "longitude": arret["longitude"],
                        "distance_m": int(d),
                        "ligne_id": rid, "ligne_nom": ligne.get("nom"),
                        "type": ligne["type"], "sens": sens_label,
                        "terminus": seq[-1] if idx_sens == 0 else seq[-1],
                        "nb_arrets_total": len(seq),
                    })
    resultats.sort(key=lambda x: x["distance_m"])
    return resultats


# frontend 

def prix_connu(mode, nom_ligne, montee, descente, numero):
    if numero and str(numero) in PRIX_LIGNES:
        return PRIX_LIGNES[str(numero)]
    return PRIX_POUR_CLASSEMENT.get(mode, 250)


def normaliser_reseau(reseau: str, type_ligne: str) -> str:
    r = nettoyer_texte(reseau or "")
    if type_ligne == "BATEAU":
        if "monbato" in r:
            return "Monbato"
        if "stl" in r:
            return "STL"
        if "aqua" in r:
            return "Aqualines"
        return "Autre"
    if type_ligne == "BUS":
        if "express" in r:
            return "Express"
        if "wibus" in r:
            return "Wibus"
        if "navette" in r:
            return "Navette"
        if "monbus" in r or "mon bus" in r:
            return "Monbus"
        return "Autre"
    if type_ligne == "GBAKA":
        if "adja" in r:
            return "Adjamé"
        if "abobo" in r:
            return "Abobo"
        return "Autre"
    return "Autre"


def lignes_du_catalogue(type_ligne=None) -> list:
    resultat = []
    for i, c in enumerate(CATALOGUE):
        if type_ligne and c["type"] != type_ligne:
            continue
        numero = c["numero"]
        if c["type"] == "BUS" and numero and str(numero).isdigit():
            n = int(numero)
            if 1 <= n <= 9:
                numero = f"{n:02d}"
        if c["type"] == "GBAKA":
            ori = nettoyer_texte(c.get("origine") or "")
            if "adja" in ori:
                reseau = "Adjamé"
            elif "abobo" in ori:
                reseau = "Abobo"
            else:
                reseau = normaliser_reseau(c.get("reseau"), "GBAKA")
        else:
            reseau = normaliser_reseau(c.get("reseau"), c["type"])
        resultat.append({
            "id": f"cat:{i}", "numero": numero or None,
            "origine": c["origine"], "destination": c["destination"],
            "nom": f"{c['origine']} ↔ {c['destination']}",
            "operateur": c["operateur"], "reseau": reseau, "type": c["type"],
            "prix_fcfa": c.get("prix_fcfa") or prix_connu(c["type"], None, "", "", c["numero"]),
        })
    return resultat


def liste_lignes(type_transport=None) -> list:
    return lignes_du_catalogue(type_transport)


def lignes_pour_app(type_transport=None, q=None, zone=None, reseau=None) -> dict:
    genre = None
    if type_transport:
        t = type_transport.strip().upper()
        if t in ("BUS", "GBAKA", "BATEAU", "WORO_WORO"):
            genre = t
    toutes = lignes_du_catalogue(genre)
    comptes = Counter(l["reseau"] for l in toutes)
    filtres = [{"nom": "Toutes", "valeur": "toutes", "total": len(toutes)}]
    for nom, total in comptes.most_common():
        filtres.append({"nom": nom, "valeur": nom.lower(), "total": total})
    choisies = toutes
    if reseau and reseau.lower() not in ("toutes", "tous", ""):
        reseau_clean = nettoyer_texte(reseau)
        choisies = [l for l in choisies if nettoyer_texte(l["reseau"]) == reseau_clean]
    if q:
        mots = extraire_mots(q)
        choisies = [l for l in choisies if all(
            m in nettoyer_texte(f"{l.get('numero') or ''} {l['origine']} {l['destination']}")
            for m in mots)]
    for l in choisies:
        if l["type"] in ("GBAKA", "WORO_WORO"):
            l["sens"] = _arrets_ligne_pour_api(l["id"])
    return {"type": genre, "reseau": reseau or "toutes",
            "total": len(choisies), "total_type": len(toutes),
            "filtres": filtres, "lignes": choisies}


def _mots_cle(texte: str) -> list:
    return [m for m in nettoyer_texte(texte or "").split()
            if len(m) > 2 and m not in ("terminus", "gare", "arret")]


def _score_nom(cible: str, nom: str) -> float:
    import difflib
    mots_c, mots_n = _mots_cle(cible), _mots_cle(nom)
    if not mots_c:
        return 0.0
    ok = 0
    for mc in mots_c:
        for mn in mots_n:
            if mc == mn or (len(mc) >= 5 and len(mn) >= 5
                            and (mc.startswith(mn[:5]) or mn.startswith(mc[:5]))) \
               or difflib.SequenceMatcher(None, mc, mn).ratio() >= 0.82:
                ok += 1
                break
    return ok / len(mots_c)


#  _sens_ligne_bus — corrige le bug 85
def _sens_ligne_bus(fiche, index_catalogue, origine, destination, numero) -> list:
    """Aller / retour d'une ligne BUS/BATEAU.
    PRIORITÉ au cat:{index} (séquence propre au CSV : deux lignes partageant
    un même numéro — ex. 85 Phalène/Gare Campus et 85 Gare Kouté/Gare Campus —
    ne doivent JAMAIS hériter des arrêts de l'autre).
    Fallback GTFS uniquement si le catalogue n'a pas de séquence, avec filtre
    supplémentaire par nom de ligne pour éviter les confusions."""
    rid_cat = f"cat:{index_catalogue}" if index_catalogue is not None else None

    candidates = []
    # La ligne 610 possède une trace catalogue en boucle qui mélange les deux
    # sens. Ses séquences GTFS décrivent correctement Adjamé Liberté ↔ Bingerville.
    if str(numero).strip() == "610":
        for rid, info in RESEAU["lignes"].items():
            if (rid.startswith("cat:") or info.get("type") != fiche["type"]
                    or str(info.get("numero") or "").strip() != "610"):
                continue
            candidates.extend(RESEAU["sequences"].get(rid, []))

    # 1) Séquences propres au catalogue (garanti : bonne ligne)
    if not candidates and rid_cat and RESEAU["sequences"].get(rid_cat):
        candidates.extend(RESEAU["sequences"][rid_cat])

    # 2) Fallback GTFS si et seulement si le catalogue n'a rien
    if not candidates and numero:
        variantes = {numero}
        if numero.isdigit():
            variantes |= {str(int(numero)), f"{int(numero):02d}"}
        for rid, info in RESEAU["lignes"].items():
            if info.get("type") != fiche["type"] or rid.startswith("cat:"):
                continue
            num = str(info.get("numero") or "").strip()
            if num not in variantes and not (num.isdigit() and numero.isdigit()
                                              and int(num) == int(numero)):
                continue
            nom_ligne = nettoyer_texte(info.get("nom") or "")
            if _score_nom(origine, nom_ligne) < 0.5 and _score_nom(destination, nom_ligne) < 0.5:
                continue
            for seq in RESEAU["sequences"].get(rid, []):
                candidates.append(seq)

    if not candidates and numero:
        # dernier recours : une seule ligne GTFS portant ce numéro -> on la prend
        rids = [rid for rid, info in RESEAU["lignes"].items()
                if info.get("type") == fiche["type"] and not rid.startswith("cat:")
                and str(info.get("numero") or "").strip() == str(numero).strip()]
        if len(rids) == 1:
            candidates.extend(RESEAU["sequences"].get(rids[0], []))

    if not candidates:
        return []

    def nettoyee(seq):
        out, prec = [], None
        for sid in seq:
            a = RESEAU["arrets"].get(sid)
            if not a:
                continue
            cle = nettoyer_texte(a["nom"])
            if cle == prec:
                continue
            prec = cle
            out.append(sid)
        return out

    def pos(sid):
        a = RESEAU["arrets"][sid]
        return a["latitude"], a["longitude"]

    def dist_ids(x, y):
        return distance_m(*pos(x), *pos(y))

    def moitie_aller_retour(seq):
        """Une séquence qui revient à son point de départ contient l'aller ET le retour :
        on la coupe au point de rebroussement (arrêt le plus éloigné du départ)."""
        if len(seq) < 6 or dist_ids(seq[0], seq[-1]) > 500:
            return [seq]
        eloignement = [dist_ids(seq[0], sid) for sid in seq]
        m = max(range(len(seq)), key=lambda i: eloignement[i])
        # autour du point de rebroussement, on préfère l'arrêt qui porte le nom de la destination
        fenetre = range(max(1, m - 3), min(len(seq) - 1, m + 3) + 1)
        m = max(fenetre, key=lambda i: (_score_nom(destination, RESEAU["arrets"][seq[i]]["nom"]),
                                         eloignement[i]))
        if m < 2 or len(seq) - m < 3:
            return [seq]
        return [seq[:m + 1], seq[m:]]

    cands = []
    for sq in candidates:
        for moitie in moitie_aller_retour(nettoyee(sq)):
            if len(moitie) >= 2 and moitie not in cands:
                cands.append(moitie)
    if not cands:
        return []

    # points de repère : arrêts qui portent le nom de l'origine / de la destination du catalogue
    def repere(nom):
        meilleur, sc = None, 0.5
        for sq in cands:
            for sid in sq:
                v = _score_nom(nom, RESEAU["arrets"][sid]["nom"])
                if v > sc or (v == sc and meilleur is None and v >= 0.5):
                    meilleur, sc = sid, v
        return meilleur

    rep_ori, rep_dest = repere(origine), repere(destination)

    def ecart(seq, ori, dest, rep_o, rep_d):
        e = 0.0
        if rep_o:
            e += dist_ids(seq[0], rep_o)
        if rep_d:
            e += dist_ids(seq[-1], rep_d)
        return e

    def cle_sens(seq, ori, dest, rep_o, rep_d):
        sc = _score_nom(ori, RESEAU["arrets"][seq[0]]["nom"]) + _score_nom(dest, RESEAU["arrets"][seq[-1]]["nom"])
        return (round(sc, 2), -ecart(seq, ori, dest, rep_o, rep_d), len(seq))

    aller = max(cands, key=lambda c: cle_sens(c, origine, destination, rep_ori, rep_dest))
    if str(numero).strip() == "610":
        retour = list(reversed(aller))
    else:
        # retour = séquence qui part de l'arrivée de l'aller et revient à son départ
        def ecart_retour(c):
            return dist_ids(c[0], aller[-1]) + dist_ids(c[-1], aller[0])
        autres = [c for c in cands if c != aller]
        retour = min(autres, key=ecart_retour) if autres else None
        if retour is None or ecart_retour(retour) > 1500 \
                or [RESEAU["arrets"][x]["nom"] for x in retour] == [RESEAU["arrets"][x]["nom"] for x in aller]:
            retour = list(reversed(aller))

    def fabrique(seq, label, direction, nom_depart, nom_terminus):
        arrets = []
        for sid in seq:
            a = RESEAU["arrets"][sid]
            arrets.append({"id": a["id"], "nom": a["nom"],
                           "latitude": a["latitude"], "longitude": a["longitude"],
                           "role": "arret"})
        arrets[0]["role"], arrets[-1]["role"] = "depart", "arrivee"
        # le départ / terminus affichés dans la liste = ceux affichés en haut de la fiche
        if nom_depart:
            arrets[0]["nom"] = nom_depart
        if nom_terminus:
            arrets[-1]["nom"] = nom_terminus
        return {"direction": direction, "sens": label,
                "titre": f"{arrets[0]['nom']} → {arrets[-1]['nom']}",
                "depart": arrets[0]["nom"], "terminus": arrets[-1]["nom"],
                "nb_arrets": len(arrets), "arrets": arrets, "trace": []}

    return [fabrique(aller, "aller", "A", origine, destination),
            fabrique(retour, "retour", "R", destination, origine)]


def detail_ligne(ident: str):
    if not RESEAU["charge"]:
        charger_reseau_local()

    fiche = None
    index_catalogue = None

    if ident.startswith("cat:") and ident[4:].isdigit():
        i = int(ident[4:])
        if 0 <= i < len(CATALOGUE):
            fiche = CATALOGUE[i]
            index_catalogue = i
    if not fiche:
        for i, c in enumerate(CATALOGUE):
            if str(c.get("numero") or "") == str(ident):
                fiche = c
                index_catalogue = i
                break
    if not fiche:
        return None

    origine = (fiche.get("origine") or "").strip()
    destination = (fiche.get("destination") or "").strip()
    numero = str(fiche.get("numero") or "").strip()

    resultat = {
        "id": ident if ident.startswith("cat:") else f"cat:{index_catalogue}",
        "numero": numero or None, "type": fiche["type"],
        "operateur": fiche.get("operateur"),
        "reseau": normaliser_reseau(fiche.get("reseau"), fiche["type"]),
        "nom": f"{origine} ↔ {destination}",
        "depart": origine, "arrivee": destination,
        "prix_fcfa": fiche.get("prix_fcfa") or prix_connu(fiche["type"], None, "", "", numero),
        "nb_arrets": None, "sens": [],
    }

    if fiche["type"] in ("GBAKA", "WORO_WORO"):
        sens_csv = _arrets_ligne_pour_api(f"cat:{index_catalogue}")
        if sens_csv:
            resultat["nb_arrets"] = max(s["nb_arrets"] for s in sens_csv)
            for s in sens_csv:
                arrets = [dict(a) for a in s["arrets"]]
                if arrets:
                    arrets[0]["role"] = "depart"
                    arrets[-1]["role"] = "arrivee"
                resultat["sens"].append({
                    "direction": "A" if s["sens"] == "aller" else "R",
                    "sens": s["sens"],
                    "titre": f"{s['depart']} → {s['arrivee']}",
                    "depart": s["depart"], "terminus": s["arrivee"],
                    "nb_arrets": s["nb_arrets"], "arrets": arrets, "trace": [],
                })
            return resultat

    if numero or index_catalogue is not None:
        sens_bus = _sens_ligne_bus(fiche, index_catalogue, origine, destination, numero)
        if sens_bus:
            resultat["sens"] = sens_bus
            resultat["nb_arrets"] = max(x["nb_arrets"] for x in sens_bus)

    if not resultat["sens"]:
        resultat["nb_arrets"] = 2
        resultat["sens"] = [{
            "direction": "A", "sens": "aller", "titre": f"{origine} → {destination}",
            "depart": origine, "terminus": destination, "nb_arrets": 2,
            "arrets": [
                {"id": None, "nom": origine, "latitude": None, "longitude": None, "role": "depart"},
                {"id": None, "nom": destination, "latitude": None, "longitude": None, "role": "arrivee"},
            ],
            "trace": [],
        }]
    return resultat


def recherche_globale(q: str, type_transport=None, maximum: int = 20) -> dict:
    mots = extraire_mots(q)
    if not mots:
        return {"lignes": [], "arrets": []}
    lignes = lignes_pour_app(type_transport, q)["lignes"][:maximum]
    arrets = []
    for a in RESEAU["arrets"].values():
        if a["id"] in RESEAU["points_libres"]:
            continue
        if all(m in nettoyer_texte(a["nom"]) for m in mots):
            arrets.append(a)
    return {"lignes": lignes, "arrets": arrets[:maximum]}


def trajets_pour_demande(candidats, priorite=None, prix_max=None, budget=None):
    if not candidats:
        return []
    if budget:
        candidats = [c for c in candidats if _cout_tri(c) <= budget]
    if priorite == "prix":
        candidats.sort(key=lambda c: (_cout_tri(c), c.get("duree_minutes", 999)))
    else:
        candidats.sort(key=lambda c: (c.get("duree_minutes", 999), _cout_tri(c)))
    return candidats


def instructions_trajet(candidat: dict) -> list:
    if not candidat or not candidat.get("etapes"):
        return []
    instructions = []
    for e in candidat["etapes"]:
        if e["mode"] == "MARCHE":
            instructions.append(f"marchez {e['distance_m']} m jusqu a {e['arrivee']}")
        else:
            nom = e.get("numero_ligne") or e.get("nom_ligne") or e["mode"]
            instructions.append(f"prenez le {e['mode'].lower()} {nom} de {e['depart']} a {e['arrivee']}")
    return instructions


def _texte_trajet(candidat, mode="TOUS"):
    if not candidat:
        return "aucun trajet trouve."
    parts = []
    for e in candidat.get("etapes", []):
        if e["mode"] == "MARCHE":
            if e.get("distance_m", 0) > 0:
                parts.append(f"marchez {e['distance_m']} m jusqu a {e['arrivee']}")
        else:
            nom = e.get("numero_ligne") or e.get("nom_ligne") or ""
            parts.append(f"{e['mode']} {nom} : {e['depart']} → {e['arrivee']}")
    return " puis ".join(parts) + f" (environ {candidat.get('duree_minutes')} min)"


# agent ia

client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

SESSIONS = defaultdict(lambda: {
    "depart": None, "destination": None, "dernier_trajet": None,
    "budget_fcfa": None, "mode_transport": None, "attente_mode": False,
})


def maj_session(session_id: str, intention: dict, trajet: dict = None, attente_mode=None):
    if not session_id:
        return
    s = SESSIONS[session_id]
    for cle in ("depart", "destination"):
        if intention.get(cle):
            s[cle] = intention[cle]
    if intention.get("budget_fcfa"):
        s["budget_fcfa"] = intention["budget_fcfa"]
    if intention.get("mode_precise") and intention.get("mode_transport"):
        s["mode_transport"] = intention["mode_transport"]
    if trajet:
        s["dernier_trajet"] = trajet
    if attente_mode is not None:
        s["attente_mode"] = attente_mode
        if attente_mode:
            s["mode_demande"] = True  # la question « quel mode ? » n'est posée qu'une fois par conversation


def analyser_intention(message: str, historique: list = None, session_id: str = None) -> dict:
    s = SESSIONS[session_id] if session_id else {}
    _EXCLURE.set(None)
    attente = bool(s.get("attente_mode"))
    mode_msg = extraire_mode(message, accepter_tous=attente)
    budget_msg = extraire_budget(message)

    if re.search(r"(comment (tu|vous) (t|vous )?.?appel|qui (es|etes).?(tu|vous)|tu es qui|ton nom|"
                 r"c.?est quoi ton nom|tu fais quoi|presente.?toi)", nettoyer_texte(message)):
        return _completer_analyse({"intention": "CONVERSATION"}, s, mode_msg, budget_msg)

    if s.get("dernieres_options") and re.search(
            r"(pas convaincu|ne me convient pas|ça ne me convient pas|ca ne me convient pas|"
            r"autre trajet|autre option|change|changer|propose autre|quelque chose d.?autre|"
            r"non,? merci|je n.?aime pas|je (ne )?veux (pas|plus)|pas (ce|cet|celui|ces|ceux)|"
            r"enleve|retire|sans ce trajet)", nettoyer_texte(message)):
        _EXCLURE.set(list(s.get("dernieres_options") or []))
        def signature_option(option):
            return tuple(
                (e.get("mode"), e.get("numero_ligne") or e.get("nom_ligne"),
                 e.get("depart"), e.get("arrivee"))
                for e in option.get("etapes", [])
                if e.get("mode") != "MARCHE"
            )

        return {
            "intention": "TRAJET", "depart": s.get("depart"),
            "destination": s.get("destination"),
            "mode_transport": mode_msg or "TOUS", "mode_precise": bool(mode_msg),
            "priorite": None, "budget_fcfa": budget_msg or s.get("budget_fcfa"),
            "prix_max_ligne": None,
            "exclure_signatures": [signature_option(option)
                                   for option in s.get("dernieres_options", [])],
        }

    if attente and mode_msg and s.get("destination"):
        return {
            "intention": "TRAJET", "depart": s.get("depart"),
            "destination": s.get("destination"),
            "mode_transport": mode_msg, "mode_precise": True, "priorite": None,
            "budget_fcfa": budget_msg or s.get("budget_fcfa"), "prix_max_ligne": None,
        }

    if mode_msg and s.get("destination") and not re.search(
            r"(c.?est quoi|qu.?est.?ce|\bpourquoi\b|comment fonctionne|\bcombien\b|est.?ce que|"
            r"\bou est\b|\bquel(le)?s? (est|sont)\b)", nettoyer_texte(message)):
        return {
            "intention": "TRAJET", "depart": s.get("depart"),
            "destination": s.get("destination"),
            "mode_transport": mode_msg, "mode_precise": True, "priorite": None,
            "budget_fcfa": budget_msg or s.get("budget_fcfa"), "prix_max_ligne": None,
        }

    if budget_msg and s.get("destination") and (s.get("depart") or s.get("dernieres_options")) \
            and not re.search(r"\bcombien\b|c.?est quoi|\bpourquoi\b|\bcomment fonctionne\b", nettoyer_texte(message)):
        mode_session = s.get("mode_transport")
        return {
            "intention": "TRAJET", "depart": s.get("depart"),
            "destination": s.get("destination"),
            "mode_transport": mode_msg or mode_session or "TOUS",
            "mode_precise": bool(mode_msg or mode_session), "priorite": None,
            "budget_fcfa": budget_msg, "prix_max_ligne": None,
        }

    if not client:
        msg = message.lower()
        if any(m in msg for m in ("bonjour", "salut", "merci", "hello", "bonsoir")):
            return _completer_analyse({"intention": "CONVERSATION"}, s, mode_msg, budget_msg)
        return _completer_analyse({"intention": "INCONNUE"}, s, mode_msg, budget_msg)

    historique = historique or []
    historique_texte = "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}"
                                   for m in historique[-10:])

    memoire_txt = ""
    if s.get("depart") or s.get("destination"):
        memoire_txt = f"""
Mémoire de la session :
- Dernier départ connu : {s.get('depart') or 'inconnu'}
- Dernière destination connue : {s.get('destination') or 'inconnue'}
- Budget connu : {s.get('budget_fcfa') or 'aucun'}
- Mode connu : {s.get('mode_transport') or 'non précisé'}

RÈGLES DE SUIVI :
- Le message CHANGE ou RELANCE la recherche (autre mode : "et en gbaka ?", autre départ : "et depuis Yopougon ?",
  autre budget, "je suis pressé, donne-moi un autre trajet", "propose-moi autre chose") -> intention = TRAJET,
  en réutilisant départ et destination ci-dessus s'ils ne sont pas redonnés.
- Le message pose une QUESTION sur les trajets déjà proposés ("c'est combien ?", "où est-ce que je descends ?",
  "combien de temps ?", "pourquoi ce bus ?") -> intention = SUIVI.
- Tout le reste (salutation, merci, blague, culture générale, avis, conseil, question sans rapport avec ce trajet,
  même juste après un trajet) -> intention = CONVERSATION. Dans ce cas on ne renvoie JAMAIS d'itinéraire.
"""

    prompt = f"""Tu es un assistant de transport à Abidjan.
Analyse le message de l'utilisateur et réponds UNIQUEMENT avec un JSON valide.

Message : "{message}"

Historique récent :
{historique_texte}
{memoire_txt}

Réponds exactement avec ce format JSON :
{{
  "intention": "TRAJET" ou "SUIVI" ou "CONVERSATION" ou "LIGNES_TRAJET" ou "INCONNUE",
  "depart": "lieu de départ ou null",
  "destination": "lieu d'arrivée ou null",
  "priorite": "duree" ou "prix" ou null,
  "prix_max_ligne": nombre ou null
}}

Règles importantes :
- Si l'utilisateur dit "ma position", "ici", "je suis là", "près de moi" alors depart = "ma position actuelle"
- intention = "CONVERSATION" pour TOUT ce qui n'est pas une demande de déplacement : salutation,
  remerciement, discussion libre, culture générale, blague, question sur toi, etc.
- intention = "TRAJET" uniquement si l'utilisateur veut aller quelque part ou relance/modifie la recherche de trajet
- N'invente jamais un lieu : mets null si le lieu n'est pas dans le message ou dans la mémoire
- Ne mets jamais de texte en dehors du JSON
"""

    try:
        reponse = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "Tu réponds uniquement en JSON valide, sans aucun texte autour."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1, max_tokens=400,
        )
        texte = reponse.choices[0].message.content.strip()
        debut = texte.find("{")
        fin = texte.rfind("}") + 1
        if debut >= 0 and fin > debut:
            data = json.loads(texte[debut:fin])
            return _completer_analyse(data, s, mode_msg, budget_msg)
    except Exception as e:
        log.error("erreur analyse intention : %s", e)
    return _completer_analyse({"intention": "INCONNUE"}, s, mode_msg, budget_msg)


def _extraire_infos_conditions(contexte: dict) -> str:
    infos = []
    trafic = contexte.get("trafic_niveau")
    if trafic is None:
        trafic = contexte.get("trafic")
    pluie = contexte.get("pluie")
    if pluie is None:
        pluie = contexte.get("weather")
    if pluie is None:
        pluie = contexte.get("meteo")

    facteur = contexte.get("trafic_facteur")
    if isinstance(trafic, str):
        libelle = {"eleve": "dense", "moyen": "modéré", "faible": "fluide"}.get(
            nettoyer_texte(trafic), trafic)
        extra = f", les trajets durent environ x{facteur:.2f}" \
            if isinstance(facteur, (int, float)) and facteur > 1.05 else ""
        infos.append(f"le trafic est {libelle}{extra}")
        trafic = None

    if trafic is not None:
        if isinstance(trafic, (int, float)):
            if trafic >= 0.7:
                infos.append(f"le trafic est dense (niveau {trafic:.2f})")
            elif trafic >= 0.4:
                infos.append(f"le trafic est modéré (niveau {trafic:.2f})")
            else:
                infos.append(f"le trafic est fluide (niveau {trafic:.2f})")
        else:
            infos.append(f"niveau de trafic : {trafic}")

    if pluie is not None:
        if isinstance(pluie, bool):
            infos.append("il pleut" if pluie else "il ne pleut pas")
        elif isinstance(pluie, (int, float)):
            if pluie > 0.5:
                infos.append(f"il pleut (intensité {pluie:.2f})")
            else:
                infos.append(f"il ne pleut pas (intensité {pluie:.2f})")
        else:
            infos.append(f"météo : {pluie}")

    return " ; ".join(infos) if infos else "conditions normales (non calculées)"


def _est_question_conditions(message: str) -> bool:
    m = (message or "").lower()
    if re.search(r"(quel temps|le temps qu|il fait quel temps|niveau de (?:la )?pluie)", m):
        return True
    return any(w in m for w in (
        "embouteillage", "trafic", "circulation", "bouchon", "pressé", "presse",
        "pressée", "pluie", "pleut", "météo", "meteo",
    ))


def extraire_budget(texte: str) -> Optional[int]:
    """Comprend le langage naturel : « j'ai 400 », « 400 balles », « je n'ai que 500f »,
    « moins de 300 », « 2 mille », « 1 500 francs »..."""
    t = (texte or "").lower().replace("\u202f", " ").replace("\u00a0", " ")
    nombre = r"(\d{1,3}(?:[ .]\d{3})+|\d+)"
    valeur = None
    m = re.search(nombre + r"\s*(?:mille|milles|k)\b", t)          # « 2 mille », « 2k »
    if m:
        valeur = int(re.sub(r"\D", "", m.group(1))) * 1000
    if valeur is None:
        m = re.search(nombre + r"\s*(?:fcfa|f\.?\s?cfa|cfa|francs?|frs?\b|f\b|balles?|piastres?|billets?)", t)
        if m:
            valeur = int(re.sub(r"\D", "", m.group(1)))
    if valeur is None:
        m = re.search(r"(?:budget|maximum|max|co[uû]t|prix|dans les|moins de|pas plus de|jusqu['’]?\s?[àa]|"
                      r"j['’]?ai(?: seulement| juste| que)?|je n['’]?ai que|il me reste|me reste|je dispose de|"
                      r"je poss[eè]de|avec|(?:un )?petit budget de)\D{0,12}" + nombre, t)
        if m:
            valeur = int(re.sub(r"\D", "", m.group(1)))
    if valeur is None:
        m = re.fullmatch(r"\s*" + nombre + r"\s*[.!]?\s*", t)       # message = juste « 400 »
        if m:
            valeur = int(re.sub(r"\D", "", m.group(1)))
    return valeur if valeur is not None and 50 <= valeur <= 1000000 else None


def extraire_mode(texte: str, accepter_tous: bool = False) -> Optional[str]:
    """BUS / GBAKA / BATEAU / TOUS, ou None si l'utilisateur n'a rien précisé.
    Gère la négation : « pas de bus », « sans gbaka », « au lieu du bus »."""
    t = nettoyer_texte(texte)
    t = re.sub(r"bateau[\s_-]*bus", "bateau", t)

    def _nie(mot_pat: str) -> bool:
        return bool(re.search(
            rf"(?:pas|sans|au lieu (?:de |du |des |d')|plut[oô]t que |non)\s*"
            rf"(?:le |la |les |un |une |des )?{mot_pat}", t))

    trouves = set()
    if re.search(r"\b(bateau|bateaux|lagunaire|bato|monbato)\b", t) \
       and not _nie(r"\b(bateau|bateaux)\b"):
        trouves.add("BATEAU")
    if re.search(r"\b(gbaka|gbakas|woro|woros)\b", t) \
       and not _nie(r"\b(gbaka|gbakas|woro|woros)\b"):
        trouves.add("GBAKA")
    if re.search(r"\b(bus|sotra|autobus|monbus)\b", t) \
       and not _nie(r"\b(bus|sotra|autobus|monbus)\b"):
        trouves.add("BUS")

    if len(trouves) == 1:
        return next(iter(trouves))
    if len(trouves) > 1:
        return "TOUS"
    if re.search(r"(tous les (modes|transports|moyens)|n.?importe (quel|lequel)|peu importe|indifferent|pas de preference)", t):
        return "TOUS"
    if accepter_tous and re.search(r"\b(tous|tout|les deux)\b", t):
        return "TOUS"
    return None


def _valeur_vide(v) -> bool:
    return v is None or (isinstance(v, str) and v.strip().lower() in ("", "null", "none", "aucun"))


#  _completer_analyse — pas de réutilisation du mode pour une nouvelle demande
def _completer_analyse(data: dict, s: dict, mode_msg, budget_msg) -> dict:
    data.setdefault("intention", "INCONNUE")
    if data["intention"] == "INCONNUE":
        data["intention"] = "CONVERSATION"   # jamais de réponse figée : on discute normalement
    if data["intention"] == "SUIVI" and not s.get("dernieres_options"):
        data["intention"] = "CONVERSATION"
    data.setdefault("depart", None)
    data.setdefault("destination", None)
    data.setdefault("priorite", None)
    data.setdefault("prix_max_ligne", None)
    data["mode_transport"] = "TOUS"
    data["mode_precise"] = False
    data["budget_fcfa"] = None

    if data["intention"] in ("TRAJET", "LIGNES_TRAJET"):
        for cle in ("depart", "destination"):
            valeur = None if _valeur_vide(data.get(cle)) else data[cle]
            data[cle] = valeur or s.get(cle)
        if mode_msg:
            data["mode_transport"], data["mode_precise"] = mode_msg, True
        elif s.get("mode_demande"):
            data["mode_precise"] = True  # déjà demandé plus tôt : on ne repose pas la question, tous les modes
        # On ne réutilise PAS le mode de session pour une nouvelle demande :
        # si l'utilisateur ne précise pas le mode, on cherche dans TOUS les modes.
        data["budget_fcfa"] = budget_msg or s.get("budget_fcfa")
    return data


def _prix_texte(c: dict) -> str:
    if not c:
        return "inconnu"
    p = c.get("cout_fcfa")
    if p is not None:
        return f"{p} FCFA"
    base = "variable (le tarif des gbakas / woro-woro dépend du chauffeur ou de l'apprenti)"
    partiel = c.get("cout_partiel_fcfa") or 0
    return f"{base}, partie bus/bateau : {partiel} FCFA" if partiel else base


def _points_cles(c: dict) -> str:
    out = []
    for e in c.get("etapes", []):
        if e["mode"] == "MARCHE":
            continue
        ligne = e.get("numero_ligne") or e.get("nom_ligne") or e["mode"]
        n = e.get("nb_arrets") or 0
        rang = f" (au {'1er' if n == 1 else str(n) + 'e'} arrêt)" if n > 0 else ""
        out.append(f"{e['mode'].lower()} {ligne} : montée à {e['depart']}, descente à {e['arrivee']}{rang}")
    return " ; ".join(out)


def _messages_llm(systeme: str, prompt: str, historique: list) -> list:
    msgs = [{"role": "system", "content": systeme}]
    for m in (historique or [])[-10:]:
        role = "assistant" if m.get("role") in ("assistant", "model", "bot") else "user"
        contenu = str(m.get("content", "")).strip()
        if contenu:
            msgs.append({"role": role, "content": contenu})
    msgs.append({"role": "user", "content": prompt})
    return msgs


SYSTEME_AMI = (
    "Tu es MobiGuide, un assistant sympa basé à Abidjan. Tu discutes naturellement, comme un ami : "
    "tu réponds à toutes les questions (pas seulement les transports), en français, de façon courte "
    "et chaleureuse. Tu ne proposes JAMAIS d'itinéraire si on ne t'en demande pas, et tu ne ramènes "
    "pas tout aux trajets. Tu ne répètes jamais une formule d'accueil (« avec plaisir », « bonne "
    "nouvelle »...) ni un conseil que tu as déjà donné plus haut dans la conversation : tu varies et "
    "tu réponds précisément à ce que l'utilisateur demande maintenant. "
    "Tu t'appelles MobiGuide, l'assistant de l'application MobiGo. Tu es toujours accueillant, chaleureux et tu "
    "guides l'utilisateur ; tu ne refuses jamais de répondre à une question simple. "
    "Si tu ne connais pas la réponse, dis-le clairement et demande une précision ; "
    "n'invente jamais une information, une ligne, un prix ou une fonctionnalité."
)


SYSTEME_TRAJET = (
    "Tu es MobiGuide, assistant de transport à Abidjan, chaleureux et naturel, comme un ami qui "
    "connaît bien la ville. Tu réponds toujours en français. Tu ne répètes jamais une formule "
    "d'accueil ni un conseil déjà dit dans la conversation. Tu n'inventes jamais de prix ni "
    "d'arrêt et tu ne renvoies jamais vers Google Maps ou Waze."
)


def _conseils_liste(options: list, budget=None) -> list:
    """Conseils pratiques selon les modes réellement proposés : liste de (clé, texte)."""
    modes = set()
    for c in (options or [])[:4]:
        for e in c.get("etapes", []):
            if e.get("mode") != "MARCHE":
                modes.add(e["mode"])
    out = []
    if "BUS" in modes:
        out.append(("BUS", "En bus, prends ton ticket ou badge ta carte auprès du machiniste."))
    if "GBAKA" in modes:
        out.append(("GBAKA", "Pour les gbakas, le prix n'est pas fixe : demande-le à l'apprenti avant de monter."))
    if "WORO_WORO" in modes:
        out.append(("WORO_WORO", "Pour un woro-woro, demande le prix au chauffeur avant de monter."))
    if "BATEAU" in modes:
        out.append(("BATEAU", "Pour le bateau-bus, garde ton ticket jusqu'à la descente."))
    if modes:
        out.append(("SIGNALER", "Signale ton arrêt de descente au chauffeur ou à l'apprenti."))
    if budget and modes & {"GBAKA", "WORO_WORO"}:
        out.append((f"BUDGET{budget}", f"Prévois un peu de marge sur tes {budget} FCFA pour les gbakas et woro-woro."))
    return out


def conseils_trajet(options: list, budget=None) -> str:
    return " ".join(t for _, t in _conseils_liste(options, budget))


def _phrase_complete(texte: str, fini_par_limite: bool) -> str:
    """Si le LLM a été coupé (max_tokens), on garde seulement les phrases terminées."""
    texte = (texte or "").strip()
    if not fini_par_limite or not texte:
        return texte
    fin = max(texte.rfind(c) for c in ".!?…")
    emoji = re.search(r"[.!?…][^\w\s]{0,3}\s*$", texte)
    if emoji:
        return texte
    return texte[:fin + 1].strip() if fin > 20 else texte


def _habiller_reponse(texte: str, options: list, budget=None) -> str:
    texte = (texte or "").strip()
    conseils = conseils_trajet(options, budget)
    if conseils:
        bas = texte.lower()
        if sum(1 for k in ("ticket", "badg", "apprenti", "chauffeur", "signale") if k in bas) < 2:
            texte += "\n\n💡 " + conseils
    return texte


def _dist_txt(m) -> str:
    m = int(m or 0)
    return f"{m / 1000:.1f} km".replace(".", ",") if m >= 1000 else f"{m} m"


def _ordinal(n: int) -> str:
    return "1er" if n == 1 else f"{n}e"


_NOM_MODE = {"BUS": "le bus", "GBAKA": "le gbaka", "BATEAU": "le bateau-bus"}


def decrire_trajet(c: dict) -> str:
    """Raconte le trajet étape par étape : marche jusqu'à l'arrêt, montée, rang de l'arrêt de
    descente (avec son nom), correspondance. Tout vient des données calculées, rien d'inventé."""
    etapes = c.get("etapes", []) or []
    idx_transports = [i for i, e in enumerate(etapes) if e.get("mode") not in ("MARCHE", "WORO_WORO")]
    dernier_t = idx_transports[-1] if idx_transports else -1
    lignes = []
    derniere_arrivee = None
    for i, e in enumerate(etapes):
        mode, d = e.get("mode"), e.get("distance_m", 0) or 0
        if mode == "MARCHE":
            if d < 30:
                continue
            minutes = e.get("duree_minutes")
            duree = f" (environ {minutes} min)" if minutes else ""
            avant_transport = any(x.get("mode") != "MARCHE" for x in etapes[i + 1:])
            if avant_transport:
                t = f"Marche {_dist_txt(d)} jusqu'à l'arrêt « {e.get('arrivee')} »{duree}."
                if d > SEUIL_MARCHE_LONGUE_M:
                    t += " C'est assez loin : tu peux aussi prendre un taxi ou un gbaka/woro-woro pour t'y rendre."
            else:
                t = f"Il te restera {_dist_txt(d)} à pied jusqu'à ta destination{duree}."
                if d > SEUIL_MARCHE_LONGUE_M:
                    t += " C'est assez long : un taxi ou un woro-woro peut t'y déposer."
            lignes.append(t)
        elif mode == "WORO_WORO":
            lignes.append(f"Pour rejoindre l'arrêt « {e.get('arrivee')} » ({_dist_txt(d)}), "
                          f"prends un woro-woro ou un taxi.")
        else:
            if derniere_arrivee and derniere_arrivee != e.get("depart") and i > 0 \
                    and etapes[i - 1].get("mode") != "MARCHE":
                lignes.append(f"Rejoins l'arrêt « {e.get('depart')} » (tout près, à pied).")
            ligne = e.get("numero_ligne")
            nom = _NOM_MODE.get(mode, str(mode).lower()) + (f" {ligne}" if ligne else "")
            n = e.get("nb_arrets") or 0
            descente = (f"descends au {_ordinal(n)} arrêt, à « {e.get('arrivee')} »" if n > 0
                        else f"descends à « {e.get('arrivee')} »")
            t = f"Monte dans {nom} à « {e.get('depart')} », puis {descente}"
            t += " pour prendre la correspondance." if i < dernier_t else "."
            lignes.append(t)
            derniere_arrivee = e.get("arrivee")
    return "\n".join(f"{k}. {t}" for k, t in enumerate(lignes, 1))


_LIBELLE_MODE = {"BUS": "bus", "GBAKA": "gbaka", "BATEAU": "bateau-bus"}


def _intro_trajet(message: str, contexte: dict, candidats: list, mode: str,
                  historique: list, premiere: bool, memoire_txt: str, infos_conditions: str) -> str:
    modele = "Voici ce que je te propose !" if premiere else "Voici ce que j'ai trouvé."
    if not client:
        return modele
    consigne = (
        "C'est la première fois que tu présentes des trajets dans cette conversation : accueille "
        "l'utilisateur chaleureusement."
        if premiere else
        "Tu as déjà présenté des trajets dans cette conversation : fais une phrase courte et "
        "naturelle, SANS formule d'accueil (pas de « avec plaisir », « bonne nouvelle », « super »), "
        "qui reconnaît ce qu'il vient de demander (autre mode, autre budget, autre trajet...)."
    )
    prompt = (
        f'L\'utilisateur vient de demander : "{message}".\n'
        f"Tu as trouvé {len(candidats)} option(s). Mode demandé : {mode}. "
        f"Budget : {contexte.get('budget_fcfa') or 'aucun'}.\n"
        f"Conditions actuelles (à ne mentionner que si c'est vraiment utile) : {infos_conditions or 'aucune'}\n"
        f"{consigne}\n"
        "Écris UNE seule phrase de 25 mots maximum. Ne décris pas les étapes (elles seront écrites "
        "juste après), ne donne aucun conseil, n'invente rien."
    )
    try:
        r = client.chat.completions.create(
            model=MODEL,
            messages=_messages_llm(SYSTEME_TRAJET + memoire_txt, prompt, historique),
            temperature=0.6, max_tokens=140,
        )
        texte = (r.choices[0].message.content or "").strip().strip('"').strip()
        texte = texte.split("\n")[0].strip()
        texte = _phrase_complete(texte, r.choices[0].finish_reason == "length")
        return texte or modele
    except Exception as e:
        log.error("erreur intro trajet : %s", e)
        return modele


def _reponse_trajet(message: str, contexte: dict, candidats: list, mode: str, historique: list,
                    s: dict, memoire_txt: str, infos_conditions: str) -> str:
    premiere = not s.get("nb_reponses_trajet")
    budget = contexte.get("budget_fcfa")

    if not candidats:
        if _EXCLURE.get():
            return ("Je n'ai pas d'autre trajet vraiment différent entre ces deux lieux. "
                    "Tu veux essayer un autre mode (bus, gbaka, bateau-bus) ?")
        return ("Je n'ai trouvé aucun trajet pour l'instant. Vérifie ton départ et ta destination, "
                "ou essaie un autre mode de transport.")

    combo = candidats[0].get("combo_mode")
    if combo:
        lib = _LIBELLE_MODE.get(combo, combo.lower())
        intro = (f"Je n'ai pas de trajet 100 % {lib} entre ces deux lieux, mais voici des trajets "
                 f"qui combinent le {lib} avec un autre mode.")
    else:
        intro = _intro_trajet(message, contexte, candidats, mode, historique, premiere,
                              memoire_txt, infos_conditions)

    if budget and extraire_budget(message) == budget:
        intro += f" Tous ces trajets respectent ton budget de {budget} FCFA."
    parts = [intro, "Option 1 :\n" + decrire_trajet(candidats[0])]
    autres = len(candidats) - 1
    if autres > 0:
        parts.append(f"J'ai aussi {autres} autre{'s' if autres > 1 else ''} option{'s' if autres > 1 else ''} "
                     f"dans les cartes.")

    deja = set(s.get("conseils_donnes") or [])
    nouveaux = [(k, t) for k, t in _conseils_liste(candidats, budget) if k not in deja]
    if nouveaux:
        parts.append("💡 " + " ".join(t for _, t in nouveaux))
        s["conseils_donnes"] = list(deja | {k for k, _ in nouveaux})
    s["nb_reponses_trajet"] = (s.get("nb_reponses_trajet") or 0) + 1
    return "\n\n".join(parts)


#  generer_reponse — naturel, sans répétition 
def generer_reponse(message: str, contexte: dict, historique: list = None,
                    session_id: str = None) -> str:
    historique = historique or []
    statut = contexte.get("statut", "")
    candidats = contexte.get("candidats") or []
    mode = contexte.get("mode", "TOUS")
    s = SESSIONS[session_id] if session_id else {}

    infos_conditions = _extraire_infos_conditions(contexte)
    question_conditions = _est_question_conditions(message)

    memoire_txt = ""
    if session_id and session_id in SESSIONS:
        if s.get("depart") or s.get("destination"):
            memoire_txt = (f"\nContexte de la discussion : départ={s.get('depart')}, "
                           f"destination={s.get('destination')}, "
                           f"budget={s.get('budget_fcfa') or 'aucun'}, "
                           f"mode={s.get('mode_transport') or 'non précisé'}")

    if statut == "TRAJET":
        return _reponse_trajet(message, contexte, candidats, mode, historique, s,
                               memoire_txt, infos_conditions)

    if not client:
        if statut == "CONVERSATION":
            return ("Bonjour ! 😊 Je suis MobiGuide, ton guide à Abidjan. "
                    "Dis-moi où tu veux aller et je te propose le meilleur trajet !")
        return ("Je suis MobiGuide. Dis-moi ton point de départ et ta destination, "
                "et je m'occupe du reste 😊")

    if statut == "SUIVI":
        systeme = SYSTEME_AMI + memoire_txt
        resume = "\n".join(
            f"- {c.get('libelle', 'proposition')} : {c.get('duree_minutes', '?')} min, "
            f"prix {_prix_texte(c)}, {c.get('nb_correspondances', 0)} correspondance(s), "
            f"{c.get('marche_m', 0)} m de marche ; {_points_cles(c)}"
            for c in candidats)
        prompt = f"""{message}

[Trajets que tu viens de proposer à l'utilisateur (cartes déjà affichées) :
{resume or 'aucun trajet en mémoire'}
Conditions actuelles : {infos_conditions}]
Réponds directement à SA question, en 2-3 phrases, avec ces infos, sur un ton chaleureux.
Ne répète pas les conseils ni l'accueil déjà donnés plus haut dans la conversation. N'ajoute un conseil
que s'il est nouveau et vraiment utile. Ne recopie pas tout l'itinéraire, n'invente rien."""
    elif statut == "CONVERSATION":
        systeme = SYSTEME_AMI + memoire_txt
        if question_conditions:
            prompt = f"""{message}

[Infos pour répondre : conditions actuelles calculées par notre modèle ML : {infos_conditions}]
Utilise UNIQUEMENT ces infos pour trafic / météo, ne renvoie jamais vers Google Maps ou Waze,
donne un conseil concret en 2-3 phrases maximum."""
        else:
            prompt = f"""{message}

Réponds directement et naturellement à ce message, comme un ami chaleureux, en 2 ou 3 phrases maximum.
Si on te demande ton nom ou qui tu es : tu es MobiGuide, l'assistant de l'application MobiGo pour se déplacer à Abidjan.
Pour une question hors transport (culture générale, blague, conseil...), réponds-y vraiment ; ne dis jamais que tu
ne réponds pas à ce genre de questions. Termine, quand c'est naturel, par une courte invitation à t'aider pour un trajet
(sans imposer de position ni d'itinéraire). Ne répète pas ce que tu as déjà dit."""
    else:
        systeme = SYSTEME_AMI + memoire_txt
        prompt = message

    try:
        reponse = client.chat.completions.create(
            model=MODEL,
            messages=_messages_llm(systeme, prompt, historique),
            temperature=0.5, max_tokens=600,
        )
        return _phrase_complete(reponse.choices[0].message.content,
                                reponse.choices[0].finish_reason == "length")
    except Exception as e:
        log.error("erreur generation reponse : %s", e)
        return "Je suis MobiGuide. Dis-moi où tu veux aller et je te propose un trajet."