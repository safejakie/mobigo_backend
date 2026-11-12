import copy
import contextvars
import csv
import difflib
import functools
import json
import os
import re
import threading
import unicodedata
from collections import Counter, defaultdict
from typing import List, Optional

import serpapi
from groq import Groq

from config import (
    DATA_DIR, GROQ_API_KEY, SERPAPI_KEY, CENTRE_ABIDJAN, MODEL,
    PRIX_LIGNES, PRIX_POUR_CLASSEMENT, RAYON_PROXIMITE_M,
    cache_get, cache_set, distance_m, log,
)

# RÉGLAGES (on les change ici, à un seul endroit) 
DESCENTE_MAX_M = 1000          # on descend à 1000 m maximum de la destination
DESCENTE_MAX_GBAKA_M = 500     # un gbaka dépose plus près (500 m)
RAYON_DEPART_M = 1500          # on cherche les arrêts de départ dans ce rayon
SEUIL_MARCHE_LONGUE_M = 1000   # au-delà, on conseille un taxi / woro-woro
ATTENTE_MINUTES = {"BUS": 8, "GBAKA": 3, "WORO_WORO": 3, "BATEAU": 15}
NOM_MODE = {"BUS": "bus", "GBAKA": "gbaka", "WORO_WORO": "woro-woro", "BATEAU": "bateau-bus"}
# Cadre géographique d'Abidjan : un lieu trouvé hors de ce cadre est refusé.
ABIDJAN_LAT = (5.15, 5.65)
ABIDJAN_LON = (-4.35, -3.75)

# ── 1. OUTILS ──
def nettoyer_texte(texte: str) -> str:
    """Minuscules + sans accents : « Adjamé » devient « adjame »."""
    if not texte:
        return ""
    texte = unicodedata.normalize("NFD", str(texte).lower())
    return "".join(c for c in texte if unicodedata.category(c) != "Mn")

def extraire_mots(texte: str) -> list:
    """Les mots importants d'un texte (sans « de », « la », « abidjan »...)."""
    vides = {"de", "du", "des", "d", "l", "la", "le", "les", "a", "au", "abidjan"}
    return [m for m in re.findall(r"\w+", nettoyer_texte(texte)) if m not in vides]

def _charger_trace(trace_wkt, max_points: int = 1000):
    """Lit un tracé « LINESTRING(lon lat, ...) » -> (liste de (lat, lon), cadre min/max)."""
    if not trace_wkt or not isinstance(trace_wkt, str):
        return [], None
    paires = re.findall(r"(-?\d+\.?\d+)\s+(-?\d+\.?\d+)", trace_wkt)
    points = [(float(lat), float(lon)) for lon, lat in paires]
    if not points:
        return [], None
    if len(points) > max_points:                      # on allège les tracés trop longs
        points = points[::max(1, len(points) // max_points)]
    lats, lons = [p[0] for p in points], [p[1] for p in points]
    return points, (min(lats), min(lons), max(lats), max(lons))

def _liste_arrets(texte: str) -> list:
    """« A → B → C » devient ['A', 'B', 'C']."""
    if not texte or not isinstance(texte, str):
        return []
    parties = re.split(r"→|->", texte)
    return [p.strip() for p in parties if p.strip()]

def _liste_coords(texte: str) -> list:
    """« lon,lat | lon,lat » devient [(lon, lat), ...]."""
    sortie = []
    for paire in (texte or "").split("|"):
        morceaux = [p.strip() for p in paire.split(",")]
        if len(morceaux) == 2:
            try:
                sortie.append((float(morceaux[0]), float(morceaux[1])))
            except ValueError:
                pass
    return sortie

def est_gare(nom: str) -> bool:
    """Vrai si le nom parle d'une gare (« gare d'Adjamé », « Adjamé gare »...)."""
    return "gare" in nettoyer_texte(nom or "")

# 2. LIEUX : Google Maps (SerpAPI) d'abord, nos arrêts locaux en secours 
def _dans_abidjan(lat, lon) -> bool:
    return ABIDJAN_LAT[0] <= lat <= ABIDJAN_LAT[1] and ABIDJAN_LON[0] <= lon <= ABIDJAN_LON[1]

def _google_maps(requete: str) -> list:
    """Interroge Google Maps (via SerpAPI). Retourne [{nom, adresse, latitude, longitude}]
    en gardant seulement les résultats situés dans Abidjan."""
    try:
        res = serpapi.search({"api_key": SERPAPI_KEY, "engine": "google_maps", "type": "search",
                              "q": requete, "ll": CENTRE_ABIDJAN, "hl": "fr", "gl": "ci"}).as_dict()
    except Exception as e:
        log.error("erreur serpapi : %s", e)
        return []
    bruts = ([res["place_results"]] if res.get("place_results") else []) + (res.get("local_results") or [])
    sortie = []
    for r in bruts:
        c = r.get("gps_coordinates") or {}
        if c.get("latitude") is None or c.get("longitude") is None:
            continue
        lat, lon = float(c["latitude"]), float(c["longitude"])
        if _dans_abidjan(lat, lon):
            sortie.append({"nom": (r.get("title") or "lieu").strip(), "adresse": r.get("address") or "",
                           "latitude": lat, "longitude": lon})
    return sortie

def _lieu_dans_nos_arrets(lieu: str):
    """Secours : cherche le lieu dans les NOMS de nos arrêts (ex : « Azito » -> « Shell Azito »).
    Accepte les petites fautes de frappe (« Zitoa »). Retourne le centre des arrêts trouvés."""
    if not RESEAU["charge"]:
        charger_reseau_local()
    mots = [m for m in extraire_mots(lieu) if len(m) >= 3]
    if not mots:
        return None
    trouves = []
    for a in RESEAU["arrets"].values():
        if a["id"] in RESEAU["points_libres"]:
            continue
        mots_arret = extraire_mots(a["nom"])
        ok = all(any(m == x or m in x or difflib.SequenceMatcher(None, m, x).ratio() >= 0.75
                     for x in mots_arret) for m in mots)
        if ok:
            trouves.append(a)
    if not trouves:
        return None
    lat = sum(a["latitude"] for a in trouves) / len(trouves)
    lon = sum(a["longitude"] for a in trouves) / len(trouves)
    return {"nom": lieu.strip().title(), "latitude": lat, "longitude": lon, "alternatives": []}

def rechercher_lieu(lieu: Optional[str]):
    """Transforme un nom de lieu en coordonnées. Google Maps d'abord, sinon nos arrêts."""
    lieu = (lieu or "").strip()
    if not lieu:
        return None
    cle = ("geo", lieu.lower())
    deja = cache_get(cle)
    if deja:
        return deja
    resultat = None
    for requete in (f"{lieu}, Abidjan, Côte d'Ivoire", f"{lieu} Abidjan"):
        trouves = _google_maps(requete)
        if trouves:
            t = trouves[0]
            resultat = {"nom": t["nom"], "latitude": t["latitude"], "longitude": t["longitude"],
                        "alternatives": []}
            break
    if not resultat:
        resultat = _lieu_dans_nos_arrets(lieu)      # secours local
    if resultat:
        cache_set(cle, resultat)
    return resultat

def suggestions_lieu(lieu: str, limite: int = 8) -> list:
    """Suggestions pendant que l'utilisateur tape. Ordre : 1) nos arrêts qui contiennent TOUS les mots,
    2) les vrais lieux de Google Maps (ex : « Université Félix Houphouët-Boigny »), 3) arrêts avec au moins
    un mot. (Avant, les arrêts « Cocody » remplissaient les 8 places et Google n'était jamais interrogé.)"""
    lieu = (lieu or "").strip()
    if len(lieu) < 2:
        return []
    cle = ("suggestions", lieu.lower())
    deja = cache_get(cle)
    if deja is not None:
        return deja
    if not RESEAU["charge"]:
        charger_reseau_local()
    mots = extraire_mots(lieu)
    tous_mots, un_mot, vus = [], [], set()
    for a in RESEAU["arrets"].values():
        if a["id"] in RESEAU["points_libres"] or a["nom"] in vus:
            continue
        nom = nettoyer_texte(a["nom"])
        fiche = {"nom": a["nom"], "adresse": "", "latitude": a["latitude"], "longitude": a["longitude"]}
        if mots and all(m in nom for m in mots):
            vus.add(a["nom"])
            tous_mots.append(fiche)
        elif any(m in nom for m in mots if len(m) > 3):
            vus.add(a["nom"])
            un_mot.append(fiche)
    resultats = tous_mots[:4]
    for t in _google_maps(f"{lieu}, Abidjan, Côte d'Ivoire"):
        if t["nom"] not in vus and len(resultats) < limite:
            vus.add(t["nom"])
            resultats.append(t)
    resultats += tous_mots[4:] + un_mot
    resultats = resultats[:limite]
    cache_set(cle, resultats)
    return resultats


MOTS_POSITION = ("position", "ici", "je suis la", "je suis ici", "pres de moi", "autour de moi",
                 "chez moi", "mon emplacement", "ma localisation", "localisation")

def determiner_depart(dep_texte: Optional[str], lat: Optional[float] = None, lon: Optional[float] = None):
    """Point de départ : le lieu écrit, sinon la position GPS du téléphone."""
    texte = nettoyer_texte((dep_texte or "").strip())
    a_gps = lat is not None and lon is not None
    ma_position = {"nom": "ma position actuelle", "latitude": lat, "longitude": lon, "alternatives": []}
    parle_position = (not texte or texte in ("null", "none", "aucun")
                      or any(m in texte for m in MOTS_POSITION))
    if a_gps and parle_position:
        return ma_position
    if not parle_position:
        lieu = rechercher_lieu(dep_texte)
        if lieu:
            return lieu
    return ma_position if a_gps else None

# 3. DONNÉES : chargement des fichiers CSV 
CATALOGUE = []                 # une entrée par ligne (bus, bateau, gbaka/woro)
RESEAU = {
    "arrets": {},              # id -> {id, nom, latitude, longitude}
    "lignes": {},              # id de ligne -> infos de la ligne
    "sequences": {},           # id de ligne -> liste ordonnée d'ids d'arrêts
    "lignes_par_arret": defaultdict(list),
    "gares": set(), "points_libres": set(),
    "arrets_non_bus": set(), "arrets_bus": set(), "arrets_gbaka": set(),
    "charge": False,
}
_lock = threading.Lock()

def _trace_estimee(row) -> bool:
    origine = nettoyer_texte(row.get("origine_trace") or "")
    return ("approximatif" in origine or "introuvable" in origine
            or nettoyer_texte(row.get("trace_complete") or "") == "non")

def _lire_csv(nom_fichier: str):
    """Lit un CSV (séparateur « ; ») et rend la liste des lignes (dictionnaires)."""
    chemin = os.path.join(DATA_DIR, nom_fichier)
    if not os.path.exists(chemin):
        return []
    with open(chemin, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f, delimiter=";"))

def _entier(valeur, defaut: int) -> int:
    try:
        return int(float(valeur or defaut))
    except Exception:
        return defaut

def charger_catalogue():
    """Remplit CATALOGUE avec les bus, les bateaux-bus et les gbakas/woro-woro."""
    global CATALOGUE
    CATALOGUE = []
    # Bus et bateau-bus : même structure, seul le fichier et la colonne du prix changent.
    for fichier, type_ligne, col_prix, prix_def, reseau_def in (
            ("bus.csv", "BUS", "prix_fcfa", 200, "Monbus"),
            ("bateau_bus.csv", "BATEAU", "prix_minimum_fcfa", 150, "Monbato")):
        for row in _lire_csv(fichier):
            origine, destination = (row.get("depart") or "").strip(), (row.get("arrivee") or "").strip()
            if not origine or not destination:
                continue
            trace, bbox = _charger_trace(row.get("trace_wkt") or "")
            CATALOGUE.append({
                "numero": (row.get("numero_ligne") or "").strip(), "origine": origine,
                "destination": destination, "operateur": (row.get("operateur") or "SOTRA").strip(),
                "reseau": (row.get("reseau") or reseau_def).strip(), "type": type_ligne,
                "prix_fcfa": _entier(row.get(col_prix), prix_def), "trace": trace, "bbox": bbox,
                "estime": _trace_estimee(row)})
    # Gbaka / woro-woro : une ligne a deux sens (aller et retour) avec leurs arrêts.
    entrees = {}
    for row in _lire_csv("gbaka_woro.csv"):
        origine, destination = (row.get("depart") or "").strip(), (row.get("arrivee") or "").strip()
        if not origine or not destination:
            continue
        type_ligne = "WORO_WORO" if "woro" in (row.get("type_transport") or "").lower() else "GBAKA"
        sens = {}
        for nom_sens, pref in (("sens_aller", "sens_aller"), ("sens_retour", "sens_retour")):
            sens[nom_sens] = {
                "destination": (row.get(pref + "_destination") or "").strip(),
                "liste_arrets": _liste_arrets(row.get(pref + "_liste_arrets") or ""),
                "coordonnees": _liste_coords(row.get(pref + "_coordonnees_gps") or "")}
        nb_stops = len(sens["sens_aller"]["liste_arrets"]) + len(sens["sens_retour"]["liste_arrets"])
        if nb_stops == 0:
            continue
        id_osm = (row.get("id_osm") or "").strip()
        cle = id_osm or (type_ligne, origine, destination)
        if cle in entrees and nb_stops <= entrees[cle]["_nb_stops"]:
            continue                                   # doublon : on garde le plus complet
        trace, bbox = _charger_trace(row.get("trace_wkt") or "")
        entrees[cle] = {
            "numero": None, "origine": origine, "destination": destination,
            "operateur": (row.get("operateur") or "divers").strip(),
            "reseau": (row.get("reseau_commune") or "Autre").strip(), "type": type_ligne,
            "prix_fcfa": 200, "trace": trace, "bbox": bbox, "estime": _trace_estimee(row),
            "id_osm": id_osm, "origine_trace": (row.get("origine_trace") or "").strip(),
            "sens_aller": sens["sens_aller"], "sens_retour": sens["sens_retour"],
            "_nb_stops": nb_stops}
    for e in entrees.values():
        e.pop("_nb_stops")
        CATALOGUE.append(e)
    log.info("catalogue charge : %d lignes", len(CATALOGUE))

# 4. Petits calculs sur la carte 
def _dist_rapide(lat1, lon1, lat2, lon2) -> float:
    """Distance approximative en mètres (rapide, assez juste à Abidjan)."""
    dy = (lat1 - lat2) * 111320.0
    dx = (lon1 - lon2) * 111320.0 * 0.9956
    return (dx * dx + dy * dy) ** 0.5

def _cellule(lat, lon, taille):
    """Case de la grille où se trouve un point (sert à chercher vite les voisins)."""
    return (int(lat / taille), int(lon / taille))

def _grille_arrets(taille, ids=None):
    """Range les arrêts dans une grille : on ne compare qu'avec les cases voisines."""
    grille = defaultdict(list)
    for sid in (ids if ids is not None else RESEAU["arrets"]):
        a = RESEAU["arrets"][sid]
        if sid not in RESEAU["points_libres"]:
            grille[_cellule(a["latitude"], a["longitude"], taille)].append(sid)
    return grille

def _voisins(grille, lat, lon, taille):
    """Tous les arrêts des 9 cases autour d'un point."""
    ci, cj = _cellule(lat, lon, taille)
    return [sid for di in (-1, 0, 1) for dj in (-1, 0, 1) for sid in grille.get((ci + di, cj + dj), ())]

def _point_proche_du_trace(lat, lon, trace, bbox, seuil_m=300) -> bool:
    """Vrai si le point est à moins de seuil_m du tracé d'une ligne."""
    if not trace or not bbox:
        return False
    marge = seuil_m / 100000.0 + 0.001
    if not (bbox[0] - marge <= lat <= bbox[2] + marge and bbox[1] - marge <= lon <= bbox[3] + marge):
        return False
    return any(distance_m(lat, lon, tl, tn) <= seuil_m for tl, tn in trace)

# ── Séquences d'arrêts : dans quel ordre une ligne passe par ses arrêts ─────
def _arrets_le_long_du_trace(trace, grille, taille, seuil_m, filtre=None):
    """Liste ordonnée des arrêts situés à moins de seuil_m du tracé (sans doublons proches)."""
    passages = defaultdict(list)                       # arrêt -> [(position sur le tracé, distance)]
    for idx, (lat, lon) in enumerate(trace):
        for sid in _voisins(grille, lat, lon, taille):
            a = RESEAU["arrets"][sid]
            d = _dist_rapide(lat, lon, a["latitude"], a["longitude"])
            if d <= seuil_m and (not filtre or filtre(sid, a["nom"])):
                passages[sid].append((idx, d))
    evenements = []                                    # (position, arrêt) : 1 passage par arrêt
    for sid, hits in passages.items():
        groupe = [hits[0]]
        for h in hits[1:] + [None]:
            if h is not None and h[0] - groupe[-1][0] <= 6:
                groupe.append(h)
                continue
            evenements.append((min(groupe, key=lambda x: x[1])[0], sid))
            groupe = [h] if h else []
    evenements.sort()
    gardes = []
    for _, sid in evenements:                         
        a = RESEAU["arrets"][sid]
        nom = nettoyer_texte(a["nom"])
        if not any(nettoyer_texte(RESEAU["arrets"][b]["nom"]) == nom and _dist_rapide(
                a["latitude"], a["longitude"], RESEAU["arrets"][b]["latitude"],
                RESEAU["arrets"][b]["longitude"]) < 250 for b in gardes[-6:]):
            gardes.append(sid)
    return gardes

def _est_arret_de_bus(sid: str, nom: str) -> bool:
    return "lagunaire" not in nettoyer_texte(nom) and sid in RESEAU["arrets_bus"]

def _nom_point(lat, lon, grille_noms) -> str:
    """Nom d'un point « bord de route » : on cite l'arrêt connu le plus proche."""
    proche, dmin = None, 400
    for sid in _voisins(grille_noms, lat, lon, 0.004):
        a = RESEAU["arrets"][sid]
        d = _dist_rapide(lat, lon, a["latitude"], a["longitude"])
        if d < dmin and len(a["nom"].strip()) >= 4 and not a["nom"].strip().isdigit():
            proche, dmin = a["nom"], d
    return f"Bord de route près de {proche}" if proche else "Bord de route"

def _boucle_asymetrique(trace) -> bool:
    """Vrai si la ligne fait une boucle (le retour ne repasse pas par l'aller)."""
    if len(trace) < 20 or _dist_rapide(*trace[0], *trace[-1]) > 500:
        return False
    loin = max(range(len(trace)), key=lambda j: _dist_rapide(*trace[0], *trace[j]))
    retour = trace[loin:]
    grille = defaultdict(list)
    for p in trace[:loin + 1]:
        grille[_cellule(p[0], p[1], 0.002)].append(p)
    pareil = 0
    for p in retour:
        ci, cj = _cellule(p[0], p[1], 0.002)
        if any(_dist_rapide(*p, *q) < 200 for di in (-1, 0, 1) for dj in (-1, 0, 1)
               for q in grille.get((ci + di, cj + dj), ())):
            pareil += 1
    return pareil < 0.6 * len(retour)

def _sequence_gbaka(i, trace, grille_gbaka, grille_noms):
    """Gbaka sans liste d'arrêts : vrais arrêts le long du tracé + un point tous les 300 m."""
    reels = _arrets_le_long_du_trace(trace, grille_gbaka, 0.0006, 40)
    evenements = []
    for sid in reels:
        a = RESEAU["arrets"][sid]
        k = min(range(len(trace)), key=lambda j: _dist_rapide(a["latitude"], a["longitude"], *trace[j]))
        evenements.append((k, sid))
    cumul = 0.0
    for k, (lat, lon) in enumerate(trace):
        if k:
            cumul += _dist_rapide(*trace[k - 1], lat, lon)
        if k and cumul < 300 and k != len(trace) - 1:
            continue
        cumul = 0.0
        if any(_dist_rapide(lat, lon, RESEAU["arrets"][s]["latitude"], RESEAU["arrets"][s]["longitude"]) < 150
               for s in reels):
            continue
        pid = f"g:{i}:{k}"
        RESEAU["arrets"][pid] = {"id": pid, "nom": _nom_point(lat, lon, grille_noms),
                                 "latitude": lat, "longitude": lon}
        RESEAU["points_libres"].add(pid)
        evenements.append((k, pid))
    return [sid for _, sid in sorted(evenements, key=lambda x: x[0])]

def _resoudre_stops_csv(sens: dict, i: int, label: str, index: dict) -> list:
    """Transforme les noms d'arrêts d'un sens gbaka en ids d'arrêts (existants ou créés)."""
    coords = sens.get("coordonnees") or []
    ids = []
    for idx, nom in enumerate(sens.get("liste_arrets") or []):
        nom = (nom or "").strip()
        if not nom:
            continue
        lon, lat = coords[idx] if idx < len(coords) else (None, None)
        existants = index.get(nettoyer_texte(nom), [])
        choisi = None
        if existants and lat is None:
            choisi = existants[0]
        elif existants:                                # le même nom à moins de 120 m
            d, choisi = min((_dist_rapide(lat, lon, RESEAU["arrets"][s]["latitude"],
                                          RESEAU["arrets"][s]["longitude"]), s) for s in existants)
            choisi = choisi if d < 120 else None
        if not choisi and lat is not None:
            choisi = f"csv:{i}:{label}:{idx}"
            RESEAU["arrets"][choisi] = {"id": choisi, "nom": nom, "latitude": float(lat), "longitude": float(lon)}
            index.setdefault(nettoyer_texte(nom), []).append(choisi)
        if choisi:
            ids.append(choisi)
    return ids

def _arrets_ligne_pour_api(rid: str) -> list:
    """Pour l'écran « détail d'une ligne » : les arrêts de chaque sens d'un gbaka/woro."""
    if not isinstance(rid, str) or not rid.startswith("cat:") or not rid[4:].isdigit():
        return []
    i = int(rid[4:])
    if i >= len(CATALOGUE) or CATALOGUE[i]["type"] not in ("GBAKA", "WORO_WORO"):
        return []
    c, sens_list = CATALOGUE[i], []
    for label in ("aller", "retour"):
        data = c.get("sens_" + label) or {}
        noms, coords = data.get("liste_arrets") or [], data.get("coordonnees") or []
        if len(noms) < 2:
            continue
        arrets = []
        for idx, nom in enumerate(noms):
            lon, lat = coords[idx] if idx < len(coords) else (None, None)
            arrets.append({"id": f"csv:{i}:{label}:{idx}", "nom": (nom or "").strip(),
                           "latitude": lat, "longitude": lon})
        sens_list.append({"sens": label, "depart": arrets[0]["nom"], "arrivee": arrets[-1]["nom"],
                          "nb_arrets": len(arrets), "arrets": arrets})
    return sens_list

def _enregistrer_sequences(rid: str, sequences: list):
    """Mémorise les séquences d'une ligne et l'ajoute à chacun de ses arrêts."""
    RESEAU["sequences"][rid] = sequences
    for seq in sequences:
        for sid in seq:
            if rid not in RESEAU["lignes_par_arret"][sid]:
                RESEAU["lignes_par_arret"][sid].append(rid)

def _construire_sequences_depuis_traces():
    """Pour les lignes du catalogue (CSV) : trouve les arrêts le long de chaque tracé."""
    taille = 0.0006
    grille, grille_noms = _grille_arrets(taille), _grille_arrets(0.004)
    grille_gbaka = _grille_arrets(taille, RESEAU["arrets_gbaka"])
    index_noms = defaultdict(list)
    for sid, a in RESEAU["arrets"].items():
        index_noms[nettoyer_texte(a["nom"])].append(sid)
    nb = 0
    for i, c in enumerate(CATALOGUE):
        rid, trace, type_ligne = f"cat:{i}", c.get("trace") or [], c["type"]
        if rid in RESEAU["sequences"]:
            continue
        if type_ligne in ("GBAKA", "WORO_WORO"):
            # 1) on utilise d'abord les listes d'arrêts écrites dans le CSV
            seqs = []
            for label in ("aller", "retour"):
                s = _resoudre_stops_csv(c.get("sens_" + label) or {}, i, label, index_noms)
                if len(s) >= 2 and (not seqs or s != seqs[0]):
                    seqs.append(s)
            if not seqs and len(trace) >= 2 and not c.get("estime"):
                # 2) sinon on déduit les arrêts du tracé
                s = _sequence_gbaka(i, trace, grille_gbaka, grille_noms)
                if len(s) >= 2:
                    seqs = [s] if _boucle_asymetrique(trace) else [s, s[::-1]]
            if seqs:
                _enregistrer_sequences(rid, seqs)
                nb += 1
            continue
        if len(trace) < 2 or c.get("estime"):
            continue
        if type_ligne == "BUS":
            s = _arrets_le_long_du_trace(trace, grille, taille, 40, filtre=_est_arret_de_bus)
        else:                                          # BATEAU : arrêts « lagunaire »
            s = _arrets_le_long_du_trace(trace, grille, taille * 6, 400,
                                         filtre=lambda sid, n: "lagunaire" in nettoyer_texte(n))
        if len(s) >= 2:
            _enregistrer_sequences(rid, [s, s[::-1]])
            nb += 1
    log.info("sequences construites depuis les traces : %d lignes", nb)

def _lire_arrets_gtfs(nom_fichier: str):
    """Ajoute à RESEAU les arrêts d'un fichier (arret.csv, stops.txt...)."""
    chemin = os.path.join(DATA_DIR, nom_fichier)
    if not os.path.exists(chemin):
        return False
    with open(chemin, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            try:
                sid = row["stop_id"].strip()
                if sid in RESEAU["arrets"]:
                    continue
                nom = (row.get("stop_name") or "").strip()
                RESEAU["arrets"][sid] = {"id": sid, "nom": nom, "latitude": float(row["stop_lat"]),
                                         "longitude": float(row["stop_lon"])}
                if est_gare(nom):
                    RESEAU["gares"].add(sid)
            except Exception:
                continue
    return True

def _type_ligne_gtfs(agence: str, nom: str) -> str:
    texte = f"{agence} {nom}".lower()
    if "gbaka" in texte:
        return "GBAKA"
    if "woro" in texte:
        return "WORO_WORO"
    if any(x in texte for x in ("bateau", "stl", "aqua", "monbato")):
        return "BATEAU"
    return "BUS"

def charger_reseau_local():
    """Charge TOUT le réseau en mémoire (appelé une seule fois, au démarrage)."""
    with _lock:
        if RESEAU["charge"]:
            return
        log.info("chargement du reseau local...")
        charger_catalogue()
        if not _lire_arrets_gtfs("arret.csv"):
            log.warning("arret.csv introuvable")
        for nom_fichier in ("stops.txt", "stops-2.txt"):
            if _lire_arrets_gtfs(nom_fichier):
                log.info("arrets ajoutes depuis %s", nom_fichier)

        def lire(nom_fichier):
            chemin = os.path.join(DATA_DIR, nom_fichier)
            if not os.path.exists(chemin):
                return []
            with open(chemin, encoding="utf-8") as f:
                return list(csv.DictReader(f))
        for row in lire("routes.txt"):                 # les lignes GTFS
            rid = row.get("route_id")
            if rid:
                numero, nom = (row.get("route_short_name") or "").strip(), (row.get("route_long_name") or "").strip()
                RESEAU["lignes"][rid] = {
                    "id": rid, "numero": numero or None, "nom": nom,
                    "type": _type_ligne_gtfs((row.get("agency_id") or ""), nom),
                    "operateur": row.get("agency_id") or "inconnu"}
        trips = defaultdict(list)                      # ligne -> ses voyages
        for row in lire("trips.txt"):
            trips[row["route_id"]].append(row["trip_id"])
        passages = defaultdict(list)                   # voyage -> arrêts avec leur rang
        for row in lire("stop_times.txt"):
            try:
                passages[row["trip_id"]].append((int(row["stop_sequence"]), row["stop_id"]))
            except Exception:
                continue
        for rid, trip_ids in trips.items():
            seqs = []
            for tid in trip_ids:
                s = [sid for _, sid in sorted(passages.get(tid, [])) if sid in RESEAU["arrets"]]
                if len(s) >= 2 and s not in seqs:
                    seqs.append(s)
            if seqs:                                   # on ajoute aussi le sens inverse
                seqs += [s[::-1] for s in list(seqs) if s[::-1] not in seqs]
                _enregistrer_sequences(rid, seqs)
        # Quels arrêts servent aux bus ? aux gbakas ? (sert à filtrer les tracés)
        bus, autres = set(), set()
        for rid, seqs in RESEAU["sequences"].items():
            type_l = RESEAU["lignes"].get(rid, {}).get("type")
            (bus if type_l == "BUS" else autres).update(s for seq in seqs for s in seq)
            if type_l in ("GBAKA", "WORO_WORO"):
                RESEAU["arrets_gbaka"].update(s for seq in seqs for s in seq)
        RESEAU["arrets_bus"], RESEAU["arrets_non_bus"] = bus, autres - bus
        for i, c in enumerate(CATALOGUE):              # les lignes du catalogue CSV
            rid = f"cat:{i}"
            RESEAU["lignes"].setdefault(rid, {
                "id": rid, "numero": c.get("numero"), "nom": f"{c['origine']} ↔ {c['destination']}",
                "type": c["type"], "operateur": c.get("operateur", "")})
        _construire_sequences_depuis_traces()
        RESEAU["charge"] = True
        log.info("reseau pret : %d arrets, %d lignes, %d sequences, %d gares", len(RESEAU["arrets"]),
                 len(RESEAU["lignes"]), len(RESEAU["sequences"]), len(RESEAU["gares"]))

# 4. ARRÊTS À PROXIMITÉ
def _types_de_l_arret(sid: str) -> set:
    """Les modes (BUS, GBAKA, BATEAU...) des lignes qui passent à cet arrêt."""
    return {RESEAU["lignes"][r]["type"] for r in RESEAU["lignes_par_arret"].get(sid, [])
            if r in RESEAU["lignes"]}

def arrets_proximite(lat: float, lon: float, rayon_m: int = 5000, tri_par_distance: bool = False) -> List[dict]:
    """Arrêts (bus, gbaka, bateau) autour d'un point. Un seul par nom (le plus proche)."""
    if not RESEAU["charge"]:
        charger_reseau_local()
    uniques = {}
    for a in RESEAU["arrets"].values():
        d = distance_m(lat, lon, a["latitude"], a["longitude"])
        types = _types_de_l_arret(a["id"]) & {"BUS", "GBAKA", "BATEAU"}   # jamais les woro-woro
        if d > rayon_m or not types:
            continue
        principal = "BUS" if "BUS" in types else ("GBAKA" if "GBAKA" in types else "BATEAU")
        if a["nom"] not in uniques or d < uniques[a["nom"]]["distance_m"]:
            uniques[a["nom"]] = {"id": a["id"], "nom": a["nom"], "latitude": a["latitude"],
                                 "longitude": a["longitude"], "distance_m": int(d),
                                 "type": principal, "types": list(types)}
    ordre = {"BUS": 0, "GBAKA": 1, "BATEAU": 2}
    cle = (lambda x: x["distance_m"]) if tri_par_distance else (lambda x: (ordre.get(x["type"], 9), x["distance_m"]))
    return sorted(uniques.values(), key=cle)

def arrets_proximite_par_modes(lat: float, lon: float, rayon_m: int = 1500,
                               modes: Optional[List[str]] = None) -> List[dict]:
    """Comme arrets_proximite, mais pour les modes demandés, triés du plus proche au plus loin."""
    if not RESEAU["charge"]:
        charger_reseau_local()
    modes = set(modes) if modes else {"BUS", "GBAKA", "BATEAU"}
    uniques = {}
    for a in RESEAU["arrets"].values():
        d = distance_m(lat, lon, a["latitude"], a["longitude"])
        if d <= rayon_m and _types_de_l_arret(a["id"]) & modes:
            if a["nom"] not in uniques or d < uniques[a["nom"]]["distance_m"]:
                uniques[a["nom"]] = {"id": a["id"], "nom": a["nom"], "latitude": a["latitude"],
                                     "longitude": a["longitude"], "distance_m": int(d)}
    return sorted(uniques.values(), key=lambda x: x["distance_m"])

def lignes_par_arret(stop_id: str) -> Optional[dict]:
    """Toutes les lignes qui passent à un arrêt (pour l'écran « arrêts à proximité »)."""
    if not RESEAU["charge"]:
        charger_reseau_local()
    arret = RESEAU["arrets"].get(stop_id)
    if not arret:
        return None
    lignes = [{"id": l["id"], "numero": l["numero"], "nom": l["nom"], "type": l["type"],
               "operateur": l["operateur"]}
              for l in (RESEAU["lignes"].get(r) for r in RESEAU["lignes_par_arret"].get(stop_id, [])) if l]
    lignes.sort(key=lambda l: (0 if l["type"] == "BUS" else 1, l["numero"] or "zzz"))
    return {"arret": arret, "lignes": lignes}

# ── 5. PRIX, TEXTES ET CHOIX DES PROPOSITIONS ──
PRIX_CATALOGUE = {}            # (mode, numéro de ligne) -> prix lu dans le CSV

def _norm_num(n) -> str:
    n = str(n or "").strip().upper()
    return str(int(n)) if n.isdigit() else n

def _prix_csv(type_ligne, numero):
    """Prix officiel (bus / bateau-bus) lu dans bus.csv et bateau_bus.csv. None si inconnu."""
    if not PRIX_CATALOGUE:
        for fichier, mode, colonne in (("bus.csv", "BUS", "prix_fcfa"),
                                       ("bateau_bus.csv", "BATEAU", "prix_minimum_fcfa")):
            for row in _lire_csv(fichier):
                try:
                    PRIX_CATALOGUE.setdefault((mode, _norm_num(row.get("numero_ligne"))),
                                              int(float(row.get(colonne))))
                except Exception:
                    continue
    return PRIX_CATALOGUE.get((type_ligne, _norm_num(numero)))

def _prix_ligne(ligne) -> int:
    """Prix d'une ligne pour le calcul (0 pour gbaka/woro : le prix n'est pas fixe)."""
    if ligne.get("type") not in ("BUS", "BATEAU"):
        return 0
    return _prix_csv(ligne["type"], ligne.get("numero")) or 0

def _rids_approximatifs() -> set:
    """Lignes du catalogue CSV dont une version GTFS plus précise existe : on les ignore."""
    if "rids_approx" not in RESEAU:
        gtfs = {(l.get("type"), _norm_num(l.get("numero"))) for rid, l in RESEAU["lignes"].items()
                if not str(rid).startswith("cat:") and RESEAU["sequences"].get(rid)}
        RESEAU["rids_approx"] = {rid for rid, l in RESEAU["lignes"].items()
                                 if str(rid).startswith("cat:") and l.get("type") in ("BUS", "BATEAU")
                                 and (l.get("type"), _norm_num(l.get("numero"))) in gtfs}
    return RESEAU["rids_approx"]

def _etapes_transport(c: dict) -> list:
    """Les étapes en véhicule d'un trajet (sans la marche ni le woro-woro d'approche)."""
    return [e for e in c.get("etapes", []) if e.get("mode") not in ("MARCHE", "WORO_WORO")]

def _acces_m(c: dict) -> int:
    """Distance à pied (ou en woro-woro d'approche) d'un trajet."""
    return c.get("marche_m", 0) + sum(e.get("distance_m", 0) for e in c.get("etapes", [])
                                       if e.get("mode") == "WORO_WORO")

def _nom_court_ligne(e: dict) -> str:
    if e.get("numero_ligne"):
        return str(e["numero_ligne"])
    nom = str(e.get("nom_ligne") or "").replace("gbaka :", "").strip()
    return nom or NOM_MODE.get(e.get("mode"), "ligne")

def _finaliser_candidat(c: dict):
    """Calcule le coût total (None si un prix est inconnu, ex : gbaka) et le nombre d'arrêts."""
    prix = [e.get("cout_fcfa") for e in c["etapes"] if e["mode"] != "MARCHE"]
    c["cout_partiel_fcfa"] = sum(p for p in prix if p is not None)
    c["prix_inconnu"] = any(p is None for p in prix)
    c["cout_fcfa"] = None if c["prix_inconnu"] else c["cout_partiel_fcfa"]
    c["nb_arrets"] = sum(e.get("nb_arrets", 0) for e in c["etapes"] if e["mode"] != "MARCHE")

def _normaliser_candidat(c: dict):
    """Prix officiels pour bus/bateau, inconnu pour gbaka, puis totaux."""
    for e in c.get("etapes", []):
        if e.get("mode") == "MARCHE":
            continue
        if e.get("nb_arrets") is None:
            e["nb_arrets"] = max(0, len(e["arrets"]) - 1) if e.get("arrets") else 0
        e["cout_fcfa"] = _prix_csv(e["mode"], e.get("numero_ligne")) if e["mode"] in ("BUS", "BATEAU") else None
    _finaliser_candidat(c)

def _ajouter_textes(c: dict):
    """Écrit la phrase de chaque étape (« Marchez 120 m jusqu'à l'arrêt X »...)."""
    etapes = c.get("etapes", [])
    for i, e in enumerate(etapes):
        if e["mode"] == "MARCHE":
            d = int(e.get("distance_m", 0))
            if i == 0:
                e["texte"] = (f"Vous êtes déjà à l'arrêt {e['arrivee']} (point de montée)." if d < 10 else
                              f"Marchez {d} m jusqu'à l'arrêt {e['arrivee']} (point de montée).")
            elif i == len(etapes) - 1:
                e["texte"] = f"Marchez {d} m jusqu'à votre destination."
            else:
                e["texte"] = f"Marchez {d} m de {e['depart']} à {e['arrivee']} pour la correspondance."
        else:
            lib = NOM_MODE.get(e["mode"], e["mode"].lower())
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
                e["texte"] = (f"Montez à {e['depart']} dans le {lib} {ligne}, descendez à "
                              f"{e['arrivee']}{suite}.").replace("  ", " ").replace(" ,", ",")
        if e.get("conseil"):
            e["texte"] += " " + e["conseil"]

# Trajets déjà proposés à exclure quand l'utilisateur dit « pas celui-là, autre chose ».
# (ContextVar : valable seulement pour la requête en cours, aucun changement dans main.py)
_EXCLURE = contextvars.ContextVar("mobigo_exclure", default=None)

def _quasi_identiques(a: dict, b: dict) -> bool:
    """Deux trajets presque identiques (mêmes modes, mêmes montées/descentes à ≤ 500 m)."""
    ta, tb = _etapes_transport(a), _etapes_transport(b)
    if not ta or len(ta) != len(tb):
        return False

    def proches(p, q):
        return bool(p and q) and distance_m(p["latitude"], p["longitude"], q["latitude"], q["longitude"]) <= 500
    return all(x.get("mode") == y.get("mode")
               and (x.get("depart") == y.get("depart") or proches(x.get("depart_pos"), y.get("depart_pos")))
               and (x.get("arrivee") == y.get("arrivee") or proches(x.get("arrivee_pos"), y.get("arrivee_pos")))
               for x, y in zip(ta, tb))

def choisir_propositions(candidats: list, mode_filtre: str = "TOUS",
                         priorite=None, budget=None, maxi: int = 4) -> list:
    """Choisit les 4 meilleures propositions, logiques et différentes :
    1. le moins de correspondances d'abord ; 2. on regroupe les trajets presque identiques ;
    3. en mode TOUS, on ajoute un autre mode seulement s'il n'est pas trop long (+50 % max)."""
    cands = [c for c in (candidats or []) if _etapes_transport(c)]
    if budget:
        cands = [c for c in cands if c.get("cout_partiel_fcfa", 0) <= budget]
    deja_proposes = _EXCLURE.get()
    if deja_proposes:                                  # « autre chose » : on retire ce qui a déjà été proposé
        cands = [c for c in cands if not any(_quasi_identiques(c, p) for p in deja_proposes)]
    if not cands:
        return []

    def marche_finale(c):
        e = c["etapes"][-1]
        return e.get("distance_m", 0) if e.get("mode") == "MARCHE" else 0
    courtes = [c for c in cands if marche_finale(c) <= SEUIL_MARCHE_LONGUE_M]
    cands = courtes or cands

    def cle(c):                                        # plus petit = meilleur
        base = c.get("nb_arrets", 0) + _acces_m(c) / 80.0     # 80 m à pied = 1 min = environ 1 arrêt en bus
        corr, duree = c.get("nb_correspondances", 0), c.get("duree_minutes", 999)
        if priorite == "prix":
            return (corr, c.get("cout_partiel_fcfa", 0), base, duree)
        return (corr, duree, base) if priorite == "duree" else (corr, base, duree)

    def famille(c):
        return frozenset(e["mode"] for e in _etapes_transport(c))

    def cle_groupe(c):                                 # deux trajets de même clé = même proposition
        tr = _etapes_transport(c)
        if len(tr) == 1:
            return ("D", tr[0]["depart"], tr[0]["arrivee"])
        return tuple((e["mode"], "" if e["mode"] in ("GBAKA", "WORO_WORO") else _nom_court_ligne(e),
                      e["depart"], e["arrivee"]) for e in tr[:-1]) + (tr[-1]["depart"],)

    def grouper(liste):
        groupes = {}
        for c in sorted(liste, key=cle):
            groupes.setdefault(cle_groupe(c), []).append(c)
        fusion = []
        for g in sorted(groupes.values(), key=lambda g: cle(g[0])):
            for k in fusion:                           # quasi-doublon : on fusionne
                if _quasi_identiques(k[0], g[0]):
                    k.extend(g)
                    break
            else:
                fusion.append(g)
        return fusion
    mini = min(c.get("nb_correspondances", 0) for c in cands)
    choisis = grouper([c for c in cands if c.get("nb_correspondances", 0) == mini])[:maxi]
    if mode_filtre == "TOUS" and choisis:              # on ajoute de la variété (autre mode)
        deja = {m for g in choisis for m in famille(g[0])}
        dur_ref = choisis[0][0].get("duree_minutes", 999) or 999
        for g in grouper([c for c in cands if c.get("nb_correspondances", 0) <= mini + 1]):
            if len(choisis) >= maxi:
                break
            if (famille(g[0]) - deja) and all(g is not x for x in choisis) \
                    and (g[0].get("duree_minutes", 999) or 999) <= dur_ref * 1.5:
                choisis.append(g)
                deja |= famille(g[0])
    resultat = []
    for n, g in enumerate(choisis, 1):
        rep = copy.deepcopy(g[0])
        tr = _etapes_transport(rep)
        derniere = tr[-1]
        noms = []
        for e in tr:
            nom_l = _nom_court_ligne(e)
            nom_l = nom_l[:25].rstrip() + "…" if len(nom_l) > 26 else nom_l
            noms.append(f"{NOM_MODE.get(e['mode'], e['mode'].lower())} {nom_l}".strip())
        rep["libelle"] = f"Option {n} · " + " puis ".join(noms)
        nb_c = rep.get("nb_correspondances", 0)
        rep["resume"] = (("Trajet direct" if nb_c == 0 else f"{nb_c} correspondance{'s' if nb_c > 1 else ''}")
                         + f" · {rep.get('nb_arrets', 0)} arrêts")
        # autres lignes qui font le même trajet (même arrêt de descente) : proposées en « alternatives »
        alternatives, vus = [], {_nom_court_ligne(derniere)}
        for autre in g[1:]:
            ta = _etapes_transport(autre)[-1]
            nom = _nom_court_ligne(ta)
            if nom in vus or ta.get("mode") != derniere.get("mode") or ta["arrivee"] != derniere["arrivee"] \
                    or ta["depart"] != derniere["depart"]:
                continue
            vus.add(nom)
            alternatives.append({"ligne": nom, "mode": ta["mode"], "nb_arrets": ta.get("nb_arrets", 0)})
        alternatives = sorted(alternatives, key=lambda a: a["nb_arrets"])[:5]
        rep["alternatives"] = alternatives
        if alternatives:
            liste = ", ".join(f"{NOM_MODE.get(a['mode'], a['mode'].lower())} {a['ligne']}"
                              + (f" ({a['nb_arrets']} arrêts)" if a["nb_arrets"] else "") for a in alternatives)
            derniere["texte"] = (derniere.get("texte") or "") + \
                f" Autres lignes possibles depuis le même arrêt : {liste}."
        resultat.append(rep)
    return resultat

# 6. CONSTRUCTION DES TRAJETS
@functools.lru_cache(maxsize=None)
def _arrets_proches_de(sid: str, rayon_m: float = 300) -> tuple:
    """Arrêts à moins de rayon_m d'un arrêt donné (pour les correspondances à pied)."""
    if "grille_corr" not in RESEAU:
        RESEAU["grille_corr"] = _grille_arrets(0.003, [s for s in RESEAU["arrets"]
                                                       if RESEAU["lignes_par_arret"].get(s)])
    a = RESEAU["arrets"][sid]
    res = []
    for autre in _voisins(RESEAU["grille_corr"], a["latitude"], a["longitude"], 0.003):
        b = RESEAU["arrets"][autre]
        d = _dist_rapide(a["latitude"], a["longitude"], b["latitude"], b["longitude"])
        if d <= rayon_m:
            res.append((autre, d))
    return tuple(res)

def _lignes_voisines() -> dict:
    """Pour chaque ligne : les lignes qu'on peut prendre en correspondance (arrêts à ≤ 250 m)."""
    if "voisines" not in RESEAU:
        voisines, approx = defaultdict(set), _rids_approximatifs()
        ok = lambda r: r not in approx and RESEAU["lignes"].get(r, {}).get("type") != "WORO_WORO"
        for rid, sequences in RESEAU["sequences"].items():
            if not ok(rid):
                continue
            for sid in [x for x in sequences[0] if x not in RESEAU["points_libres"]][::2]:
                for autre, _ in _arrets_proches_de(sid, 250):
                    voisines[rid].update(r2 for r2 in RESEAU["lignes_par_arret"].get(autre, []) if r2 != rid and ok(r2))
        RESEAU["voisines"] = voisines
    return RESEAU["voisines"]

def _lignes_proches(point: dict, rayon_m: float, modes: set) -> dict:
    """TOUTES les lignes (bus, gbaka, bateau) qui passent à moins de rayon_m d'un point.
    Pour chaque ligne on garde SON arrêt le plus proche du point.
    Résultat : {(id_ligne, n° du sens): (id_arret, distance_m, rang_dans_la_sequence)}"""
    approx, res = _rids_approximatifs(), {}
    for a in RESEAU["arrets"].values():
        d = distance_m(point["latitude"], point["longitude"], a["latitude"], a["longitude"])
        if d > rayon_m:
            continue
        for rid in RESEAU["lignes_par_arret"].get(a["id"], ()):
            ligne = RESEAU["lignes"].get(rid)
            if rid in approx or not ligne or ligne["type"] not in modes or ligne["type"] == "WORO_WORO":
                continue
            for s, seq in enumerate(RESEAU["sequences"].get(rid, [])):
                if a["id"] in seq and ((rid, s) not in res or d < res[(rid, s)][1]):
                    res[(rid, s)] = (a["id"], int(d), seq.index(a["id"]))
    return res

def _garder_par_mode(lignes: dict, n: int = 12) -> list:
    """Garde, pour chaque mode (bus, gbaka, bateau), les n lignes dont l'arrêt est le plus proche."""
    gardes = []
    for mode in ("BUS", "GBAKA", "BATEAU"):
        gardes += sorted((kv for kv in lignes.items() if RESEAU["lignes"][kv[0][0]]["type"] == mode),
                         key=lambda kv: kv[1][1])[:n]
    return gardes

def _meilleure_descente(seq: list, i_montee: int, dest: dict, maxi_m: float):
    """Après la montée, quel arrêt de la ligne est le plus proche de la destination ?
    Retourne (rang, distance) ou None si tous sont à plus de maxi_m."""
    meilleur = None
    for j in range(i_montee + 1, len(seq)):
        a = RESEAU["arrets"].get(seq[j])
        if a:
            d = distance_m(dest["latitude"], dest["longitude"], a["latitude"], a["longitude"])
            if meilleur is None or d < meilleur[1]:
                meilleur = (j, d)
    return meilleur if meilleur and meilleur[1] <= maxi_m else None

def _meilleur_transfert(seq_a, apres_a, seq_b, avant_b, rayon=300, ligne_b_bus=False):
    """Meilleur endroit pour passer de la ligne A à la ligne B (arrêts proches à pied).
    Retourne (rang de descente sur A, rang de montée sur B, distance à pied, id_A, id_B)."""
    pos_b = {}
    for k, sid in enumerate(seq_b):
        if k <= avant_b and sid not in pos_b:
            pos_b[sid] = k
    meilleur = None
    for i in range(apres_a + 1, len(seq_a)):
        if seq_a[i] in RESEAU["points_libres"]:
            continue
        for autre, d in _arrets_proches_de(seq_a[i], max(rayon, 800) if ligne_b_bus else rayon):
            k = pos_b.get(autre)
            if k is None:
                continue
            terminus = ligne_b_bus and k == 0          # monter au terminus de la ligne B : on tolère 800 m
            if d > rayon and not (terminus and d <= 800):
                continue
            score = (i - apres_a) + (avant_b - k) + d / 150.0 - (4 if terminus else 0)
            if not meilleur or score < meilleur[0]:
                meilleur = (score, i, k, d, seq_a[i], autre)
    return meilleur[1:] if meilleur else None

def _pos(a: dict) -> dict:
    return {"latitude": a["latitude"], "longitude": a["longitude"]}

def _etape_marche(nom_dep, nom_arr, p, q, distance, minimum=1) -> dict:
    return {"mode": "MARCHE", "depart": nom_dep, "arrivee": nom_arr, "distance_m": int(distance),
            "duree_minutes": max(minimum, int(distance) // 80), "cout_fcfa": 0,
            "depart_pos": _pos(p), "arrivee_pos": _pos(q)}

def _duree_sur_ligne(ids: list) -> int:
    """Durée en minutes : on compte 350 m par minute le long des arrêts."""
    total = sum(_dist_rapide(RESEAU["arrets"][a]["latitude"], RESEAU["arrets"][a]["longitude"],
                             RESEAU["arrets"][b]["latitude"], RESEAU["arrets"][b]["longitude"])
                for a, b in zip(ids, ids[1:]) if a in RESEAU["arrets"] and b in RESEAU["arrets"])
    return max(5, int(total // 350))

def _assembler(dep: dict, dest: dict, trajets_lignes: list) -> dict:
    """Fabrique un trajet complet à partir d'une liste de (id_ligne, arrêts_parcourus) :
    marche -> ligne 1 -> (marche -> ligne 2 ...) -> marche. Calcule durée, marche, correspondances."""
    A = RESEAU["arrets"]
    etapes = []
    precedent = None                                   # (nom, position) du dernier point où l'on se trouve
    precedent_pos, precedent_nom = dep, dep.get("nom", "depart")
    for n, (rid, ids) in enumerate(trajets_lignes):
        m, d, ligne = A[ids[0]], A[ids[-1]], RESEAU["lignes"][rid]
        d_marche = distance_m(precedent_pos["latitude"], precedent_pos["longitude"], m["latitude"], m["longitude"])
        etapes.append(_etape_marche(precedent_nom, m["nom"], precedent_pos, m, d_marche, 2 if n else 1))
        etape = {"mode": ligne["type"], "numero_ligne": ligne.get("numero"), "nom_ligne": ligne.get("nom"),
                 "depart": m["nom"], "arrivee": d["nom"], "depart_pos": _pos(m), "arrivee_pos": _pos(d),
                 "distance_m": distance_m(m["latitude"], m["longitude"], d["latitude"], d["longitude"]),
                 "duree_minutes": _duree_sur_ligne(ids), "cout_fcfa": _prix_ligne(ligne),
                 "nb_arrets": len(ids) - 1}
        if len(trajets_lignes) == 1:                   # trajet direct : on détaille tous les arrêts
            etape["arrets"] = [{"id": s, "nom": A[s]["nom"], "latitude": A[s]["latitude"],
                                "longitude": A[s]["longitude"]} for s in ids if s in A]
        etapes.append(etape)
        precedent_pos, precedent_nom = d, d["nom"]
    d_fin = distance_m(precedent_pos["latitude"], precedent_pos["longitude"], dest["latitude"], dest["longitude"])
    etapes.append(_etape_marche(precedent_nom, dest.get("nom", "destination"), precedent_pos, dest, d_fin))
    return {"duree_minutes": sum(e["duree_minutes"] for e in etapes),
            "cout_fcfa": sum(e["cout_fcfa"] for e in etapes),
            "marche_m": sum(e["distance_m"] for e in etapes if e["mode"] == "MARCHE"),
            "nb_correspondances": len(trajets_lignes) - 1, "etapes": etapes,
            "arrets_itineraire": etapes[1].get("arrets", []) if len(trajets_lignes) == 1 else []}

def _score_trajet(c: dict) -> int:
    """Plus petit = meilleur : durée + (marche convertie en minutes)."""
    return c["duree_minutes"] + c.get("marche_m", 0) // 160

def _candidats_deux_correspondances(dep, dest, departs, arrivees, prix_max_ligne=None) -> list:
    """Trajets avec 2 correspondances (3 lignes) : ligne 1 -> ligne 2 -> ligne 3."""
    voisines, A = _lignes_voisines(), RESEAU["arrets"]
    seqs = RESEAU["sequences"]
    par_rid_arr = defaultdict(list)
    for (r, s), v in arrivees.items():
        par_rid_arr[r].append((s, v))
    resultats, essais = {}, 0
    for (r1, s1), (sid1, _, i1) in sorted(departs.items(), key=lambda kv: kv[1][1])[:25]:
        for r2 in voisines.get(r1, ()):
            for r3 in voisines.get(r2, ()):
                if r3 in (r1, r2) or r3 not in par_rid_arr:
                    continue
                essais += 1
                if essais > 400:
                    return list(resultats.values())
                if prix_max_ligne and max(_prix_ligne(RESEAU["lignes"][r]) for r in (r1, r2, r3)) > prix_max_ligne:
                    continue
                seq1 = seqs[r1][s1]
                for seq2 in seqs[r2]:
                    for s3, (_, _, k_desc) in sorted(par_rid_arr[r3], key=lambda x: x[1][1])[:3]:
                        seq3 = seqs[r3][s3]
                        t1 = _meilleur_transfert(seq1, i1, seq2, len(seq2) - 2)
                        t2 = _meilleur_transfert(seq2, t1[1], seq3, k_desc - 1) if t1 else None
                        if not t2:
                            continue
                        ids = [seq1[i1:t1[0] + 1], seq2[t1[1]:t2[0] + 1], seq3[t2[1]:k_desc + 1]]
                        if any(len(x) < 2 or _dist_rapide(A[x[0]]["latitude"], A[x[0]]["longitude"],
                                                          A[x[-1]]["latitude"], A[x[-1]]["longitude"]) < 1500
                               for x in ids):          # une ligne trop courte ne vaut pas le détour
                            continue
                        cand = _assembler(dep, dest, list(zip((r1, r2, r3), ids)))
                        if (r1, r2, r3) not in resultats or _score_trajet(cand) < _score_trajet(resultats[(r1, r2, r3)]):
                            resultats[(r1, r2, r3)] = cand
    return list(resultats.values())

def _candidats_depuis_catalogue(dep, dest, modes, arrets_dep, arrets_arr) -> list:
    """Dernier recours : lignes du catalogue CSV sans séquence d'arrêts. On les accepte si un arrêt
    proche du départ ET un arrêt proche de la destination sont sur son tracé et dans ses noms."""
    def lie(arrets, nom_ligne):                        # un arrêt dont le nom ressemble au terminus
        for a in arrets:
            nom = nettoyer_texte(a["nom"])
            if any(m in nom for m in nom_ligne.split() if len(m) > 3) or \
               any(m in nom_ligne for m in nom.split() if len(m) > 3):
                return a
        return None
    sortie = []
    for c in CATALOGUE:
        ori, des = nettoyer_texte(c.get("origine") or ""), nettoyer_texte(c.get("destination") or "")
        if c["type"] not in modes or not ori or not des or not c.get("trace") or not c.get("bbox"):
            continue
        montee, descente = lie(arrets_dep, ori), lie(arrets_arr, des)
        if not (montee and descente):                  # on essaie dans l'autre sens
            montee, descente = lie(arrets_dep, des), lie(arrets_arr, ori)
        if not (montee and descente) or not all(
                _point_proche_du_trace(a["latitude"], a["longitude"], c["trace"], c["bbox"])
                for a in (montee, descente)):
            continue
        prix = c.get("prix_fcfa") or _prix_ligne({"numero": c.get("numero"), "type": c["type"]})
        etape = {"mode": c["type"], "numero_ligne": c.get("numero"),
                 "nom_ligne": f"{c['origine']} ↔ {c['destination']}",
                 "depart": montee["nom"], "arrivee": descente["nom"], "depart_pos": _pos(montee),
                 "arrivee_pos": _pos(descente), "duree_minutes": 40, "cout_fcfa": prix,
                 "distance_m": distance_m(montee["latitude"], montee["longitude"],
                                          descente["latitude"], descente["longitude"])}
        fin = _etape_marche(descente["nom"], dest.get("nom", "destination"), descente, dest, descente["distance_m"])
        debut = _etape_marche(dep.get("nom", "depart"), montee["nom"], dep, montee, montee["distance_m"])
        sortie.append({"duree_minutes": debut["duree_minutes"] + 40 + fin["duree_minutes"], "cout_fcfa": prix,
                       "marche_m": montee["distance_m"] + descente["distance_m"], "nb_correspondances": 0,
                       "etapes": [debut, etape, fin], "arrets_itineraire": []})
        if len(sortie) >= 5:
            break
    return sortie

def _ajouter_conseils_marche(c: dict):
    """Si une marche dépasse 1 km, on conseille un gbaka / woro-woro / taxi."""
    for position, e in enumerate(c["etapes"]):
        if e["mode"] != "MARCHE" or e.get("distance_m", 0) <= SEUIL_MARCHE_LONGUE_M:
            continue
        km = e["distance_m"] / 1000
        lignes = _lignes_libres_proches(e["depart_pos"], e["arrivee_pos"])
        debut = f"L'arrêt est à {km:.1f} km" if position == 0 else f"Il reste {km:.1f} km à parcourir"
        if lignes:
            genre = "gbaka" if lignes[0][0] == "GBAKA" else "woro-woro"
            e["conseil"] = (f"{debut} : prenez plutôt le {genre} « {lignes[0][1]} » qui passe "
                            + ("près de vous pour le rejoindre." if position == 0 else
                               "dans ce secteur si vous ne voulez pas marcher.")).replace("prenez plutôt", "prenez plutôt", 1)
        else:
            e["conseil"] = f"{debut} : prenez plutôt un woro-woro, un gbaka ou un taxi."

def _lignes_libres_proches(pos_a: dict, pos_b: dict) -> list:
    """Gbakas / woro-woro dont le tracé passe près des deux points (pour éviter une longue marche)."""
    noms = []
    for c in CATALOGUE:
        if c["type"] in ("GBAKA", "WORO_WORO") and not c.get("estime") and c.get("trace") and c.get("bbox") \
                and _point_proche_du_trace(pos_a["latitude"], pos_a["longitude"], c["trace"], c["bbox"], 250) \
                and _point_proche_du_trace(pos_b["latitude"], pos_b["longitude"], c["trace"], c["bbox"], 500):
            noms.append((c["type"], f"{c['origine']} ↔ {c['destination']}"))
            if len(noms) >= 2:
                break
    return noms

def conseil_marche_chat(options: list) -> str:
    """Phrase pour le chat quand une marche est longue (même idée que _ajouter_conseils_marche)."""
    conseils, vus = [], set()
    for c in options or []:
        for e in c.get("etapes", []):
            cle = (e.get("depart"), e.get("arrivee"))
            if e.get("mode") != "MARCHE" or e.get("distance_m", 0) <= SEUIL_MARCHE_LONGUE_M or cle in vus \
                    or not e.get("depart_pos") or not e.get("arrivee_pos"):
                continue
            vus.add(cle)
            km, lignes = e["distance_m"] / 1000, _lignes_libres_proches(e["depart_pos"], e["arrivee_pos"])
            if lignes:
                genre = "gbaka" if lignes[0][0] == "GBAKA" else "woro-woro"
                conseils.append(f"la marche de {km:.1f} km vers {e['arrivee']} est longue : le {genre} "
                                f"« {lignes[0][1]} » passe dans ce secteur (prix variable selon le chauffeur)")
            else:
                conseils.append(f"la marche de {km:.1f} km vers {e['arrivee']} est longue et je n'ai pas de "
                                f"gbaka/woro-woro connu sur ce tronçon : prévois de marcher")
            if len(conseils) >= 2:
                break
    return " ; ".join(conseils)

def _variante_woro_si_longue_marche(c: dict):
    """Copie du trajet où la longue marche du début est remplacée par un woro-woro."""
    etapes = c.get("etapes", [])
    if not etapes or etapes[0].get("mode") != "MARCHE" or etapes[0].get("distance_m", 0) <= SEUIL_MARCHE_LONGUE_M:
        return None
    v = copy.deepcopy(c)
    m = v["etapes"].pop(0)
    duree, attente = max(3, m["distance_m"] // 400), ATTENTE_MINUTES["WORO_WORO"]
    v["etapes"].insert(0, {"mode": "WORO_WORO", "depart": m["depart"], "arrivee": m["arrivee"],
                           "depart_pos": m["depart_pos"], "arrivee_pos": m["arrivee_pos"],
                           "distance_m": m["distance_m"], "duree_minutes": duree, "cout_fcfa": 0,
                           "attente_minutes": attente})
    v["duree_minutes"] += duree + attente - m.get("duree_minutes", 0)
    v["marche_m"] = max(0, v.get("marche_m", 0) - m["distance_m"])
    v["variante"] = "woro_remplace_marche"
    return v

def _filtrer_trajets_gbaka(candidats: list) -> list:
    """Mode gbaka seul : on écarte les trajets avec marche entre deux gbakas > 250 m ou grand détour."""
    marche_finale = lambda c: c["etapes"][-1].get("distance_m", 0) if c["etapes"][-1]["mode"] == "MARCHE" else 0
    marche_entre = lambda c: max([e.get("distance_m", 0) for e in c["etapes"][1:-1] if e["mode"] == "MARCHE"] or [0])
    detour = lambda c: any(e["mode"] == "GBAKA" and e["duree_minutes"] * 350 > 2.2 * e["distance_m"] + 1500
                           for e in c["etapes"])
    bons = [c for c in candidats if marche_entre(c) <= 250 and not detour(c)]
    if any(marche_finale(c) <= 800 for c in bons):
        bons = [c for c in bons if marche_finale(c) <= 800]
    return sorted(bons, key=lambda c: (c["nb_correspondances"], _score_trajet(c)))

def _signature(c: dict) -> tuple:
    """Identité d'un trajet : la suite (ligne, montée, descente) de ses étapes en véhicule."""
    return tuple((e.get("mode"), e.get("numero_ligne") or e.get("nom_ligne"), e.get("depart"), e.get("arrivee"))
                 for e in c.get("etapes", []) if e.get("mode") != "MARCHE")

def _construire_candidats_rayon(dep: dict, dest: dict, mode_filtre: str, prix_max_ligne,
                                rayon: int, descente_max: int) -> List[dict]:
    """Cherche les trajets : on monte à moins de `rayon` m du départ et on descend à moins
    de `descente_max` m de la destination."""
    if not RESEAU["charge"]:
        charger_reseau_local()
    modes = {"BUS", "GBAKA", "BATEAU"} if mode_filtre == "TOUS" else {mode_filtre}
    # 1) Toutes les lignes qui passent près du départ / près de la destination
    #    (pour chaque ligne : son arrêt le plus proche).
    departs = _lignes_proches(dep, rayon, modes)
    arrivees = _lignes_proches(dest, descente_max, modes)
    arrets_dep = arrets_proximite(dep["latitude"], dep["longitude"], rayon, tri_par_distance=True)
    arrets_arr = arrets_proximite(dest["latitude"], dest["longitude"], descente_max, tri_par_distance=True)
    log.info("[trajets] mode=%s rayon=%s lignes_dep=%d lignes_arr=%d", mode_filtre, rayon, len(departs), len(arrivees))
    if not arrets_dep or not arrets_arr:
        return []
    # Un gbaka dépose plus près : 500 m (ou le gbaka le plus proche de la destination + 100 m)
    d_gbaka = min((v[1] for (r, _), v in arrivees.items() if RESEAU["lignes"][r]["type"] == "GBAKA"), default=0)
    limite_gbaka = min(descente_max, max(DESCENTE_MAX_GBAKA_M, d_gbaka + 100))
    S = RESEAU["sequences"]
    prix_ok = lambda rid: not prix_max_ligne or _prix_ligne(RESEAU["lignes"][rid]) <= prix_max_ligne
    # 2) DIRECT : on monte à l'arrêt de la ligne le plus proche de moi, on descend au plus proche de la destination
    candidats, meilleur = [], {}
    for (rid, s), (sid, _, i) in departs.items():
        fin = _meilleure_descente(S[rid][s], i, dest, descente_max)
        if fin and prix_ok(rid):
            c = _assembler(dep, dest, [(rid, S[rid][s][i:fin[0] + 1])])
            if rid not in meilleur or _score_trajet(c) < _score_trajet(meilleur[rid]):
                meilleur[rid] = c
    candidats += meilleur.values()
    # 3) UNE CORRESPONDANCE : ligne 1 (près du départ) -> ligne 2 (près de la destination)
    dist_corr = 250 if modes == {"GBAKA"} else 500
    meilleur = {}
    premiers, seconds = _garder_par_mode(departs), _garder_par_mode(arrivees)
    for (r1, s1), (_, _, i1) in premiers:
        for (r2, s2), (_, _, i2) in seconds:
            if r1 == r2 or not (prix_ok(r1) and prix_ok(r2)):
                continue
            t = _meilleur_transfert(S[r1][s1], i1, S[r2][s2], i2 - 1, dist_corr,
                                    RESEAU["lignes"][r2]["type"] == "BUS")
            if t and t[0] > i1 and i2 > t[1]:
                c = _assembler(dep, dest, [(r1, S[r1][s1][i1:t[0] + 1]), (r2, S[r2][s2][t[1]:i2 + 1])])
                if (r1, r2) not in meilleur or _score_trajet(c) < _score_trajet(meilleur[(r1, r2)]):
                    meilleur[(r1, r2)] = c
    candidats += list(meilleur.values())[:25]
    # 4) DEUX CORRESPONDANCES (seulement si le départ est loin des arrêts) puis dernier recours : catalogue
    if not any(c["etapes"][0].get("distance_m", 0) <= SEUIL_MARCHE_LONGUE_M for c in candidats):
        candidats += _candidats_deux_correspondances(dep, dest, departs, arrivees, prix_max_ligne)
    if not candidats:
        candidats = _candidats_depuis_catalogue(dep, dest, modes, arrets_dep, arrets_arr)
    # 5) Nettoyage : descente trop loin pour un gbaka, attente, variantes woro-woro
    def descente_trop_loin(c):
        tr = [e for e in c["etapes"] if e["mode"] != "MARCHE"]
        return bool(tr) and tr[-1]["mode"] in ("GBAKA", "WORO_WORO") and c["etapes"][-1]["mode"] == "MARCHE" \
            and c["etapes"][-1].get("distance_m", 0) > limite_gbaka
    candidats = [c for c in candidats if not descente_trop_loin(c)]
    if departs:                                        # on ne propose pas de monter beaucoup plus loin que l'arrêt le plus proche
        limite_montee = max(600, 3 * min(v[1] for v in departs.values()))
        proches = [c for c in candidats if c["etapes"][0].get("distance_m", 0) <= limite_montee]
        candidats = proches or candidats
    if modes == {"GBAKA"}:
        candidats = _filtrer_trajets_gbaka(candidats)
    for c in candidats:
        for e in c["etapes"]:
            if e["mode"] != "MARCHE" and "attente_minutes" not in e:
                e["attente_minutes"] = ATTENTE_MINUTES.get(e["mode"], 5)
                c["duree_minutes"] += e["attente_minutes"]
    plus_proche = min(departs.items(), key=lambda kv: kv[1][1])[1][0] if departs else None
    for c in list(candidats):                          # variante : woro-woro au lieu de la longue marche
        tr = _etapes_transport(c)
        if plus_proche and tr and (tr[0].get("depart_pos") or {}).get("latitude") is not None:
            a = RESEAU["arrets"][plus_proche]
            if distance_m(a["latitude"], a["longitude"], tr[0]["depart_pos"]["latitude"],
                          tr[0]["depart_pos"]["longitude"]) > 100:
                continue
        v = _variante_woro_si_longue_marche(c)
        if v:
            candidats.append(v)
    # 6) Tri : (a) moins de correspondances/vite, (b) en pénalisant la longue marche ; on mélange les deux listes
    par_vehicules = sorted(candidats, key=lambda c: (c["nb_correspondances"], _score_trajet(c), c["cout_fcfa"]))
    par_marche = sorted(candidats, key=lambda c: (
        c["nb_correspondances"] + (1 if c["etapes"][0].get("distance_m", 0) > SEUIL_MARCHE_LONGUE_M else 0),
        _score_trajet(c), c["cout_fcfa"]))
    uniques, vus = [], set()
    for c in [x for paire in zip(par_vehicules, par_marche) for x in paire]:
        sig = tuple((e.get("numero_ligne") or e.get("nom_ligne"), e["depart"], e["arrivee"])
                    for e in c["etapes"] if e["mode"] != "MARCHE")
        if sig not in vus:
            vus.add(sig)
            uniques.append(c)
    candidats = uniques[:10]
    for c in candidats:                                # on enlève les marches de correspondance < 30 m
        n = len(c["etapes"])
        c["etapes"] = [e for i, e in enumerate(c["etapes"])
                       if not (e["mode"] == "MARCHE" and 0 < i < n - 1 and e.get("distance_m", 0) < 30)]
        _ajouter_conseils_marche(c)
    return candidats

def construire_candidats(dep: dict, dest: dict, mode_filtre: str = "TOUS", prix_max_ligne=None) -> List[dict]:
    """Point d'entrée : on cherche d'abord un trajet qui dépose à ≤ 1000 m de la destination ;
    si rien, on élargit la zone de départ, et en tout dernier recours la distance de descente."""
    for rayon, descente in ((RAYON_DEPART_M, DESCENTE_MAX_M), (2500, DESCENTE_MAX_M),
                            (4000, DESCENTE_MAX_M), (4000, 2 * DESCENTE_MAX_M)):
        resultats = _construire_candidats_rayon(dep, dest, mode_filtre, prix_max_ligne, rayon, descente)
        if resultats:
            for c in resultats:
                _normaliser_candidat(c)
            return resultats
    return []

def trajets_par_mode(dep: dict, dest: dict, prix_max_ligne=None) -> dict:
    """Les meilleurs trajets séparés par mode (pour l'écran « explorer »)."""
    return {"bus": construire_candidats(dep, dest, "BUS", prix_max_ligne)[:5],
            "gbakas": construire_candidats(dep, dest, "GBAKA", prix_max_ligne)[:5],
            "bateaux": construire_candidats(dep, dest, "BATEAU", prix_max_ligne)[:5]}

def candidats_multimodaux(dep: dict, dest: dict, mode_filtre: str = "TOUS", prix_max_ligne=None) -> list:
    """Mode imposé -> uniquement ce mode (sinon trajets qui le COMBINENT avec un autre).
    Mode TOUS -> bus + gbaka + bateau-bus + trajets mixtes."""
    if mode_filtre in ("BUS", "GBAKA", "BATEAU"):
        autorises = {"BUS": {"BUS"}, "GBAKA": {"GBAKA", "WORO_WORO"}, "BATEAU": {"BATEAU"}}[mode_filtre]
        modes_de = lambda c: {e["mode"] for e in c.get("etapes", []) if e["mode"] != "MARCHE"}
        purs = [c for c in construire_candidats(dep, dest, mode_filtre, prix_max_ligne) if modes_de(c) <= autorises]
        if not purs:                                   # aucun trajet 100 % dans ce mode : trajets qui le combinent
            purs = [c for c in construire_candidats(dep, dest, "TOUS", prix_max_ligne)
                    if modes_de(c) & autorises and not modes_de(c) <= autorises]
            for c in purs:
                c["combo_mode"] = mode_filtre
        for c in purs:
            _ajouter_textes(c)
        return purs
    par_mode = trajets_par_mode(dep, dest, prix_max_ligne)
    sortie, vus = [], set()
    for groupe in (construire_candidats(dep, dest, "TOUS", prix_max_ligne),
                   par_mode["bus"], par_mode["gbakas"], par_mode["bateaux"]):
        for c in groupe:
            if _signature(c) not in vus:
                vus.add(_signature(c))
                _ajouter_textes(c)
                sortie.append(c)
    return sortie

def arrets_proches_trajet(dep: dict, dest: dict, mode_filtre: str = "TOUS", rayon_m: int = 3000) -> dict:
    """Les 5 arrêts les plus proches du départ et de la destination (pour l'écran carte)."""
    modes = None if mode_filtre == "TOUS" else [mode_filtre]
    return {"depart": arrets_proximite_par_modes(dep["latitude"], dep["longitude"], rayon_m, modes)[:5],
            "arrivee": arrets_proximite_par_modes(dest["latitude"], dest["longitude"], rayon_m, modes)[:5]}


# ── 7. API POUR LE FRONTEND (écrans « lignes », « détail d'une ligne », recherche) ──
def prix_connu(mode, nom_ligne, montee, descente, numero):
    if numero and str(numero) in PRIX_LIGNES:
        return PRIX_LIGNES[str(numero)]
    return PRIX_POUR_CLASSEMENT.get(mode, 250)

def normaliser_reseau(reseau: str, type_ligne: str) -> str:
    """Nom propre du réseau (Monbus, Wibus, Monbato...), « Autre » si inconnu."""
    r = nettoyer_texte(reseau or "")
    mots = {"BATEAU": (("monbato", "Monbato"), ("stl", "STL"), ("aqua", "Aqualines")),
            "BUS": (("express", "Express"), ("wibus", "Wibus"), ("navette", "Navette"),
                    ("monbus", "Monbus"), ("mon bus", "Monbus")),
            "GBAKA": (("adja", "Adjamé"), ("abobo", "Abobo"))}
    for cle, nom in mots.get(type_ligne, ()):
        if cle in r:
            return nom
    return "Autre"

def lignes_du_catalogue(type_ligne=None) -> list:
    """Liste des lignes du catalogue, au format attendu par l'application."""
    resultat = []
    for i, c in enumerate(CATALOGUE):
        if type_ligne and c["type"] != type_ligne:
            continue
        numero = c["numero"]
        if c["type"] == "BUS" and numero and str(numero).isdigit() and 1 <= int(numero) <= 9:
            numero = f"{int(numero):02d}"                  # 5 -> « 05 »
        ori = nettoyer_texte(c.get("origine") or "")
        reseau = normaliser_reseau("adja" if c["type"] == "GBAKA" and "adja" in ori else
                                   "abobo" if c["type"] == "GBAKA" and "abobo" in ori else c.get("reseau"),
                                   c["type"])
        resultat.append({"id": f"cat:{i}", "numero": numero or None, "origine": c["origine"],
                         "destination": c["destination"], "nom": f"{c['origine']} ↔ {c['destination']}",
                         "operateur": c["operateur"], "reseau": reseau, "type": c["type"],
                         "prix_fcfa": c.get("prix_fcfa") or prix_connu(c["type"], None, "", "", c["numero"])})
    return resultat

def liste_lignes(type_transport=None) -> list:
    return lignes_du_catalogue(type_transport)

def lignes_pour_app(type_transport=None, q=None, zone=None, reseau=None) -> dict:
    """Écran « Explorer » : lignes filtrées par type, réseau et mot-clé, avec les compteurs."""
    t = (type_transport or "").strip().upper()
    genre = t if t in ("BUS", "GBAKA", "BATEAU", "WORO_WORO") else None
    toutes = lignes_du_catalogue(genre)
    filtres = [{"nom": "Toutes", "valeur": "toutes", "total": len(toutes)}]
    filtres += [{"nom": n, "valeur": n.lower(), "total": t2}
                for n, t2 in Counter(l["reseau"] for l in toutes).most_common()]
    choisies = toutes
    if reseau and reseau.lower() not in ("toutes", "tous", ""):
        choisies = [l for l in choisies if nettoyer_texte(l["reseau"]) == nettoyer_texte(reseau)]
    if q:
        mots = extraire_mots(q)
        choisies = [l for l in choisies if all(
            m in nettoyer_texte(f"{l.get('numero') or ''} {l['origine']} {l['destination']}") for m in mots)]
    for l in choisies:
        if l["type"] in ("GBAKA", "WORO_WORO"):
            l["sens"] = _arrets_ligne_pour_api(l["id"])
    return {"type": genre, "reseau": reseau or "toutes", "total": len(choisies),
            "total_type": len(toutes), "filtres": filtres, "lignes": choisies}

def _mots_cle(texte: str) -> list:
    return [m for m in nettoyer_texte(texte or "").split()
            if len(m) > 2 and m not in ("terminus", "gare", "arret")]

def _score_nom(cible: str, nom: str) -> float:
    """Ressemblance entre deux noms de lieux (0 à 1), tolère les petites fautes."""
    mots_c, mots_n = _mots_cle(cible), _mots_cle(nom)
    if not mots_c:
        return 0.0
    ok = sum(1 for mc in mots_c if any(
        mc == mn or (len(mc) >= 5 and len(mn) >= 5 and (mc.startswith(mn[:5]) or mn.startswith(mc[:5])))
        or difflib.SequenceMatcher(None, mc, mn).ratio() >= 0.82 for mn in mots_n))
    return ok / len(mots_c)

def _sens_ligne_bus(fiche, index_catalogue, origine, destination, numero) -> list:
    """Aller / retour d'une ligne BUS ou BATEAU (les arrêts affichés dans l'écran détail).
    On prend d'abord les arrêts propres à la ligne du CSV (cat:N) pour ne jamais mélanger
    deux lignes qui ont le même numéro ; sinon on se rabat sur le GTFS."""
    rid_cat = f"cat:{index_catalogue}" if index_catalogue is not None else None
    S, L, A = RESEAU["sequences"], RESEAU["lignes"], RESEAU["arrets"]
    gtfs = [rid for rid, i in L.items() if not rid.startswith("cat:") and i.get("type") == fiche["type"]]
    candidates = []
    if str(numero).strip() == "610":                   # la 610 : le GTFS est plus fiable que le tracé en boucle
        for rid in gtfs:
            if str(L[rid].get("numero") or "").strip() == "610":
                candidates += S.get(rid, [])
    if not candidates and rid_cat and S.get(rid_cat):
        candidates += S[rid_cat]
    if not candidates and numero:                      # repli GTFS : même numéro + nom de ligne qui ressemble
        variantes = {numero} | ({str(int(numero)), f"{int(numero):02d}"} if numero.isdigit() else set())
        for rid in gtfs:
            num = str(L[rid].get("numero") or "").strip()
            if num in variantes or (num.isdigit() and numero.isdigit() and int(num) == int(numero)):
                nom_l = nettoyer_texte(L[rid].get("nom") or "")
                if _score_nom(origine, nom_l) >= 0.5 or _score_nom(destination, nom_l) >= 0.5:
                    candidates += S.get(rid, [])
    if not candidates and numero:                      # dernier recours : une seule ligne GTFS avec ce numéro
        rids = [r for r in gtfs if str(L[r].get("numero") or "").strip() == str(numero).strip()]
        if len(rids) == 1:
            candidates += S.get(rids[0], [])
    if not candidates:
        return []
    dist_ids = lambda x, y: distance_m(A[x]["latitude"], A[x]["longitude"], A[y]["latitude"], A[y]["longitude"])

    def nettoyee(seq):                                 # enlève les arrêts répétés de même nom
        out, prec = [], None
        for sid in seq:
            if sid in A and nettoyer_texte(A[sid]["nom"]) != prec:
                prec = nettoyer_texte(A[sid]["nom"])
                out.append(sid)
        return out

    def moities(seq):                                  # une boucle contient l'aller ET le retour : on la coupe
        if len(seq) < 6 or dist_ids(seq[0], seq[-1]) > 500:
            return [seq]
        loin = [dist_ids(seq[0], s) for s in seq]
        m = max(range(len(seq)), key=lambda i: loin[i])
        m = max(range(max(1, m - 3), min(len(seq) - 1, m + 3) + 1),
                key=lambda i: (_score_nom(destination, A[seq[i]]["nom"]), loin[i]))
        return [seq] if m < 2 or len(seq) - m < 3 else [seq[:m + 1], seq[m:]]
    cands = []
    for sq in candidates:
        for m in moities(nettoyee(sq)):
            if len(m) >= 2 and m not in cands:
                cands.append(m)
    if not cands:
        return []

    def repere(nom):                                   # l'arrêt qui porte le nom de l'origine / destination
        meilleur, sc = None, 0.5
        for sq in cands:
            for sid in sq:
                v = _score_nom(nom, A[sid]["nom"])
                if v > sc or (v == sc and meilleur is None):
                    meilleur, sc = sid, v
        return meilleur
    rep_o, rep_d = repere(origine), repere(destination)

    def cle_sens(seq):
        sc = _score_nom(origine, A[seq[0]]["nom"]) + _score_nom(destination, A[seq[-1]]["nom"])
        ecart = (dist_ids(seq[0], rep_o) if rep_o else 0) + (dist_ids(seq[-1], rep_d) if rep_d else 0)
        return (round(sc, 2), -ecart, len(seq))
    aller = max(cands, key=cle_sens)
    ecart_retour = lambda c: dist_ids(c[0], aller[-1]) + dist_ids(c[-1], aller[0])
    autres = [c for c in cands if c != aller]
    retour = min(autres, key=ecart_retour) if autres and str(numero).strip() != "610" else None
    if retour is None or ecart_retour(retour) > 1500 \
            or [A[x]["nom"] for x in retour] == [A[x]["nom"] for x in aller]:
        retour = list(reversed(aller))

    def fabrique(seq, label, direction, nom_depart, nom_terminus):
        arrets = [{"id": A[s]["id"], "nom": A[s]["nom"], "latitude": A[s]["latitude"],
                   "longitude": A[s]["longitude"], "role": "arret"} for s in seq]
        arrets[0]["role"], arrets[-1]["role"] = "depart", "arrivee"
        arrets[0]["nom"], arrets[-1]["nom"] = nom_depart or arrets[0]["nom"], nom_terminus or arrets[-1]["nom"]
        return {"direction": direction, "sens": label, "titre": f"{arrets[0]['nom']} → {arrets[-1]['nom']}",
                "depart": arrets[0]["nom"], "terminus": arrets[-1]["nom"], "nb_arrets": len(arrets),
                "arrets": arrets, "trace": []}
    return [fabrique(aller, "aller", "A", origine, destination),
            fabrique(retour, "retour", "R", destination, origine)]

def detail_ligne(ident: str):
    """Écran « détail d'une ligne » : infos + arrêts de l'aller et du retour."""
    if not RESEAU["charge"]:
        charger_reseau_local()
    index = int(ident[4:]) if ident.startswith("cat:") and ident[4:].isdigit() and int(ident[4:]) < len(CATALOGUE) \
        else next((i for i, c in enumerate(CATALOGUE) if str(c.get("numero") or "") == str(ident)), None)
    if index is None:
        return None
    fiche = CATALOGUE[index]
    origine, destination = (fiche.get("origine") or "").strip(), (fiche.get("destination") or "").strip()
    numero = str(fiche.get("numero") or "").strip()
    res = {"id": f"cat:{index}", "numero": numero or None, "type": fiche["type"],
           "operateur": fiche.get("operateur"), "reseau": normaliser_reseau(fiche.get("reseau"), fiche["type"]),
           "nom": f"{origine} ↔ {destination}", "depart": origine, "arrivee": destination,
           "prix_fcfa": fiche.get("prix_fcfa") or prix_connu(fiche["type"], None, "", "", numero),
           "nb_arrets": None, "sens": []}
    if fiche["type"] in ("GBAKA", "WORO_WORO"):
        sens_csv = _arrets_ligne_pour_api(f"cat:{index}")
        if sens_csv:
            res["nb_arrets"] = max(s["nb_arrets"] for s in sens_csv)
            for s in sens_csv:
                arrets = [dict(a) for a in s["arrets"]]
                arrets[0]["role"], arrets[-1]["role"] = "depart", "arrivee"
                res["sens"].append({"direction": "A" if s["sens"] == "aller" else "R", "sens": s["sens"],
                                    "titre": f"{s['depart']} → {s['arrivee']}", "depart": s["depart"],
                                    "terminus": s["arrivee"], "nb_arrets": s["nb_arrets"],
                                    "arrets": arrets, "trace": []})
            return res
    sens_bus = _sens_ligne_bus(fiche, index, origine, destination, numero)
    if sens_bus:
        res["sens"], res["nb_arrets"] = sens_bus, max(x["nb_arrets"] for x in sens_bus)
    else:                                              # aucun arrêt connu : on affiche au moins départ/arrivée
        res["nb_arrets"] = 2
        res["sens"] = [{"direction": "A", "sens": "aller", "titre": f"{origine} → {destination}",
                        "depart": origine, "terminus": destination, "nb_arrets": 2, "trace": [],
                        "arrets": [{"id": None, "nom": origine, "latitude": None, "longitude": None, "role": "depart"},
                                   {"id": None, "nom": destination, "latitude": None, "longitude": None,
                                    "role": "arrivee"}]}]
    return res

def recherche_globale(q: str, type_transport=None, maximum: int = 20) -> dict:
    """Barre de recherche : lignes + arrêts qui contiennent tous les mots tapés."""
    mots = extraire_mots(q)
    if not mots:
        return {"lignes": [], "arrets": []}
    arrets = [a for a in RESEAU["arrets"].values() if a["id"] not in RESEAU["points_libres"]
              and all(m in nettoyer_texte(a["nom"]) for m in mots)]
    return {"lignes": lignes_pour_app(type_transport, q)["lignes"][:maximum], "arrets": arrets[:maximum]}

def trajets_pour_demande(candidats, priorite=None, prix_max=None, budget=None):
    """Trie les trajets par prix ou par durée (et applique le budget)."""
    if not candidats:
        return []
    cout = lambda c: c.get("cout_partiel_fcfa", 0)
    if budget:
        candidats = [c for c in candidats if cout(c) <= budget]
    candidats.sort(key=(lambda c: (cout(c), c.get("duree_minutes", 999))) if priorite == "prix"
                   else (lambda c: (c.get("duree_minutes", 999), cout(c))))
    return candidats

def instructions_trajet(candidat: dict) -> list:
    """Liste de phrases simples décrivant le trajet."""
    if not candidat or not candidat.get("etapes"):
        return []
    return [f"marchez {e['distance_m']} m jusqu a {e['arrivee']}" if e["mode"] == "MARCHE" else
            f"prenez le {e['mode'].lower()} {e.get('numero_ligne') or e.get('nom_ligne') or e['mode']} "
            f"de {e['depart']} a {e['arrivee']}" for e in candidat["etapes"]]


# 8. ASSISTANT MOBIGUIDE (comprendre le message + répondre avec le LLM Groq)
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
# Mémoire de conversation : une case par session_id (départ, destination, budget, mode...)
SESSIONS = defaultdict(lambda: {"depart": None, "destination": None, "dernier_trajet": None,
                                "budget_fcfa": None, "mode_transport": None, "attente_mode": False})

def maj_session(session_id: str, intention: dict, trajet: dict = None, attente_mode=None):
    """Met à jour la mémoire de la conversation avec ce que l'on vient de comprendre."""
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
            s["mode_demande"] = True                   # la question « quel mode ? » n'est posée qu'une fois

def extraire_budget(texte: str) -> Optional[int]:
    """Comprend « j'ai 400 », « 400 balles », « moins de 300 », « 2 mille », « 1 500 francs »..."""
    t = (texte or "").lower().replace(" ", " ").replace(" ", " ")
    nombre = r"(\d{1,3}(?:[ .]\d{3})+|\d+)"
    motifs = (nombre + r"\s*(?:mille|milles|k)\b",                                      # « 2 mille »
              nombre + r"\s*(?:fcfa|f\.?\s?cfa|cfa|francs?|frs?\b|f\b|balles?|piastres?|billets?)",
              r"(?:budget|maximum|max|co[uû]t|prix|dans les|moins de|pas plus de|jusqu['’]?\s?[àa]|"
              r"j['’]?ai(?: seulement| juste| que)?|je n['’]?ai que|il me reste|me reste|je dispose de|"
              r"je poss[eè]de|avec|(?:un )?petit budget de)\D{0,12}" + nombre,
              r"\s*" + nombre + r"\s*[.!]?\s*")                                         # message = « 400 »
    for n, motif in enumerate(motifs):
        m = re.fullmatch(motif, t) if n == 3 else re.search(motif, t)
        if m:
            valeur = int(re.sub(r"\D", "", m.group(1))) * (1000 if n == 0 else 1)
            return valeur if 50 <= valeur <= 1000000 else None
    return None

def extraire_mode(texte: str, accepter_tous: bool = False) -> Optional[str]:
    """BUS / GBAKA / BATEAU / TOUS, ou None. Comprend la négation (« pas de bus », « sans gbaka »)."""
    t = re.sub(r"bateau[\s_-]*bus", "bateau", nettoyer_texte(texte))
    trouves = set()
    for mode, mots in (("BATEAU", r"(bateau|bateaux|lagunaire|bato|monbato)"),
                       ("GBAKA", r"(gbaka|gbakas|woro|woros)"), ("BUS", r"(bus|sotra|autobus|monbus)")):
        nie = re.search(rf"(?:pas|sans|au lieu (?:de |du |des |d')|plut[oô]t que |non)\s*"
                        rf"(?:le |la |les |un |une |des )?\b{mots}\b", t)
        if re.search(rf"\b{mots}\b", t) and not nie:
            trouves.add(mode)
    if len(trouves) == 1:
        return next(iter(trouves))
    if len(trouves) > 1 or re.search(r"(tous les (modes|transports|moyens)|n.?importe (quel|lequel)|peu importe|"
                                     r"indifferent|pas de preference)", t) \
            or (accepter_tous and re.search(r"\b(tous|tout|les deux)\b", t)):
        return "TOUS"
    return None

def _est_question_conditions(message: str) -> bool:
    """Le message parle-t-il de trafic, de pluie, de météo ?"""
    m = (message or "").lower()
    return bool(re.search(r"(quel temps|le temps qu|il fait quel temps|niveau de (?:la )?pluie)", m)) or any(
        w in m for w in ("embouteillage", "trafic", "circulation", "bouchon", "pressé", "presse",
                         "pressée", "pluie", "pleut", "météo", "meteo"))

def _extraire_infos_conditions(contexte: dict) -> str:
    """Résume en une phrase le trafic et la météo calculés par le modèle ML."""
    def premier(*cles):
        return next((contexte[c] for c in cles if contexte.get(c) is not None), None)
    trafic, pluie, facteur = premier("trafic_niveau", "trafic"), premier("pluie", "weather", "meteo"), contexte.get("trafic_facteur")
    infos = []
    if isinstance(trafic, str):
        libelle = {"eleve": "dense", "moyen": "modéré", "faible": "fluide"}.get(nettoyer_texte(trafic), trafic)
        extra = f", les trajets durent environ x{facteur:.2f}" if isinstance(facteur, (int, float)) and facteur > 1.05 else ""
        infos.append(f"le trafic est {libelle}{extra}")
    elif isinstance(trafic, (int, float)):
        etat = "dense" if trafic >= 0.7 else "modéré" if trafic >= 0.4 else "fluide"
        infos.append(f"le trafic est {etat} (niveau {trafic:.2f})")
    elif trafic is not None:
        infos.append(f"niveau de trafic : {trafic}")
    if isinstance(pluie, bool):
        infos.append("il pleut" if pluie else "il ne pleut pas")
    elif isinstance(pluie, (int, float)):
        infos.append(f"il pleut (intensité {pluie:.2f})" if pluie > 0.5 else f"il ne pleut pas (intensité {pluie:.2f})")
    elif pluie is not None:
        infos.append(f"météo : {pluie}")
    return " ; ".join(infos) if infos else "conditions normales (non calculées)"

def _completer_analyse(data: dict, s: dict, mode_msg, budget_msg) -> dict:
    """Complète le résultat de l'analyse avec la mémoire de la session (départ, budget, mode...)."""
    if data.setdefault("intention", "INCONNUE") == "INCONNUE":
        data["intention"] = "CONVERSATION"             # jamais de réponse figée : on discute normalement
    if data["intention"] == "SUIVI" and not s.get("dernieres_options"):
        data["intention"] = "CONVERSATION"
    for cle in ("depart", "destination", "priorite", "prix_max_ligne"):
        data.setdefault(cle, None)
    data.update(mode_transport="TOUS", mode_precise=False, budget_fcfa=None)
    if data["intention"] in ("TRAJET", "LIGNES_TRAJET"):
        for cle in ("depart", "destination"):
            vide = data.get(cle) is None or (isinstance(data[cle], str) and data[cle].strip().lower() in ("", "null", "none", "aucun"))
            data[cle] = (None if vide else data[cle]) or s.get(cle)
        if mode_msg:
            data["mode_transport"], data["mode_precise"] = mode_msg, True
        elif s.get("mode_demande"):
            data["mode_precise"] = True                # question « quel mode ? » déjà posée : on cherche dans TOUS
        data["budget_fcfa"] = budget_msg or s.get("budget_fcfa")   # (le mode de session n'est PAS réutilisé)
    return data

def _trajet_demande(s, mode, mode_precise, budget, **plus) -> dict:
    """Intention « TRAJET » qui réutilise le départ et la destination déjà connus."""
    return {"intention": "TRAJET", "depart": s.get("depart"), "destination": s.get("destination"),
            "mode_transport": mode, "mode_precise": mode_precise, "priorite": None,
            "budget_fcfa": budget, "prix_max_ligne": None, **plus}

def analyser_intention(message: str, historique: list = None, session_id: str = None) -> dict:
    """Comprend le message : TRAJET, SUIVI (question sur les trajets), CONVERSATION... + lieux/budget/mode."""
    s = SESSIONS[session_id] if session_id else {}
    _EXCLURE.set(None)
    msg = nettoyer_texte(message)
    mode_msg, budget_msg = extraire_mode(message, accepter_tous=bool(s.get("attente_mode"))), extraire_budget(message)
    question = re.search(r"(c.?est quoi|qu.?est.?ce|\bpourquoi\b|comment fonctionne|\bcombien\b|est.?ce que|"
                         r"\bou est\b|\bquel(le)?s? (est|sont)\b)", msg)
    # 1) Règles rapides (sans LLM)
    if re.search(r"(comment (tu|vous) (t|vous )?.?appel|qui (es|etes).?(tu|vous)|tu es qui|ton nom|"
                 r"c.?est quoi ton nom|tu fais quoi|presente.?toi)", msg):
        return _completer_analyse({"intention": "CONVERSATION"}, s, mode_msg, budget_msg)
    if s.get("dernieres_options") and re.search(           # « pas celui-là, autre chose »
            r"(pas convaincu|ne me convient pas|ça ne me convient pas|ca ne me convient pas|autre trajet|"
            r"autre option|change|changer|propose autre|quelque chose d.?autre|non,? merci|je n.?aime pas|"
            r"je (ne )?veux (pas|plus)|pas (ce|cet|celui|ces|ceux)|enleve|retire|sans ce trajet)", msg):
        _EXCLURE.set(list(s.get("dernieres_options") or []))
        return _trajet_demande(s, mode_msg or "TOUS", bool(mode_msg), budget_msg or s.get("budget_fcfa"),
                               exclure_signatures=[_signature(o) for o in s["dernieres_options"]])
    if mode_msg and s.get("destination") and (s.get("attente_mode") or not question):   # « et en gbaka ? »
        return _trajet_demande(s, mode_msg, True, budget_msg or s.get("budget_fcfa"))
    if budget_msg and s.get("destination") and (s.get("depart") or s.get("dernieres_options")) \
            and not re.search(r"\bcombien\b|c.?est quoi|\bpourquoi\b|\bcomment fonctionne\b", msg):
        mode_session = s.get("mode_transport")                                           # « j'ai 400 »
        return _trajet_demande(s, mode_msg or mode_session or "TOUS", bool(mode_msg or mode_session), budget_msg)
    if not client:
        salut = any(m in message.lower() for m in ("bonjour", "salut", "merci", "hello", "bonsoir"))
        return _completer_analyse({"intention": "CONVERSATION" if salut else "INCONNUE"}, s, mode_msg, budget_msg)
    # 2) Sinon on demande au LLM de classer le message (réponse JSON)
    historique_texte = "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in (historique or [])[-10:])
    memoire = ""
    if s.get("depart") or s.get("destination"):
        memoire = f"""
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
    prompt = f"""Tu es un assistant de transport à Abidjan. Réponds UNIQUEMENT avec un JSON valide.
Message : "{message}"
Historique récent :
{historique_texte}
{memoire}
Format exact :
{{"intention": "TRAJET" ou "SUIVI" ou "CONVERSATION" ou "LIGNES_TRAJET" ou "INCONNUE",
  "depart": "lieu de départ ou null", "destination": "lieu d'arrivée ou null",
  "priorite": "duree" ou "prix" ou null, "prix_max_ligne": nombre ou null}}
Règles :
- "ma position", "ici", "je suis là", "près de moi" -> depart = "ma position actuelle"
- CONVERSATION pour TOUT ce qui n'est pas un déplacement (salutation, merci, culture générale, blague, question sur toi...)
- TRAJET uniquement si l'utilisateur veut aller quelque part ou relance/modifie la recherche
- N'invente jamais un lieu : null s'il n'est ni dans le message ni dans la mémoire. Rien en dehors du JSON.
"""
    try:
        r = client.chat.completions.create(
            model=MODEL, temperature=0.1, max_tokens=400,
            messages=[{"role": "system", "content": "Tu réponds uniquement en JSON valide, sans aucun texte autour."},
                      {"role": "user", "content": prompt}])
        texte = r.choices[0].message.content.strip()
        debut, fin = texte.find("{"), texte.rfind("}") + 1
        if debut >= 0 and fin > debut:
            return _completer_analyse(json.loads(texte[debut:fin]), s, mode_msg, budget_msg)
    except Exception as e:
        log.error("erreur analyse intention : %s", e)
    return _completer_analyse({"intention": "INCONNUE"}, s, mode_msg, budget_msg)

# Textes de réponse 
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
    "n'invente jamais une information, une ligne, un prix ou une fonctionnalité.")
SYSTEME_TRAJET = (
    "Tu es MobiGuide, assistant de transport à Abidjan, chaleureux et naturel, comme un ami qui "
    "connaît bien la ville. Tu réponds toujours en français. Tu ne répètes jamais une formule "
    "d'accueil ni un conseil déjà dit dans la conversation. Tu n'inventes jamais de prix ni "
    "d'arrêt et tu ne renvoies jamais vers Google Maps ou Waze. "
    "Quand un mode de transport est demandé (bateau-bus, bus, gbaka) et qu'aucun trajet n'existe "
    "dans ce mode, dis-le clairement et propose d'autres modes uniquement si l'utilisateur le demande. "
    "Ne dis jamais « il n'y a pas de trajet en bateau-bus » puis propose encore des trajets qui contiennent du bateau-bus.")

def _messages_llm(systeme: str, prompt: str, historique: list) -> list:
    """Messages envoyés au LLM : consigne + 10 derniers échanges + question."""
    msgs = [{"role": "system", "content": systeme}]
    for m in (historique or [])[-10:]:
        contenu = str(m.get("content", "")).strip()
        if contenu:
            msgs.append({"role": "assistant" if m.get("role") in ("assistant", "model", "bot") else "user",
                         "content": contenu})
    return msgs + [{"role": "user", "content": prompt}]

def _phrase_complete(texte: str, fini_par_limite: bool) -> str:
    """Si le LLM a été coupé (max_tokens), on garde seulement les phrases terminées."""
    texte = (texte or "").strip()
    if not fini_par_limite or not texte or re.search(r"[.!?…][^\w\s]{0,3}\s*$", texte):
        return texte
    fin = max(texte.rfind(c) for c in ".!?…")
    return texte[:fin + 1].strip() if fin > 20 else texte

def _prix_texte(c: dict) -> str:
    if not c:
        return "inconnu"
    if c.get("cout_fcfa") is not None:
        return f"{c['cout_fcfa']} FCFA"
    base = "variable (le tarif des gbakas / woro-woro dépend du chauffeur ou de l'apprenti)"
    partiel = c.get("cout_partiel_fcfa") or 0
    return f"{base}, partie bus/bateau : {partiel} FCFA" if partiel else base

def _ordinal(n: int) -> str:
    return "1er" if n == 1 else f"{n}e"

def _points_cles(c: dict) -> str:
    return " ; ".join(
        f"{e['mode'].lower()} {e.get('numero_ligne') or e.get('nom_ligne') or e['mode']} : montée à {e['depart']}, "
        f"descente à {e['arrivee']}" + (f" (au {_ordinal(e['nb_arrets'])} arrêt)" if (e.get("nb_arrets") or 0) > 0 else "")
        for e in c.get("etapes", []) if e["mode"] != "MARCHE")

def _conseils_liste(options: list, budget=None) -> list:
    """Conseils pratiques selon les modes réellement proposés : liste de (clé, texte)."""
    modes = {e["mode"] for c in (options or [])[:4] for e in c.get("etapes", []) if e.get("mode") != "MARCHE"}
    textes = [("BUS", "En bus, prends ton ticket ou badge ta carte auprès du machiniste."),
              ("GBAKA", "Pour les gbakas, le prix n'est pas fixe : demande-le à l'apprenti avant de monter."),
              ("WORO_WORO", "Pour un woro-woro, demande le prix au chauffeur avant de monter."),
              ("BATEAU", "Pour le bateau-bus, garde ton ticket jusqu'à la descente.")]
    out = [(m, t) for m, t in textes if m in modes]
    if modes:
        out.append(("SIGNALER", "Signale ton arrêt de descente au chauffeur ou à l'apprenti."))
    if budget and modes & {"GBAKA", "WORO_WORO"}:
        out.append((f"BUDGET{budget}", f"Prévois un peu de marge sur tes {budget} FCFA pour les gbakas et woro-woro."))
    return out


def _dist_txt(m) -> str:
    m = int(m or 0)
    return f"{m / 1000:.1f} km".replace(".", ",") if m >= 1000 else f"{m} m"

def decrire_trajet(c: dict) -> str:
    """Raconte le trajet étape par étape (marche, montée, rang de l'arrêt de descente, correspondance).
    Tout vient des données calculées, rien n'est inventé."""
    etapes = c.get("etapes", []) or []
    transports = [i for i, e in enumerate(etapes) if e.get("mode") not in ("MARCHE", "WORO_WORO")]
    dernier_t = transports[-1] if transports else -1
    lignes, derniere_arrivee = [], None
    for i, e in enumerate(etapes):
        mode, d = e.get("mode"), e.get("distance_m", 0) or 0
        if mode == "MARCHE":
            if d < 30:
                continue
            duree = f" (environ {e['duree_minutes']} min)" if e.get("duree_minutes") else ""
            if any(x.get("mode") != "MARCHE" for x in etapes[i + 1:]):
                t = f"Marche {_dist_txt(d)} jusqu'à l'arrêt « {e.get('arrivee')} »{duree}."
                if d > SEUIL_MARCHE_LONGUE_M:
                    t += " C'est assez loin : tu peux aussi prendre un taxi ou un gbaka/woro-woro pour t'y rendre."
            else:
                t = f"Il te restera {_dist_txt(d)} à pied jusqu'à ta destination{duree}."
                if d > SEUIL_MARCHE_LONGUE_M:
                    t += " C'est assez long : un taxi ou un woro-woro peut t'y déposer."
            lignes.append(t)
        elif mode == "WORO_WORO":
            lignes.append(f"Pour rejoindre l'arrêt « {e.get('arrivee')} » ({_dist_txt(d)}), prends un woro-woro ou un taxi.")
        else:
            if derniere_arrivee and derniere_arrivee != e.get("depart") and etapes[i - 1].get("mode") != "MARCHE":
                lignes.append(f"Rejoins l'arrêt « {e.get('depart')} » (tout près, à pied).")
            nom = "le " + NOM_MODE.get(mode, str(mode).lower()) + (f" {e['numero_ligne']}" if e.get("numero_ligne") else "")
            n = e.get("nb_arrets") or 0
            descente = f"descends au {_ordinal(n)} arrêt, à « {e.get('arrivee')} »" if n > 0 else f"descends à « {e.get('arrivee')} »"
            lignes.append(f"Monte dans {nom} à « {e.get('depart')} », puis {descente}"
                          + (" pour prendre la correspondance." if i < dernier_t else "."))
            derniere_arrivee = e.get("arrivee")
    return "\n".join(f"{k}. {t}" for k, t in enumerate(lignes, 1))

def message_aucun_trajet(mode: str, dep: dict = None, dest: dict = None) -> str:
    """Message CLAIR quand aucun trajet n'existe. Dans un mode précis (ex : bateau-bus) : on dit non, on cite
    l'arrêt de ce mode le plus proche et on propose UNIQUEMENT les AUTRES modes (pas de boucle)."""
    precis = mode in ("BUS", "GBAKA", "BATEAU")
    lib = NOM_MODE[mode] if precis else "transport"
    texte = (f"Je n'ai trouvé aucun trajet en {lib} entre ces deux lieux." if precis else
             "Je n'ai trouvé aucun trajet entre ces deux lieux.")
    if dep and dest:
        proches = []
        for qui, lieu in (("ton départ", dep), ("ta destination", dest)):
            a = arrets_proximite_par_modes(lieu["latitude"], lieu["longitude"], 5000, [mode] if precis else None)[:1]
            if a:
                proches.append(f"{qui} : « {a[0]['nom']} » à {_dist_txt(a[0]['distance_m'])}")
        texte += (f" L'arrêt de {lib} le plus proche de " + " et de ".join(proches) + ".") if proches else \
            f" Il n'y a pas d'arrêt de {lib} près de ces lieux."
    if not precis:
        return texte + " Vérifie ton départ et ta destination, ou précise un quartier."
    autres = [NOM_MODE[m] for m in ("BUS", "GBAKA", "BATEAU") if m != mode]
    return texte + f" Je peux chercher en {' ou en '.join(autres)}, ou avec tous les modes : tu veux ?"


def _intro_trajet(message, contexte, candidats, mode, historique, premiere, memoire_txt, infos_conditions) -> str:
    """Une phrase d'introduction naturelle (LLM) avant la liste des étapes."""
    modele = "Voici ce que je te propose !" if premiere else "Voici ce que j'ai trouvé."
    if not client:
        return modele
    consigne = ("C'est la première fois que tu présentes des trajets dans cette conversation : accueille "
                "l'utilisateur chaleureusement." if premiere else
                "Tu as déjà présenté des trajets dans cette conversation : fais une phrase courte et "
                "naturelle, SANS formule d'accueil (pas de « avec plaisir », « bonne nouvelle », « super »), "
                "qui reconnaît ce qu'il vient de demander (autre mode, autre budget, autre trajet...).")
    prompt = (f'L\'utilisateur vient de demander : "{message}".\n'
              f"Tu as trouvé {len(candidats)} option(s). Mode demandé : {mode}. "
              f"Budget : {contexte.get('budget_fcfa') or 'aucun'}.\n"
              f"Conditions actuelles (à ne mentionner que si c'est vraiment utile) : {infos_conditions or 'aucune'}\n"
              f"{consigne}\n"
              "Écris UNE seule phrase de 25 mots maximum. Ne décris pas les étapes (elles seront écrites "
              "juste après), ne donne aucun conseil, n'invente rien.")
    try:
        r = client.chat.completions.create(model=MODEL, temperature=0.6, max_tokens=140,
                                           messages=_messages_llm(SYSTEME_TRAJET + memoire_txt, prompt, historique))
        texte = (r.choices[0].message.content or "").strip().strip('"').strip().split("\n")[0].strip()
        return _phrase_complete(texte, r.choices[0].finish_reason == "length") or modele
    except Exception as e:
        log.error("erreur intro trajet : %s", e)
        return modele

def _reponse_trajet(message, contexte, candidats, mode, historique, s, memoire_txt, infos_conditions) -> str:
    """Réponse quand on présente des trajets : intro + option 1 détaillée + conseils nouveaux."""
    premiere, budget = not s.get("nb_reponses_trajet"), contexte.get("budget_fcfa")
    if not candidats:
        if _EXCLURE.get():
            return ("Je n'ai pas d'autre trajet vraiment différent entre ces deux lieux. "
                    "Tu veux essayer un autre mode (bus, gbaka, bateau-bus) ?")
        if mode and mode != "TOUS":
            return message_aucun_trajet(mode, contexte.get("depart_lieu"), contexte.get("destination_lieu"))
        return ("Je n'ai trouvé aucun trajet pour l'instant. Vérifie ton départ et ta destination, "
                "ou essaie un autre mode de transport.")
    combo = candidats[0].get("combo_mode")
    if combo:
        lib = NOM_MODE.get(combo, combo.lower())
        intro = (f"Je n'ai pas de trajet 100 % {lib} entre ces deux lieux, mais voici des trajets "
                 f"qui combinent le {lib} avec un autre mode.")
    else:
        intro = _intro_trajet(message, contexte, candidats, mode, historique, premiere, memoire_txt, infos_conditions)
    if budget and extraire_budget(message) == budget:
        intro += f" Tous ces trajets respectent ton budget de {budget} FCFA."
    position = (candidats[0]["etapes"][0].get("depart_pos") if candidats[0].get("etapes") else None)
    situe = _situer_sur_trajet(position, contexte.get("trajet_precedent"))
    parts = [intro] + ([situe] if situe else []) + ["Option 1 :\n" + decrire_trajet(candidats[0])]
    autres = len(candidats) - 1
    if autres > 0:
        parts.append(f"J'ai aussi {autres} autre{'s' if autres > 1 else ''} option{'s' if autres > 1 else ''} dans les cartes.")
    deja = set(s.get("conseils_donnes") or [])
    nouveaux = [(k, t) for k, t in _conseils_liste(candidats, budget) if k not in deja]
    if nouveaux:                                       # on ne répète jamais un conseil déjà donné
        parts.append("💡 " + " ".join(t for _, t in nouveaux))
        s["conseils_donnes"] = list(deja | {k for k, _ in nouveaux})
    s["nb_reponses_trajet"] = (s.get("nb_reponses_trajet") or 0) + 1
    return "\n\n".join(parts)

def _situer_sur_trajet(position: dict, options: list) -> str:
    """Accompagnement : la personne dit où elle est. On regarde si cet endroit est sur le trajet
    déjà proposé (arrêts en coordonnées GPS) et on lui dit où elle en est."""
    if not position or not options:
        return ""
    meilleur = None
    for n, e in enumerate([e for e in options[0].get("etapes", []) if e.get("mode") not in ("MARCHE", "WORO_WORO")], 1):
        points = [(e["depart"], e.get("depart_pos")), (e["arrivee"], e.get("arrivee_pos"))] + \
                 [(a["nom"], a) for a in e.get("arrets", [])]
        for nom, p in points:
            if p and p.get("latitude") is not None:
                d = distance_m(position["latitude"], position["longitude"], p["latitude"], p["longitude"])
                if d <= 400 and (not meilleur or d < meilleur[0]):
                    meilleur = (d, nom, n, e)
    if not meilleur:
        return ""
    d, nom, n, e = meilleur
    ligne = f"{NOM_MODE.get(e['mode'], '')} {e.get('numero_ligne') or ''}".strip()
    if nom == e["arrivee"]:
        return (f"Tu es tout près de « {nom} » ({_dist_txt(d)}), l'arrêt de descente de l'étape {n} de ton trajet "
                f"({ligne}) : descends-y, puis suis la suite ci-dessous.")
    return (f"Tu es tout près de « {nom} » ({_dist_txt(d)}), sur l'étape {n} de ton trajet ({ligne}) : "
            f"tu peux la continuer et descendre à « {e['arrivee']} ».")

def generer_reponse(message: str, contexte: dict, historique: list = None, session_id: str = None) -> str:
    """Réponse finale de MobiGuide selon le statut : TRAJET, SUIVI (question sur un trajet) ou CONVERSATION."""
    historique, statut = historique or [], contexte.get("statut", "")
    candidats, mode = contexte.get("candidats") or [], contexte.get("mode", "TOUS")
    s = SESSIONS[session_id] if session_id else {}
    infos_conditions = _extraire_infos_conditions(contexte)
    memoire_txt = ""
    if session_id and (s.get("depart") or s.get("destination")):
        memoire_txt = (f"\nContexte de la discussion : départ={s.get('depart')}, destination={s.get('destination')}, "
                       f"budget={s.get('budget_fcfa') or 'aucun'}, mode={s.get('mode_transport') or 'non précisé'}")
    if statut == "TRAJET":
        return _reponse_trajet(message, contexte, candidats, mode, historique, s, memoire_txt, infos_conditions)
    if not client:
        return ("Bonjour ! 😊 Je suis MobiGuide, ton guide à Abidjan. Dis-moi où tu veux aller et je te propose le meilleur trajet !"
                if statut == "CONVERSATION" else
                "Je suis MobiGuide. Dis-moi ton point de départ et ta destination, et je m'occupe du reste 😊")
    prompt = message
    if statut == "SUIVI":
        resume = "\n".join(f"- {c.get('libelle', 'proposition')} : {c.get('duree_minutes', '?')} min, "
                           f"prix {_prix_texte(c)}, {c.get('nb_correspondances', 0)} correspondance(s), "
                           f"{c.get('marche_m', 0)} m de marche ; {_points_cles(c)}" for c in candidats)
        prompt = f"""{message}

[Trajets que tu viens de proposer à l'utilisateur (cartes déjà affichées) :
{resume or 'aucun trajet en mémoire'}
Conditions actuelles : {infos_conditions}]
Réponds directement à SA question, en 2-3 phrases, avec ces infos, sur un ton chaleureux.
Ne répète pas les conseils ni l'accueil déjà donnés plus haut dans la conversation. N'ajoute un conseil
que s'il est nouveau et vraiment utile. Ne recopie pas tout l'itinéraire, n'invente rien."""
    elif statut == "CONVERSATION" and _est_question_conditions(message):
        prompt = f"""{message}

[Infos pour répondre : conditions actuelles calculées par notre modèle ML : {infos_conditions}]
Utilise UNIQUEMENT ces infos pour trafic / météo, ne renvoie jamais vers Google Maps ou Waze,
donne un conseil concret en 2-3 phrases maximum."""
    elif statut == "CONVERSATION":
        prompt = f"""{message}

Réponds directement et naturellement à ce message, comme un ami chaleureux, en 2 ou 3 phrases maximum.
Si on te demande ton nom ou qui tu es : tu es MobiGuide, l'assistant de l'application MobiGo pour se déplacer à Abidjan.
Pour une question hors transport (culture générale, blague, conseil...), réponds-y vraiment ; ne dis jamais que tu
ne réponds pas à ce genre de questions. Termine, quand c'est naturel, par une courte invitation à t'aider pour un trajet
(sans imposer de position ni d'itinéraire). Ne répète pas ce que tu as déjà dit."""
    try:
        r = client.chat.completions.create(model=MODEL, temperature=0.5, max_tokens=600,
                                           messages=_messages_llm(SYSTEME_AMI + memoire_txt, prompt, historique))
        return _phrase_complete(r.choices[0].message.content, r.choices[0].finish_reason == "length")
    except Exception as e:
        log.error("erreur generation reponse : %s", e)
        return "Je suis MobiGuide. Dis-moi où tu veux aller et je te propose un trajet."


choisir_options = choisir_propositions    # ancien nom utilisé par main.py