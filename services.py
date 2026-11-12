import csv, json, os, re, threading, unicodedata, copy, contextvars
from collections import Counter, defaultdict
from typing import Optional, List
import serpapi
from groq import Groq
from config import (DATA_DIR, GROQ_API_KEY, SERPAPI_KEY, CENTRE_ABIDJAN, MODEL,
    PRIX_LIGNES, PRIX_POUR_CLASSEMENT, RAYON_PROXIMITE_M,
    cache_get, cache_set, distance_m, log)


# SECTION OUTILS DE TEXTE
# Ces fonctions preparent le texte pour la comparaison

def nettoyer_texte(texte):
    # Enleve les accents et met tout en minuscules
    if not texte: return ""
    t = unicodedata.normalize("NFD", str(texte).lower())
    return "".join(c for c in t if unicodedata.category(c) != "Mn")

def extraire_mots(texte):
    # Decoupe un texte en mots utiles
    vides = {"de", "du", "des", "d", "l", "la", "le", "les", "a", "au", "abidjan"}
    return [m for m in re.findall(r"\w+", nettoyer_texte(texte)) if m not in vides]

def _charger_trace(wkt, max_points=1000):
    # Transforme un texte WKT en liste de points latitude longitude
    # et calcule la zone qui contient tous les points
    if not wkt or not isinstance(wkt, str): return [], None
    try:
        coords = re.findall(r"(-?\d+\.?\d+)\s+(-?\d+\.?\d+)", wkt)
        if not coords: return [], None
        points = [(float(lat), float(lon)) for lon, lat in coords]
        if len(points) > max_points:
            step = max(1, len(points) // max_points)
            points = points[::step]
        lats = [p[0] for p in points]
        lons = [p[1] for p in points]
        return points, (min(lats), min(lons), max(lats), max(lons))
    except Exception:
        return [], None

def _point_proche_du_trace(lat, lon, trace, bbox, seuil=300):
    # Verifie si un point est proche d'une trace
    if not trace or not bbox: return False
    mn_lat, mn_lon, mx_lat, mx_lon = bbox
    marge = seuil / 100000.0 + 0.001
    if not (mn_lat - marge <= lat <= mx_lat + marge): return False
    if not (mn_lon - marge <= lon <= mx_lon + marge): return False
    for tlat, tlon in trace:
        if distance_m(lat, lon, tlat, tlon) <= seuil: return True
    return False

def _parse_liste_arrets(s):
    # Separe une liste d arrets separes par des fleches
    if not s or not isinstance(s, str): return []
    s = s.strip()
    if "→" in s: parts = s.split("→")
    elif "->" in s: parts = s.split("->")
    else: parts = [s]
    return [p.strip() for p in parts if p.strip()]

def _parse_coords_str(s):
    # Transforme un texte longitude latitude en liste de tuples
    if not s or not isinstance(s, str): return []
    out = []
    for pair in s.split("|"):
        pair = pair.strip()
        if not pair: continue
        parts = [p.strip() for p in pair.split(",")]
        if len(parts) != 2: continue
        try: out.append((float(parts[0]), float(parts[1])))
        except ValueError: continue
    return out


# SECTION GARES
NOMS_GARES = ["gare", "gare nord", "gare sud", "gare routiere", "gare adjame",
    "adjame gare", "gare d adjame", "gare abobo", "abobo gare", "gare d abobo",
    "gare cocody", "gare de cocody", "gare plateau", "gare de plateau",
    "gare du plateau", "gare treichville", "gare de treichville", "treichville gare",
    "gare marcory", "gare de marcory", "marcory gare", "gare koumassi",
    "gare de koumassi", "koumassi gare", "gare port bouet", "gare de port bouet",
    "port bouet gare", "gare yopougon", "gare de yopougon", "yopougon gare",
    "gare attecoube", "gare d attecoube", "attecoube gare", "gare bassam", "gare de bassam"]

def est_gare(nom):
    # Verifie si un nom correspond a une gare
    n = nettoyer_texte(nom or "")
    if not n: return False
    if "gare" in n: return True
    return any(g in n for g in NOMS_GARES if g != "gare")


# SECTION GEOCODAGE
# Transforme un nom de lieu en coordonnees GPS

def rechercher_lieu(lieu):
    # Cherche les coordonnees GPS d un lieu avec serpapi
    if not lieu: return None
    lieu = lieu.strip()
    if not lieu: return None
    cle = ("geo", lieu.lower())
    deja = cache_get(cle)
    if deja: return deja
    for q in [f"{lieu}, Abidjan, Côte d'Ivoire", f"{lieu}, Abidjan", f"{lieu} Abidjan"]:
        try:
            res = serpapi.search({"api_key": SERPAPI_KEY, "engine": "google_maps",
                "type": "search", "q": q, "ll": CENTRE_ABIDJAN, "hl": "fr", "gl": "ci"}).as_dict()
        except Exception as e:
            log.error("erreur serpapi %s", e)
            continue
        cands = ([res["place_results"]] if res.get("place_results") else []) + (res.get("local_results") or [])
        for r in cands:
            c = r.get("gps_coordinates") or {}
            if c.get("latitude") is None or c.get("longitude") is None: continue
            resultat = {"nom": (r.get("title") or lieu).strip(),
                "latitude": float(c["latitude"]), "longitude": float(c["longitude"]),
                "alternatives": []}
            cache_set(cle, resultat)
            return resultat
    return None

def suggestions_lieu(lieu, limite=8):
    # Propose des lieux qui ressemblent au texte tape
    if not lieu or len(lieu.strip()) < 2: return []
    lieu = lieu.strip()
    cle = ("suggestions", lieu.lower())
    deja = cache_get(cle)
    if deja is not None: return deja
    resultats, vus = [], set()
    if not RESEAU["charge"]: charger_reseau_local()
    net = nettoyer_texte(lieu)
    for arret in RESEAU["arrets"].values():
        if arret["id"] in RESEAU["points_libres"]: continue
        nom_n = nettoyer_texte(arret["nom"])
        if net in nom_n or any(m in nom_n for m in net.split() if len(m) > 3):
            if arret["nom"] in vus: continue
            vus.add(arret["nom"])
            resultats.append({"nom": arret["nom"], "adresse": "",
                "latitude": arret["latitude"], "longitude": arret["longitude"]})
            if len(resultats) >= limite: break
    if len(resultats) < limite:
        try:
            res = serpapi.search({"api_key": SERPAPI_KEY, "engine": "google_maps",
                "type": "search", "q": f"{lieu}, Abidjan, Côte d'Ivoire",
                "ll": CENTRE_ABIDJAN, "hl": "fr", "gl": "ci"}).as_dict()
            cands = (res.get("local_results") or []) + ([res["place_results"]] if res.get("place_results") else [])
            for r in cands:
                c = r.get("gps_coordinates") or {}
                if not (c.get("latitude") and c.get("longitude")): continue
                nom = r.get("title") or "lieu"
                if nom in vus: continue
                vus.add(nom)
                resultats.append({"nom": nom, "adresse": r.get("address") or "",
                    "latitude": float(c["latitude"]), "longitude": float(c["longitude"])})
                if len(resultats) >= limite: break
        except Exception as e:
            log.warning("suggestions serpapi echouee %s", e)
    cache_set(cle, resultats)
    return resultats

MOTS_POSITION = ("position", "ici", "je suis la", "je suis ici", "pres de moi",
    "autour de moi", "chez moi", "mon emplacement", "ma localisation", "localisation")

def determiner_depart(dep_texte, lat=None, lon=None):
    # Determine le point de depart d un trajet
    a_gps = lat is not None and lon is not None
    texte = (dep_texte or "").strip()
    net = nettoyer_texte(texte)
    position = {"nom": "ma position actuelle",
        "latitude": float(lat) if a_gps else None,
        "longitude": float(lon) if a_gps else None, "alternatives": []}
    if a_gps and (not net or net in ("null", "none", "aucun")
                  or any(m in net for m in MOTS_POSITION)):
        return position
    if net and net not in ("null", "none", "aucun") and not any(m in net for m in MOTS_POSITION):
        lieu = rechercher_lieu(texte)
        if lieu: return lieu
    return position if a_gps else None


# SECTION DONNEES GLOBALES
CATALOGUE = []
RESEAU = {"arrets": {}, "lignes": {}, "sequences": {},
    "lignes_par_arret": defaultdict(list), "gares": set(), "points_libres": set(),
    "arrets_non_bus": set(), "arrets_bus": set(), "arrets_gbaka": set(), "charge": False}
_lock = threading.Lock()

def _trace_estimee(row):
    # Dit si la trace GPS a ete estimee et non mesuree
    o = nettoyer_texte(row.get("origine_trace") or "")
    c = nettoyer_texte(row.get("trace_complete") or "")
    return "approximatif" in o or "introuvable" in o or c == "non"

def charger_catalogue():
    # Lit les fichiers bus.csv, bateau_bus.csv et gbaka_woro.csv
    global CATALOGUE
    CATALOGUE = []
    for fichier, typ, col_prix, defaut, oper, res in [
        ("bus.csv", "BUS", "prix_fcfa", 200, "SOTRA", "Monbus"),
        ("bateau_bus.csv", "BATEAU", "prix_minimum_fcfa", 150, "SOTRA", "Monbato")]:
        chemin = os.path.join(DATA_DIR, fichier)
        if not os.path.exists(chemin): continue
        with open(chemin, encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f, delimiter=";"):
                origine = (row.get("depart") or "").strip()
                dest = (row.get("arrivee") or "").strip()
                if not origine or not dest: continue
                try: prix = int(float(row.get(col_prix) or defaut))
                except Exception: prix = defaut
                trace, bbox = _charger_trace(row.get("trace_wkt") or "")
                CATALOGUE.append({"numero": (row.get("numero_ligne") or "").strip(),
                    "origine": origine, "destination": dest,
                    "operateur": (row.get("operateur") or oper).strip(),
                    "reseau": (row.get("reseau") or res).strip(), "type": typ,
                    "prix_fcfa": prix, "trace": trace, "bbox": bbox,
                    "estime": _trace_estimee(row)})
    chemin = os.path.join(DATA_DIR, "gbaka_woro.csv")
    if os.path.exists(chemin):
        entrees = {}
        with open(chemin, encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f, delimiter=";"):
                typ_raw = (row.get("type_transport") or "").strip().lower()
                origine = (row.get("depart") or "").strip()
                dest = (row.get("arrivee") or "").strip()
                if not origine or not dest: continue
                typ = "WORO_WORO" if "woro" in typ_raw else "GBAKA"
                sa = {"destination": (row.get("sens_aller_destination") or "").strip(),
                    "liste_arrets": _parse_liste_arrets(row.get("sens_aller_liste_arrets") or ""),
                    "coordonnees": _parse_coords_str(row.get("sens_aller_coordonnees_gps") or "")}
                sr = {"destination": (row.get("sens_retour_destination") or "").strip(),
                    "liste_arrets": _parse_liste_arrets(row.get("sens_retour_liste_arrets") or ""),
                    "coordonnees": _parse_coords_str(row.get("sens_retour_coordonnees_gps") or "")}
                if not sa["liste_arrets"] and not sr["liste_arrets"]: continue
                id_osm = (row.get("id_osm") or "").strip()
                cle = id_osm or (typ, origine, dest)
                nb = len(sa["liste_arrets"]) + len(sr["liste_arrets"])
                if cle in entrees and nb <= entrees[cle]["_nb"]: continue
                trace, bbox = _charger_trace(row.get("trace_wkt") or "")
                entrees[cle] = {"numero": None, "origine": origine, "destination": dest,
                    "operateur": (row.get("operateur") or "divers").strip(),
                    "reseau": (row.get("reseau_commune") or "Autre").strip(), "type": typ,
                    "prix_fcfa": 200, "trace": trace, "bbox": bbox,
                    "estime": _trace_estimee(row), "id_osm": id_osm,
                    "origine_trace": (row.get("origine_trace") or "").strip(),
                    "sens_aller": sa, "sens_retour": sr, "_nb": nb}
        for e in entrees.values():
            e.pop("_nb")
            CATALOGUE.append(e)
    log.info("catalogue charge %d lignes", len(CATALOGUE))


# SECTION CONSTRUCTION DES SEQUENCES D ARRETS
def _dist_rapide(lat1, lon1, lat2, lon2):
    # Calcule une distance rapide en metres (approximation locale)
    dy = (lat1 - lat2) * 111320.0
    dx = (lon1 - lon2) * 111320.0 * 0.9956
    return (dx * dx + dy * dy) ** 0.5

def _cellule(lat, lon, taille):
    return (int(lat / taille), int(lon / taille))

def _grille_arrets(taille):
    # Groupe les arrets dans une grille pour accelerer les recherches
    g = defaultdict(list)
    for sid, a in RESEAU["arrets"].items():
        if sid in RESEAU["points_libres"]: continue
        g[_cellule(a["latitude"], a["longitude"], taille)].append(sid)
    return g

def _arrets_le_long_du_trace(trace, grille, taille, seuil, filtre=None):
    # Trouve les arrets qui sont sur le chemin d une trace
    passages = defaultdict(list)
    for idx, (lat, lon) in enumerate(trace):
        ci, cj = _cellule(lat, lon, taille)
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                for sid in grille.get((ci + di, cj + dj), ()):
                    a = RESEAU["arrets"][sid]
                    d = _dist_rapide(lat, lon, a["latitude"], a["longitude"])
                    if d > seuil: continue
                    if filtre and not filtre(sid, a["nom"]): continue
                    passages[sid].append((idx, d))
    evts = []
    for sid, hits in passages.items():
        grp = [hits[0]]
        for h in hits[1:]:
            if h[0] - grp[-1][0] <= 6: grp.append(h)
            else:
                evts.append((min(grp, key=lambda x: x[1])[0], sid))
                grp = [h]
        evts.append((min(grp, key=lambda x: x[1])[0], sid))
    evts.sort()
    gardes = []
    for idx, sid in evts:
        a = RESEAU["arrets"][sid]
        nom = nettoyer_texte(a["nom"])
        doublon = False
        for aut in gardes[-6:]:
            b = RESEAU["arrets"][aut]
            if nettoyer_texte(b["nom"]) == nom and _dist_rapide(a["latitude"],
                    a["longitude"], b["latitude"], b["longitude"]) < 250:
                doublon = True
                break
        if not doublon: gardes.append(sid)
    return gardes

def _est_arret_de_bus(sid, nom):
    if "lagunaire" in nettoyer_texte(nom): return False
    return sid in RESEAU["arrets_bus"] if RESEAU["arrets_bus"] else False

def _nom_point(lat, lon, grille_noms):
    # Donne un nom au point de bord de route le plus proche
    proche, dmin = None, 400
    ci, cj = _cellule(lat, lon, 0.004)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            for sid in grille_noms.get((ci + di, cj + dj), ()):
                a = RESEAU["arrets"][sid]
                d = _dist_rapide(lat, lon, a["latitude"], a["longitude"])
                if d < dmin and len(a["nom"].strip()) >= 4 and not a["nom"].strip().isdigit():
                    proche, dmin = a["nom"], d
    return f"Bord de route pres de {proche}" if proche else "Bord de route"

def _boucle_asymetrique(trace):
    # Detecte une trace qui ne revient pas sur ses pas
    if len(trace) < 20: return False
    if _dist_rapide(trace[0][0], trace[0][1], trace[-1][0], trace[-1][1]) > 500: return False
    loin = max(range(len(trace)), key=lambda j: _dist_rapide(
        trace[0][0], trace[0][1], trace[j][0], trace[j][1]))
    aller, retour = trace[:loin + 1], trace[loin:]
    grille = defaultdict(list)
    for p in aller: grille[_cellule(p[0], p[1], 0.002)].append(p)
    pareil = 0
    for p in retour:
        ci, cj = _cellule(p[0], p[1], 0.002)
        if any(_dist_rapide(p[0], p[1], q[0], q[1]) < 200
                for di in (-1, 0, 1) for dj in (-1, 0, 1)
                for q in grille.get((ci + di, cj + dj), ())):
            pareil += 1
    return pareil < 0.6 * len(retour)

def _sequence_gbaka(i, trace, grille_gbaka, grille_noms):
    # Construit la suite d arrets d une ligne gbaka
    reels = _arrets_le_long_du_trace(trace, grille_gbaka, 0.0006, 40)
    evts = []
    for sid in reels:
        a = RESEAU["arrets"][sid]
        k = min(range(len(trace)), key=lambda j: _dist_rapide(
            a["latitude"], a["longitude"], trace[j][0], trace[j][1]))
        evts.append((k, sid))
    cumul = 0.0
    for k, (lat, lon) in enumerate(trace):
        if k: cumul += _dist_rapide(trace[k-1][0], trace[k-1][1], lat, lon)
        if k and cumul < 300 and k != len(trace) - 1: continue
        cumul = 0.0
        if any(_dist_rapide(lat, lon, RESEAU["arrets"][s]["latitude"],
                RESEAU["arrets"][s]["longitude"]) < 150 for s in reels): continue
        pid = f"g:{i}:{k}"
        RESEAU["arrets"][pid] = {"id": pid, "nom": _nom_point(lat, lon, grille_noms),
            "latitude": lat, "longitude": lon}
        RESEAU["points_libres"].add(pid)
        evts.append((k, pid))
    return [sid for _, sid in sorted(evts, key=lambda x: x[0])]

def _index_stops_par_nom():
    idx = defaultdict(list)
    for sid, a in RESEAU["arrets"].items():
        idx[nettoyer_texte(a["nom"])].append(sid)
    return idx

def _trouver_stop_existant(nom, lat, lon, index):
    # Cherche un arret existant avec le meme nom
    cands = index.get(nettoyer_texte(nom), [])
    if not cands: return None
    if lat is None or lon is None: return cands[0]
    best, dmin = None, 120.0
    for sid in cands:
        a = RESEAU["arrets"][sid]
        d = _dist_rapide(lat, lon, a["latitude"], a["longitude"])
        if d < dmin: dmin, best = d, sid
    return best

def _resoudre_stops_csv(sens, i, label, index):
    # Transforme les noms d arrets du CSV en identifiants reels
    noms = sens.get("liste_arrets") or []
    coords = sens.get("coordonnees") or []
    if not noms: return []
    ids = []
    for idx, nom in enumerate(noms):
        nom = (nom or "").strip()
        if not nom: continue
        lat = lon = None
        if idx < len(coords): lon, lat = coords[idx]
        sid = _trouver_stop_existant(nom, lat, lon, index)
        if sid:
            ids.append(sid)
            continue
        if lat is None or lon is None: continue
        new = f"csv:{i}:{label}:{idx}"
        RESEAU["arrets"][new] = {"id": new, "nom": nom,
            "latitude": float(lat), "longitude": float(lon)}
        index.setdefault(nettoyer_texte(nom), []).append(new)
        ids.append(new)
    return ids

def _arrets_ligne_pour_api(rid):
    # Renvoie les arrets d une ligne gbaka pour l API du frontend
    if not isinstance(rid, str) or not rid.startswith("cat:"): return []
    try: i = int(rid[4:])
    except ValueError: return []
    if i < 0 or i >= len(CATALOGUE): return []
    c = CATALOGUE[i]
    if c["type"] not in ("GBAKA", "WORO_WORO"): return []
    result = []
    for label, sd in (("aller", c.get("sens_aller") or {}),
                      ("retour", c.get("sens_retour") or {})):
        noms = sd.get("liste_arrets") or []
        coords = sd.get("coordonnees") or []
        if len(noms) < 2: continue
        arrets = []
        for idx, n in enumerate(noms):
            if idx < len(coords):
                lon, lat = coords[idx]
            else:
                lat = lon = None
            arrets.append({"id": f"csv:{i}:{label}:{idx}",
                "nom": (n or "").strip(), "latitude": lat, "longitude": lon})
        if len(arrets) < 2: continue
        result.append({"sens": label, "depart": arrets[0]["nom"],
            "arrivee": arrets[-1]["nom"], "nb_arrets": len(arrets), "arrets": arrets})
    return result

def _construire_sequences_depuis_traces():
    # Construit les suites d arrets de toutes les lignes du catalogue
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
        if rid in RESEAU["sequences"]: continue
        typ = c["type"]
        trace = c.get("trace") or []
        if typ in ("GBAKA", "WORO_WORO"):
            seq_a = _resoudre_stops_csv(c.get("sens_aller") or {}, i, "aller", index_noms)
            seq_r = _resoudre_stops_csv(c.get("sens_retour") or {}, i, "retour", index_noms)
            seqs = []
            if len(seq_a) >= 2: seqs.append(seq_a)
            if len(seq_r) >= 2 and (not seqs or seq_r != seqs[0]): seqs.append(seq_r)
            if seqs:
                RESEAU["sequences"][rid] = seqs
                for s in seqs:
                    for sid in s:
                        if rid not in RESEAU["lignes_par_arret"][sid]:
                            RESEAU["lignes_par_arret"][sid].append(rid)
                nb += 1
                continue
            if len(trace) < 2 or c.get("estime"): continue
            seq = _sequence_gbaka(i, trace, grille_gbaka, grille_noms)
            if len(seq) < 2: continue
            RESEAU["sequences"][rid] = [seq] if _boucle_asymetrique(trace) else [seq, seq[::-1]]
            for sid in seq:
                if rid not in RESEAU["lignes_par_arret"][sid]:
                    RESEAU["lignes_par_arret"][sid].append(rid)
            nb += 1
            continue
        if len(trace) < 2 or c.get("estime"): continue
        if typ == "BUS":
            seq = _arrets_le_long_du_trace(trace, grille, taille, 40, filtre=_est_arret_de_bus)
        elif typ == "BATEAU":
            seq = _arrets_le_long_du_trace(trace, grille, taille * 6, 400,
                filtre=lambda sid, n: "lagunaire" in nettoyer_texte(n))
        else: continue
        if len(seq) < 2: continue
        RESEAU["sequences"][rid] = [seq, seq[::-1]]
        for sid in seq:
            if rid not in RESEAU["lignes_par_arret"][sid]:
                RESEAU["lignes_par_arret"][sid].append(rid)
        nb += 1
    log.info("sequences construites %d lignes", nb)

def charger_reseau_local():
    # Charge tout le reseau : arrets, lignes, sequences
    # Cette fonction ne fait rien si le reseau est deja charge
    with _lock:
        if RESEAU["charge"]: return
        log.info("chargement du reseau local")
        charger_catalogue()
        for nom_fichier in ("arret.csv", "stops.txt", "stops-2.txt"):
            chemin = os.path.join(DATA_DIR, nom_fichier)
            if not os.path.exists(chemin): continue
            with open(chemin, encoding="utf-8-sig") as f:
                for row in csv.DictReader(f):
                    try:
                        sid = row["stop_id"].strip()
                        if sid in RESEAU["arrets"]: continue
                        nom = (row.get("stop_name") or "").strip()
                        RESEAU["arrets"][sid] = {"id": sid, "nom": nom,
                            "latitude": float(row["stop_lat"]),
                            "longitude": float(row["stop_lon"])}
                        if est_gare(nom): RESEAU["gares"].add(sid)
                    except Exception: continue
        chemin = os.path.join(DATA_DIR, "routes.txt")
        if os.path.exists(chemin):
            with open(chemin, encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    rid = row.get("route_id")
                    if not rid: continue
                    nom = (row.get("route_long_name") or "").strip()
                    agency = (row.get("agency_id") or "").lower()
                    if "gbaka" in agency or "gbaka" in nom.lower(): typ = "GBAKA"
                    elif "woro" in agency or "woro" in nom.lower(): typ = "WORO_WORO"
                    elif any(x in agency or x in nom.lower()
                             for x in ("bateau", "stl", "aqua", "monbato")): typ = "BATEAU"
                    else: typ = "BUS"
                    RESEAU["lignes"][rid] = {"id": rid,
                        "numero": (row.get("route_short_name") or "").strip() or None,
                        "nom": nom, "type": typ,
                        "operateur": row.get("agency_id") or "inconnu"}
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
                        stop_times[row["trip_id"]].append(
                            {"stop_id": row["stop_id"], "seq": int(row["stop_sequence"])})
                    except Exception: continue
        for rid, tids in trips.items():
            if not tids: continue
            all_seqs, arrets_ligne = [], set()
            for tid in tids:
                ordonnes = sorted(stop_times.get(tid, []), key=lambda x: x["seq"])
                seq = [a["stop_id"] for a in ordonnes if a["stop_id"] in RESEAU["arrets"]]
                if len(seq) >= 2:
                    if seq not in all_seqs: all_seqs.append(seq)
                    arrets_ligne.update(seq)
            if all_seqs:
                for s in list(all_seqs):
                    if s[::-1] not in all_seqs: all_seqs.append(s[::-1])
                RESEAU["sequences"][rid] = all_seqs
                for sid in arrets_ligne:
                    if rid not in RESEAU["lignes_par_arret"][sid]:
                        RESEAU["lignes_par_arret"][sid].append(rid)
        arrets_bus, arrets_autres = set(), set()
        for rid_g, seqs in RESEAU["sequences"].items():
            cible = arrets_bus if RESEAU["lignes"].get(rid_g, {}).get("type") == "BUS" else arrets_autres
            for s in seqs: cible.update(s)
        RESEAU["arrets_non_bus"] = arrets_autres - arrets_bus
        RESEAU["arrets_bus"] = arrets_bus
        for rid_g, seqs in RESEAU["sequences"].items():
            if RESEAU["lignes"].get(rid_g, {}).get("type") in ("GBAKA", "WORO_WORO"):
                for s in seqs: RESEAU["arrets_gbaka"].update(s)
        for i, c in enumerate(CATALOGUE):
            rid = f"cat:{i}"
            if rid not in RESEAU["lignes"]:
                RESEAU["lignes"][rid] = {"id": rid, "numero": c.get("numero"),
                    "nom": f"{c['origine']} ↔ {c['destination']}", "type": c["type"],
                    "operateur": c.get("operateur", "")}
        _construire_sequences_depuis_traces()
        RESEAU["charge"] = True
        log.info("reseau pret %d arrets %d lignes %d sequences %d gares",
            len(RESEAU["arrets"]), len(RESEAU["lignes"]),
            len(RESEAU["sequences"]), len(RESEAU["gares"]))


# SECTION PROXIMITE DES ARRETS
def arrets_proximite(lat, lon, rayon_m=5000, tri_par_distance=False):
    # Renvoie les arrets autour d une position
    if not RESEAU["charge"]: charger_reseau_local()
    uniques = {}
    for arret in RESEAU["arrets"].values():
        d = distance_m(lat, lon, arret["latitude"], arret["longitude"])
        if d > rayon_m: continue
        types = set()
        for rid in RESEAU["lignes_par_arret"].get(arret["id"], []):
            l = RESEAU["lignes"].get(rid)
            if l: types.add(l["type"])
        types &= {"BUS", "GBAKA", "BATEAU"}
        if not types: continue
        principal = "BUS" if "BUS" in types else ("GBAKA" if "GBAKA" in types else "BATEAU")
        nom = arret["nom"]
        if nom not in uniques or d < uniques[nom]["distance_m"]:
            uniques[nom] = {"id": arret["id"], "nom": nom,
                "latitude": arret["latitude"], "longitude": arret["longitude"],
                "distance_m": int(d), "type": principal, "types": list(types)}
    res = list(uniques.values())
    if tri_par_distance: res.sort(key=lambda x: x["distance_m"])
    else:
        ordre = {"BUS": 0, "GBAKA": 1, "BATEAU": 2}
        res.sort(key=lambda x: (ordre.get(x["type"], 9), x["distance_m"]))
    return res

def arrets_proximite_par_modes(lat, lon, rayon_m=1500, modes=None):
    # Comme arrets_proximite mais on filtre par mode
    if not RESEAU["charge"]: charger_reseau_local()
    autorises = set(modes) if modes else {"BUS", "GBAKA", "BATEAU"}
    uniques = {}
    for arret in RESEAU["arrets"].values():
        d = distance_m(lat, lon, arret["latitude"], arret["longitude"])
        if d > rayon_m: continue
        if any(RESEAU["lignes"].get(rid, {}).get("type") in autorises
                for rid in RESEAU["lignes_par_arret"].get(arret["id"], [])):
            nom = arret["nom"]
            if nom not in uniques or d < uniques[nom]["distance_m"]:
                uniques[nom] = {"id": arret["id"], "nom": nom,
                    "latitude": arret["latitude"], "longitude": arret["longitude"],
                    "distance_m": int(d)}
    res = list(uniques.values())
    res.sort(key=lambda x: x["distance_m"])
    return res

def lignes_par_arret(stop_id):
    # Donne les lignes qui passent par un arret
    if not RESEAU["charge"]: charger_reseau_local()
    arret = RESEAU["arrets"].get(stop_id)
    if not arret: return None
    lignes = []
    for rid in RESEAU["lignes_par_arret"].get(stop_id, []):
        l = RESEAU["lignes"].get(rid)
        if l:
            lignes.append({"id": l["id"], "numero": l["numero"], "nom": l["nom"],
                "type": l["type"], "operateur": l["operateur"]})
    lignes.sort(key=lambda l: (0 if l["type"] == "BUS" else 1, l["numero"] or "zzz"))
    return {"arret": arret, "lignes": lignes}


# SECTION PRIX
PRIX_CATALOGUE = {}
_prix_charges = False

def _norm_num(n):
    n = str(n or "").strip().upper()
    return str(int(n)) if n.isdigit() else n

def _charger_prix():
    global _prix_charges
    _prix_charges = True
    PRIX_CATALOGUE.clear()
    for fichier, typ, col in (("bus.csv", "BUS", "prix_fcfa"),
                              ("bateau_bus.csv", "BATEAU", "prix_minimum_fcfa")):
        chemin = os.path.join(DATA_DIR, fichier)
        if not os.path.exists(chemin): continue
        with open(chemin, encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f, delimiter=";"):
                try: p = int(float(row.get(col)))
                except Exception: continue
                PRIX_CATALOGUE.setdefault((typ, _norm_num(row.get("numero_ligne"))), p)

def _prix_csv(typ, numero):
    if not _prix_charges: _charger_prix()
    return PRIX_CATALOGUE.get((typ, _norm_num(numero)))

def _prix_ligne(ligne):
    if ligne.get("type") not in ("BUS", "BATEAU"): return 0
    return _prix_csv(ligne["type"], ligne.get("numero")) or 0

def prix_connu(mode, nom_ligne, montee, descente, numero):
    # Prix d une ligne avec valeurs par defaut
    if numero and str(numero) in PRIX_LIGNES: return PRIX_LIGNES[str(numero)]
    return PRIX_POUR_CLASSEMENT.get(mode, 250)


# SECTION AIDES POUR CONSTRUIRE LES TRAJETS
ATTENTE_MINUTES = {"BUS": 8, "GBAKA": 3, "WORO_WORO": 3, "BATEAU": 15}
SEUIL_MARCHE_LONGUE_M = 1000
DESCENTE_MAX_GBAKA_M = 500
_LIBELLES_MODE = {"BUS": "bus", "GBAKA": "gbaka", "WORO_WORO": "woro-woro", "BATEAU": "bateau-bus"}
_MODE_COURT = _LIBELLES_MODE
_NOM_MODE = {"BUS": "le bus", "GBAKA": "le gbaka", "BATEAU": "le bateau-bus"}
_LIBELLE_MODE = {"BUS": "bus", "GBAKA": "gbaka", "BATEAU": "bateau-bus"}

def _rids_approximatifs():
    # Identifiants de lignes dont la trace est approximative
    if "rids_approx" not in RESEAU:
        gtfs = set()
        for rid, l in RESEAU["lignes"].items():
            if not str(rid).startswith("cat:") and RESEAU["sequences"].get(rid):
                gtfs.add((l.get("type"), _norm_num(l.get("numero"))))
        RESEAU["rids_approx"] = {rid for rid, l in RESEAU["lignes"].items()
            if str(rid).startswith("cat:") and l.get("type") in ("BUS", "BATEAU")
            and (l.get("type"), _norm_num(l.get("numero"))) in gtfs}
    return RESEAU["rids_approx"]

def _finaliser_candidat(c):
    # Calcule les totaux d un trajet (prix, duree, arrets)
    connus, inconnu = 0, False
    for e in c["etapes"]:
        if e["mode"] == "MARCHE": continue
        p = e.get("cout_fcfa")
        if p is None: inconnu = True
        else: connus += p
    c["cout_partiel_fcfa"] = connus
    c["cout_fcfa"] = None if inconnu else connus
    c["prix_inconnu"] = inconnu
    c["nb_arrets"] = sum(e.get("nb_arrets", 0) for e in c["etapes"] if e["mode"] != "MARCHE")

def _cout_tri(c):
    return c.get("cout_partiel_fcfa", 0)

def _normaliser_candidat(c):
    # Remplit les prix et calcule les totaux
    for e in c.get("etapes", []):
        if e.get("mode") == "MARCHE": continue
        if e.get("nb_arrets") is None:
            e["nb_arrets"] = max(0, len(e["arrets"]) - 1) if e.get("arrets") else 0
        e["cout_fcfa"] = (_prix_csv(e["mode"], e.get("numero_ligne"))
                          if e["mode"] in ("BUS", "BATEAU") else None)
    _finaliser_candidat(c)

def _ajouter_textes(c):
    # Ajoute un texte lisible a chaque etape
    etapes = c.get("etapes", [])
    n = len(etapes)
    for i, e in enumerate(etapes):
        if e["mode"] == "MARCHE":
            d = int(e.get("distance_m", 0))
            if i == 0:
                e["texte"] = (f"Vous etes deja a l arret {e['arrivee']} (point de montee)."
                    if d < 10 else f"Marchez {d} m jusqu a l arret {e['arrivee']} (point de montee).")
            elif i == n - 1:
                e["texte"] = f"Marchez {d} m jusqu a votre destination."
            else:
                e["texte"] = f"Marchez {d} m de {e['depart']} a {e['arrivee']} pour la correspondance."
        else:
            lib = _LIBELLES_MODE.get(e["mode"], e["mode"].lower())
            ligne = str(e.get("numero_ligne") or e.get("nom_ligne") or "")
            for p in ("gbaka :", "gbaka:", "woro-woro :", "woro :"):
                if ligne.lower().startswith(p): ligne = ligne[len(p):].strip()
            nb = e.get("nb_arrets") or 0
            suite = f" ({nb} arret{'s' if nb > 1 else ''})" if nb else ""
            if e["mode"] == "WORO_WORO" and i == 0:
                dep = "votre position" if "position" in str(e.get("depart", "")).lower() else e["depart"]
                e["texte"] = (f"Prenez un woro-woro ou un gbaka depuis {dep} "
                    f"jusqu a {e['arrivee']} (prix variable selon le chauffeur).")
            else:
                e["texte"] = (f"Montez a {e['depart']} dans le {lib} {ligne}, "
                    f"descendez a {e['arrivee']}{suite}.").replace("  ", " ").replace(" ,", ",")
        if e.get("conseil"): e["texte"] += " " + e["conseil"]

def _nom_court_ligne(e):
    if e.get("numero_ligne"): return str(e["numero_ligne"])
    nom = str(e.get("nom_ligne") or "").replace("gbaka :", "").strip()
    return nom or _MODE_COURT.get(e.get("mode"), "ligne")

def _etapes_transport(c):
    return [e for e in c.get("etapes", []) if e.get("mode") not in ("MARCHE", "WORO_WORO")]

def _acces_m(c):
    return c.get("marche_m", 0) + sum(e.get("distance_m", 0)
        for e in c.get("etapes", []) if e.get("mode") == "WORO_WORO")

# Varible qui stocke les trajets deja proposes quand l utilisateur dit "pas celui la"
_EXCLURE = contextvars.ContextVar("excl", default=None)

def _pos_proches(p, q, seuil=500):
    if not p or not q: return False
    return distance_m(p["latitude"], p["longitude"], q["latitude"], q["longitude"]) <= seuil

def _quasi_identiques(a, b):
    # Deux trajets qui font presque la meme chose
    ta, tb = _etapes_transport(a), _etapes_transport(b)
    if not ta or len(ta) != len(tb): return False
    for x, y in zip(ta, tb):
        if x.get("mode") != y.get("mode"): return False
        if not (x.get("depart") == y.get("depart")
                or _pos_proches(x.get("depart_pos"), y.get("depart_pos"))): return False
        if not (x.get("arrivee") == y.get("arrivee")
                or _pos_proches(x.get("arrivee_pos"), y.get("arrivee_pos"))): return False
    return True

def _signature_candidat(c):
    return tuple((e.get("mode"), e.get("numero_ligne") or e.get("nom_ligne"),
        e.get("depart"), e.get("arrivee"))
        for e in c.get("etapes", []) if e.get("mode") != "MARCHE")


# SECTION CHOISIR LES PROPOSITIONS A AFFICHER
def choisir_propositions(candidats, mode_filtre="TOUS", priorite=None, budget=None, maxi=4):
    # Prend une liste de trajets et choisit les plus interessants
    # 1. Les moins de correspondances d abord
    # 2. Les trajets qui se ressemblent sont groupes
    # 3. On ajoute un autre mode seulement s il n est pas trop mauvais
    cands = [c for c in (candidats or []) if _etapes_transport(c)]
    if budget: cands = [c for c in cands if c.get("cout_partiel_fcfa", 0) <= budget]
    if not cands: return []
    deja = _EXCLURE.get()
    if deja:
        cands = [c for c in cands if not any(_quasi_identiques(c, p) for p in deja)]
        if not cands: return []

    def marche_finale(c):
        e = c["etapes"][-1]
        return e.get("distance_m", 0) if e.get("mode") == "MARCHE" else 0
    courtes = [c for c in cands if marche_finale(c) <= SEUIL_MARCHE_LONGUE_M]
    if courtes: cands = courtes

    def cle(c):
        base = c.get("nb_arrets", 0) + _acces_m(c) / 200.0
        if priorite == "prix":
            return (c.get("nb_correspondances", 0), c.get("cout_partiel_fcfa", 0),
                    base, c.get("duree_minutes", 999))
        if priorite == "duree":
            return (c.get("nb_correspondances", 0), c.get("duree_minutes", 999), base)
        return (c.get("nb_correspondances", 0), base, c.get("duree_minutes", 999))

    def famille(c):
        return frozenset(e["mode"] for e in _etapes_transport(c))

    def cle_groupe(c):
        tr = _etapes_transport(c)
        if len(tr) == 1: return ("D", tr[0]["depart"], tr[0]["arrivee"])
        return tuple((e["mode"],
            "" if e["mode"] in ("GBAKA", "WORO_WORO") else _nom_court_ligne(e),
            e["depart"], e["arrivee"]) for e in tr[:-1]) + (tr[-1]["depart"],)

    def grouper(liste):
        groupes = {}
        for c in sorted(liste, key=cle):
            groupes.setdefault(cle_groupe(c), []).append(c)
        tries = sorted(groupes.values(), key=lambda g: cle(g[0]))
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
    if mode_filtre == "TOUS" and choisis:
        deja_modes = {m for g in choisis for m in famille(g[0])}
        dur_ref = choisis[0][0].get("duree_minutes", 999) or 999
        autres = grouper([c for c in cands if c.get("nb_correspondances", 0) <= mini + 1])
        for g in autres:
            if len(choisis) >= maxi: break
            if (famille(g[0]) - deja_modes) and all(g is not x for x in choisis):
                if (g[0].get("duree_minutes", 999) or 999) <= dur_ref * 1.5:
                    choisis.append(g)
                    deja_modes |= famille(g[0])
        if len(choisis) > maxi: choisis = choisis[:maxi]
    res = []
    for n, g in enumerate(choisis, 1):
        rep = copy.deepcopy(g[0])
        tr = _etapes_transport(rep)
        derniere = tr[-1]
        noms = []
        for e in tr:
            nom_l = _nom_court_ligne(e)
            if len(nom_l) > 26: nom_l = nom_l[:25].rstrip() + "..."
            noms.append(f"{_MODE_COURT.get(e['mode'], e['mode'].lower())} {nom_l}".strip())
        rep["libelle"] = f"Option {n} " + " puis ".join(noms)
        nb_c = rep.get("nb_correspondances", 0)
        rep["resume"] = ("Trajet direct" if nb_c == 0
            else f"{nb_c} correspondance{'s' if nb_c > 1 else ''}") + \
            f" ({rep.get('nb_arrets', 0)} arrets)"
        alts, vus = [], {_nom_court_ligne(derniere)}
        for aut in g[1:]:
            ta = _etapes_transport(aut)[-1]
            nom = _nom_court_ligne(ta)
            if nom in vus or ta.get("mode") != derniere.get("mode") \
                    or ta["arrivee"] != derniere["arrivee"]: continue
            vus.add(nom)
            alts.append({"ligne": nom, "mode": ta["mode"], "nb_arrets": ta.get("nb_arrets", 0)})
        alts.sort(key=lambda a: a["nb_arrets"])
        rep["alternatives"] = alts[:5]
        if alts:
            liste = ", ".join(f"{_MODE_COURT.get(a['mode'], a['mode'].lower())} {a['ligne']}" +
                (f" ({a['nb_arrets']} arrets)" if a["nb_arrets"] else "") for a in alts[:5])
            derniere["texte"] = (derniere.get("texte") or "") + \
                f" Autres lignes depuis le meme arret : {liste}."
        res.append(rep)
    return res

def conseil_marche_chat(options):
    # Donne un conseil quand il y a une longue marche
    conseils, vus = [], set()
    for c in options or []:
        for e in c.get("etapes", []):
            if e.get("mode") != "MARCHE" or e.get("distance_m", 0) <= SEUIL_MARCHE_LONGUE_M: continue
            if not e.get("depart_pos") or not e.get("arrivee_pos"): continue
            k = (e.get("depart"), e.get("arrivee"))
            if k in vus: continue
            vus.add(k)
            km = e["distance_m"] / 1000
            lignes = _lignes_libres_proches(e["depart_pos"], e["arrivee_pos"])
            if lignes:
                genre = "gbaka" if lignes[0][0] == "GBAKA" else "woro-woro"
                conseils.append(f"la marche de {km:.1f} km vers {e['arrivee']} est longue : "
                    f"le {genre} « {lignes[0][1]} » passe dans ce secteur "
                    f"(prix variable selon le chauffeur)")
            else:
                conseils.append(f"la marche de {km:.1f} km vers {e['arrivee']} est longue "
                    f"et je n ai pas de gbaka connu sur ce troncon : prevois de marcher")
            if len(conseils) >= 2: return " ; ".join(conseils)
    return " ; ".join(conseils)


# SECTION CONSTRUCTION DES TRAJETS
def _lignes_passant_par(arret_id, modes_autorises):
    res = []
    approx = _rids_approximatifs()
    for rid in RESEAU["lignes_par_arret"].get(arret_id, []):
        if rid in approx: continue
        l = RESEAU["lignes"].get(rid)
        if not l or l["type"] not in modes_autorises: continue
        if l["type"] == "WORO_WORO": continue
        for seq in RESEAU["sequences"].get(rid, []):
            if arret_id in seq: res.append({"ligne": l, "rid": rid, "sequence": seq})
    return res

def _arrets_utiles(arrets, modes, maxi=60):
    # Garde des arrets utiles en assurant une representation de chaque mode
    modes_reels = {m for m in modes if m != "WORO_WORO"}
    if not modes_reels: return []
    par_mode = {m: [] for m in modes_reels}
    for a in arrets:
        types = {RESEAU["lignes"].get(rid, {}).get("type")
                 for rid in RESEAU["lignes_par_arret"].get(a["id"], [])} & modes_reels
        for t in types: par_mode[t].append(a)
    gardes, vus_ids, lignes_vues = [], set(), set()

    def ajouter(a):
        if a["id"] in vus_ids: return False
        nouvelles = {rid for rid in RESEAU["lignes_par_arret"].get(a["id"], [])
            if (l := RESEAU["lignes"].get(rid))
            and l["type"] in modes_reels and rid not in lignes_vues}
        if not nouvelles: return False
        gardes.append(a)
        vus_ids.add(a["id"])
        lignes_vues.update(nouvelles)
        return True

    quota = min(10, max(2, maxi // len(modes_reels)))
    for mode in modes_reels:
        cnt = 0
        for a in par_mode[mode]:
            if len(gardes) >= maxi: break
            if ajouter(a):
                cnt += 1
                if cnt >= quota: break
    for a in arrets:
        if len(gardes) >= maxi: break
        ajouter(a)
    ordre = {a["id"]: i for i, a in enumerate(arrets)}
    gardes.sort(key=lambda a: ordre.get(a["id"], 10**9))
    return gardes

def _duree_sur_ligne(ids):
    # Duree estimee du trajet sur un segment en minutes
    total = 0.0
    for a, b in zip(ids, ids[1:]):
        p, q = RESEAU["arrets"].get(a), RESEAU["arrets"].get(b)
        if p and q:
            total += _dist_rapide(p["latitude"], p["longitude"], q["latitude"], q["longitude"])
    return max(5, int(total // 350))

def _ajouter_attente(candidat):
    # Ajoute le temps d attente pour chaque transport
    total = 0
    for e in candidat["etapes"]:
        if e["mode"] != "MARCHE" and "attente_minutes" not in e:
            e["attente_minutes"] = ATTENTE_MINUTES.get(e["mode"], 5)
            total += e["attente_minutes"]
    candidat["duree_minutes"] += total

def _lignes_libres_proches(pos_a, pos_b):
    # Cherche des lignes gbaka dont la trace passe pres de deux points
    noms = []
    for c in CATALOGUE:
        if c["type"] not in ("GBAKA", "WORO_WORO") or c.get("estime"): continue
        trace, bbox = c.get("trace"), c.get("bbox")
        if not trace or not bbox: continue
        if _point_proche_du_trace(pos_a["latitude"], pos_a["longitude"], trace, bbox, 250) and \
           _point_proche_du_trace(pos_b["latitude"], pos_b["longitude"], trace, bbox, 500):
            noms.append((c["type"], f"{c['origine']} ↔ {c['destination']}"))
            if len(noms) >= 2: break
    return noms

def _ajouter_conseils_marche(candidat):
    # Ajoute des conseils quand il y a une longue marche
    for pos, e in enumerate(candidat["etapes"]):
        if e["mode"] != "MARCHE" or e.get("distance_m", 0) <= SEUIL_MARCHE_LONGUE_M: continue
        au_depart = pos == 0
        km = e["distance_m"] / 1000
        lignes = _lignes_libres_proches(e["depart_pos"], e["arrivee_pos"])
        if lignes:
            ligne = lignes[0]
            genre = "gbaka" if ligne[0] == "GBAKA" else "woro-woro"
            if au_depart:
                e["conseil"] = (f"L arret est a {km:.1f} km : prenez plutot "
                    f"le {genre} « {ligne[1]} » qui passe pres de vous.")
            else:
                e["conseil"] = (f"Il reste {km:.1f} km : le {genre} « {ligne[1]} » "
                    f"dessert ce secteur si vous ne voulez pas marcher.")
        else:
            if au_depart:
                e["conseil"] = (f"L arret est a {km:.1f} km : prenez plutot un "
                    f"woro-woro, un gbaka ou un taxi pour le rejoindre.")
            else:
                e["conseil"] = (f"Il reste {km:.1f} km : prenez un woro-woro, "
                    f"un gbaka ou un taxi si vous ne voulez pas marcher.")

def _variante_woro_si_longue_marche(candidat):
    # Remplace une longue marche initiale par un woro-woro
    etapes = candidat.get("etapes", [])
    if not etapes or etapes[0].get("mode") != "MARCHE": return None
    marche = etapes[0]
    if marche.get("distance_m", 0) <= SEUIL_MARCHE_LONGUE_M: return None
    v = copy.deepcopy(candidat)
    m = v["etapes"].pop(0)
    duree_woro = max(3, m["distance_m"] // 400)
    wait = ATTENTE_MINUTES.get("WORO_WORO", 3)
    v["etapes"].insert(0, {"mode": "WORO_WORO", "depart": m["depart"],
        "arrivee": m["arrivee"], "depart_pos": m["depart_pos"],
        "arrivee_pos": m["arrivee_pos"], "distance_m": m["distance_m"],
        "duree_minutes": duree_woro, "cout_fcfa": 0, "attente_minutes": wait})
    v["duree_minutes"] = v["duree_minutes"] - m.get("duree_minutes", 0) + duree_woro + wait
    v["marche_m"] = max(0, v.get("marche_m", 0) - m["distance_m"])
    v["variante"] = "woro_remplace_marche"
    return v

def _grille_correspondance():
    if "grille_corr" not in RESEAU:
        g = defaultdict(list)
        for sid, a in RESEAU["arrets"].items():
            if RESEAU["lignes_par_arret"].get(sid) and sid not in RESEAU["points_libres"]:
                g[_cellule(a["latitude"], a["longitude"], 0.003)].append(sid)
        RESEAU["grille_corr"] = g
    return RESEAU["grille_corr"]

def _arrets_proches_de(sid, rayon=300):
    # Trouve les arrets proches d un arret donne
    a = RESEAU["arrets"][sid]
    grille = _grille_correspondance()
    ci, cj = _cellule(a["latitude"], a["longitude"], 0.003)
    res = []
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            for aut in grille.get((ci + di, cj + dj), ()):
                b = RESEAU["arrets"][aut]
                d = _dist_rapide(a["latitude"], a["longitude"], b["latitude"], b["longitude"])
                if d <= rayon: res.append((aut, d))
    return res

def _lignes_voisines():
    # Pour chaque ligne, les lignes qui passent pres d un de ses arrets
    if "voisines" not in RESEAU:
        voisines = defaultdict(set)
        approx = _rids_approximatifs()
        for rid, seqs in RESEAU["sequences"].items():
            if rid in approx or RESEAU["lignes"].get(rid, {}).get("type") == "WORO_WORO": continue
            for sid in [x for x in seqs[0] if x not in RESEAU["points_libres"]][::2]:
                for aut, _ in _arrets_proches_de(sid, 250):
                    for rid2 in RESEAU["lignes_par_arret"].get(aut, []):
                        if (rid2 != rid and rid2 not in approx
                                and RESEAU["lignes"].get(rid2, {}).get("type") != "WORO_WORO"):
                            voisines[rid].add(rid2)
        RESEAU["voisines"] = voisines
    return RESEAU["voisines"]

def _etape_marche(dep_nom, arr_nom, p, q, dist):
    return {"mode": "MARCHE", "depart": dep_nom, "arrivee": arr_nom,
        "distance_m": int(dist), "duree_minutes": max(1, int(dist) // 80), "cout_fcfa": 0,
        "depart_pos": {"latitude": p["latitude"], "longitude": p["longitude"]},
        "arrivee_pos": {"latitude": q["latitude"], "longitude": q["longitude"]}}

def _etape_ligne(ligne, rid, ids):
    m, d = RESEAU["arrets"][ids[0]], RESEAU["arrets"][ids[-1]]
    return {"mode": ligne["type"], "numero_ligne": ligne.get("numero"),
        "nom_ligne": ligne.get("nom"), "depart": m["nom"], "arrivee": d["nom"],
        "depart_pos": {"latitude": m["latitude"], "longitude": m["longitude"]},
        "arrivee_pos": {"latitude": d["latitude"], "longitude": d["longitude"]},
        "distance_m": distance_m(m["latitude"], m["longitude"], d["latitude"], d["longitude"]),
        "duree_minutes": _duree_sur_ligne(ids), "cout_fcfa": _prix_ligne(ligne),
        "nb_arrets": len(ids) - 1}

def _meilleur_transfert(seq_a, apres_a, seq_b, avant_b, rayon=300, ligne_b_bus=False):
    # Trouve le meilleur point de correspondance entre deux lignes
    pos_b = {}
    for k, sid in enumerate(seq_b):
        if k <= avant_b and sid not in pos_b: pos_b[sid] = k
    rayon_max = max(rayon, 800) if ligne_b_bus else rayon
    meilleur = None
    for i in range(apres_a + 1, len(seq_a)):
        sid_a = seq_a[i]
        if sid_a in RESEAU["points_libres"]: continue
        for aut, d in _arrets_proches_de(sid_a, rayon_max):
            k = pos_b.get(aut)
            if k is None: continue
            terminus = ligne_b_bus and k == 0
            if d > rayon and not (terminus and d <= 800): continue
            score = (i - apres_a) + (avant_b - k) + d / 150.0 - (4 if terminus else 0)
            if not meilleur or score < meilleur[0]:
                meilleur = (score, i, k, d, sid_a, aut)
    if not meilleur: return None
    _, i, k, d, sid_a, aut = meilleur
    return (i, k, d, sid_a, aut)

def _candidats_deux_correspondances(dep, dest, modes, arrets_dep, arrets_arr, prix_max_ligne=None):
    # Trajets avec deux changements de bus
    voisines = _lignes_voisines()
    perm = lambda rid: RESEAU["lignes"].get(rid, {}).get("type") in modes
    lignes_dep, lignes_arr = {}, {}
    for a in arrets_dep:
        for info in _lignes_passant_par(a["id"], modes):
            lignes_dep.setdefault(info["rid"], []).append((a, info))
    for a in arrets_arr:
        for info in _lignes_passant_par(a["id"], modes):
            lignes_arr.setdefault(info["rid"], []).append((a, info))
    res, triples = {}, 0
    for r1, departs in lignes_dep.items():
        for r2 in voisines.get(r1, ()):
            if r2 == r1 or not perm(r2): continue
            for r3 in voisines.get(r2, ()):
                if r3 in (r1, r2) or r3 not in lignes_arr: continue
                triples += 1
                if triples > 400: return list(res.values())
                l1, l2, l3 = RESEAU["lignes"][r1], RESEAU["lignes"][r2], RESEAU["lignes"][r3]
                if prix_max_ligne and max(_prix_ligne(l1), _prix_ligne(l2),
                        _prix_ligne(l3)) > prix_max_ligne: continue
                a_dep, info1 = min(departs, key=lambda x: x[0]["distance_m"])
                seq1 = info1["sequence"]
                i_mont = seq1.index(a_dep["id"])
                for seq2 in RESEAU["sequences"][r2]:
                    for a_arr, info3 in sorted(lignes_arr[r3],
                            key=lambda x: x[0]["distance_m"])[:3]:
                        seq3 = info3["sequence"]
                        k_desc = seq3.index(a_arr["id"])
                        t1 = _meilleur_transfert(seq1, i_mont, seq2, len(seq2) - 2)
                        if not t1: continue
                        i_d1, j_m2, d12, s_d1, s_m2 = t1
                        t2 = _meilleur_transfert(seq2, j_m2, seq3, k_desc - 1)
                        if not t2: continue
                        j_d2, k_m3, d23, s_d2, s_m3 = t2
                        ids1 = seq1[i_mont:i_d1 + 1]
                        ids2 = seq2[j_m2:j_d2 + 1]
                        ids3 = seq3[k_m3:k_desc + 1]
                        if len(ids1) < 2 or len(ids2) < 2 or len(ids3) < 2: continue
                        A = lambda sid: RESEAU["arrets"][sid]
                        trop_court = False
                        for ids in (ids1, ids2, ids3):
                            if _dist_rapide(A(ids[0])["latitude"], A(ids[0])["longitude"],
                                    A(ids[-1])["latitude"], A(ids[-1])["longitude"]) < 1500:
                                trop_court = True
                        if trop_court: continue
                        etapes = [
                            _etape_marche(dep.get("nom", "depart"), A(ids1[0])["nom"],
                                dep, A(ids1[0]), a_dep["distance_m"]),
                            _etape_ligne(l1, r1, ids1),
                            _etape_marche(A(s_d1)["nom"], A(s_m2)["nom"], A(s_d1), A(s_m2), d12),
                            _etape_ligne(l2, r2, ids2),
                            _etape_marche(A(s_d2)["nom"], A(s_m3)["nom"], A(s_d2), A(s_m3), d23),
                            _etape_ligne(l3, r3, ids3),
                            _etape_marche(A(ids3[-1])["nom"], dest.get("nom", "destination"),
                                A(ids3[-1]), dest, a_arr["distance_m"])]
                        for e in (etapes[2], etapes[4]):
                            e["duree_minutes"] = max(2, e["duree_minutes"])
                        cand = {"duree_minutes": sum(e["duree_minutes"] for e in etapes),
                            "cout_fcfa": sum(e["cout_fcfa"] for e in etapes),
                            "marche_m": sum(e["distance_m"] for e in etapes
                                            if e["mode"] == "MARCHE"),
                            "nb_correspondances": 2, "etapes": etapes,
                            "arrets_itineraire": []}
                        k = (r1, r2, r3)
                        if k not in res or _score_trajet(cand) < _score_trajet(res[k]):
                            res[k] = cand
    return list(res.values())

def _score_trajet(c):
    # Score plus petit = meilleur trajet
    # On penalise fortement la marche finale pour ne pas deposer loin
    etapes = c.get("etapes") or []
    final = etapes[-1].get("distance_m", 0) if etapes and etapes[-1].get("mode") == "MARCHE" else 0
    return c["duree_minutes"] + c.get("marche_m", 0) // 160 + final // 60

def _filtrer_trajets_gbaka(candidats):
    # Filtre les trajets gbaka pour ne garder que les plus logiques
    def m_finale(c):
        e = c["etapes"][-1]
        return e.get("distance_m", 0) if e["mode"] == "MARCHE" else 0
    def m_entre(c):
        return max([e.get("distance_m", 0) for e in c["etapes"][1:-1]
                    if e["mode"] == "MARCHE"] or [0])
    def detour(c):
        return any(e["mode"] == "GBAKA" and e["duree_minutes"] * 350 >
                   2.2 * e["distance_m"] + 1500 for e in c["etapes"])
    bons = [c for c in candidats if m_entre(c) <= 250 and not detour(c)]
    if any(m_finale(c) <= 800 for c in bons):
        bons = [c for c in bons if m_finale(c) <= 800]
    bons.sort(key=lambda c: (c["nb_correspondances"], _score_trajet(c)))
    uniques, vus = [], set()
    for c in bons:
        k = _signature_candidat(c)
        if k in vus: continue
        vus.add(k)
        uniques.append(c)
    return uniques


# SECTION CONSTRUCTION DES TRAJETS CANDIDATS
def _construire_candidats_rayon(dep, dest, mode_filtre, prix_max_ligne, rayon):
    # Construit les trajets dans un rayon donne autour du depart et de la destination
    if not RESEAU["charge"]: charger_reseau_local()
    modes = {"BUS", "GBAKA", "BATEAU"}
    if mode_filtre == "BUS": modes = {"BUS"}
    elif mode_filtre == "GBAKA": modes = {"GBAKA"}
    elif mode_filtre == "BATEAU": modes = {"BATEAU"}

    # CORRECTION DU BUG : on prend 20 arrets de depart, plus seulement 1 par mode
    # Sinon une ligne qui part de Shell Azito (100 m plus loin) etait ignoree
    # et on se retrouvait avec un trajet qui finit a Terminus 42
    arrets_dep = arrets_proximite(dep["latitude"], dep["longitude"], rayon,
                                   tri_par_distance=True)[:20]
    arret_depart_prioritaire = arrets_dep[0] if arrets_dep else None
    arrets_arr_prox = arrets_proximite(dest["latitude"], dest["longitude"], rayon,
                                        tri_par_distance=True)
    arrets_arr = _arrets_utiles(arrets_arr_prox, modes)
    if not arrets_dep or not arrets_arr: return []
    log.info("[v2] mode=%s rayon=%s arr_dep=%d arr_arr=%d",
             mode_filtre, rayon, len(arrets_dep), len(arrets_arr))

    candidats, vus, meilleur_direct = [], set(), {}
    descente_max_gbaka = DESCENTE_MAX_GBAKA_M
    if modes & {"GBAKA", "WORO_WORO"}:
        dmin = min((distance_m(dest["latitude"], dest["longitude"],
                               ar["latitude"], ar["longitude"])
            for ar in RESEAU["arrets"].values()
            if any(RESEAU["lignes"].get(rid, {}).get("type") == "GBAKA"
                   for rid in RESEAU["lignes_par_arret"].get(ar["id"], []))), default=0)
        descente_max_gbaka = max(DESCENTE_MAX_GBAKA_M, int(dmin) + 100)

    # TRAJETS DIRECTS : un seul transport, on teste chaque arret de depart
    sens_pris = set()
    for a_dep in arrets_dep:
        for info in _lignes_passant_par(a_dep["id"], modes):
            ligne, seq = info["ligne"], info["sequence"]
            if ligne["type"] in ("GBAKA", "WORO_WORO"):
                k = (info["rid"], tuple(seq))
                if k in sens_pris: continue
                sens_pris.add(k)
            prix = _prix_ligne(ligne)
            if prix_max_ligne and prix > prix_max_ligne: continue
            if a_dep["id"] not in seq: continue
            montee = RESEAU["arrets"][a_dep["id"]]
            md = distance_m(dep["latitude"], dep["longitude"],
                            montee["latitude"], montee["longitude"])
            if md > rayon: continue
            im = seq.index(a_dep["id"])
            best_sid, best_d = None, float("inf")
            for sid in seq[im + 1:]:
                ar = RESEAU["arrets"].get(sid)
                if not ar: continue
                d = distance_m(dest["latitude"], dest["longitude"],
                               ar["latitude"], ar["longitude"])
                if d < best_d: best_d, best_sid = d, sid
            if not best_sid or best_d > rayon: continue
            k = ("D", info["rid"], a_dep["id"], best_sid)
            if k in vus: continue
            vus.add(k)
            descente = RESEAU["arrets"][best_sid]
            seg = seq[im:seq.index(best_sid) + 1]
            seq_arrets = [{"id": RESEAU["arrets"][s]["id"], "nom": RESEAU["arrets"][s]["nom"],
                "latitude": RESEAU["arrets"][s]["latitude"],
                "longitude": RESEAU["arrets"][s]["longitude"]}
                for s in seg if s in RESEAU["arrets"]]
            md_deb = max(1, int(md) // 80)
            md_fin = max(1, int(best_d) // 80)
            duree_transp = _duree_sur_ligne(seg)
            nouveau = {"duree_minutes": md_deb + duree_transp + md_fin,
                "cout_fcfa": prix, "marche_m": int(md) + int(best_d),
                "nb_correspondances": 0,
                "etapes": [
                    {"mode": "MARCHE", "depart": dep.get("nom", "depart"),
                     "arrivee": montee["nom"], "distance_m": int(md),
                     "duree_minutes": md_deb, "cout_fcfa": 0,
                     "depart_pos": {"latitude": dep["latitude"], "longitude": dep["longitude"]},
                     "arrivee_pos": {"latitude": montee["latitude"], "longitude": montee["longitude"]}},
                    {"mode": ligne["type"], "numero_ligne": ligne.get("numero"),
                     "nom_ligne": ligne.get("nom"), "depart": montee["nom"],
                     "arrivee": descente["nom"],
                     "depart_pos": {"latitude": montee["latitude"], "longitude": montee["longitude"]},
                     "arrivee_pos": {"latitude": descente["latitude"], "longitude": descente["longitude"]},
                     "distance_m": distance_m(montee["latitude"], montee["longitude"],
                                              descente["latitude"], descente["longitude"]),
                     "duree_minutes": duree_transp, "cout_fcfa": prix,
                     "arrets": seq_arrets},
                    {"mode": "MARCHE", "depart": descente["nom"],
                     "arrivee": dest.get("nom", "destination"),
                     "distance_m": int(best_d), "duree_minutes": md_fin, "cout_fcfa": 0,
                     "depart_pos": {"latitude": descente["latitude"],
                                    "longitude": descente["longitude"]},
                     "arrivee_pos": {"latitude": dest["latitude"],
                                     "longitude": dest["longitude"]}}],
                "arrets_itineraire": seq_arrets}
            anc = meilleur_direct.get(info["rid"])
            if not anc or _score_trajet(nouveau) < _score_trajet(anc):
                meilleur_direct[info["rid"]] = nouveau
    candidats.extend(meilleur_direct.values())

    # TRAJETS AVEC 1 CORRESPONDANCE
    DIST_C = 250 if modes == {"GBAKA"} else 500
    nb_corr, meilleur_corr = 0, {}
    arrets_arr_c = list(arrets_arr)
    # CORRECTION : on ajoute 20 arrets proches de la destination pour trouver la bonne correspondance
    vus_arr = {a["id"] for a in arrets_arr_c}
    for a in arrets_arr_prox:
        if a["id"] in vus_arr: continue
        arrets_arr_c.append(a)
        vus_arr.add(a["id"])
        if len(arrets_arr_c) >= 20: break
    if modes & {"GBAKA"}:
        proches = []
        for ar in RESEAU["arrets"].values():
            d = distance_m(dest["latitude"], dest["longitude"],
                           ar["latitude"], ar["longitude"])
            if d > descente_max_gbaka: continue
            if any(RESEAU["lignes"].get(rid, {}).get("type") == "GBAKA"
                   for rid in RESEAU["lignes_par_arret"].get(ar["id"], [])):
                proches.append({"id": ar["id"], "nom": ar["nom"],
                    "latitude": ar["latitude"], "longitude": ar["longitude"],
                    "distance_m": int(d)})
        proches.sort(key=lambda a: a["distance_m"])
        ids_gb = {a["id"] for a in proches}
        arrets_arr_c = proches + [a for a in arrets_arr_c if a["id"] not in ids_gb]
    sens_c = set()
    for a_dep in arrets_dep[:40]:
        if nb_corr >= 25: break
        for info1 in _lignes_passant_par(a_dep["id"], modes):
            if nb_corr >= 25: break
            l1, s1 = info1["ligne"], info1["sequence"]
            if l1["type"] in ("GBAKA", "WORO_WORO"):
                k = (info1["rid"], tuple(s1))
                if k in sens_c: continue
                sens_c.add(k)
            prix1 = _prix_ligne(l1)
            if a_dep["id"] not in s1: continue
            montee1 = RESEAU["arrets"][a_dep["id"]]
            md = distance_m(dep["latitude"], dep["longitude"],
                            montee1["latitude"], montee1["longitude"])
            if md > rayon: continue
            im = s1.index(a_dep["id"])
            for a_arr in arrets_arr_c[:20]:
                for info2 in _lignes_passant_par(a_arr["id"], modes):
                    l2 = info2["ligne"]
                    if l2["id"] == l1["id"]: continue
                    s2 = info2["sequence"]
                    if a_arr["id"] not in s2: continue
                    ia = s2.index(a_arr["id"])
                    t = _meilleur_transfert(s1, im, s2, ia - 1, rayon=DIST_C,
                                            ligne_b_bus=(l2["type"] == "BUS"))
                    if not t: continue
                    sid_d1, sid_m2, dc = t[3], t[4], t[2]
                    ar_d1 = RESEAU["arrets"][sid_d1]
                    ar_m2 = RESEAU["arrets"][sid_m2]
                    descente2 = RESEAU["arrets"][a_arr["id"]]
                    nb1 = s1.index(sid_d1) - im
                    nb2 = ia - s2.index(sid_m2)
                    if nb1 < 1 or nb2 < 1: continue
                    prix2 = _prix_ligne(l2)
                    d1 = _duree_sur_ligne(s1[im:im + nb1 + 1])
                    d2 = _duree_sur_ligne(s2[ia - nb2:ia + 1])
                    dc_min = max(2, int(dc) // 80)
                    k = ("C", l1["id"], l2["id"], a_dep["id"], sid_d1, sid_m2, a_arr["id"])
                    if k in vus: continue
                    vus.add(k)
                    md_min = max(1, int(md) // 80)
                    md_fin_min = max(1, int(a_arr["distance_m"]) // 80)
                    etapes = [
                        {"mode": "MARCHE", "depart": dep.get("nom", "depart"),
                         "arrivee": montee1["nom"], "distance_m": int(md),
                         "duree_minutes": md_min, "cout_fcfa": 0,
                         "depart_pos": {"latitude": dep["latitude"],
                                        "longitude": dep["longitude"]},
                         "arrivee_pos": {"latitude": montee1["latitude"],
                                         "longitude": montee1["longitude"]}},
                        {"mode": l1["type"], "numero_ligne": l1.get("numero"),
                         "nom_ligne": l1.get("nom"), "depart": montee1["nom"],
                         "arrivee": ar_d1["nom"],
                         "depart_pos": {"latitude": montee1["latitude"],
                                        "longitude": montee1["longitude"]},
                         "arrivee_pos": {"latitude": ar_d1["latitude"],
                                         "longitude": ar_d1["longitude"]},
                         "distance_m": distance_m(montee1["latitude"], montee1["longitude"],
                                                  ar_d1["latitude"], ar_d1["longitude"]),
                         "duree_minutes": d1, "cout_fcfa": prix1, "nb_arrets": nb1},
                        {"mode": "MARCHE", "depart": ar_d1["nom"],
                         "arrivee": ar_m2["nom"], "distance_m": int(dc),
                         "duree_minutes": dc_min, "cout_fcfa": 0,
                         "depart_pos": {"latitude": ar_d1["latitude"],
                                        "longitude": ar_d1["longitude"]},
                         "arrivee_pos": {"latitude": ar_m2["latitude"],
                                         "longitude": ar_m2["longitude"]}},
                        {"mode": l2["type"], "numero_ligne": l2.get("numero"),
                         "nom_ligne": l2.get("nom"), "depart": ar_m2["nom"],
                         "arrivee": descente2["nom"],
                         "depart_pos": {"latitude": ar_m2["latitude"],
                                        "longitude": ar_m2["longitude"]},
                         "arrivee_pos": {"latitude": descente2["latitude"],
                                         "longitude": descente2["longitude"]},
                         "distance_m": distance_m(ar_m2["latitude"], ar_m2["longitude"],
                                                  descente2["latitude"], descente2["longitude"]),
                         "duree_minutes": d2, "cout_fcfa": prix2, "nb_arrets": nb2},
                        {"mode": "MARCHE", "depart": descente2["nom"],
                         "arrivee": dest.get("nom", "destination"),
                         "distance_m": a_arr["distance_m"], "duree_minutes": md_fin_min,
                         "cout_fcfa": 0,
                         "depart_pos": {"latitude": descente2["latitude"],
                                        "longitude": descente2["longitude"]},
                         "arrivee_pos": {"latitude": dest["latitude"],
                                         "longitude": dest["longitude"]}}]
                    nouveau = {"duree_minutes": md_min + d1 + dc_min + d2 + md_fin_min,
                        "cout_fcfa": prix1 + prix2,
                        "marche_m": int(md) + int(dc) + a_arr["distance_m"],
                        "nb_correspondances": 1, "etapes": etapes,
                        "arrets_itineraire": []}
                    paire = (l1["id"], l2["id"])
                    anc = meilleur_corr.get(paire)
                    if not anc or _score_trajet(nouveau) < _score_trajet(anc):
                        meilleur_corr[paire] = nouveau
                    nb_corr = len(meilleur_corr)
                    if nb_corr >= 25: break
    candidats.extend(meilleur_corr.values())
    if not any(c["etapes"][0].get("distance_m", 0) <= SEUIL_MARCHE_LONGUE_M
               for c in candidats):
        candidats.extend(_candidats_deux_correspondances(dep, dest, modes,
            arrets_dep, arrets_arr, prix_max_ligne))

    # FALLBACK : si vraiment rien, on cherche dans le catalogue par nom
    if not candidats:
        for c in CATALOGUE:
            if c["type"] not in modes: continue
            ori = nettoyer_texte(c.get("origine") or "")
            des = nettoyer_texte(c.get("destination") or "")
            if not ori or not des: continue
            def match(nom_cat, arrets):
                for a in arrets:
                    n = nettoyer_texte(a["nom"])
                    if any(m in n for m in nom_cat.split() if len(m) > 3) or \
                       any(m in nom_cat for m in n.split() if len(m) > 3): return a
                return None
            montee = match(ori, arrets_dep)
            descente = match(des, arrets_arr)
            if not (montee and descente):
                montee = match(des, arrets_dep)
                descente = match(ori, arrets_arr)
            if not (montee and descente): continue
            trace, bbox = c.get("trace") or [], c.get("bbox")
            if not trace or not bbox: continue
            if not _point_proche_du_trace(montee["latitude"], montee["longitude"], trace, bbox): continue
            if not _point_proche_du_trace(descente["latitude"], descente["longitude"], trace, bbox): continue
            prix = c.get("prix_fcfa") or _prix_ligne({"numero": c.get("numero"), "type": c["type"]})
            candidats.append({
                "duree_minutes": max(1, montee["distance_m"] // 80) + 40 + max(1, descente["distance_m"] // 80),
                "cout_fcfa": prix, "marche_m": montee["distance_m"] + descente["distance_m"],
                "nb_correspondances": 0,
                "etapes": [
                    {"mode": "MARCHE", "depart": dep.get("nom", "depart"),
                     "arrivee": montee["nom"], "distance_m": montee["distance_m"],
                     "duree_minutes": max(1, montee["distance_m"] // 80), "cout_fcfa": 0,
                     "depart_pos": {"latitude": dep["latitude"], "longitude": dep["longitude"]},
                     "arrivee_pos": {"latitude": montee["latitude"], "longitude": montee["longitude"]}},
                    {"mode": c["type"], "numero_ligne": c.get("numero"),
                     "nom_ligne": f"{c['origine']} ↔ {c['destination']}",
                     "depart": montee["nom"], "arrivee": descente["nom"],
                     "depart_pos": {"latitude": montee["latitude"], "longitude": montee["longitude"]},
                     "arrivee_pos": {"latitude": descente["latitude"], "longitude": descente["longitude"]},
                     "distance_m": distance_m(montee["latitude"], montee["longitude"],
                                              descente["latitude"], descente["longitude"]),
                     "duree_minutes": 40, "cout_fcfa": prix},
                    {"mode": "MARCHE", "depart": descente["nom"],
                     "arrivee": dest.get("nom", "destination"),
                     "distance_m": descente["distance_m"],
                     "duree_minutes": max(1, descente["distance_m"] // 80), "cout_fcfa": 0,
                     "depart_pos": {"latitude": descente["latitude"],
                                    "longitude": descente["longitude"]},
                     "arrivee_pos": {"latitude": dest["latitude"],
                                     "longitude": dest["longitude"]}}],
                "arrets_itineraire": []})
            if len(candidats) >= 5: break

    # Filtre gbaka : on descend pres de la destination
    if modes & {"GBAKA", "WORO_WORO"}:
        def trop_loin(c):
            tr = [e for e in c["etapes"] if e["mode"] != "MARCHE"]
            fin = c["etapes"][-1]
            return bool(tr) and tr[-1]["mode"] in ("GBAKA", "WORO_WORO") \
                and fin["mode"] == "MARCHE" and fin.get("distance_m", 0) > descente_max_gbaka
        candidats = [c for c in candidats if not trop_loin(c)]
    if modes == {"GBAKA"}: candidats = _filtrer_trajets_gbaka(candidats)
    for c in candidats: _ajouter_attente(c)

    # Variantes avec woro-woro si marche initiale trop longue
    variantes = []
    for c in list(candidats):
        if arret_depart_prioritaire:
            premier = next((e for e in c.get("etapes", [])
                            if e.get("mode") != "MARCHE"), None)
            if not premier: continue
            dp = premier.get("depart_pos") or {}
            if dp.get("latitude") is None or dp.get("longitude") is None: continue
            if distance_m(arret_depart_prioritaire["latitude"],
                          arret_depart_prioritaire["longitude"],
                          dp["latitude"], dp["longitude"]) > 100: continue
        v = _variante_woro_si_longue_marche(c)
        if v: variantes.append(v)
    candidats.extend(variantes)

    # CORRECTION : si on a des trajets qui deposent a moins de 500 m, on prefere ceux-la
    def m_fin(c):
        e = c["etapes"][-1]
        return e.get("distance_m", 0) if e["mode"] == "MARCHE" else 0
    courts = [c for c in candidats if m_fin(c) <= 500]
    if courts: candidats = courts

    # Tri et deduplication
    par_v = sorted(candidats, key=lambda c: (c["nb_correspondances"],
                                              _score_trajet(c), c["cout_fcfa"]))
    par_m = sorted(candidats, key=lambda c: (
        c["nb_correspondances"] +
        (1 if c["etapes"][0].get("distance_m", 0) > SEUIL_MARCHE_LONGUE_M else 0),
        _score_trajet(c), c["cout_fcfa"]))
    fusion = []
    for a, b in zip(par_v, par_m): fusion.extend([a, b])
    uniques, vus_sig = [], set()
    for c in fusion:
        sig = tuple((e.get("numero_ligne") or e.get("nom_ligne"), e["depart"], e["arrivee"])
                    for e in c["etapes"] if e["mode"] != "MARCHE")
        if sig in vus_sig: continue
        vus_sig.add(sig)
        uniques.append(c)
    candidats = uniques[:10]
    for c in candidats:
        c["etapes"] = [e for i, e in enumerate(c["etapes"])
            if not (e["mode"] == "MARCHE" and 0 < i < len(c["etapes"]) - 1
                    and e.get("distance_m", 0) < 30)]
    for c in candidats: _ajouter_conseils_marche(c)
    return candidats

def construire_candidats(dep, dest, mode_filtre="TOUS", prix_max_ligne=None):
    # Essaie plusieurs rayons de recherche du plus petit au plus grand
    for rayon in (1500, 2500, 4000):
        res = _construire_candidats_rayon(dep, dest, mode_filtre, prix_max_ligne, rayon)
        if res:
            for c in res: _normaliser_candidat(c)
            return res
    return []

def candidats_multimodaux(dep, dest, mode_filtre="TOUS", prix_max_ligne=None):
    # Si un mode est impose, on cherche uniquement dans ce mode
    # Sinon on combine tous les modes
    if mode_filtre in ("BUS", "GBAKA", "BATEAU"):
        autorises = {"BUS": {"BUS"}, "GBAKA": {"GBAKA", "WORO_WORO"},
                     "BATEAU": {"BATEAU"}}[mode_filtre]
        def modes_trajet(c):
            return {e["mode"] for e in c.get("etapes", []) if e["mode"] != "MARCHE"}
        out = construire_candidats(dep, dest, mode_filtre, prix_max_ligne)
        purs = [c for c in out if modes_trajet(c) <= autorises]
        if not purs:
            mixtes = construire_candidats(dep, dest, "TOUS", prix_max_ligne)
            purs = [c for c in mixtes
                    if modes_trajet(c) & autorises and not modes_trajet(c) <= autorises]
            for c in purs: c["combo_mode"] = mode_filtre
        for c in purs: _ajouter_textes(c)
        return purs
    out, vus = [], set()
    groupes = [construire_candidats(dep, dest, "TOUS", prix_max_ligne)]
    pm = trajets_par_mode(dep, dest, prix_max_ligne)
    groupes += [pm.get("bus", []), pm.get("gbakas", []), pm.get("bateaux", [])]
    for groupe in groupes:
        for c in groupe:
            sig = _signature_candidat(c)
            if sig in vus: continue
            vus.add(sig)
            _ajouter_textes(c)
            out.append(c)
    return out

def arrets_proches_trajet(dep, dest, mode_filtre="TOUS", rayon_m=3000):
    modes = None if mode_filtre == "TOUS" else [mode_filtre]
    a_d = arrets_proximite_par_modes(dep["latitude"], dep["longitude"], rayon_m, modes)
    a_a = arrets_proximite_par_modes(dest["latitude"], dest["longitude"], rayon_m, modes)
    return {"depart": a_d[:5], "arrivee": a_a[:5]}

def trajets_par_mode(dep, dest, prix_max_ligne=None):
    return {"bus": construire_candidats(dep, dest, "BUS", prix_max_ligne)[:5],
        "gbakas": construire_candidats(dep, dest, "GBAKA", prix_max_ligne)[:5],
        "bateaux": construire_candidats(dep, dest, "BATEAU", prix_max_ligne)[:5]}

def passages_gbaka_pres_de(lat, lon, rayon_m=800):
    # Liste les passages de gbaka pres d une position
    if not RESEAU["charge"]: charger_reseau_local()
    res = []
    for rid, seqs in RESEAU["sequences"].items():
        l = RESEAU["lignes"].get(rid)
        if not l or l["type"] not in ("GBAKA", "WORO_WORO"): continue
        for idx, seq in enumerate(seqs):
            label = "aller" if idx == 0 else "retour"
            for sid in seq:
                a = RESEAU["arrets"].get(sid)
                if not a: continue
                d = distance_m(lat, lon, a["latitude"], a["longitude"])
                if d <= rayon_m:
                    res.append({"arret_id": sid, "arret_nom": a["nom"],
                        "latitude": a["latitude"], "longitude": a["longitude"],
                        "distance_m": int(d), "ligne_id": rid, "ligne_nom": l.get("nom"),
                        "type": l["type"], "sens": label, "terminus": seq[-1],
                        "nb_arrets_total": len(seq)})
    res.sort(key=lambda x: x["distance_m"])
    return res


# SECTION FRONTEND
def normaliser_reseau(reseau, type_ligne):
    r = nettoyer_texte(reseau or "")
    if type_ligne == "BATEAU":
        if "monbato" in r: return "Monbato"
        if "stl" in r: return "STL"
        if "aqua" in r: return "Aqualines"
        return "Autre"
    if type_ligne == "BUS":
        if "express" in r: return "Express"
        if "wibus" in r: return "Wibus"
        if "navette" in r: return "Navette"
        if "monbus" in r or "mon bus" in r: return "Monbus"
        return "Autre"
    if type_ligne == "GBAKA":
        if "adja" in r: return "Adjamé"
        if "abobo" in r: return "Abobo"
        return "Autre"
    return "Autre"

def lignes_du_catalogue(type_ligne=None):
    res = []
    for i, c in enumerate(CATALOGUE):
        if type_ligne and c["type"] != type_ligne: continue
        numero = c["numero"]
        if c["type"] == "BUS" and numero and str(numero).isdigit() and 1 <= int(numero) <= 9:
            numero = f"{int(numero):02d}"
        if c["type"] == "GBAKA":
            ori = nettoyer_texte(c.get("origine") or "")
            reseau = ("Adjamé" if "adja" in ori else
                     ("Abobo" if "abobo" in ori else normaliser_reseau(c.get("reseau"), "GBAKA")))
        else:
            reseau = normaliser_reseau(c.get("reseau"), c["type"])
        res.append({"id": f"cat:{i}", "numero": numero or None,
            "origine": c["origine"], "destination": c["destination"],
            "nom": f"{c['origine']} ↔ {c['destination']}", "operateur": c["operateur"],
            "reseau": reseau, "type": c["type"],
            "prix_fcfa": c.get("prix_fcfa") or prix_connu(c["type"], None, "", "", c["numero"])})
    return res

def liste_lignes(type_transport=None):
    return lignes_du_catalogue(type_transport)

def lignes_pour_app(type_transport=None, q=None, zone=None, reseau=None):
    genre = None
    if type_transport:
        t = type_transport.strip().upper()
        if t in ("BUS", "GBAKA", "BATEAU", "WORO_WORO"): genre = t
    toutes = lignes_du_catalogue(genre)
    comptes = Counter(l["reseau"] for l in toutes)
    filtres = [{"nom": "Toutes", "valeur": "toutes", "total": len(toutes)}]
    for nom, tot in comptes.most_common():
        filtres.append({"nom": nom, "valeur": nom.lower(), "total": tot})
    choisies = toutes
    if reseau and reseau.lower() not in ("toutes", "tous", ""):
        rn = nettoyer_texte(reseau)
        choisies = [l for l in choisies if nettoyer_texte(l["reseau"]) == rn]
    if q:
        mots = extraire_mots(q)
        choisies = [l for l in choisies if all(
            m in nettoyer_texte(f"{l.get('numero') or ''} {l['origine']} {l['destination']}")
            for m in mots)]
    for l in choisies:
        if l["type"] in ("GBAKA", "WORO_WORO"):
            l["sens"] = _arrets_ligne_pour_api(l["id"])
    return {"type": genre, "reseau": reseau or "toutes", "total": len(choisies),
        "total_type": len(toutes), "filtres": filtres, "lignes": choisies}

def _mots_cle(texte):
    return [m for m in nettoyer_texte(texte or "").split()
            if len(m) > 2 and m not in ("terminus", "gare", "arret")]

def _score_nom(cible, nom):
    # Score de similarite entre deux noms, de 0 a 1
    import difflib
    mc, mn = _mots_cle(cible), _mots_cle(nom)
    if not mc: return 0.0
    ok = 0
    for a in mc:
        for b in mn:
            if a == b or (len(a) >= 5 and len(b) >= 5
                          and (a.startswith(b[:5]) or b.startswith(a[:5]))) \
               or difflib.SequenceMatcher(None, a, b).ratio() >= 0.82:
                ok += 1
                break
    return ok / len(mc)

def _sens_ligne_bus(fiche, index_cat, origine, destination, numero):
    # Renvoie aller et retour d une ligne bus ou bateau
    rid_cat = f"cat:{index_cat}" if index_cat is not None else None
    candidates = []
    if str(numero).strip() == "610":
        for rid, info in RESEAU["lignes"].items():
            if not rid.startswith("cat:") and info.get("type") == fiche["type"] \
                    and str(info.get("numero") or "").strip() == "610":
                candidates.extend(RESEAU["sequences"].get(rid, []))
    if not candidates and rid_cat and RESEAU["sequences"].get(rid_cat):
        candidates.extend(RESEAU["sequences"][rid_cat])
    if not candidates and numero:
        variantes = {numero}
        if numero.isdigit(): variantes |= {str(int(numero)), f"{int(numero):02d}"}
        for rid, info in RESEAU["lignes"].items():
            if info.get("type") != fiche["type"] or rid.startswith("cat:"): continue
            num = str(info.get("numero") or "").strip()
            if num not in variantes and not (num.isdigit() and numero.isdigit()
                                              and int(num) == int(numero)): continue
            nom_l = nettoyer_texte(info.get("nom") or "")
            if _score_nom(origine, nom_l) < 0.5 and _score_nom(destination, nom_l) < 0.5: continue
            for seq in RESEAU["sequences"].get(rid, []): candidates.append(seq)
    if not candidates and numero:
        rids = [rid for rid, info in RESEAU["lignes"].items()
            if info.get("type") == fiche["type"] and not rid.startswith("cat:")
            and str(info.get("numero") or "").strip() == str(numero).strip()]
        if len(rids) == 1: candidates.extend(RESEAU["sequences"].get(rids[0], []))
    if not candidates: return []

    def nettoyee(seq):
        out, prec = [], None
        for sid in seq:
            a = RESEAU["arrets"].get(sid)
            if not a: continue
            k = nettoyer_texte(a["nom"])
            if k == prec: continue
            prec = k
            out.append(sid)
        return out

    def pos(sid):
        a = RESEAU["arrets"][sid]
        return a["latitude"], a["longitude"]

    def dist_ids(x, y): return distance_m(*pos(x), *pos(y))

    def moitie(seq):
        # Coupe une sequence en boucle en aller et retour
        if len(seq) < 6 or dist_ids(seq[0], seq[-1]) > 500: return [seq]
        eloig = [dist_ids(seq[0], sid) for sid in seq]
        m = max(range(len(seq)), key=lambda i: eloig[i])
        f = range(max(1, m - 3), min(len(seq) - 1, m + 3) + 1)
        m = max(f, key=lambda i: (_score_nom(destination,
                                              RESEAU["arrets"][seq[i]]["nom"]), eloig[i]))
        if m < 2 or len(seq) - m < 3: return [seq]
        return [seq[:m + 1], seq[m:]]

    cands = []
    for sq in candidates:
        for mo in moitie(nettoyee(sq)):
            if len(mo) >= 2 and mo not in cands: cands.append(mo)
    if not cands: return []

    def repere(nom):
        best, sc = None, 0.5
        for sq in cands:
            for sid in sq:
                v = _score_nom(nom, RESEAU["arrets"][sid]["nom"])
                if v > sc or (v == sc and best is None and v >= 0.5): best, sc = sid, v
        return best

    rep_o, rep_d = repere(origine), repere(destination)

    def ecart(seq, ro, rd):
        e = 0.0
        if ro: e += dist_ids(seq[0], ro)
        if rd: e += dist_ids(seq[-1], rd)
        return e

    def cle_sens(seq, ro, rd):
        s = _score_nom(origine, RESEAU["arrets"][seq[0]]["nom"]) + \
            _score_nom(destination, RESEAU["arrets"][seq[-1]]["nom"])
        return (round(s, 2), -ecart(seq, ro, rd), len(seq))

    aller = max(cands, key=lambda c: cle_sens(c, rep_o, rep_d))
    if str(numero).strip() == "610":
        retour = list(reversed(aller))
    else:
        def ecart_ret(c): return dist_ids(c[0], aller[-1]) + dist_ids(c[-1], aller[0])
        autres = [c for c in cands if c != aller]
        retour = min(autres, key=ecart_ret) if autres else None
        if retour is None or ecart_ret(retour) > 1500 \
                or [RESEAU["arrets"][x]["nom"] for x in retour] == \
                   [RESEAU["arrets"][x]["nom"] for x in aller]:
            retour = list(reversed(aller))

    def fabrique(seq, label, direction, nom_dep, nom_term):
        arrets = [{"id": RESEAU["arrets"][s]["id"], "nom": RESEAU["arrets"][s]["nom"],
            "latitude": RESEAU["arrets"][s]["latitude"],
            "longitude": RESEAU["arrets"][s]["longitude"], "role": "arret"} for s in seq]
        arrets[0]["role"], arrets[-1]["role"] = "depart", "arrivee"
        if nom_dep: arrets[0]["nom"] = nom_dep
        if nom_term: arrets[-1]["nom"] = nom_term
        return {"direction": direction, "sens": label,
            "titre": f"{arrets[0]['nom']} -> {arrets[-1]['nom']}",
            "depart": arrets[0]["nom"], "terminus": arrets[-1]["nom"],
            "nb_arrets": len(arrets), "arrets": arrets, "trace": []}

    return [fabrique(aller, "aller", "A", origine, destination),
            fabrique(retour, "retour", "R", destination, origine)]

def detail_ligne(ident):
    # Renvoie la fiche complete d une ligne
    if not RESEAU["charge"]: charger_reseau_local()
    fiche, index_cat = None, None
    if ident.startswith("cat:") and ident[4:].isdigit():
        i = int(ident[4:])
        if 0 <= i < len(CATALOGUE): fiche, index_cat = CATALOGUE[i], i
    if not fiche:
        for i, c in enumerate(CATALOGUE):
            if str(c.get("numero") or "") == str(ident):
                fiche, index_cat = c, i
                break
    if not fiche: return None
    origine = (fiche.get("origine") or "").strip()
    destination = (fiche.get("destination") or "").strip()
    numero = str(fiche.get("numero") or "").strip()
    resultat = {"id": ident if ident.startswith("cat:") else f"cat:{index_cat}",
        "numero": numero or None, "type": fiche["type"],
        "operateur": fiche.get("operateur"),
        "reseau": normaliser_reseau(fiche.get("reseau"), fiche["type"]),
        "nom": f"{origine} ↔ {destination}", "depart": origine, "arrivee": destination,
        "prix_fcfa": fiche.get("prix_fcfa") or prix_connu(fiche["type"], None, "", "", numero),
        "nb_arrets": None, "sens": []}
    if fiche["type"] in ("GBAKA", "WORO_WORO"):
        sens_csv = _arrets_ligne_pour_api(f"cat:{index_cat}")
        if sens_csv:
            resultat["nb_arrets"] = max(s["nb_arrets"] for s in sens_csv)
            for s in sens_csv:
                arrets = [dict(a) for a in s["arrets"]]
                if arrets: arrets[0]["role"], arrets[-1]["role"] = "depart", "arrivee"
                resultat["sens"].append({"direction": "A" if s["sens"] == "aller" else "R",
                    "sens": s["sens"], "titre": f"{s['depart']} -> {s['arrivee']}",
                    "depart": s["depart"], "terminus": s["arrivee"],
                    "nb_arrets": s["nb_arrets"], "arrets": arrets, "trace": []})
            return resultat
    if numero or index_cat is not None:
        sb = _sens_ligne_bus(fiche, index_cat, origine, destination, numero)
        if sb:
            resultat["sens"] = sb
            resultat["nb_arrets"] = max(x["nb_arrets"] for x in sb)
    if not resultat["sens"]:
        resultat["nb_arrets"] = 2
        resultat["sens"] = [{"direction": "A", "sens": "aller",
            "titre": f"{origine} -> {destination}", "depart": origine,
            "terminus": destination, "nb_arrets": 2,
            "arrets": [{"id": None, "nom": origine, "latitude": None,
                        "longitude": None, "role": "depart"},
                       {"id": None, "nom": destination, "latitude": None,
                        "longitude": None, "role": "arrivee"}], "trace": []}]
    return resultat

def recherche_globale(q, type_transport=None, maximum=20):
    mots = extraire_mots(q)
    if not mots: return {"lignes": [], "arrets": []}
    lignes = lignes_pour_app(type_transport, q)["lignes"][:maximum]
    arrets = [a for a in RESEAU["arrets"].values()
              if a["id"] not in RESEAU["points_libres"]
              and all(m in nettoyer_texte(a["nom"]) for m in mots)]
    return {"lignes": lignes, "arrets": arrets[:maximum]}

def trajets_pour_demande(candidats, priorite=None, prix_max=None, budget=None):
    if not candidats: return []
    if budget: candidats = [c for c in candidats if _cout_tri(c) <= budget]
    if priorite == "prix":
        candidats.sort(key=lambda c: (_cout_tri(c), c.get("duree_minutes", 999)))
    else:
        candidats.sort(key=lambda c: (c.get("duree_minutes", 999), _cout_tri(c)))
    return candidats

def instructions_trajet(candidat):
    # Renvoie une liste d instructions courtes pour le trajet
    if not candidat or not candidat.get("etapes"): return []
    out = []
    for e in candidat["etapes"]:
        if e["mode"] == "MARCHE":
            out.append(f"marchez {e['distance_m']} m jusqu a {e['arrivee']}")
        else:
            nom = e.get("numero_ligne") or e.get("nom_ligne") or e["mode"]
            out.append(f"prenez le {e['mode'].lower()} {nom} de {e['depart']} a {e['arrivee']}")
    return out

def _texte_trajet(candidat, mode="TOUS"):
    if not candidat: return "aucun trajet trouve."
    parts = []
    for e in candidat.get("etapes", []):
        if e["mode"] == "MARCHE":
            if e.get("distance_m", 0) > 0:
                parts.append(f"marchez {e['distance_m']} m jusqu a {e['arrivee']}")
        else:
            nom = e.get("numero_ligne") or e.get("nom_ligne") or ""
            parts.append(f"{e['mode']} {nom} : {e['depart']} -> {e['arrivee']}")
    return " puis ".join(parts) + f" (environ {candidat.get('duree_minutes')} min)"


# SECTION AGENT IA
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
SESSIONS = defaultdict(lambda: {"depart": None, "destination": None,
    "dernier_trajet": None, "budget_fcfa": None, "mode_transport": None,
    "attente_mode": False})

def maj_session(session_id, intention, trajet=None, attente_mode=None):
    # Met a jour les infos de la session utilisateur
    if not session_id: return
    s = SESSIONS[session_id]
    for k in ("depart", "destination"):
        if intention.get(k): s[k] = intention[k]
    if intention.get("budget_fcfa"): s["budget_fcfa"] = intention["budget_fcfa"]
    if intention.get("mode_precise") and intention.get("mode_transport"):
        s["mode_transport"] = intention["mode_transport"]
    if trajet: s["dernier_trajet"] = trajet
    if attente_mode is not None:
        s["attente_mode"] = attente_mode
        if attente_mode: s["mode_demande"] = True

def extraire_budget(texte):
    # Comprend le budget dans une phrase comme "j ai 400" ou "2 mille"
    t = (texte or "").lower().replace("\u202f", " ").replace("\u00a0", " ")
    nombre = r"(\d{1,3}(?:[ .]\d{3})+|\d+)"
    val = None
    m = re.search(nombre + r"\s*(?:mille|milles|k)\b", t)
    if m: val = int(re.sub(r"\D", "", m.group(1))) * 1000
    if val is None:
        m = re.search(nombre + r"\s*(?:fcfa|f\.?\s?cfa|cfa|francs?|frs?\b|f\b|balles?|piastres?|billets?)", t)
        if m: val = int(re.sub(r"\D", "", m.group(1)))
    if val is None:
        m = re.search(r"(?:budget|maximum|max|co[uû]t|prix|dans les|moins de|pas plus de|"
            r"jusqu['’]?\s?[àa]|j['’]?ai(?: seulement| juste| que)?|je n['’]?ai que|"
            r"il me reste|me reste|je dispose de|je poss[eè]de|avec|"
            r"(?:un )?petit budget de)\D{0,12}" + nombre, t)
        if m: val = int(re.sub(r"\D", "", m.group(1)))
    if val is None:
        m = re.fullmatch(r"\s*" + nombre + r"\s*[.!]?\s*", t)
        if m: val = int(re.sub(r"\D", "", m.group(1)))
    return val if val is not None and 50 <= val <= 1000000 else None

def extraire_mode(texte, accepter_tous=False):
    # Trouve le mode de transport demande : BUS, GBAKA, BATEAU ou TOUS
    t = nettoyer_texte(texte)
    t = re.sub(r"bateau[\s_-]*bus", "bateau", t)

    def nie(pat):
        return bool(re.search(rf"(?:pas|sans|au lieu (?:de |du |des |d')|plut[oô]t que |non)\s*"
            rf"(?:le |la |les |un |une |des )?{pat}", t))

    trouves = set()
    if re.search(r"\b(bateau|bateaux|lagunaire|bato|monbato)\b", t) \
            and not nie(r"\b(bateau|bateaux)\b"): trouves.add("BATEAU")
    if re.search(r"\b(gbaka|gbakas|woro|woros)\b", t) \
            and not nie(r"\b(gbaka|gbakas|woro|woros)\b"): trouves.add("GBAKA")
    if re.search(r"\b(bus|sotra|autobus|monbus)\b", t) \
            and not nie(r"\b(bus|sotra|autobus|monbus)\b"): trouves.add("BUS")
    if len(trouves) == 1: return next(iter(trouves))
    if len(trouves) > 1: return "TOUS"
    if re.search(r"(tous les (modes|transports|moyens)|n.?importe (quel|lequel)|"
                 r"peu importe|indifferent|pas de preference)", t): return "TOUS"
    if accepter_tous and re.search(r"\b(tous|tout|les deux)\b", t): return "TOUS"
    return None

def _valeur_vide(v):
    return v is None or (isinstance(v, str)
                         and v.strip().lower() in ("", "null", "none", "aucun"))

def _completer_analyse(data, s, mode_msg, budget_msg):
    # Complete les infos de l analyse avant de renvoyer au frontend
    data.setdefault("intention", "INCONNUE")
    if data["intention"] == "INCONNUE": data["intention"] = "CONVERSATION"
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
        for k in ("depart", "destination"):
            v = None if _valeur_vide(data.get(k)) else data[k]
            data[k] = v or s.get(k)
        if mode_msg:
            data["mode_transport"], data["mode_precise"] = mode_msg, True
        elif s.get("mode_demande"): data["mode_precise"] = True
        data["budget_fcfa"] = budget_msg or s.get("budget_fcfa")
    return data

def _extraire_infos_conditions(contexte):
    # Transforme les infos trafic et meteo en texte lisible
    infos = []
    trafic = contexte.get("trafic_niveau") or contexte.get("trafic")
    pluie = contexte.get("pluie") or contexte.get("weather") or contexte.get("meteo")
    facteur = contexte.get("trafic_facteur")
    if isinstance(trafic, str):
        libelle = {"eleve": "dense", "moyen": "modere", "faible": "fluide"}.get(
            nettoyer_texte(trafic), trafic)
        extra = (f", les trajets durent environ x{facteur:.2f}"
                 if isinstance(facteur, (int, float)) and facteur > 1.05 else "")
        infos.append(f"le trafic est {libelle}{extra}")
        trafic = None
    if trafic is not None:
        if isinstance(trafic, (int, float)):
            if trafic >= 0.7: infos.append(f"le trafic est dense (niveau {trafic:.2f})")
            elif trafic >= 0.4: infos.append(f"le trafic est modere (niveau {trafic:.2f})")
            else: infos.append(f"le trafic est fluide (niveau {trafic:.2f})")
        else: infos.append(f"niveau de trafic {trafic}")
    if pluie is not None:
        if isinstance(pluie, bool): infos.append("il pleut" if pluie else "il ne pleut pas")
        elif isinstance(pluie, (int, float)):
            if pluie > 0.5: infos.append(f"il pleut (intensite {pluie:.2f})")
            else: infos.append(f"il ne pleut pas (intensite {pluie:.2f})")
        else: infos.append(f"meteo {pluie}")
    return " ; ".join(infos) if infos else "conditions normales"

def _est_question_conditions(message):
    m = (message or "").lower()
    if re.search(r"(quel temps|le temps qu|il fait quel temps|niveau de (?:la )?pluie)", m):
        return True
    return any(w in m for w in ("embouteillage", "trafic", "circulation", "bouchon",
        "pressé", "presse", "pressée", "pluie", "pleut", "météo", "meteo"))

def analyser_intention(message, historique=None, session_id=None):
    # Comprend ce que veut faire l utilisateur
    s = SESSIONS[session_id] if session_id else {}
    _EXCLURE.set(None)
    attente = bool(s.get("attente_mode"))
    mode_msg = extraire_mode(message, accepter_tous=attente)
    budget_msg = extraire_budget(message)
    if re.search(r"(comment (tu|vous) (t|vous )?.?appel|qui (es|etes).?(tu|vous)|"
                 r"tu es qui|ton nom|c.?est quoi ton nom|tu fais quoi|presente.?toi)",
                 nettoyer_texte(message)):
        return _completer_analyse({"intention": "CONVERSATION"}, s, mode_msg, budget_msg)
    if s.get("dernieres_options") and re.search(
            r"(pas convaincu|ne me convient pas|autre trajet|autre option|change|changer|"
            r"propose autre|quelque chose d.?autre|non,? merci|je n.?aime pas|"
            r"je (ne )?veux (pas|plus)|pas (ce|cet|celui|ces|ceux)|"
            r"enleve|retire|sans ce trajet)", nettoyer_texte(message)):
        _EXCLURE.set(list(s.get("dernieres_options") or []))
        def sig(option):
            return tuple((e.get("mode"), e.get("numero_ligne") or e.get("nom_ligne"),
                e.get("depart"), e.get("arrivee"))
                for e in option.get("etapes", []) if e.get("mode") != "MARCHE")
        return {"intention": "TRAJET", "depart": s.get("depart"),
            "destination": s.get("destination"),
            "mode_transport": mode_msg or "TOUS", "mode_precise": bool(mode_msg),
            "priorite": None, "budget_fcfa": budget_msg or s.get("budget_fcfa"),
            "prix_max_ligne": None,
            "exclure_signatures": [sig(o) for o in s.get("dernieres_options", [])]}
    if attente and mode_msg and s.get("destination"):
        return {"intention": "TRAJET", "depart": s.get("depart"),
            "destination": s.get("destination"), "mode_transport": mode_msg,
            "mode_precise": True, "priorite": None,
            "budget_fcfa": budget_msg or s.get("budget_fcfa"), "prix_max_ligne": None}
    if mode_msg and s.get("destination") and not re.search(
            r"(c.?est quoi|qu.?est.?ce|\bpourquoi\b|comment fonctionne|\bcombien\b|"
            r"est.?ce que|\bou est\b|\bquel(le)?s? (est|sont)\b)", nettoyer_texte(message)):
        return {"intention": "TRAJET", "depart": s.get("depart"),
            "destination": s.get("destination"), "mode_transport": mode_msg,
            "mode_precise": True, "priorite": None,
            "budget_fcfa": budget_msg or s.get("budget_fcfa"), "prix_max_ligne": None}
    if budget_msg and s.get("destination") and (s.get("depart") or s.get("dernieres_options")) \
            and not re.search(r"\bcombien\b|c.?est quoi|\bpourquoi\b|\bcomment fonctionne\b",
                              nettoyer_texte(message)):
        mode_session = s.get("mode_transport")
        return {"intention": "TRAJET", "depart": s.get("depart"),
            "destination": s.get("destination"),
            "mode_transport": mode_msg or mode_session or "TOUS",
            "mode_precise": bool(mode_msg or mode_session),
            "priorite": None, "budget_fcfa": budget_msg, "prix_max_ligne": None}
    if not client:
        msg = message.lower()
        if any(m in msg for m in ("bonjour", "salut", "merci", "hello", "bonsoir")):
            return _completer_analyse({"intention": "CONVERSATION"}, s, mode_msg, budget_msg)
        return _completer_analyse({"intention": "INCONNUE"}, s, mode_msg, budget_msg)
    historique = historique or []
    histo_txt = "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}"
                          for m in historique[-10:])
    memoire_txt = ""
    if s.get("depart") or s.get("destination"):
        memoire_txt = f"""
Memoire de la session :
Dernier depart {s.get('depart') or 'inconnu'}
Derniere destination {s.get('destination') or 'inconnue'}
Budget {s.get('budget_fcfa') or 'aucun'}
Mode {s.get('mode_transport') or 'non precise'}

REGLES DE SUIVI :
Si le message CHANGE la recherche (autre mode, autre depart, autre budget, autre trajet),
mets intention TRAJET en reutilisant depart et destination ci dessus s ils ne sont pas redonnes.
Si le message pose une QUESTION sur les trajets deja proposes, mets intention SUIVI.
Tout le reste (salutation, merci, blague, culture generale, question sans rapport),
mets intention CONVERSATION. On ne renvoie JAMAIS d itineraire dans ce cas.
"""
    prompt = f"""Tu es un assistant de transport a Abidjan.
Analyse le message et reponds UNIQUEMENT avec un JSON valide.

Message : "{message}"

Historique recent :
{histo_txt}
{memoire_txt}

Format JSON attendu :
{{
  "intention": "TRAJET" ou "SUIVI" ou "CONVERSATION" ou "LIGNES_TRAJET" ou "INCONNUE",
  "depart": "lieu de depart ou null",
  "destination": "lieu d arrivee ou null",
  "priorite": "duree" ou "prix" ou null,
  "prix_max_ligne": nombre ou null
}}

Regles :
Si l utilisateur dit "ma position", "ici", "je suis la", "pres de moi",
alors depart vaut "ma position actuelle".
intention CONVERSATION pour tout ce qui n est pas une demande de deplacement.
intention TRAJET uniquement si l utilisateur veut aller quelque part ou relance la recherche.
N invente jamais un lieu, mets null si le lieu n est pas dans le message ou la memoire.
Ne mets jamais de texte en dehors du JSON.
"""
    try:
        r = client.chat.completions.create(model=MODEL,
            messages=[{"role": "system",
                       "content": "Tu reponds uniquement en JSON valide, sans texte autour."},
                      {"role": "user", "content": prompt}],
            temperature=0.1, max_tokens=400)
        texte = r.choices[0].message.content.strip()
        debut, fin = texte.find("{"), texte.rfind("}") + 1
        if debut >= 0 and fin > debut:
            data = json.loads(texte[debut:fin])
            return _completer_analyse(data, s, mode_msg, budget_msg)
    except Exception as e:
        log.error("erreur analyse intention %s", e)
    return _completer_analyse({"intention": "INCONNUE"}, s, mode_msg, budget_msg)


SYSTEME_AMI = (
    "Tu es MobiGuide, un assistant sympa base a Abidjan. Tu discutes naturellement, comme un ami. "
    "Tu reponds a toutes les questions (pas seulement les transports), en francais, de facon "
    "courte et chaleureuse. Tu ne proposes JAMAIS d itineraire si on ne t en demande pas. "
    "Tu ne repetes jamais une formule d accueil ni un conseil deja donne plus haut dans la "
    "conversation. Tu t appelles MobiGuide, l assistant de l application MobiGo. "
    "Si tu ne connais pas la reponse, dis le clairement et demande une precision. "
    "N invente jamais une information, une ligne, un prix ou une fonctionnalite."
)

SYSTEME_TRAJET = (
    "Tu es MobiGuide, assistant de transport a Abidjan, chaleureux et naturel, comme un ami "
    "qui connait bien la ville. Tu reponds toujours en francais. Tu ne repetes jamais une "
    "formule d accueil ni un conseil deja dit. Tu n inventes jamais de prix ni d arret. "
    "Tu ne renvoies jamais vers Google Maps ou Waze. "
    "Quand un mode est demande et qu aucun trajet n existe dans ce mode, dis le clairement "
    "et propose d autres modes uniquement si l utilisateur le demande."
)

def _conseils_liste(options, budget=None):
    # Conseils pratiques selon les modes reels des trajets
    modes = set()
    for c in (options or [])[:4]:
        for e in c.get("etapes", []):
            if e.get("mode") != "MARCHE": modes.add(e["mode"])
    out = []
    if "BUS" in modes:
        out.append(("BUS", "En bus, prends ton ticket ou badge ta carte aupres du machiniste."))
    if "GBAKA" in modes:
        out.append(("GBAKA", "Pour les gbakas, le prix n est pas fixe : demande le a l apprenti avant de monter."))
    if "WORO_WORO" in modes:
        out.append(("WORO_WORO", "Pour un woro-woro, demande le prix au chauffeur avant de monter."))
    if "BATEAU" in modes:
        out.append(("BATEAU", "Pour le bateau-bus, garde ton ticket jusqu a la descente."))
    if modes:
        out.append(("SIGNALER", "Signale ton arret de descente au chauffeur ou a l apprenti."))
    if budget and modes & {"GBAKA", "WORO_WORO"}:
        out.append((f"BUDGET{budget}",
            f"Prevois un peu de marge sur tes {budget} FCFA pour les gbakas et woro-woro."))
    return out

def _phrase_complete(texte, fini_par_limite):
    # Si le LLM a ete coupe par max_tokens, on garde seulement les phrases finies
    texte = (texte or "").strip()
    if not fini_par_limite or not texte: return texte
    fin = max(texte.rfind(c) for c in ".!?...")
    emoji = re.search(r"[.!?...][^\w\s]{0,3}\s*$", texte)
    if emoji: return texte
    return texte[:fin + 1].strip() if fin > 20 else texte

def _dist_txt(m):
    m = int(m or 0)
    return f"{m / 1000:.1f} km".replace(".", ",") if m >= 1000 else f"{m} m"

def _ordinal(n): return "1er" if n == 1 else f"{n}e"

def decrire_trajet(c):
    # Raconte le trajet etape par etape
    etapes = c.get("etapes", []) or []
    idx_transports = [i for i, e in enumerate(etapes)
                      if e.get("mode") not in ("MARCHE", "WORO_WORO")]
    dernier_t = idx_transports[-1] if idx_transports else -1
    lignes = []
    derniere_arrivee = None
    for i, e in enumerate(etapes):
        mode, d = e.get("mode"), e.get("distance_m", 0) or 0
        if mode == "MARCHE":
            if d < 30: continue
            minutes = e.get("duree_minutes")
            duree = f" (environ {minutes} min)" if minutes else ""
            avant_transport = any(x.get("mode") != "MARCHE" for x in etapes[i + 1:])
            if avant_transport:
                t = f"Marche {_dist_txt(d)} jusqu a l arret « {e.get('arrivee')} »{duree}."
                if d > SEUIL_MARCHE_LONGUE_M:
                    t += " C est assez loin : tu peux aussi prendre un taxi ou un gbaka."
            else:
                t = f"Il te restera {_dist_txt(d)} a pied jusqu a ta destination{duree}."
                if d > SEUIL_MARCHE_LONGUE_M:
                    t += " C est assez long : un taxi ou un woro-woro peut t y deposer."
            lignes.append(t)
        elif mode == "WORO_WORO":
            lignes.append(f"Pour rejoindre l arret « {e.get('arrivee')} » ({_dist_txt(d)}), "
                          f"prends un woro-woro ou un taxi.")
        else:
            if derniere_arrivee and derniere_arrivee != e.get("depart") and i > 0 \
                    and etapes[i - 1].get("mode") != "MARCHE":
                lignes.append(f"Rejoins l arret « {e.get('depart')} » (tout pres, a pied).")
            ligne = e.get("numero_ligne")
            nom = _NOM_MODE.get(mode, str(mode).lower()) + (f" {ligne}" if ligne else "")
            n = e.get("nb_arrets") or 0
            descente = (f"descends au {_ordinal(n)} arret, a « {e.get('arrivee')} »"
                        if n > 0 else f"descends a « {e.get('arrivee')} »")
            t = f"Monte dans {nom} a « {e.get('depart')} », puis {descente}"
            t += " pour prendre la correspondance." if i < dernier_t else "."
            lignes.append(t)
            derniere_arrivee = e.get("arrivee")
    return "\n".join(f"{k}. {t}" for k, t in enumerate(lignes, 1))

def _intro_trajet(message, contexte, candidats, mode, historique, premiere,
                  memoire_txt, infos_conditions):
    # Petite phrase d introduction generee par le LLM
    modele = "Voici ce que je te propose." if premiere else "Voici ce que j ai trouve."
    if not client: return modele
    consigne = ("C est la premiere fois que tu presentes des trajets : accueille "
                "l utilisateur chaleureusement." if premiere else
                "Tu as deja presente des trajets : fais une phrase courte SANS formule "
                "d accueil qui reconnait ce qu il vient de demander.")
    prompt = (f'L utilisateur vient de demander : "{message}".\n'
        f"Tu as trouve {len(candidats)} option(s). Mode demande : {mode}. "
        f"Budget : {contexte.get('budget_fcfa') or 'aucun'}.\n"
        f"Conditions : {infos_conditions or 'aucune'}\n{consigne}\n"
        "Ecris UNE seule phrase de 25 mots maximum. Ne decris pas les etapes, "
        "ne donne aucun conseil, n invente rien.")
    try:
        r = client.chat.completions.create(model=MODEL,
            messages=_messages_llm(SYSTEME_TRAJET + memoire_txt, prompt, historique),
            temperature=0.6, max_tokens=140)
        texte = (r.choices[0].message.content or "").strip().strip('"').strip()
        texte = texte.split("\n")[0].strip()
        texte = _phrase_complete(texte, r.choices[0].finish_reason == "length")
        return texte or modele
    except Exception as e:
        log.error("erreur intro trajet %s", e)
        return modele

def _messages_llm(systeme, prompt, historique):
    msgs = [{"role": "system", "content": systeme}]
    for m in (historique or [])[-10:]:
        role = "assistant" if m.get("role") in ("assistant", "model", "bot") else "user"
        c = str(m.get("content", "")).strip()
        if c: msgs.append({"role": role, "content": c})
    msgs.append({"role": "user", "content": prompt})
    return msgs

def _prix_texte(c):
    if not c: return "inconnu"
    p = c.get("cout_fcfa")
    if p is not None: return f"{p} FCFA"
    base = "variable (le tarif des gbakas depend du chauffeur)"
    partiel = c.get("cout_partiel_fcfa") or 0
    return f"{base}, partie bus : {partiel} FCFA" if partiel else base

def _points_cles(c):
    out = []
    for e in c.get("etapes", []):
        if e["mode"] == "MARCHE": continue
        ligne = e.get("numero_ligne") or e.get("nom_ligne") or e["mode"]
        n = e.get("nb_arrets") or 0
        rang = f" (au {'1er' if n == 1 else str(n) + 'e'} arret)" if n > 0 else ""
        out.append(f"{e['mode'].lower()} {ligne} : montee a {e['depart']}, "
                   f"descente a {e['arrivee']}{rang}")
    return " ; ".join(out)

def _reponse_trajet(message, contexte, candidats, mode, historique, s,
                    memoire_txt, infos_conditions):
    premiere = not s.get("nb_reponses_trajet")
    budget = contexte.get("budget_fcfa")
    if not candidats:
        if _EXCLURE.get():
            return ("Je n ai pas d autre trajet vraiment different entre ces deux lieux. "
                    "Tu veux essayer un autre mode (bus, gbaka, bateau-bus) ?")
        if mode and mode != "TOUS":
            lib = _LIBELLE_MODE.get(mode, mode.lower())
            return (f"Je n ai trouve aucun trajet en {lib} entre ces deux lieux. "
                    f"Tu veux que je cherche en bus, gbaka ou tous les modes ?")
        return ("Je n ai trouve aucun trajet. Verifie ton depart et ta destination, "
                "ou essaie un autre mode de transport.")
    combo = candidats[0].get("combo_mode")
    if combo:
        lib = _LIBELLE_MODE.get(combo, combo.lower())
        intro = (f"Je n ai pas de trajet 100 pour cent {lib} mais voici des trajets "
                 f"qui combinent le {lib} avec un autre mode.")
    else:
        intro = _intro_trajet(message, contexte, candidats, mode, historique,
                              premiere, memoire_txt, infos_conditions)
    if budget and extraire_budget(message) == budget:
        intro += f" Tous ces trajets respectent ton budget de {budget} FCFA."
    parts = [intro, "Option 1 :\n" + decrire_trajet(candidats[0])]
    autres = len(candidats) - 1
    if autres > 0:
        parts.append(f"J ai aussi {autres} autre{'s' if autres > 1 else ''} "
                     f"option{'s' if autres > 1 else ''} dans les cartes.")
    deja = set(s.get("conseils_donnes") or [])
    nouveaux = [(k, t) for k, t in _conseils_liste(candidats, budget) if k not in deja]
    if nouveaux:
        parts.append("Conseil : " + " ".join(t for _, t in nouveaux))
        s["conseils_donnes"] = list(deja | {k for k, _ in nouveaux})
    s["nb_reponses_trajet"] = (s.get("nb_reponses_trajet") or 0) + 1
    return "\n\n".join(parts)

def generer_reponse(message, contexte, historique=None, session_id=None):
    # Genere la reponse finale pour l utilisateur
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
            memoire_txt = (f"\nContexte : depart={s.get('depart')}, "
                f"destination={s.get('destination')}, "
                f"budget={s.get('budget_fcfa') or 'aucun'}, "
                f"mode={s.get('mode_transport') or 'non precise'}")
    if statut == "TRAJET":
        return _reponse_trajet(message, contexte, candidats, mode, historique, s,
                               memoire_txt, infos_conditions)
    if not client:
        if statut == "CONVERSATION":
            return ("Bonjour ! Je suis MobiGuide, ton guide a Abidjan. "
                    "Dis-moi ou tu veux aller et je te propose le meilleur trajet !")
        return ("Je suis MobiGuide. Dis-moi ton point de depart et ta destination, "
                "et je m occupe du reste.")
    if statut == "SUIVI":
        systeme = SYSTEME_AMI + memoire_txt
        resume = "\n".join(
            f"{c.get('libelle', 'proposition')} : {c.get('duree_minutes', '?')} min, "
            f"prix {_prix_texte(c)}, {c.get('nb_correspondances', 0)} correspondance(s), "
            f"{c.get('marche_m', 0)} m de marche ; {_points_cles(c)}"
            for c in candidats)
        prompt = f"""{message}

[Trajets deja proposes a l utilisateur :
{resume or 'aucun trajet en memoire'}
Conditions : {infos_conditions}]
Reponds directement a SA question, en 2 ou 3 phrases, sur un ton chaleureux.
Ne repete pas les conseils ni l accueil deja donnes plus haut. Ne recopie pas tout
l itineraire, n invente rien."""
    elif statut == "CONVERSATION":
        systeme = SYSTEME_AMI + memoire_txt
        if question_conditions:
            prompt = f"""{message}

[Conditions calculees : {infos_conditions}]
Utilise UNIQUEMENT ces infos pour trafic et meteo, ne renvoie jamais vers Google Maps
ou Waze, donne un conseil concret en 2 ou 3 phrases maximum."""
        else:
            prompt = f"""{message}

Reponds directement et naturellement a ce message, comme un ami chaleureux,
en 2 ou 3 phrases maximum. Si on te demande qui tu es : tu es MobiGuide,
l assistant de l application MobiGo. Pour une question hors transport, reponds y
vraiment. Termine, si c est naturel, par une courte invitation a t aider pour un trajet.
Ne repete pas ce que tu as deja dit."""
    else:
        systeme = SYSTEME_AMI + memoire_txt
        prompt = message
    try:
        r = client.chat.completions.create(model=MODEL,
            messages=_messages_llm(systeme, prompt, historique),
            temperature=0.5, max_tokens=600)
        return _phrase_complete(r.choices[0].message.content,
                                r.choices[0].finish_reason == "length")
    except Exception as e:
        log.error("erreur generation reponse %s", e)
        return "Je suis MobiGuide. Dis-moi ou tu veux aller et je te propose un trajet."