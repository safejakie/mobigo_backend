import csv
import os
from datetime import datetime
from math import radians, sin, cos, asin, sqrt

import numpy as np
import pandas as pd
import requests
import joblib
import serpapi

from config import SERPAPI_KEY, OPENWEATHER_API_KEY, log


# CONSTANTES
OPENWEATHER_KEY = OPENWEATHER_API_KEY

MODES = ["BUS", "GBAKA", "BATEAU", "WORO_WORO"]

FEATURES_BASE = [
    "duree_theorique", "cout_fcfa", "marche_m", "nb_correspondances",
    "distance_transport_m", "duree_transport_min", "nb_arrets",
    "nb_bus", "nb_gbaka", "nb_bateau", "nb_marche", "part_transport",
    "heure", "jour_semaine", "mois", "pluie",
    "trafic_facteur", "distance_vol_oiseau_km",
]
FEATURES_DERIVEES = ["heure_sin", "heure_cos", "est_pointe", "est_weekend"]
FEATURES = FEATURES_BASE + FEATURES_DERIVEES
CIBLE = "duree_reelle_estimee"

CHEMIN_DATASET = "dataset.csv"
CHEMIN_MODELE = os.path.join("models", "duration_rf.pkl")


# CONTEXTE TEMPS RÉEL

def get_meteo(lat: float, lon: float) -> dict:
    if not OPENWEATHER_KEY:
        return {"pluie": False, "condition": "unknown"}
    try:
        r = requests.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={"lat": lat, "lon": lon, "appid": OPENWEATHER_KEY,
                    "units": "metric", "lang": "fr"},
            timeout=5,
        )
        d = r.json()
        cond = d["weather"][0]["main"]
        return {"pluie": cond in ("Rain", "Drizzle", "Thunderstorm"),
                "condition": cond}
    except Exception as e:
        log.warning("openweather: %s", e)
        return {"pluie": False, "condition": "unknown"}


def estimer_trafic(heure=None, pluie=False, user_signale=False) -> dict:
    h = heure if heure is not None else datetime.now().hour
    if user_signale:
        return {"niveau": "eleve", "facteur": 1.5}
    if 7 <= h <= 9 or 17 <= h <= 19:
        base = {"niveau": "eleve", "facteur": 1.4}
    elif 12 <= h <= 14:
        base = {"niveau": "moyen", "facteur": 1.15}
    else:
        base = {"niveau": "faible", "facteur": 1.0}
    if pluie:
        base["facteur"] *= 1.3
    return base


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    dlat, dlon = radians(lat2 - lat1), radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return 2 * R * asin(sqrt(a))


def construire_contexte(origine: dict, destination: dict) -> dict:
    now = datetime.now()
    meteo = get_meteo(origine["latitude"], origine["longitude"])
    trafic = estimer_trafic(pluie=meteo.get("pluie", False))
    return {
        "heure": now.hour,
        "jour_semaine": now.weekday(),
        "mois": now.month,
        "pluie": meteo["pluie"],
        "trafic_facteur": trafic["facteur"],
        "trafic_niveau": trafic["niveau"],
        "distance_vol_oiseau_km": haversine_km(
            origine["latitude"], origine["longitude"],
            destination["latitude"], destination["longitude"],
        ),
    }

# FEATURES

def extraire_features(candidat: dict, contexte: dict) -> dict:
    modes_count = {m: 0 for m in MODES}
    distance_transport = distance_marche = duree_transport = nb_arrets = 0

    for e in candidat.get("etapes", []):
        mode = e.get("mode", "MARCHE")
        if mode == "MARCHE":
            distance_marche += e.get("distance_m", 0)
        else:
            modes_count[mode] = modes_count.get(mode, 0) + 1
            distance_transport += e.get("distance_m", 0)
            duree_transport += e.get("duree_minutes", 0)
            if "arrets" in e:
                nb_arrets += len(e["arrets"])

    total = distance_transport + distance_marche
    heure = contexte.get("heure", 12)
    jour = contexte.get("jour_semaine", 0)

    return {
        "duree_theorique": candidat.get("duree_minutes", 0),
        "cout_fcfa": (candidat.get("cout_fcfa") if candidat.get("cout_fcfa") is not None
                      else candidat.get("cout_partiel_fcfa", 0)),
        "marche_m": candidat.get("marche_m", 0),
        "nb_correspondances": candidat.get("nb_correspondances", 0),
        "distance_transport_m": distance_transport,
        "duree_transport_min": duree_transport,
        "nb_arrets": nb_arrets,
        "nb_bus": modes_count.get("BUS", 0),
        "nb_gbaka": modes_count.get("GBAKA", 0),
        "nb_bateau": modes_count.get("BATEAU", 0),
        "nb_marche": sum(1 for e in candidat.get("etapes", [])
                         if e.get("mode") == "MARCHE"),
        "part_transport": (distance_transport / total) if total else 0,
        "heure": heure,
        "jour_semaine": jour,
        "mois": contexte.get("mois", 1),
        "pluie": int(contexte.get("pluie", False)),
        "trafic_facteur": contexte.get("trafic_facteur", 1.0),
        "distance_vol_oiseau_km": contexte.get("distance_vol_oiseau_km", 0),
        "heure_sin": np.sin(2 * np.pi * heure / 24),
        "heure_cos": np.cos(2 * np.pi * heure / 24),
        "est_pointe": int(heure in (7, 8, 9, 17, 18, 19)),
        "est_weekend": int(jour >= 5),
    }


def calculer_cible(candidat: dict, google_min: float,
                   distance_vol_km: float) -> float:
    free_flow = max(distance_vol_km / 30 * 60, 1.0)
    congestion = max(google_min / free_flow, 0.5)

    modes = [e["mode"] for e in candidat.get("etapes", [])
             if e.get("mode") != "MARCHE"]
    if "BATEAU" in modes:
        sens = 0.15
    elif "GBAKA" in modes:
        sens = 0.85
    elif "WORO_WORO" in modes:
        sens = 0.90
    else:
        sens = 1.0

    facteur = 1 + (congestion - 1) * sens
    return candidat.get("duree_minutes", 0) * max(facteur, 0.5)


# SERPAPI

def _extraire_duree_min(directions):
    if not directions:
        return None
    if isinstance(directions, dict):
        routes = directions.get("routes") or []
        if not routes:
            return None
        dur = routes[0].get("duration") or routes[0].get("duration_in_traffic")
    else:
        dur = (directions[0].get("duration")
               or directions[0].get("duration_in_traffic"))

    if dur is None:
        return None
    if isinstance(dur, str):
        nums = "".join(c for c in dur if c.isdigit())
        return int(nums) if nums else None
    return float(dur) / 60.0


def get_google_duration(origine: dict, destination: dict,
                        travel_mode="driving", usage="prod"):
    if usage == "dataset":
        api_key = os.getenv("SERPAPI_KEY_DATASET", SERPAPI_KEY)
    else:
        api_key = SERPAPI_KEY

    client = serpapi.Client(api_key=api_key)

    params = {
        "engine": "google_maps_directions",
        "start_coords": f"{origine['latitude']},{origine['longitude']}",
        "end_coords": f"{destination['latitude']},{destination['longitude']}",
        "travel_mode": "0",
        "hl": "fr",
        "gl": "ci",
    }

    try:
        res = client.search(params)
    except Exception as e:
        log.warning("serpapi directions: %s", e)
        return None

    if res.get("error"):
        log.warning("serpapi directions: %s", res["error"])
        return None

    return _extraire_duree_min(res.get("directions"))


# COLLECTE

def collecter(origine, destination, candidats, contexte, google_min=None):
    if not candidats:
        return

    if google_min is None:
        google_min = get_google_duration(origine, destination)
    if google_min is None:
        log.info("collecte: google indisponible, on saute")
        return

    nouveau = not os.path.exists(CHEMIN_DATASET)
    with open(CHEMIN_DATASET, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FEATURES + [CIBLE])
        if nouveau:
            writer.writeheader()
        for c in candidats:
            row = extraire_features(c, contexte)
            row[CIBLE] = round(calculer_cible(
                c, google_min, contexte["distance_vol_oiseau_km"]
            ), 2)
            writer.writerow(row)
    log.info("collecte: %d candidats (google=%.1f min)",
             len(candidats), google_min)


# PRÉDICTEUR

_model_package = None

def charger():
    global _model_package
    if os.path.exists(CHEMIN_MODELE):
        _model_package = joblib.load(CHEMIN_MODELE)
        mae = _model_package.get("mae_test", "?")
        print(f" Modèle chargé ({CHEMIN_MODELE}) — MAE test = {mae} min")
    else:
        print(" Pas de modèle. L'agent utilise la durée théorique.")

def predire_duree(candidat: dict, contexte: dict):
    if _model_package is None:
        return None
    try:
        feats = extraire_features(candidat, contexte)
        X = pd.DataFrame([feats])[_model_package["features"]]
        return float(_model_package["model"].predict(X)[0])
    except Exception as e:
        log.error("prédiction: %s", e)
        return None

def ajuster_candidats(candidats: list, contexte: dict) -> list:
    for c in candidats:
        p = predire_duree(c, contexte)
        if p is not None:
            c.setdefault("duree_theorique", c["duree_minutes"])
            c["duree_minutes"] = round(p)
    return candidats