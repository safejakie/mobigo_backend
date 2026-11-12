import argparse
import csv
import json
import random
import time
from datetime import datetime
from math import asin, cos, radians, sin, sqrt
from pathlib import Path

import ml_mobigo
import services

LIMITE_APPELS = 80
MAX_ECHECS_CONSECUTIFS = 3

_compteur_appels = 0
_echecs_consecutifs = 0

# Cache des durées Google : une paire déjà interrogée ne coûte plus jamais de crédit SerpAPI
CACHE_GOOGLE = Path("google_cache.json")
_cache_google = {}
if CACHE_GOOGLE.exists():
    try:
        _cache_google = json.loads(CACHE_GOOGLE.read_text(encoding="utf-8"))
    except Exception:
        _cache_google = {}


def _sauver_cache():
    """Écriture atomique du cache (évite la corruption en cas d'arrêt brutal)."""
    tmp = CACHE_GOOGLE.with_suffix(".tmp")
    tmp.write_text(json.dumps(_cache_google, ensure_ascii=False), encoding="utf-8")
    tmp.replace(CACHE_GOOGLE)


def google_avec_cache(dep, dest):
    """Retourne (durée_min, appel_reel). N'appelle SerpAPI que si la paire n'est pas en cache."""
    cle = (f"{dep['latitude']:.5f},{dep['longitude']:.5f}>"
           f"{dest['latitude']:.5f},{dest['longitude']:.5f}")
    if cle in _cache_google:
        return _cache_google[cle], False

    try:
        val = ml_mobigo.get_google_duration(dep, dest, usage="prod")
    except Exception as e:
        raise RuntimeError(f"SerpAPI: {type(e).__name__}: {e}") from e

    if val is not None:
        _cache_google[cle] = val
        _sauver_cache()
    return val, True


def charger_geojson(chemin: str) -> list:
    with open(chemin, "r", encoding="utf-8") as f:
        data = json.load(f)
    points = []
    for feature in data.get("features", []):
        geom = feature.get("geometry", {})
        if geom.get("type") != "LineString":
            continue
        for lon, lat in geom.get("coordinates", []):
            points.append((float(lat), float(lon)))
    return points


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return 2 * R * asin(sqrt(a))


def echantillonner_points(points, n, dist_min_km=1.0):
    random.shuffle(points)
    sel = []
    for lat, lon in points:
        if len(sel) >= n:
            break
        if all(haversine_km(lat, lon, a, b) >= dist_min_km for a, b in sel):
            sel.append((lat, lon))
    return sel


def generer_paires(points, n_pairs, dist_min_km=2.0, dist_max_km=15.0):
    paires, essais = [], 0
    while len(paires) < n_pairs and essais < n_pairs * 30:
        essais += 1
        a, b = random.sample(points, 2)
        d = haversine_km(a[0], a[1], b[0], b[1])
        if dist_min_km <= d <= dist_max_km:
            paires.append((a, b))
    return paires


def generer_contextes(dist_km):
    """4 contextes variés : matin pointe, midi, soir pointe, nuit."""
    return [
        {  # matin pointe
            "heure": random.choice([7, 8, 9]),
            "jour_semaine": random.randint(0, 4),
            "mois": datetime.now().month,
            "pluie": random.random() < 0.4,
            "distance_vol_oiseau_km": dist_km,
        },
        {  # midi
            "heure": random.choice([12, 13, 14]),
            "jour_semaine": random.randint(0, 6),
            "mois": datetime.now().month,
            "pluie": random.random() < 0.2,
            "distance_vol_oiseau_km": dist_km,
        },
        {  # soir pointe
            "heure": random.choice([17, 18, 19]),
            "jour_semaine": random.randint(0, 4),
            "mois": datetime.now().month,
            "pluie": random.random() < 0.4,
            "distance_vol_oiseau_km": dist_km,
        },
        {  # nuit
            "heure": random.choice([21, 22, 23]),
            "jour_semaine": random.randint(0, 6),
            "mois": datetime.now().month,
            "pluie": random.random() < 0.15,
            "distance_vol_oiseau_km": dist_km,
        },
    ]


def ajuster_google_pour_contexte(google_min, ctx):
    """Ajustement documenté : pluie ×1.25, pointe ×1.30."""
    g = google_min
    if ctx["pluie"]:
        g *= 1.25
    if ctx["heure"] in (7, 8, 9, 17, 18, 19):
        g *= 1.30
    return g


def traiter_paire(o, d, idx, total, writer):
    global _compteur_appels, _echecs_consecutifs

    if _compteur_appels >= LIMITE_APPELS:
        print(f"\n Limite de {LIMITE_APPELS} appels atteinte. Arrêt.")
        return -1

    o_lat, o_lon = o
    d_lat, d_lon = d
    dist_vol = haversine_km(o_lat, o_lon, d_lat, d_lon)

    print(f"\n[{idx:2d}/{total}] ({o_lat:.4f},{o_lon:.4f}) → "
          f"({d_lat:.4f},{d_lon:.4f}) | {dist_vol:.1f} km")

    dep = {"nom": "origine", "latitude": o_lat, "longitude": o_lon}
    dest = {"nom": "destination", "latitude": d_lat, "longitude": d_lon}

    try:
        candidats = services.construire_candidats(dep, dest)
    except Exception as e:
        print(f" Erreur construire_candidats : {type(e).__name__}: {e}")
        return 0

    if not candidats:
        print(" Aucun itinéraire (pas d'appel SerpAPI)")
        return 0

    print(f"  ✅ {len(candidats)} candidats")

    # 1 SEUL appel SerpAPI par paire
    try:
        google_min, appel_reel = google_avec_cache(dep, dest)
    except RuntimeError as e:
        print(f" {e}")
        _echecs_consecutifs += 1
        if _echecs_consecutifs >= MAX_ECHECS_CONSECUTIFS:
            print(f" {_echecs_consecutifs} échecs consécutifs. "
                  f"Vérifiez la config SerpAPI dans ml_mobigo.py. Arrêt.")
            return -1
        return 0

    if google_min is None:
        _echecs_consecutifs += 1
        print(f" SerpAPI indisponible ({_echecs_consecutifs}/{MAX_ECHECS_CONSECUTIFS})")
        if _echecs_consecutifs >= MAX_ECHECS_CONSECUTIFS:
            print(f"   Arrêt après {_echecs_consecutifs} échecs consécutifs.")
            return -1
        return 0

    _echecs_consecutifs = 0
    if appel_reel:
        _compteur_appels += 1

    print(f"   Google Maps : {google_min:.1f} min "
          f"({_compteur_appels}/{LIMITE_APPELS} appels)")

    lignes = 0
    for ctx in generer_contextes(dist_vol):
        trafic = ml_mobigo.estimer_trafic(heure=ctx["heure"], pluie=ctx["pluie"])
        ctx["trafic_facteur"] = trafic["facteur"]
        google_adj = ajuster_google_pour_contexte(google_min, ctx)

        for c in candidats:
            row = ml_mobigo.extraire_features(c, ctx)
            row[ml_mobigo.CIBLE] = round(
                ml_mobigo.calculer_cible(c, google_adj, dist_vol), 2
            )
            writer.writerow(row)
            lignes += 1

    print(f"   {lignes} lignes ({len(candidats)} cand. × 4 ctx)")
    return lignes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--geojson", default="abidjan_rues.geojson")
    parser.add_argument("--n_points", type=int, default=60)
    parser.add_argument("--n_pairs", type=int, default=80,
                        help="Max 80 (limite SerpAPI)")
    parser.add_argument("--dist_min_km", type=float, default=2.0)
    parser.add_argument("--dist_max_km", type=float, default=15.0)
    parser.add_argument("--output", default="dataset.csv")
    parser.add_argument("--pause", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.n_pairs > LIMITE_APPELS:
        print(f"  {args.n_pairs} > {LIMITE_APPELS}. On réduit à {LIMITE_APPELS}.")
        args.n_pairs = LIMITE_APPELS

    random.seed(args.seed)

    print(" Chargement GeoJSON...")
    points = charger_geojson(args.geojson)
    print(f"   {len(points)} points extraits")

    print(f"\n Échantillonnage : {args.n_points} points (min 1 km)")
    pool = echantillonner_points(points, args.n_points, 1.0)
    print(f"   {len(pool)} points retenus")

    print(f"\n Génération : {args.n_pairs} paires "
          f"({args.dist_min_km}-{args.dist_max_km} km)")
    paires = generer_paires(pool, args.n_pairs, args.dist_min_km, args.dist_max_km)
    print(f"   {len(paires)} paires valides")

    chemin = Path(args.output)
    if chemin.exists():
        chemin.unlink()
        print(f"    Ancien {chemin} supprimé")

    total_lignes = 0
    t0 = time.time()
    interruption = False

    with open(chemin, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ml_mobigo.FEATURES + [ml_mobigo.CIBLE])
        writer.writeheader()

        try:
            for i, (o, d) in enumerate(paires, 1):
                r = traiter_paire(o, d, i, len(paires), writer)
                if r == -1:
                    break
                total_lignes += r
                time.sleep(args.pause)
        except KeyboardInterrupt:
            interruption = True
            print("\n\n  Interruption manuelle (Ctrl+C). Sauvegarde en cours...")

    duree = time.time() - t0
    print(f"\n{'='*55}")
    if interruption:
        print(f"  Interrompu après {duree:.1f}s")
    else:
        print(f" Terminé en {duree:.1f}s")
    print(f"   Lignes dans {chemin} : {total_lignes}")
    print(f"   Appels SerpAPI consommés : {_compteur_appels}")
    print(f"   Durées Google en cache : {len(_cache_google)}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()