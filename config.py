import csv
import io
import json
import re
import threading
import time
import urllib.request
from collections import defaultdict, deque
from math import asin, cos, radians, sin, sqrt
import logging
import os

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("mobigo")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
APP_API_KEY = os.getenv("APP_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",")]

MODEL = "openai/gpt-oss-120b"

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY manquante dans le .env")
if not SERPAPI_KEY:
    raise RuntimeError("SERPAPI_KEY manquante dans le .env")

LAT_MIN, LAT_MAX = 5.15, 5.55
LON_MIN, LON_MAX = -4.25, -3.75
CENTRE_ABIDJAN = "@5.3364,-4.0267,12z"
TIMEZONE = "Africa/Abidjan"

# ajout du contexte mémorisé (JSON) au message 
MAX_MESSAGE = 1200
RATE_LIMIT = 20
CACHE_TTL = 60 * 60 * 24

RAYON_PROXIMITE_M = 600
RAYON_CORRESPONDANCE_M = 250
MAX_TRANSFERTS = 1

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

PRIX_LIGNES = {}
PRIX_TRAJETS = {}
PRIX_POUR_CLASSEMENT = {}

_cache = {}
_cache_lock = threading.Lock()


def cache_get(cle, ttl=None):
    duree = CACHE_TTL if ttl is None else ttl
    with _cache_lock:
        item = _cache.get(cle)
        if item and time.time() - item[0] < duree:
            return item[1]
        if item:
            _cache.pop(cle, None)
    return None


def cache_set(cle, valeur):
    with _cache_lock:
        if len(_cache) > 2000:
            _cache.clear()
        _cache[cle] = (time.time(), valeur)


_hits = defaultdict(deque)
_hits_lock = threading.Lock()


def trop_de_requetes(ip: str) -> bool:
    maintenant = time.time()
    with _hits_lock:
        file = _hits[ip]
        while file and maintenant - file[0] > 60:
            file.popleft()
        if len(file) >= RATE_LIMIT:
            return True
        file.append(maintenant)
    return False


def distance_m(lat1, lon1, lat2, lon2) -> int:
    r = 6371000
    p1, p2 = radians(lat1), radians(lat2)
    dp, dl = p2 - p1, radians(lon2 - lon1)
    a = sin(dp / 2) ** 2 + cos(p1) * cos(p2) * sin(dl / 2) ** 2
    return int(2 * r * asin(sqrt(a)))


def extraire_json(texte: str) -> dict:
    try:
        return json.loads(texte)
    except Exception:
        debut, fin = texte.find("{"), texte.rfind("}") + 1
        if debut == -1 or fin == 0:
            raise ValueError("aucun json dans la reponse")
        return json.loads(texte[debut:fin])


def fix_encoding(text):
    if not text or "Ã" not in text:
        return text
    try:
        return text.encode("latin-1").decode("utf-8")
    except Exception:
        return text


def http_json(url, data=None, headers=None, timeout=15, method=None):
    req = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))