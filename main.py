import json
import re
import threading
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import ALLOWED_ORIGINS, MAX_MESSAGE, log, trop_de_requetes
from services import (
    analyser_intention,
    charger_reseau_local,
    construire_candidats,
    generer_reponse,
    detail_ligne,
    liste_lignes,
    lignes_pour_app,
    recherche_globale,
    rechercher_lieu,
    suggestions_lieu,
    instructions_trajet,
    trajets_pour_demande,
    choisir_propositions,
    candidats_multimodaux,
    arrets_proches_trajet,
    conseil_marche_chat,
    _est_question_conditions,
    arrets_proximite,
    lignes_par_arret,
    trajets_par_mode,
    arrets_proximite_par_modes,
    maj_session,
    SESSIONS,
    RESEAU,
)

from ml_mobigo import (
    charger as charger_modele,
    construire_contexte,
    ajuster_candidats,
)
charger_modele()

app = FastAPI(title="MobiGo API", version="2.4.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

threading.Thread(target=charger_reseau_local, daemon=True).start()


def _contexte_ml_sur(dep_lieu: Optional[dict], dest_lieu: Optional[dict]) -> dict:
    if not dep_lieu or not dest_lieu:
        return {}
    try:
        c = construire_contexte(dep_lieu, dest_lieu)
        return c if isinstance(c, dict) else {}
    except Exception as e:
        log.warning("construire_contexte a échoué : %s", e)
        return {}


class Demande(BaseModel):
    message: str = Field(..., min_length=2, max_length=MAX_MESSAGE)
    historique: list = Field(default=[])
    depart_lat: Optional[float] = None
    depart_lon: Optional[float] = None
    dest_lat: Optional[float] = None
    dest_lon: Optional[float] = None
    session_id: Optional[str] = None
    contexte: Optional[dict] = None


class Lieu(BaseModel):
    nom: str
    latitude: float
    longitude: float
    alternatives: list = []


class ChatResponse(BaseModel):
    demande: str
    intention: str
    reponse: str
    depart: Optional[Lieu] = None
    destination: Optional[Lieu] = None
    trajets: list = []
    instructions: list = []
    trajet: Optional[dict] = None
    arrets: list = []
    arrets_depart: list = []
    arrets_destination: list = []
    choix_economiques: list = []
    autres_options: list = []
    lignes_possibles: list = []
    lignes_ici: list = []
    budget_fcfa: Optional[int] = None
    options_budget: list = []
    debug: Optional[dict] = None
    contexte: dict = {}


def est_position_actuelle(libelle: str) -> bool:
    texte = (libelle or "").strip().lower()
    expressions = (
        "position actuelle", "ma position", "ma localisation",
        "ma position gps", "ma localisation actuelle",
        "près de moi", "pres de moi", "autour de moi",
    )
    return any(e in texte for e in expressions) or bool(re.search(r"\bici\b", texte))


def coordonnees_gps_obligatoires(lat, lon, libelle: str):
    if (lat is None) != (lon is None):
        raise HTTPException(
            status_code=422,
            detail="Les deux coordonnées GPS (latitude et longitude) sont obligatoires.",
        )
    if est_position_actuelle(libelle) and (lat is None or lon is None):
        raise HTTPException(
            status_code=422,
            detail="Position actuelle indisponible : active la localisation du téléphone "
                   "et transmets depart_lat et depart_lon.",
        )
    return lat is not None and lon is not None


def verifier_acces(request: Request):
    ip = request.client.host if request.client else "inconnu"
    if trop_de_requetes(ip):
        raise HTTPException(status_code=429, detail="trop de requetes. reessaie dans une minute.")


@app.get("/")
def accueil():
    return {"message": "MobiGo API"}


@app.get("/lignes", dependencies=[Depends(verifier_acces)])
def lignes(type: Optional[str] = None):
    return {"type": type, "lignes": liste_lignes(type)}


@app.get("/api/lignes", dependencies=[Depends(verifier_acces)])
def api_lignes(type: Optional[str] = None, reseau: Optional[str] = None,
               q: Optional[str] = None, zone: Optional[str] = None):
    return lignes_pour_app(type, q, zone, reseau)


@app.get("/api/lignes/{ident:path}", dependencies=[Depends(verifier_acces)])
def api_ligne(ident: str):
    fiche = detail_ligne(ident)
    if not fiche:
        raise HTTPException(status_code=404, detail="ligne introuvable.")
    return fiche


@app.get("/ligne", dependencies=[Depends(verifier_acces)])
def ligne(id: str):
    return api_ligne(id)


@app.get("/api/recherche", dependencies=[Depends(verifier_acces)])
def api_recherche(q: str, type: Optional[str] = None):
    return recherche_globale(q, type)


@app.get("/api/suggestions", dependencies=[Depends(verifier_acces)])
def api_suggestions(q: str = ""):
    return suggestions_lieu(q)


@app.get("/api/geocoder", dependencies=[Depends(verifier_acces)])
def api_geocoder(q: str):
    lieu = rechercher_lieu(q)
    if not lieu:
        raise HTTPException(status_code=404, detail="lieu introuvable")
    return lieu


class RequeteTrajetFlutter(BaseModel):
    depart_lat: float
    depart_lon: float
    dest_lat: float
    dest_lon: float


@app.post("/api/itineraire", dependencies=[Depends(verifier_acces)])
def api_itineraire(req: RequeteTrajetFlutter):
    dep = {"nom": "point de depart", "latitude": req.depart_lat, "longitude": req.depart_lon}
    dest = {"nom": "destination", "latitude": req.dest_lat, "longitude": req.dest_lon}
    try:
        candidats = construire_candidats(dep, dest)
        if not candidats:
            raise HTTPException(status_code=404, detail="aucun itineraire trouve")

        contexte = construire_contexte(dep, dest) or {}
        candidats = ajuster_candidats(candidats, contexte)

        meilleur_candidat = candidats[0]
        return {**meilleur_candidat, "trajet": meilleur_candidat,
                "instructions": instructions_trajet(meilleur_candidat)}
    except HTTPException:
        raise
    except Exception as e:
        log.error("erreur itineraire : %s", e)
        raise HTTPException(status_code=500, detail="erreur interne du serveur")


@app.get("/api/arrets-proximite", dependencies=[Depends(verifier_acces)])
def api_arrets_proximite(lat: float, lon: float, rayon: int = 1000):
    return {"arrets": arrets_proximite(lat, lon, rayon)}


@app.get("/api/arret/{stop_id}/lignes", dependencies=[Depends(verifier_acces)])
def api_lignes_par_arret(stop_id: str):
    data = lignes_par_arret(stop_id)
    if not data:
        raise HTTPException(status_code=404, detail="arret introuvable")
    return data


@app.get("/api/trajet-multimodal", dependencies=[Depends(verifier_acces)])
def api_trajet_multimodal(
    depart: str, destination: str,
    depart_lat: Optional[float] = None, depart_lon: Optional[float] = None,
    dest_lat: Optional[float] = None, dest_lon: Optional[float] = None,
):
    if coordonnees_gps_obligatoires(depart_lat, depart_lon, depart):
        dep = {"nom": depart, "latitude": depart_lat, "longitude": depart_lon}
    else:
        dep = rechercher_lieu(depart)
        if not dep:
            raise HTTPException(status_code=404, detail=f"Lieu de départ introuvable : {depart}")

    if (dest_lat is None) != (dest_lon is None):
        raise HTTPException(status_code=422,
                            detail="Les deux coordonnées GPS de la destination sont obligatoires.")
    if dest_lat is not None and dest_lon is not None:
        dest = {"nom": destination, "latitude": dest_lat, "longitude": dest_lon}
    else:
        dest = rechercher_lieu(destination)
        if not dest:
            raise HTTPException(status_code=404, detail=f"Destination introuvable : {destination}")

    resultats = trajets_par_mode(dep, dest)
    contexte = construire_contexte(dep, dest) or {}
    for cle in ("bus", "gbakas", "bateaux"):
        if resultats.get(cle):
            resultats[cle] = ajuster_candidats(resultats[cle], contexte)
    return resultats


@app.get("/api/debug-trajet", dependencies=[Depends(verifier_acces)])
def api_debug_trajet(depart_lat: float, depart_lon: float, dest_lat: float, dest_lon: float):
    arrets_dep = arrets_proximite_par_modes(depart_lat, depart_lon, 3000)
    arrets_arr = arrets_proximite_par_modes(dest_lat, dest_lon, 3000)
    return {
        "arrets_depart_3km": arrets_dep, "arrets_arrivee_3km": arrets_arr,
        "nb_arrets_dep": len(arrets_dep), "nb_arrets_arr": len(arrets_arr),
    }


@app.get("/api/debug-lignes", dependencies=[Depends(verifier_acces)])
def api_debug_lignes(depart_lat: float, depart_lon: float, dest_lat: float, dest_lon: float):
    if not RESEAU["charge"]:
        charger_reseau_local()
    arrets_dep = arrets_proximite(depart_lat, depart_lon, 1500)[:20]
    arrets_arr = arrets_proximite(dest_lat, dest_lon, 1500)[:20]
    lignes_dep = {}
    for a in arrets_dep:
        for rid in RESEAU["lignes_par_arret"].get(a["id"], []):
            ligne = RESEAU["lignes"].get(rid)
            if ligne:
                num = ligne.get("numero") or ligne.get("nom") or rid
                lignes_dep[str(num)] = ligne.get("type")
    lignes_arr = {}
    for a in arrets_arr:
        for rid in RESEAU["lignes_par_arret"].get(a["id"], []):
            ligne = RESEAU["lignes"].get(rid)
            if ligne:
                num = ligne.get("numero") or ligne.get("nom") or rid
                lignes_arr[str(num)] = ligne.get("type")
    communes = sorted(set(lignes_dep) & set(lignes_arr))
    return {
        "nb_sequences": len(RESEAU["sequences"]),
        "nb_arrets": len(RESEAU["arrets"]),
        "lignes_pres_depart": lignes_dep, "lignes_pres_arrivee": lignes_arr,
        "lignes_en_commun": communes, "nb_commun": len(communes),
    }


MARQUEUR_CTX = "Contexte mémorisé à respecter :"
MODES_FLUTTER = {"BUS": "BUS", "GBAKA": "GBAKA", "WORO_WORO": "GBAKA",
                 "BATEAU": "BATEAU", "BATEAU_BUS": "BATEAU"}


def _separer_contexte(message: str, session_id: str, contexte_champ) -> str:
    ctx = {}
    if MARQUEUR_CTX in message:
        message, _, brut = message.partition(MARQUEUR_CTX)
        try:
            ctx = json.loads(brut.strip())
        except Exception:
            ctx = {}
    if isinstance(contexte_champ, dict):
        ctx = {**contexte_champ, **ctx}
    s = SESSIONS[session_id]
    for cle in ("depart", "destination"):
        if ctx.get(cle) and not s.get(cle):
            s[cle] = str(ctx[cle])
    budget = ctx.get("budget_max") or ctx.get("budget")
    if isinstance(budget, (int, float)) and not s.get("budget_fcfa"):
        s["budget_fcfa"] = int(budget)
    mode = MODES_FLUTTER.get(str(ctx.get("mode_transport") or "").upper())
    if mode and not s.get("mode_transport"):
        s["mode_transport"] = mode
    return message.strip()


def _contexte_reponse(session_id: str) -> dict:
    s = SESSIONS[session_id]
    out = {}
    if s.get("depart"):
        out["depart"] = s["depart"]
    if s.get("destination"):
        out["destination"] = s["destination"]
    if s.get("budget_fcfa"):
        out["budget_max"] = s["budget_fcfa"]
    if s.get("mode_transport") and s["mode_transport"] != "TOUS":
        out["mode_transport"] = s["mode_transport"]
    return out


def construire_contexte_session(session_id: str, demande) -> dict:
    if demande.depart_lat is not None and demande.depart_lon is not None:
        p = {"nom": "ma position actuelle",
             "latitude": demande.depart_lat, "longitude": demande.depart_lon}
        return _contexte_ml_sur(p, p)
    s = SESSIONS[session_id]
    d, a = s.get("depart"), s.get("destination")
    if d and a and not est_position_actuelle(d):
        try:
            return _contexte_ml_sur(rechercher_lieu(d), rechercher_lieu(a))
        except Exception as e:
            log.warning("géocodage session échoué : %s", e)
    return {}


@app.post("/assistant/chat", response_model=ChatResponse,
          dependencies=[Depends(verifier_acces)])
def assistant_chat(demande: Demande):
    session_id = demande.session_id or "default"
    
    message = _separer_contexte(demande.message.strip(), session_id, demande.contexte)
    
    analyse = analyser_intention(message, demande.historique, session_id=session_id)
    intention = analyse["intention"]
    mode_transport = analyse.get("mode_transport", "TOUS")

    log.info("intention : %s | mode : %s (précisé=%s)",
             intention, mode_transport, analyse.get("mode_precise"))

    def repondre(texte: str, **kw) -> ChatResponse:
        return ChatResponse(demande=message, intention=kw.pop("intention", intention),
                            reponse=texte,
                            contexte=_contexte_reponse(session_id), **kw)

    # ── CONVERSATION ─────────────────────────────────────────────
    if intention in ("CONVERSATION", "INCONNUE"):
        contexte_ml = {}
        if _est_question_conditions(message):
            dep_conv, dest_conv = None, None
            if demande.depart_lat is not None and demande.depart_lon is not None:
                dep_conv = {"nom": "ma position actuelle",
                            "latitude": demande.depart_lat, "longitude": demande.depart_lon}
                dest_conv = dep_conv
            else:
                s = SESSIONS[session_id]
                d, a = s.get("depart"), s.get("destination")
                if d and a and not est_position_actuelle(d):
                    try:
                        dep_conv, dest_conv = rechercher_lieu(d), rechercher_lieu(a)
                    except Exception as e:
                        log.warning("géocodage session échoué : %s", e)
            contexte_ml = _contexte_ml_sur(dep_conv, dest_conv)
        contexte_llm = {**contexte_ml, "statut": "CONVERSATION"}
        return repondre(
            generer_reponse(message, contexte_llm, demande.historique, session_id=session_id),
            intention="CONVERSATION",
        )

    # ── SUIVI ────────────────────────────────────────────────────
    if intention == "SUIVI":
        s = SESSIONS[session_id]
        contexte_ml = {}
        if _est_question_conditions(message):
            contexte_ml = construire_contexte_session(session_id, demande)
        contexte_llm = {**contexte_ml, "statut": "SUIVI",
                        "candidats": s.get("dernieres_options") or []}
        return repondre(
            generer_reponse(message, contexte_llm, demande.historique, session_id=session_id),
            intention="CONVERSATION",
        )

    #  TRAJET / LIGNES_TRAJET 
    if intention in ("TRAJET", "LIGNES_TRAJET"):
        # ⚠️ NOUVEAU : Bloc try/except pour attraper les erreurs de calcul de trajet
        try:
            dep_texte = analyse.get("depart") or ""
            dest_texte = analyse.get("destination") or ""
            maj_session(session_id, analyse)

            try:
                gps = coordonnees_gps_obligatoires(demande.depart_lat, demande.depart_lon, dep_texte)
            except HTTPException:
                return repondre("Je n'ai pas accès à ta position. Active la localisation "
                                "ou dis-moi d'où tu pars 😊")

            if not dest_texte:
                return repondre("Vers quel quartier ou lieu veux-tu aller exactement ? 😊")
            if not dep_texte and not gps:
                return repondre("D'où pars-tu ? Dis-moi un quartier ou un lieu "
                                "(ou active ta position) 😊")

            if gps and (not dep_texte or est_position_actuelle(dep_texte)):
                dep = {"nom": "ma position actuelle",
                       "latitude": demande.depart_lat, "longitude": demande.depart_lon}
            else:
                dep = rechercher_lieu(dep_texte)

            if (demande.dest_lat is None) != (demande.dest_lon is None):
                raise HTTPException(status_code=422,
                                    detail="Les deux coordonnées GPS de la destination sont obligatoires.")
            if demande.dest_lat is not None and demande.dest_lon is not None:
                dest = {"nom": dest_texte, "latitude": demande.dest_lat, "longitude": demande.dest_lon}
            else:
                dest = rechercher_lieu(dest_texte)

            if not dep or not dest:
                return repondre("Je n'ai pas trouvé ces lieux. Tu peux me les préciser ? 😊")

            prix_max = analyse.get("prix_max_ligne")
            budget = analyse.get("budget_fcfa")

            proches = arrets_proches_trajet(dep, dest, mode_transport)
            candidats = candidats_multimodaux(dep, dest, mode_transport, prix_max)

            signatures_exclues = {
                tuple(signature) for signature in (analyse.get("exclure_signatures") or [])
            }
            if signatures_exclues:
                candidats = [
                    c for c in candidats
                    if tuple((e.get("mode"), e.get("numero_ligne") or e.get("nom_ligne"),
                              e.get("depart"), e.get("arrivee"))
                             for e in c.get("etapes", []) if e.get("mode") != "MARCHE")
                    not in signatures_exclues
                ]

            contexte = construire_contexte(dep, dest) or {}
            candidats = ajuster_candidats(candidats, contexte)

            tous = list(candidats)
            candidats = trajets_pour_demande(candidats, analyse.get("priorite"), prix_max, budget)

            if not tous:
                p_dep = proches["depart"][0] if proches["depart"] else None
                p_arr = proches["arrivee"][0] if proches["arrivee"] else None
                infos = []
                if p_dep:
                    infos.append(f"l'arrêt le plus proche de ton départ est {p_dep['nom']} "
                                 f"(~{p_dep['distance_m']} m)")
                if p_arr:
                    infos.append(f"celui de ta destination est {p_arr['nom']} "
                                 f"(~{p_arr['distance_m']} m)")
                detail = (" Pour info, " + " et ".join(infos) + ".") if infos else ""
                message_aucun = (
                    "Je n'ai pas trouvé d'autre combinaison pertinente pour ce trajet."
                    if signatures_exclues else
                    "Oups 😅 je ne trouve pas de trajet direct dans ce mode entre ces deux lieux."
                )
                return repondre(
                    message_aucun + detail +
                    " Tu veux que j'essaie avec un autre mode (gbaka, bus, bateau-bus) ?",
                    depart=Lieu(**dep), destination=Lieu(**dest),
                    arrets_depart=proches["depart"],
                    arrets_destination=proches["arrivee"],
                )

            if tous and not candidats and budget:
                moins_cher = min(c.get("cout_partiel_fcfa", 0) for c in tous)
                maj_session(session_id, analyse, attente_mode=False)
                return repondre(
                    f"Avec {budget} FCFA, je ne trouve aucun trajet 😕. "
                    f"Le moins cher que je connais coûte environ {moins_cher} FCFA. "
                    f"Tu veux augmenter un peu ton budget ?",
                    depart=Lieu(**dep), destination=Lieu(**dest), budget_fcfa=budget,
                )

            options = choisir_propositions(candidats, mode_transport,
                                            analyse.get("priorite"), budget, maxi=4)
            principal = options[0] if options else None
            maj_session(session_id, analyse, trajet=principal, attente_mode=False)
            SESSIONS[session_id]["dernieres_options"] = options

            contexte_llm = {
                **contexte,
                "statut": "TRAJET",
                "candidats": options,
                "mode": mode_transport,
                "prix_max_ligne": prix_max,
                "budget_fcfa": budget,
                "conseil_marche": conseil_marche_chat(options),
                "arrets_proches": {k: [{"nom": a["nom"], "distance_m": a["distance_m"]}
                                        for a in v[:3]]
                                    for k, v in proches.items()},
            }
            reponse = generer_reponse(message, contexte_llm, demande.historique,
                                       session_id=session_id)

            return repondre(
                reponse,
                depart=Lieu(**dep),
                destination=Lieu(**dest),
                trajets=options,
                arrets_depart=proches["depart"],
                arrets_destination=proches["arrivee"],
                instructions=instructions_trajet(principal) if principal else [],
                trajet=principal,
                budget_fcfa=budget,
            )
        except Exception as e:
            log.error("Erreur interne lors du calcul du trajet : %s", e)
            return repondre(
                "Oups, j'ai rencontré une petite erreur technique en cherchant ton trajet. "
                "Vérifie tes lieux de départ et d'arrivée, puis réessaie ! 😊"
            )

    return repondre("Je suis MobiGuide. Dis-moi où tu veux aller ! 😊", intention="INCONNUE")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)