import sqlite3
import datetime
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent
MODELE_DIR = BASE_DIR / "modele"
CHEMIN_DB = BASE_DIR / "conteneurs.db"
FRONTEND_DIR = BASE_DIR.parent / "frontend"

app = FastAPI(title="API Prediction Conteneurs")

modele = joblib.load(MODELE_DIR / "meilleur_modele.pkl")
colonnes_modele = joblib.load(MODELE_DIR / "colonnes_modele.pkl")
tables_encodage = joblib.load(MODELE_DIR / "tables_encodage.pkl")


def get_connexion():
    return sqlite3.connect(CHEMIN_DB)


with get_connexion() as conn:
    nb_lots = conn.execute("SELECT COUNT(*) FROM lots").fetchone()[0]

print(f"[DEMARRAGE] Modele charge : {type(modele).__name__}")
print(f"[DEMARRAGE] {len(colonnes_modele)} colonnes attendues par le modele")
print(f"[DEMARRAGE] Base de donnees connectee : {nb_lots} lots")


class ConteneurInput(BaseModel):
    id_marchandise: int
    id_port_chargement: int
    id_pays_chargement: int
    id_importateur: int
    lmd_lms: int
    id_terminal: int
    date_validation_bad: str
    date_pointage: str


@app.get("/")
def racine():
    return {"message": "L'API fonctionne !"}


# ------------------------------------------------------------------
# PREDICTION
# ------------------------------------------------------------------
def encoder_entree(entree: ConteneurInput, conn):
    date_validation = pd.to_datetime(entree.date_validation_bad)
    date_pointage = pd.to_datetime(entree.date_pointage)

    ligne = {
        "LMD_LMS": entree.lmd_lms,
        "ANNEE": date_validation.year,
        "MOIS_VALIDATION": date_validation.month,
        "TRIMESTRE_VALIDATION": (date_validation.month - 1) // 3 + 1,
        "JOUR_SEMAINE_VALIDATION": date_validation.dayofweek,
        "EST_WEEKEND_VALIDATION": int(date_validation.dayofweek in [5, 6]),
        "ECART_VALIDATION_POINTAGE": (date_pointage - date_validation).days,
    }

    correspondance_freq = [
        ("ID_IMPORTATEUR", entree.id_importateur, "FREQ_IMPORTATEUR"),
        ("ID_MARCHANDISE", entree.id_marchandise, "FREQ_MARCHANDISE"),
        ("ID_PAYS_CHARGEMENT", entree.id_pays_chargement, "FREQ_PAYS"),
        ("ID_PORT_CHARGEMENT", entree.id_port_chargement, "FREQ_PORT"),
    ]
    for colonne, valeur, nom_feature in correspondance_freq:
        resultat = conn.execute(f"SELECT COUNT(*) FROM lots WHERE {colonne} = ?", (valeur,)).fetchone()
        ligne[nom_feature] = resultat[0]

    ligne["IMPORTATEUR_RARE"] = int(ligne["FREQ_IMPORTATEUR"] <= 1)

    for colonne, valeur in [("ID_IMPORTATEUR", entree.id_importateur), ("ID_MARCHANDISE", entree.id_marchandise)]:
        info = tables_encodage[colonne]
        stats_valeur = info["stats"].get(valeur, {})
        ligne[f"{colonne}_MOYENNE_HIST"] = stats_valeur.get("MOYENNE_HIST", info["moyenne_globale"])
        ligne[f"{colonne}_MEDIANE_HIST"] = stats_valeur.get("MEDIANE_HIST", info["mediane_globale"])
        ligne[f"{colonne}_ECARTTYPE_HIST"] = stats_valeur.get("ECARTTYPE_HIST", info["ecarttype_global"])
        ligne[f"{colonne}_NB_HIST"] = stats_valeur.get("NB_HIST", 0)

    info_combo = tables_encodage["ID_IMPORTATEUR_ID_MARCHANDISE_combo"]
    cle_combo = f"{entree.id_importateur}_{entree.id_marchandise}"
    if cle_combo in info_combo["combos_fiables"]:
        valeur_combo = info_combo["combos_fiables"][cle_combo]
    elif entree.id_importateur in info_combo["moyenne_col1"]:
        valeur_combo = info_combo["moyenne_col1"][entree.id_importateur]
    else:
        valeur_combo = info_combo["moyenne_globale"]
    ligne["ID_IMPORTATEUR_ID_MARCHANDISE_ENC"] = valeur_combo

    for colonne, valeur in [
        ("ID_PORT_CHARGEMENT", entree.id_port_chargement),
        ("ID_PAYS_CHARGEMENT", entree.id_pays_chargement),
    ]:
        info = tables_encodage[colonne]
        ligne[f"{colonne}_ENC"] = info["table"].get(valeur, info["moyenne_globale"])

    for cat in tables_encodage["ID_TERMINAL__categories"]:
        ligne[f"ID_TERMINAL_{cat}"] = int(entree.id_terminal == cat)

    ligne_ordonnee = {col: ligne.get(col, 0) for col in colonnes_modele}
    return pd.DataFrame([ligne_ordonnee])


@app.post("/api/predict")
def predict(entree: ConteneurInput):
    with get_connexion() as conn:
        X = encoder_entree(entree, conn)
        prediction = float(modele.predict(X)[0])
        prediction = max(0, round(prediction, 1))

        conn.execute("""
            INSERT INTO predictions_historique
            (date_prediction, id_marchandise, id_port_chargement, id_pays_chargement, id_importateur,
             lmd_lms, id_terminal, date_validation_bad, date_pointage, delai_estime_jours, modele_utilise)
            VALUES (datetime('now'), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            entree.id_marchandise, entree.id_port_chargement, entree.id_pays_chargement, entree.id_importateur,
            entree.lmd_lms, entree.id_terminal, entree.date_validation_bad, entree.date_pointage,
            prediction, type(modele).__name__
        ))
        conn.commit()

    return {
        "delai_estime_jours": prediction,
        "modele_utilise": type(modele).__name__,
        "date_prediction": datetime.datetime.now().isoformat(),
    }


# ------------------------------------------------------------------
# STATISTIQUES
# ------------------------------------------------------------------
@app.get("/api/stats")
def stats():
    with get_connexion() as conn:
        total = conn.execute("SELECT COUNT(*) FROM lots").fetchone()[0]
        moyenne = conn.execute("SELECT AVG(DELAI_SEJOUR) FROM lots").fetchone()[0]
        maximum = conn.execute("SELECT MAX(DELAI_SEJOUR) FROM lots").fetchone()[0]
        mediane = conn.execute("""
            SELECT DELAI_SEJOUR FROM lots ORDER BY DELAI_SEJOUR
            LIMIT 1 OFFSET (SELECT COUNT(*) FROM lots) / 2
        """).fetchone()[0]

        top_pays_rows = conn.execute("""
            SELECT ID_PAYS_CHARGEMENT, AVG(DELAI_SEJOUR), COUNT(*)
            FROM lots GROUP BY ID_PAYS_CHARGEMENT ORDER BY COUNT(*) DESC LIMIT 5
        """).fetchall()

        evolution_rows = conn.execute("""
            SELECT MOIS_VALIDATION, AVG(DELAI_SEJOUR)
            FROM lots GROUP BY MOIS_VALIDATION ORDER BY MOIS_VALIDATION
        """).fetchall()

    evolution_dict = {int(mois): round(val, 2) for mois, val in evolution_rows}
    evolution_mensuelle = [evolution_dict.get(m, 0) for m in range(1, 13)]

    return {
        "total_lots": total,
        "delai_moyen": round(moyenne, 2),
        "delai_median": round(mediane, 2),
        "delai_max": round(maximum, 2),
        "top_pays": [
            {"pays": int(p), "delai_moyen": round(m, 2), "nb_lots": int(n)}
            for p, m, n in top_pays_rows
        ],
        "evolution_mensuelle": evolution_mensuelle,
    }


# ------------------------------------------------------------------
# TABLEAU DES LOTS
# ------------------------------------------------------------------
@app.get("/api/lots")
def lots(page: int = 1, taille_page: int = 15, recherche: str = ""):
    with get_connexion() as conn:
        colonnes = "ID_LOT, ID_MARCHANDISE, ID_PAYS_CHARGEMENT, ID_IMPORTATEUR, ID_TERMINAL, DELAI_SEJOUR, ANNEE"

        if recherche:
            try:
                valeur = int(recherche)
                condition = "WHERE ID_LOT = ? OR ID_IMPORTATEUR = ? OR ID_MARCHANDISE = ?"
                params = (valeur, valeur, valeur)
            except ValueError:
                condition, params = "WHERE 1=0", ()
        else:
            condition, params = "", ()

        total = conn.execute(f"SELECT COUNT(*) FROM lots {condition}", params).fetchone()[0]

        debut = (page - 1) * taille_page
        rows = conn.execute(
            f"SELECT {colonnes} FROM lots {condition} LIMIT ? OFFSET ?",
            params + (taille_page, debut)
        ).fetchall()

    noms_colonnes = colonnes.split(", ")
    return {
        "total": total,
        "page": page,
        "taille_page": taille_page,
        "lots": [dict(zip(noms_colonnes, row)) for row in rows],
    }


# ------------------------------------------------------------------
# HISTORIQUE DES IMPORTATEURS
# ------------------------------------------------------------------
@app.get("/api/historique-importateurs")
def historique_importateurs(page: int = 1, taille_page: int = 15, recherche: str = ""):
    with get_connexion() as conn:
        if recherche:
            try:
                valeur = int(recherche)
                condition = "WHERE ID_IMPORTATEUR = ?"
                params = (valeur,)
            except ValueError:
                condition, params = "WHERE 1=0", ()
        else:
            condition, params = "", ()

        total = conn.execute(f"SELECT COUNT(DISTINCT ID_IMPORTATEUR) FROM lots {condition}", params).fetchone()[0]

        debut = (page - 1) * taille_page
        rows = conn.execute(f"""
            SELECT ID_IMPORTATEUR, AVG(DELAI_SEJOUR), COUNT(*)
            FROM lots {condition}
            GROUP BY ID_IMPORTATEUR ORDER BY COUNT(*) DESC
            LIMIT ? OFFSET ?
        """, params + (taille_page, debut)).fetchall()

        resultats = []
        for importateur, moyenne, nb in rows:
            mediane = conn.execute("""
                SELECT DELAI_SEJOUR FROM lots WHERE ID_IMPORTATEUR = ? ORDER BY DELAI_SEJOUR
                LIMIT 1 OFFSET (SELECT COUNT(*) FROM lots WHERE ID_IMPORTATEUR = ?) / 2
            """, (importateur, importateur)).fetchone()[0]

            detail_marchandises = conn.execute("""
                SELECT ID_MARCHANDISE, AVG(DELAI_SEJOUR), COUNT(*)
                FROM lots WHERE ID_IMPORTATEUR = ?
                GROUP BY ID_MARCHANDISE ORDER BY COUNT(*) DESC
            """, (importateur,)).fetchall()

            resultats.append({
                "importateur": int(importateur),
                "delai_moyen": round(moyenne, 2),
                "delai_median": round(mediane, 2),
                "nb_operations": int(nb),
                "marchandises": [
                    {"marchandise": int(m), "delai_moyen": round(mo, 2), "nb": int(n)}
                    for m, mo, n in detail_marchandises
                ]
            })

    return {"total": total, "page": page, "taille_page": taille_page, "importateurs": resultats}


# ------------------------------------------------------------------
# HISTORIQUE DES PREDICTIONS FAITES
# ------------------------------------------------------------------
@app.get("/api/history")
def history(limite: int = 20):
    with get_connexion() as conn:
        rows = conn.execute("""
            SELECT date_prediction, id_marchandise, id_importateur, delai_estime_jours, modele_utilise
            FROM predictions_historique ORDER BY id DESC LIMIT ?
        """, (limite,)).fetchall()

    return {
        "historique": [
            {"date": r[0], "id_marchandise": r[1], "id_importateur": r[2], "delai_estime_jours": r[3], "modele": r[4]}
            for r in rows
        ]
    }


# ------------------------------------------------------------------
# FRONTEND
# ------------------------------------------------------------------
@app.get("/app")
def servir_app():
    return FileResponse(FRONTEND_DIR / "index.html")

app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend")
