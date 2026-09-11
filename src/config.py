"""
config.py
---------
Centralise tous les paramètres du projet (chemins, noms de colonnes,
format des dates...). Si un jour un nom de colonne ou un format change,
tu modifies UNIQUEMENT ce fichier, pas les autres.
"""

# ------------------------------------------------------------------
# CHEMINS DES FICHIERS
# ------------------------------------------------------------------
CHEMIN_DATA_BRUT = "../data/conteneurs_brut.csv"
CHEMIN_DATA_CLEAN = "../outputs/conteneurs_clean.csv"
CHEMIN_RAPPORT_MANQUANTS = "../outputs/rapport_valeurs_manquantes.csv"

# ------------------------------------------------------------------
# PARAMETRES DE LECTURE DU CSV
# ------------------------------------------------------------------
SEPARATEUR = ";"

# ------------------------------------------------------------------
# NOMS DES COLONNES (adapter ici si jamais un nom change)
# ------------------------------------------------------------------
COL_ID_LOT = "ID_LOT"
COL_ID_BL = "ID_BL"
COL_ID_MARCHANDISE = "ID_MARCHANDISE"
COL_ID_PORT_CHARGEMENT = "ID_PORT_CHARGEMENT"
COL_ID_PAYS_CHARGEMENT = "ID_PAYS_CHARGEMENT"
COL_ID_IMPORTATEUR = "ID_IMPORTATEUR"
COL_LMD_LMS = "LMD_LMS"
COL_ID_TERMINAL = "ID_TERMINAL"
COL_DATE_VALIDATION_BAD = "DATE_VALIDATION_BAD"
COL_DELAI_POINTAGE_BAD = "DELAI_POINTAGE_BAD"
COL_DATE_POINTAGE = "DATE_POINTAGE"
COL_DELAI_SEJOUR = "DELAI_SEJOUR"          # variable cible
COL_ANNEE = "ANNEE"

# Colonnes de dates à convertir automatiquement
COLONNES_DATES = [COL_DATE_VALIDATION_BAD, COL_DATE_POINTAGE]
FORMAT_DATE = "%d-%m-%y"   # format JJ-MM-AA, ex: 23-02-23

# Colonnes "code" (catégorielles encodées en chiffres)
COLONNES_CODES = [
    COL_ID_MARCHANDISE,
    COL_ID_PORT_CHARGEMENT,
    COL_ID_PAYS_CHARGEMENT,
    COL_ID_IMPORTATEUR,
    COL_LMD_LMS,
    COL_ID_TERMINAL,
]

# Valeurs considérées comme "manquantes" en plus des vraies cases vides
VALEURS_MANQUANTES_SUPPLEMENTAIRES = ["N/A", "n/a", "-", "NULL", "null", "None", ""]