import pandas as pd
import numpy as np
import config


def charger_donnees(chemin=config.CHEMIN_DATA_BRUT):
    df = pd.read_csv(chemin, sep=config.SEPARATEUR)
    if df.columns[0].startswith("Unnamed") or df.columns[0] == "":
        df = df.drop(columns=[df.columns[0]])
    print(f"[OK] Donnees chargees : {df.shape[0]} lignes, {df.shape[1]} colonnes")
    return df


def remplacer_faux_manquants(df):
    df = df.replace(config.VALEURS_MANQUANTES_SUPPLEMENTAIRES, np.nan)
    return df


def rapport_valeurs_manquantes(df, sauvegarder=True):
    total = df.isnull().sum()
    pourcentage = (total / len(df)) * 100
    resume = pd.DataFrame({"Nb_manquants": total, "%_manquant": pourcentage.round(2)})
    resume = resume[resume["Nb_manquants"] > 0].sort_values("%_manquant", ascending=False)
    print("\n--- Rapport des valeurs manquantes ---")
    print(resume if not resume.empty else "Aucune valeur manquante trouvee.")
    if sauvegarder and not resume.empty:
        resume.to_csv(config.CHEMIN_RAPPORT_MANQUANTS)
        print(f"[OK] Rapport sauvegarde dans {config.CHEMIN_RAPPORT_MANQUANTS}")
    return resume


def convertir_dates(df):
    for col in config.COLONNES_DATES:
        avant = df[col].isnull().sum()
        df[col] = pd.to_datetime(df[col], format=config.FORMAT_DATE, errors="coerce")
        apres = df[col].isnull().sum()
        nouveaux_invalides = apres - avant
        print(f"[INFO] {col} : {nouveaux_invalides} valeurs n'ont pas pu etre converties en date")
    return df


def supprimer_doublons(df):
    avant = len(df)
    df = df.drop_duplicates()
    print(f"[INFO] {avant - len(df)} lignes strictement dupliquees supprimees")
    print(f"[INFO] ID_LOT n'est pas un identifiant unique (un lot peut regrouper plusieurs lignes), donc pas de deduplication sur cette colonne")
    return df

def nettoyer_delai_sejour(df):
    col = config.COL_DELAI_SEJOUR
    print(f"\n--- Statistiques de {col} avant nettoyage ---")
    print(df[col].describe())
    negatifs = (df[col] < 0).sum()
    print(f"[INFO] {negatifs} lignes avec delai negatif -> supprimees")
    df = df[df[col] >= 0]
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    borne_sup = Q3 + 1.5 * IQR
    nb_extremes = (df[col] > borne_sup).sum()
    print(f"[INFO] Borne superieure normale (IQR) : {borne_sup:.2f} jours")
    print(f"[INFO] {nb_extremes} lignes au-dessus de cette borne (a examiner, pas supprimees automatiquement)")
    return df


def inspecter_colonnes_codes(df):
    print("\n--- Inspection des colonnes categorielles (codes) ---")
    for col in config.COLONNES_CODES:
        print(f"{col:25s} -> {df[col].nunique():5d} valeurs uniques | min={df[col].min()} | max={df[col].max()}")


def pipeline_nettoyage_complet(chemin_entree=config.CHEMIN_DATA_BRUT, chemin_sortie=config.CHEMIN_DATA_CLEAN):
    df = charger_donnees(chemin_entree)
    df = remplacer_faux_manquants(df)
    rapport_valeurs_manquantes(df)
    df = convertir_dates(df)
    df = supprimer_doublons(df)
    df = nettoyer_delai_sejour(df)
    inspecter_colonnes_codes(df)
    df.to_csv(chemin_sortie, sep=config.SEPARATEUR, index=False)
    print(f"\n[OK] Fichier nettoye sauvegarde : {chemin_sortie}")
    print(f"[OK] Shape final : {df.shape[0]} lignes, {df.shape[1]} colonnes")
    return df
