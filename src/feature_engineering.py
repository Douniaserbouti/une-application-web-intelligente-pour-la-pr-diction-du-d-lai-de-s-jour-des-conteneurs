import pandas as pd
import numpy as np
import config


def charger_donnees_propres():
    df = pd.read_csv(config.CHEMIN_DATA_CLEAN, sep=config.SEPARATEUR)
    for col in config.COLONNES_DATES:
        df[col] = pd.to_datetime(df[col], format="%Y-%m-%d", errors="coerce")
    print(f"[OK] {df.shape[0]} lignes chargees")
    return df


def features_temporelles(df):
    col_date = config.COL_DATE_VALIDATION_BAD
    df["MOIS_VALIDATION"] = df[col_date].dt.month
    df["TRIMESTRE_VALIDATION"] = df[col_date].dt.quarter
    df["JOUR_SEMAINE_VALIDATION"] = df[col_date].dt.dayofweek
    df["EST_WEEKEND_VALIDATION"] = df["JOUR_SEMAINE_VALIDATION"].isin([5, 6]).astype(int)
    print("[OK] Features temporelles creees")
    return df


def feature_ecart_dates(df):
    df["ECART_VALIDATION_POINTAGE"] = (df[config.COL_DATE_POINTAGE] - df[config.COL_DATE_VALIDATION_BAD]).dt.days
    print("[OK] Feature creee : ECART_VALIDATION_POINTAGE")
    print(df["ECART_VALIDATION_POINTAGE"].describe())
    return df


def features_frequence(df):
    for col, nom_feature in [
        (config.COL_ID_IMPORTATEUR, "FREQ_IMPORTATEUR"),
        (config.COL_ID_MARCHANDISE, "FREQ_MARCHANDISE"),
        (config.COL_ID_PAYS_CHARGEMENT, "FREQ_PAYS"),
        (config.COL_ID_PORT_CHARGEMENT, "FREQ_PORT"),
    ]:
        freq = df[col].value_counts()
        df[nom_feature] = df[col].map(freq)
    print("[OK] Features de frequence creees")
    return df


def feature_nouvel_importateur(df):
    df["IMPORTATEUR_RARE"] = (df["FREQ_IMPORTATEUR"] == 1).astype(int)
    print(f"[OK] Feature creee : IMPORTATEUR_RARE ({df['IMPORTATEUR_RARE'].sum()} lignes concernees)")
    return df


def resume_features(df):
    nouvelles_colonnes = [
        "MOIS_VALIDATION", "TRIMESTRE_VALIDATION", "JOUR_SEMAINE_VALIDATION",
        "EST_WEEKEND_VALIDATION", "ECART_VALIDATION_POINTAGE",
        "FREQ_IMPORTATEUR", "FREQ_MARCHANDISE", "FREQ_PAYS", "FREQ_PORT",
        "IMPORTATEUR_RARE"
    ]
    print("\n--- Resume des nouvelles features ---")
    print(df[nouvelles_colonnes].describe())


def pipeline_feature_engineering():
    df = charger_donnees_propres()
    df = features_temporelles(df)
    df = feature_ecart_dates(df)
    df = features_frequence(df)
    df = feature_nouvel_importateur(df)
    resume_features(df)
    df.to_csv(config.CHEMIN_DATA_FEATURES, sep=config.SEPARATEUR, index=False)
    print(f"\n[OK] Fichier avec features sauvegarde : {config.CHEMIN_DATA_FEATURES}")
    print(f"[OK] Shape final : {df.shape[0]} lignes, {df.shape[1]} colonnes")
    return df


if __name__ == "__main__":
    pipeline_feature_engineering()
