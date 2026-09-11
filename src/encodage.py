import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
import config

COLONNES_HISTORIQUE_RICHE = [config.COL_ID_IMPORTATEUR, config.COL_ID_MARCHANDISE]
COLONNES_TARGET_ENCODING_SIMPLE = [config.COL_ID_PORT_CHARGEMENT, config.COL_ID_PAYS_CHARGEMENT]
COLONNES_ONEHOT = [config.COL_ID_TERMINAL]
COLONNES_A_SUPPRIMER = [config.COL_ID_LOT, config.COL_ID_BL]
COLONNES_DATES_BRUTES = config.COLONNES_DATES + [config.COL_DELAI_POINTAGE_BAD]


def charger_donnees_features():
    df = pd.read_csv(config.CHEMIN_DATA_FEATURES, sep=config.SEPARATEUR)
    print(f"[OK] {df.shape[0]} lignes, {df.shape[1]} colonnes chargees")
    return df


def separer_train_test(df, taille_test=0.2, seed=42):
    train, test = train_test_split(df, test_size=taille_test, random_state=seed)
    print(f"[OK] Split effectue : {len(train)} lignes train, {len(test)} lignes test")
    return train.copy(), test.copy()


def historique_riche(train, test, colonne, tables_encodage):
    col_cible = config.COL_DELAI_SEJOUR
    stats = train.groupby(colonne)[col_cible].agg(
        MOYENNE_HIST="mean", MEDIANE_HIST="median", ECARTTYPE_HIST="std", NB_HIST="count"
    )
    stats["ECARTTYPE_HIST"] = stats["ECARTTYPE_HIST"].fillna(0)

    moyenne_globale = train[col_cible].mean()
    mediane_globale = train[col_cible].median()
    ecarttype_global = train[col_cible].std()

    fallbacks = {
        "MOYENNE_HIST": moyenne_globale, "MEDIANE_HIST": mediane_globale,
        "ECARTTYPE_HIST": ecarttype_global, "NB_HIST": 0,
    }

    for nom_stat, valeur_fallback in fallbacks.items():
        nom_colonne = f"{colonne}_{nom_stat}"
        table = stats[nom_stat]
        train[nom_colonne] = train[colonne].map(table)
        test[nom_colonne] = test[colonne].map(table).fillna(valeur_fallback)

    tables_encodage[colonne] = {
        "stats": stats.to_dict(orient="index"),
        "moyenne_globale": moyenne_globale,
        "mediane_globale": mediane_globale,
        "ecarttype_global": ecarttype_global,
    }

    print(f"[OK] Historique riche cree pour {colonne} : MOYENNE_HIST, MEDIANE_HIST, ECARTTYPE_HIST, NB_HIST")
    return train, test


def feature_croisee(train, test, col1, col2, tables_encodage, seuil_confiance=30):
    col_cible = config.COL_DELAI_SEJOUR
    nom_combo = f"{col1}_{col2}_COMBO"
    train[nom_combo] = train[col1].astype(str) + "_" + train[col2].astype(str)
    test[nom_combo] = test[col1].astype(str) + "_" + test[col2].astype(str)

    stats_combo = train.groupby(nom_combo)[col_cible].agg(moyenne="mean", nb="count")
    combos_fiables = stats_combo[stats_combo["nb"] >= seuil_confiance]["moyenne"]

    moyenne_col1 = train.groupby(col1)[col_cible].mean()
    moyenne_globale = train[col_cible].mean()

    nom_feature = f"{col1}_{col2}_ENC"

    def encoder(row_combo, row_col1):
        if row_combo in combos_fiables.index:
            return combos_fiables[row_combo]
        elif row_col1 in moyenne_col1.index:
            return moyenne_col1[row_col1]
        else:
            return moyenne_globale

    train[nom_feature] = [encoder(c, v) for c, v in zip(train[nom_combo], train[col1])]
    test[nom_feature] = [encoder(c, v) for c, v in zip(test[nom_combo], test[col1])]

    train = train.drop(columns=[nom_combo])
    test = test.drop(columns=[nom_combo])

    tables_encodage[f"{col1}_{col2}_combo"] = {
        "combos_fiables": combos_fiables.to_dict(),
        "moyenne_col1": moyenne_col1.to_dict(),
        "moyenne_globale": moyenne_globale,
        "seuil_confiance": seuil_confiance,
    }

    print(f"[OK] Feature croisee creee : {nom_feature} "
          f"({len(combos_fiables)} combinaisons fiables sur {stats_combo.shape[0]} vues, seuil={seuil_confiance})")
    return train, test


def target_encoding_simple(train, test, colonne, tables_encodage):
    col_cible = config.COL_DELAI_SEJOUR
    nom_nouvelle_colonne = f"{colonne}_ENC"
    moyenne_globale_train = train[col_cible].mean()
    table_moyennes = train.groupby(colonne)[col_cible].mean()
    train[nom_nouvelle_colonne] = train[colonne].map(table_moyennes)
    test[nom_nouvelle_colonne] = test[colonne].map(table_moyennes)
    nb_inconnues = test[nom_nouvelle_colonne].isnull().sum()
    test[nom_nouvelle_colonne] = test[nom_nouvelle_colonne].fillna(moyenne_globale_train)
    tables_encodage[colonne] = {"table": table_moyennes.to_dict(), "moyenne_globale": moyenne_globale_train}
    print(f"[OK] Target encoding sur {colonne} -> {nom_nouvelle_colonne} ({nb_inconnues} inconnues dans le test)")
    return train, test


def onehot_encoding(train, test, colonne, tables_encodage):
    categories_train = sorted(train[colonne].unique())
    for cat in categories_train:
        nom_colonne = f"{colonne}_{cat}"
        train[nom_colonne] = (train[colonne] == cat).astype(int)
        test[nom_colonne] = (test[colonne] == cat).astype(int)
    tables_encodage[f"{colonne}__categories"] = categories_train
    print(f"[OK] One-Hot encoding sur {colonne} -> {len(categories_train)} nouvelles colonnes")
    return train, test


def nettoyer_colonnes_finales(train, test):
    colonnes_a_retirer = (
        COLONNES_HISTORIQUE_RICHE + COLONNES_TARGET_ENCODING_SIMPLE +
        COLONNES_ONEHOT + COLONNES_A_SUPPRIMER + COLONNES_DATES_BRUTES
    )
    colonnes_a_retirer = [c for c in colonnes_a_retirer if c in train.columns]
    train = train.drop(columns=colonnes_a_retirer)
    test = test.drop(columns=colonnes_a_retirer)
    print(f"[OK] Colonnes brutes retirees : {colonnes_a_retirer}")
    return train, test


def verification_coherence(train, test):
    colonnes_identiques = list(train.columns) == list(test.columns)
    print(f"\n[VERIFICATION] Colonnes identiques entre train et test : {colonnes_identiques}")
    if not colonnes_identiques:
        print("[ATTENTION] Les colonnes different !")
        print("Train uniquement :", set(train.columns) - set(test.columns))
        print("Test uniquement :", set(test.columns) - set(train.columns))
    print(f"[VERIFICATION] Valeurs manquantes train : {train.isnull().sum().sum()}")
    print(f"[VERIFICATION] Valeurs manquantes test  : {test.isnull().sum().sum()}")


def pipeline_encodage():
    df = charger_donnees_features()
    train, test = separer_train_test(df)

    tables_encodage = {}

    for colonne in COLONNES_HISTORIQUE_RICHE:
        train, test = historique_riche(train, test, colonne, tables_encodage)

    train, test = feature_croisee(train, test, config.COL_ID_IMPORTATEUR, config.COL_ID_MARCHANDISE, tables_encodage, seuil_confiance=30)

    for colonne in COLONNES_TARGET_ENCODING_SIMPLE:
        train, test = target_encoding_simple(train, test, colonne, tables_encodage)

    for colonne in COLONNES_ONEHOT:
        train, test = onehot_encoding(train, test, colonne, tables_encodage)

    train, test = nettoyer_colonnes_finales(train, test)
    verification_coherence(train, test)

    train.to_csv(config.CHEMIN_TRAIN, sep=config.SEPARATEUR, index=False)
    test.to_csv(config.CHEMIN_TEST, sep=config.SEPARATEUR, index=False)
    joblib.dump(tables_encodage, config.CHEMIN_TABLES_ENCODAGE)

    print(f"\n[OK] Train sauvegarde : {config.CHEMIN_TRAIN} ({train.shape[0]} lignes, {train.shape[1]} colonnes)")
    print(f"[OK] Test sauvegarde  : {config.CHEMIN_TEST} ({test.shape[0]} lignes, {test.shape[1]} colonnes)")
    print(f"[OK] Tables d'encodage sauvegardees : {config.CHEMIN_TABLES_ENCODAGE}")

    return train, test


if __name__ == "__main__":
    pipeline_encodage()
