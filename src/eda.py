import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import config

DOSSIER_FIGURES = "../outputs/figures"
os.makedirs(DOSSIER_FIGURES, exist_ok=True)
sns.set_style("whitegrid")


def charger_donnees_propres():
    df = pd.read_csv(config.CHEMIN_DATA_CLEAN, sep=config.SEPARATEUR)
    for col in config.COLONNES_DATES:
        df[col] = pd.to_datetime(df[col], format="%Y-%m-%d", errors="coerce")
    print(f"[OK] {df.shape[0]} lignes chargees pour l'EDA")
    return df


def stats_generales(df):
    col = config.COL_DELAI_SEJOUR
    print("\n--- Statistiques generales de DELAI_SEJOUR ---")
    print(df[col].describe())
    print(f"\nMediane : {df[col].median():.2f} jours")
    print(f"Skewness (asymetrie) : {df[col].skew():.2f}  (> 1 = distribution tres etalee a droite)")


def histogramme_delai_sejour(df):
    col = config.COL_DELAI_SEJOUR
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(df[col], bins=100, color="steelblue", edgecolor="black")
    axes[0].set_title("Distribution de DELAI_SEJOUR (echelle normale)")
    axes[0].set_xlabel("Jours")
    axes[0].set_ylabel("Nombre de lots")
    axes[1].hist(df[col].clip(upper=df[col].quantile(0.99)), bins=100, color="darkorange", edgecolor="black")
    axes[1].set_title("Distribution de DELAI_SEJOUR (99e percentile, zoom)")
    axes[1].set_xlabel("Jours")
    axes[1].set_ylabel("Nombre de lots")
    plt.tight_layout()
    chemin = f"{DOSSIER_FIGURES}/histogramme_delai_sejour.png"
    plt.savefig(chemin, dpi=100)
    plt.close()
    print(f"[OK] Graphique sauvegarde : {chemin}")


def boxplot_par_colonne(df, colonne, max_categories=20):
    col_cible = config.COL_DELAI_SEJOUR
    nb_categories = df[colonne].nunique()
    if nb_categories > max_categories:
        print(f"[SKIP] {colonne} a {nb_categories} categories (> {max_categories}), boxplot ignore (illisible)")
        return
    plt.figure(figsize=(12, 6))
    ordre = df.groupby(colonne)[col_cible].median().sort_values().index
    sns.boxplot(data=df, x=colonne, y=col_cible, order=ordre, showfliers=False)
    plt.title(f"DELAI_SEJOUR par {colonne} (sans les valeurs extremes pour la lisibilite)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    chemin = f"{DOSSIER_FIGURES}/boxplot_{colonne}.png"
    plt.savefig(chemin, dpi=100)
    plt.close()
    print(f"[OK] Graphique sauvegarde : {chemin}")


def matrice_correlation(df):
    colonnes_numeriques = df.select_dtypes(include=["int64", "float64"]).columns
    corr = df[colonnes_numeriques].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Matrice de correlation (variables numeriques)")
    plt.tight_layout()
    chemin = f"{DOSSIER_FIGURES}/matrice_correlation.png"
    plt.savefig(chemin, dpi=100)
    plt.close()
    print(f"[OK] Graphique sauvegarde : {chemin}")
    print("\n--- Correlation de chaque variable avec DELAI_SEJOUR ---")
    print(corr[config.COL_DELAI_SEJOUR].sort_values(ascending=False))


def evolution_par_annee(df):
    moyenne_par_annee = df.groupby(config.COL_ANNEE)[config.COL_DELAI_SEJOUR].agg(["mean", "median", "count"])
    print("\n--- Delai de sejour moyen par annee ---")
    print(moyenne_par_annee)
    plt.figure(figsize=(10, 5))
    moyenne_par_annee["mean"].plot(marker="o", label="Moyenne")
    moyenne_par_annee["median"].plot(marker="s", label="Mediane")
    plt.title("Evolution du delai de sejour par annee")
    plt.xlabel("Annee")
    plt.ylabel("Jours")
    plt.legend()
    plt.tight_layout()
    chemin = f"{DOSSIER_FIGURES}/evolution_par_annee.png"
    plt.savefig(chemin, dpi=100)
    plt.close()
    print(f"[OK] Graphique sauvegarde : {chemin}")


def top_marchandises_plus_lentes(df, top_n=15):
    stats = df.groupby(config.COL_ID_MARCHANDISE)[config.COL_DELAI_SEJOUR].agg(["mean", "count"])
    stats = stats[stats["count"] >= 30]
    top = stats.sort_values("mean", ascending=False).head(top_n)
    print(f"\n--- Top {top_n} marchandises (codes) avec le delai moyen le plus long (min. 30 occurrences) ---")
    print(top)


def pipeline_eda_complet():
    df = charger_donnees_propres()
    stats_generales(df)
    histogramme_delai_sejour(df)
    boxplot_par_colonne(df, config.COL_LMD_LMS)
    boxplot_par_colonne(df, config.COL_ID_TERMINAL)
    boxplot_par_colonne(df, config.COL_ANNEE)
    matrice_correlation(df)
    evolution_par_annee(df)
    top_marchandises_plus_lentes(df)
    print(f"\n[OK] EDA terminee. Regarde les graphiques dans {DOSSIER_FIGURES}/")


if __name__ == "__main__":
    pipeline_eda_complet()
