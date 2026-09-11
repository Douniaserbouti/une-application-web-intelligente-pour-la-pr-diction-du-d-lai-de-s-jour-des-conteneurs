from nettoyage import pipeline_nettoyage_complet

if __name__ == "__main__":
    df_propre = pipeline_nettoyage_complet()
    print("\n--- Apercu des 5 premieres lignes du fichier nettoye ---")
    print(df_propre.head())
