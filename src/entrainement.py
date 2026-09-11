import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import config


def charger_train_test():
    train = pd.read_csv(config.CHEMIN_TRAIN, sep=config.SEPARATEUR)
    test = pd.read_csv(config.CHEMIN_TEST, sep=config.SEPARATEUR)
    print(f"[OK] Train : {train.shape[0]} lignes, {train.shape[1]} colonnes")
    print(f"[OK] Test  : {test.shape[0]} lignes, {test.shape[1]} colonnes")
    return train, test


def separer_x_y(df):
    col_cible = config.COL_DELAI_SEJOUR
    X = df.drop(columns=[col_cible])
    y = df[col_cible]
    return X, y


def evaluer_modele(nom, modele, X_train, y_train, X_test, y_test):
    print(f"\n--- Entrainement : {nom} ---")
    modele.fit(X_train, y_train)
    pred_train = modele.predict(X_train)
    pred_test = modele.predict(X_test)
    resultats = {
        "Modele": nom,
        "MAE_train": mean_absolute_error(y_train, pred_train),
        "MAE_test": mean_absolute_error(y_test, pred_test),
        "RMSE_train": np.sqrt(mean_squared_error(y_train, pred_train)),
        "RMSE_test": np.sqrt(mean_squared_error(y_test, pred_test)),
        "R2_train": r2_score(y_train, pred_train),
        "R2_test": r2_score(y_test, pred_test),
    }
    print(f"MAE  (test) : {resultats['MAE_test']:.2f} jours")
    print(f"RMSE (test) : {resultats['RMSE_test']:.2f} jours")
    print(f"R2   (test) : {resultats['R2_test']:.3f}")
    return modele, resultats


def comparer_modeles(X_train, y_train, X_test, y_test):
    modeles = {
        "Regression lineaire": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=10, min_samples_leaf=20, n_jobs=-1, random_state=42
        ),
        "Gradient Boosting (Hist)": HistGradientBoostingRegressor(
            max_iter=200, max_depth=8, min_samples_leaf=50, l2_regularization=1.0, random_state=42
        ),
    }
    resultats_liste = []
    modeles_entraines = {}
    for nom, modele in modeles.items():
        modele_entraine, resultats = evaluer_modele(nom, modele, X_train, y_train, X_test, y_test)
        resultats_liste.append(resultats)
        modeles_entraines[nom] = modele_entraine
    tableau_comparaison = pd.DataFrame(resultats_liste).sort_values("MAE_test")
    return tableau_comparaison, modeles_entraines


def afficher_importance_variables(modele, colonnes, top_n=10):
    if hasattr(modele, "feature_importances_"):
        importances = pd.Series(modele.feature_importances_, index=colonnes)
        importances = importances.sort_values(ascending=False).head(top_n)
        print(f"\n--- Top {top_n} variables les plus importantes ---")
        print(importances)
    else:
        print("\n[INFO] Ce modele ne fournit pas d'importance de variables directement.")


def pipeline_entrainement():
    train, test = charger_train_test()
    X_train, y_train = separer_x_y(train)
    X_test, y_test = separer_x_y(test)
    tableau_comparaison, modeles_entraines = comparer_modeles(X_train, y_train, X_test, y_test)
    print("\n=== TABLEAU COMPARATIF DES MODELES (trie par MAE test croissant) ===")
    print(tableau_comparaison.to_string(index=False))
    tableau_comparaison.to_csv(config.CHEMIN_COMPARAISON, sep=config.SEPARATEUR, index=False)
    print(f"\n[OK] Tableau comparatif sauvegarde : {config.CHEMIN_COMPARAISON}")
    meilleur_nom = tableau_comparaison.iloc[0]["Modele"]
    meilleur_modele = modeles_entraines[meilleur_nom]
    print(f"\n[OK] Meilleur modele selon le MAE test : {meilleur_nom}")
    afficher_importance_variables(meilleur_modele, X_train.columns)
    joblib.dump(meilleur_modele, config.CHEMIN_MEILLEUR_MODELE)
    joblib.dump(list(X_train.columns), config.CHEMIN_COLONNES_MODELE)
    print(f"\n[OK] Meilleur modele sauvegarde : {config.CHEMIN_MEILLEUR_MODELE}")
    return tableau_comparaison, meilleur_modele


if __name__ == "__main__":
    pipeline_entrainement()
