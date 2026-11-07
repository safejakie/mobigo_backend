"""
MOBIGO — Entraînement du modèle de prédiction de durée
======================================================
Couvre les 5 critères du bloc 2 de la soutenance :
  2.1 Problème ML       → Section [2.1]
  2.2 Préparation données → Section [2.2]
  2.3 Choix et entraînement → Section [2.3]
  2.4 Évaluation        → Section [2.4]
  2.5 Amélioration      → Section [2.5]
"""
import os
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, KFold,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ml_mobigo import FEATURES, CIBLE, CHEMIN_DATASET, CHEMIN_MODELE

warnings.filterwarnings("ignore")


def localiser_dataset() -> str:
    for chemin in (CHEMIN_DATASET, "ml/dataset.csv", "/content/dataset.csv"):
        if os.path.exists(chemin):
            return chemin
    raise FileNotFoundError(
        "dataset.csv introuvable. Lance d'abord generer_dataset.py."
    )


def charger_et_nettoyer(chemin: str) -> pd.DataFrame:
    print("\n[2.2] Chargement et nettoyage")
    print("-" * 55)
    df = pd.read_csv(chemin)
    print(f"  • Lignes brutes              : {len(df)}")

    manquants = [c for c in FEATURES + [CIBLE] if c not in df.columns]
    if manquants:
        raise ValueError(f"Colonnes absentes : {manquants}")

    avant = len(df); df = df.drop_duplicates()
    print(f"  • Doublons supprimés         : {avant - len(df)}")

    avant = len(df); df = df.dropna(subset=FEATURES + [CIBLE])
    print(f"  • Lignes avec NaN supprimées : {avant - len(df)}")

    avant = len(df); df = df[(df[CIBLE] >= 1) & (df[CIBLE] <= 360)]
    print(f"  • Outliers cible supprimés   : {avant - len(df)}")

    for c in FEATURES + [CIBLE]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=FEATURES + [CIBLE]).reset_index(drop=True)
    print(f"  • Lignes finales             : {len(df)}")

    print(f"\n  ➜ Cible : min={df[CIBLE].min():.1f}  "
          f"moy={df[CIBLE].mean():.1f}  "
          f"méd={df[CIBLE].median():.1f}  "
          f"max={df[CIBLE].max():.1f}")
    return df


def construire_modeles() -> dict:
    return {
        "Ridge (baseline linéaire)": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ]),
        "RandomForest": RandomForestRegressor(
            n_estimators=200, max_depth=12, min_samples_leaf=3,
            random_state=42, n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            random_state=42,
        ),
    }


def evaluer(y_true, y_pred, nom):
    return {
        "modèle": nom,
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R²": r2_score(y_true, y_pred),
    }


def ameliorer(X_train, y_train):
    print("\n[2.5] GridSearchCV sur RandomForest")
    print("-" * 55)
    grid = GridSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=-1),
        {"n_estimators": [100, 200, 300],
         "max_depth": [8, 12, 16, None],
         "min_samples_leaf": [1, 3, 5]},
        cv=3, scoring="neg_mean_absolute_error", n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    print(f"  • Meilleurs paramètres : {grid.best_params_}")
    print(f"  • MAE CV               : {-grid.best_score_:.2f} min")
    return grid.best_estimator_


def afficher_importances(model, noms):
    if not hasattr(model, "feature_importances_"):
        return
    print("\n  ➜ Top 8 features :")
    imp = sorted(zip(noms, model.feature_importances_), key=lambda x: -x[1])[:8]
    for nom, score in imp:
        barre = "█" * int(score * 40)
        print(f"    {nom:25s} {score:.3f}  {barre}")


def tracer(y_test, pred_base, pred_model):
    try:
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].scatter(y_test, pred_model, alpha=0.5, s=15)
        lim = [0, max(y_test.max(), pred_model.max()) * 1.05]
        ax[0].plot(lim, lim, "r--")
        ax[0].set_xlabel("Réel (min)"); ax[0].set_ylabel("Prédit (min)")
        ax[0].set_title("Prédictions vs Réel")

        ax[1].hist(y_test - pred_base, bins=30, alpha=0.5, label="Baseline")
        ax[1].hist(y_test - pred_model, bins=30, alpha=0.5, label="Modèle")
        ax[1].set_xlabel("Erreur (min)"); ax[1].legend()
        ax[1].set_title("Distribution des erreurs")
        plt.tight_layout()
        plt.savefig("rapport_performance.png", dpi=120)
        print("\n  ➜ Graphiques : rapport_performance.png")
    except Exception as e:
        print(f"\n  ⚠️  Graphiques : {e}")


def main():
    print("=" * 60)
    print(" MOBIGO — Entraînement modèle durée")
    print(f" {datetime.now():%Y-%m-%d %H:%M}")
    print("=" * 60)

    print("\n[2.1] Problème : RÉGRESSION supervisée")
    print("-" * 55)
    print(f"  • {len(FEATURES)} features")
    print(f"  • Cible : {CIBLE}")
    print(f"  • Métriques : MAE, RMSE, R²")

    df = charger_et_nettoyer(localiser_dataset())
    if len(df) < 100:
        print(f"\n⚠️  {len(df)} lignes seulement. "
              f"Lance plus de paires dans generer_dataset.py.")

    X, y = df[FEATURES], df[CIBLE]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
    )
    print(f"\n  • Train : {len(X_train)}  • Test : {len(X_test)}")

    print("\n[2.3] Entraînement")
    print("-" * 55)
    resultats, best_mae, best_model, best_name = [], float("inf"), None, ""
    for nom, model in construire_modeles().items():
        model.fit(X_train, y_train)
        r = evaluer(y_test, model.predict(X_test), nom)
        resultats.append(r)
        print(f"  • {nom:30s} MAE={r['MAE']:.2f}  "
              f"RMSE={r['RMSE']:.2f}  R²={r['R²']:.2f}")
        if r["MAE"] < best_mae:
            best_mae, best_model, best_name = r["MAE"], model, nom

    res_base = evaluer(y_test, X_test["duree_theorique"],
                       "Baseline (durée théorique)")
    resultats.append(res_base)
    print(f"  • {res_base['modèle']:30s} MAE={res_base['MAE']:.2f}  "
          f"RMSE={res_base['RMSE']:.2f}  R²={res_base['R²']:.2f}")

    print("\n[2.4] Tableau comparatif")
    print("-" * 55)
    print(pd.DataFrame(resultats).round(2).to_string(index=False))

    gain = res_base["MAE"] - best_mae
    print(f"\n  ➜ Meilleur : {best_name}")
    print(f"  ➜ Gain MAE vs baseline : {gain:.2f} min "
          f"({100 * gain / res_base['MAE']:.1f} %)")

    cv = cross_val_score(
        best_model, X_train, y_train,
        cv=KFold(5, shuffle=True, random_state=42),
        scoring="neg_mean_absolute_error",
    )
    print(f"  ➜ CV 5-fold MAE : {-cv.mean():.2f} ± {cv.std():.2f} min")

    print("\n[2.5] Amélioration")
    print("-" * 55)
    tuned = ameliorer(X_train, y_train)
    r_tuned = evaluer(y_test, tuned.predict(X_test), "RF tuné")
    print(f"  • MAE après tuning : {r_tuned['MAE']:.2f} min")

    final = tuned if r_tuned["MAE"] < best_mae else best_model
    pred_final = final.predict(X_test)

    afficher_importances(final, FEATURES)
    tracer(y_test, X_test["duree_theorique"], pred_final)

    print("\n[2.5] Démo prédiction")
    print("-" * 55)
    pred0 = final.predict(X.iloc[[0]])[0]
    print(f"  • Théorique : {df['duree_theorique'].iloc[0]:.0f} min")
    print(f"  • Prédit    : {pred0:.0f} min")
    print(f"  • Réel      : {df[CIBLE].iloc[0]:.0f} min")

    os.makedirs(os.path.dirname(CHEMIN_MODELE), exist_ok=True)
    joblib.dump({
        "model": final,
        "features": FEATURES,
        "mae_test": r_tuned["MAE"] if r_tuned["MAE"] < best_mae else best_mae,
        "trained_at": datetime.now().isoformat(),
    }, CHEMIN_MODELE)
        # ... (fin de la fonction main)
    print(f"\n✅ Modèle sauvegardé : {CHEMIN_MODELE}")

    print("\n" + "=" * 60)
    print(" TERMINÉ — copie le .pkl dans models/ et redémarre l'API")
    print("=" * 60)


if __name__ == "__main__":
    main()