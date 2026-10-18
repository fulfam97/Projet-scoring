"""
Modélisation : métriques d'ordonnancement, modèle concurrent (XGBoost), importance SHAP.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from utils.binning import woe_encode_regles


# ---------------------------------------------------------------------------
# Métriques d'ordonnancement (5.3)
# ---------------------------------------------------------------------------

def metriques(y, score, nom: str = "") -> dict:
    """AUC, Gini (= 2 AUC - 1) et KS (écart maximal entre fonctions de répartition des défauts et des sains)."""
    from sklearn.metrics import roc_auc_score, roc_curve

    y, score = np.asarray(y, dtype=int), np.asarray(score, dtype=float)
    fpr, tpr, _ = roc_curve(y, score)
    auc = float(roc_auc_score(y, score))
    return {"echantillon": nom, "n": int(y.size), "n_defaut": int(y.sum()), "taux_defaut": float(y.mean()),
            "auc": auc, "gini": 2 * auc - 1, "ks": float(np.max(tpr - fpr))}


# ---------------------------------------------------------------------------
# Modèle concurrent : XGBoost (5.2)
# ---------------------------------------------------------------------------

def matrice_xgb(df: pl.DataFrame, variables: list[str], regles: dict[str, dict], target: str,
                encodage: str = "brut", ref: pl.DataFrame | None = None) -> np.ndarray:
    """Matrice explicative pour XGBoost.
    encodage='brut' : valeurs d'origine (le manquant reste manquant, géré nativement par XGBoost) ;
    les variables qualitatives, sans ordre naturel, sont remplacées par la WOE de leur classe.
    encodage='woe' : chaque variable remplacée par la WOE de sa classe finale (même information que la régression).
    Les WOE sont calculées sur ref (le train)."""
    if encodage == "woe":
        return woe_encode_regles(df, {v: regles[v] for v in variables}, target, ref).to_numpy().astype(float)
    quali = [v for v in variables if not df.schema[v].is_numeric()]
    woe = woe_encode_regles(df, {v: regles[v] for v in quali}, target, ref) if quali else None
    cols = [woe[v] if v in quali else df[v].cast(pl.Float64) for v in variables]
    return pl.DataFrame(cols).to_numpy().astype(float)


def entrainer_xgb(X_train: np.ndarray, y_train, X_valid: np.ndarray, y_valid, seed: int = 0, **params):
    """XGBoost avec arrêt anticipé sur un échantillon de validation tiré du train (le test n'intervient pas).
    Arbres peu profonds et sous-échantillonnage pour limiter le surapprentissage (peu de défauts)."""
    import xgboost as xgb

    reglages = dict(n_estimators=2000, learning_rate=0.03, max_depth=3, min_child_weight=5, subsample=0.8,
                    colsample_bytree=0.8, reg_lambda=1.0, eval_metric="auc", early_stopping_rounds=100,
                    random_state=seed, n_jobs=-1) | params
    modele = xgb.XGBClassifier(**reglages)
    modele.fit(X_train, np.asarray(y_train), eval_set=[(X_valid, np.asarray(y_valid))], verbose=False)
    return modele


def importance_shap(modele, X: np.ndarray, variables: list[str]) -> pl.DataFrame:
    """Importance SHAP (10.1) : contribution absolue moyenne de chaque variable au score de XGBoost,
    et sa part dans le total. Calcul natif de XGBoost (pred_contribs), sans librairie shap."""
    import xgboost as xgb

    contrib = modele.get_booster().predict(xgb.DMatrix(X), pred_contribs=True,
                                           iteration_range=(0, modele.best_iteration + 1))
    moy = np.abs(contrib[:, :-1]).mean(axis=0)          # dernière colonne = constante
    return pl.DataFrame({"variable": variables, "shap_moyen": moy}).with_columns(
        part_shap=pl.col("shap_moyen") / pl.col("shap_moyen").sum()).sort("shap_moyen", descending=True)


def confronter_iv_shap(synthese: pl.DataFrame, shap: pl.DataFrame) -> pl.DataFrame:
    """Rang IV (test) contre rang SHAP : un rang SHAP bien plus faible que le rang IV signale une variable
    dont l'information est déjà portée par d'autres ; l'inverse, une information complémentaire."""
    return (
        synthese.select("variable", "iv_test").join(shap.select("variable", "part_shap"), on="variable")
        .with_columns(rang_iv=pl.col("iv_test").rank("ordinal", descending=True),
                      rang_shap=pl.col("part_shap").rank("ordinal", descending=True))
        .with_columns(ecart_rang=pl.col("rang_iv").cast(pl.Int64) - pl.col("rang_shap").cast(pl.Int64))
        .sort("rang_shap")
    )
