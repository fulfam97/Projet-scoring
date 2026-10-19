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


# ---------------------------------------------------------------------------
# Régression logistique sur indicatrices (5.1)
# ---------------------------------------------------------------------------

def ordre_risque(df: pl.DataFrame, var: str, regle: dict, target: str) -> list[str]:
    """Classes de la variable triées par taux de défaut croissant (sur df, le train) : la première,
    la moins risquée, sert de référence."""
    from utils.binning import appliquer_regle

    d = appliquer_regle(df, var, regle).with_columns(df[target].alias("y"))
    return d.group_by("classe").agg(pl.col("y").mean()).sort(["y", "classe"])["classe"].to_list()


def matrice_indicatrices(df: pl.DataFrame, variables: list[str], regles: dict[str, dict],
                         ordres: dict[str, list[str]]) -> tuple[np.ndarray, list[str], dict[str, list[int]]]:
    """Une indicatrice par classe, sauf la classe de référence (la moins risquée sur le train).
    Renvoie la matrice, le nom des colonnes et, pour chaque variable, l'indice de ses colonnes
    (dans l'ordre de risque croissant)."""
    from utils.binning import appliquer_regle

    cols, noms, groupes = [], [], {}
    for v in variables:
        classe = appliquer_regle(df, v, regles[v])["classe"]
        groupes[v] = []
        for c in ordres[v][1:]:
            groupes[v].append(len(noms))
            noms.append(f"{v} = {c}")
            cols.append((classe == c).cast(pl.Float64).to_numpy())
    X = np.column_stack(cols) if cols else np.empty((df.height, 0))
    return X, noms, groupes


def ajuster_logit(X: np.ndarray, y, groupes_clients=None):
    """Régression logistique par maximum de vraisemblance (statsmodels), sans régularisation.
    groupes_clients : identifiant client -> erreurs standard robustes par grappes (clients observés plusieurs mois)."""
    import statsmodels.api as sm

    Xc = sm.add_constant(X, has_constant="add")
    kw = {} if groupes_clients is None else {
        "cov_type": "cluster", "cov_kwds": {"groups": np.unique(np.asarray(groupes_clients), return_inverse=True)[1]}}
    return sm.Logit(np.asarray(y, dtype=float), Xc).fit(disp=0, maxiter=200, **kw)


def predire(res, X: np.ndarray) -> np.ndarray:
    import statsmodels.api as sm

    return np.asarray(res.predict(sm.add_constant(X, has_constant="add")))


def coefficients_coherents(res, groupes: dict[str, list[int]], tolerance: float = 0.0) -> dict[str, bool]:
    """Monotonie des coefficients (5.1) : la référence étant la classe la moins risquée et les classes étant
    rangées par risque croissant, les coefficients d'une variable doivent être positifs et croissants."""
    beta = np.asarray(res.params)[1:]
    out = {}
    for v, idx in groupes.items():
        b = beta[idx]
        out[v] = bool(len(b) == 0 or (b[0] > -tolerance and np.all(np.diff(b) >= -tolerance)))
    return out


def test_lr_variable(res_complet, X: np.ndarray, y, groupes: dict[str, list[int]], var: str) -> tuple[float, int, float]:
    """Test du rapport de vraisemblance : modèle complet contre modèle sans les indicatrices de la variable."""
    from scipy.stats import chi2

    idx = groupes[var]
    res_reduit = ajuster_logit(np.delete(X, idx, axis=1), y)
    lr = 2 * (res_complet.llf - res_reduit.llf)
    return float(lr), len(idx), float(chi2.sf(lr, len(idx)))


def stepwise_liaison(fit: pl.DataFrame, valid: pl.DataFrame, variables_triees: list[str], regles: dict[str, dict],
                     target: str, alpha: float = 0.05, gain_gini_min: float = 0.002,
                     tolerance_coef: float = 0.0) -> tuple[list[str], pl.DataFrame]:
    """Sélection pas à pas par ordre de liaison (5.5). Les variables sont essayées par IV décroissante ;
    une variable entre dans le modèle si :
    1. elle est significative (rapport de vraisemblance, seuil alpha) ;
    2. elle améliore le Gini de validation d'au moins gain_gini_min ;
    3. ses coefficients et ceux des variables déjà retenues restent cohérents (positifs, croissants avec le risque).
    Estimation sur `fit`, Gini mesuré sur `valid` : le test reste réservé à la mesure finale.
    Renvoie les variables retenues et le journal des décisions."""
    from sklearn.metrics import roc_auc_score

    ordres = {v: ordre_risque(fit, v, regles[v], target) for v in variables_triees}
    y_fit, y_val = fit[target].to_numpy(), valid[target].to_numpy()
    retenues, journal, gini_courant = [], [], 0.0
    for v in variables_triees:
        essai = retenues + [v]
        X_fit, _, groupes = matrice_indicatrices(fit, essai, regles, ordres)
        X_val, _, _ = matrice_indicatrices(valid, essai, regles, ordres)
        ligne = {"variable": v, "n_classes": len(ordres[v])}
        if not groupes[v]:
            journal.append(ligne | {"decision": "écartée", "motif": "une seule classe"}); continue
        try:
            res = ajuster_logit(X_fit, y_fit)
        except Exception as e:                       # matrice singulière, non-convergence
            journal.append(ligne | {"decision": "écartée", "motif": f"estimation impossible ({type(e).__name__})"}); continue
        lr, ddl, p = test_lr_variable(res, X_fit, y_fit, groupes, v)
        gini = 2 * roc_auc_score(y_val, predire(res, X_val)) - 1
        coh = coefficients_coherents(res, groupes, tolerance_coef)
        ligne |= {"p_lr": p, "gini_valid": gini, "gain_gini": gini - gini_courant}
        if p >= alpha:
            motif = f"non significative (p = {p:.3f})"
        elif gini - gini_courant < gain_gini_min:
            motif = f"gain de Gini insuffisant ({gini - gini_courant:+.4f})"
        elif not all(coh.values()):
            motif = "coefficients incohérents : " + ", ".join(k for k, ok in coh.items() if not ok)
        else:
            motif = None
        if motif:
            journal.append(ligne | {"decision": "écartée", "motif": motif})
        else:
            retenues, gini_courant = essai, gini
            journal.append(ligne | {"decision": "retenue", "motif": ""})
    return retenues, pl.DataFrame(journal)


# ---------------------------------------------------------------------------
# Diagnostics du modèle retenu (5.1, 5.4)
# ---------------------------------------------------------------------------

def tests_par_variable(res, X: np.ndarray, y, groupes: dict[str, list[int]]) -> pl.DataFrame:
    """Par variable : rapport de vraisemblance (retrait de toutes ses indicatrices) et Wald conjoint
    (avec la covariance de `res`, robuste par grappes si le modèle l'est)."""
    lignes = []
    for v, idx in groupes.items():
        lr, ddl, p_lr = test_lr_variable(res, X, y, groupes, v)
        R = np.zeros((len(idx), X.shape[1] + 1))
        for k, j in enumerate(idx):
            R[k, j + 1] = 1.0
        w = res.wald_test(R, scalar=True)
        lignes.append({"variable": v, "ddl": ddl, "lr": lr, "p_lr": p_lr,
                       "wald": float(w.statistic), "p_wald": float(w.pvalue)})
    return pl.DataFrame(lignes).sort("lr", descending=True)


def odds_ratios(res, noms: list[str]) -> pl.DataFrame:
    """Coefficients par modalité, p-value (Wald), odds ratio et IC à 95 % (5.1)."""
    ic = np.asarray(res.conf_int())
    return pl.DataFrame({
        "terme": ["constante"] + noms, "coefficient": np.asarray(res.params), "p_value": np.asarray(res.pvalues),
        "odds_ratio": np.exp(np.asarray(res.params)), "or_ic95_bas": np.exp(ic[:, 0]), "or_ic95_haut": np.exp(ic[:, 1]),
    })


def vif_indicatrices(X: np.ndarray, noms: list[str]) -> pl.DataFrame:
    """VIF de chaque indicatrice (5.4) : VIF > 5 préoccupant, > 10 sévère."""
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    Xc = sm.add_constant(X, has_constant="add")
    return pl.DataFrame({"indicatrice": noms,
                         "vif": [float(variance_inflation_factor(Xc, j + 1)) for j in range(X.shape[1])]}
                        ).sort("vif", descending=True)

def retirer_bornes(regle: dict, a_retirer: list[float]) -> dict:
    """Fusionne des classes adjacentes d'une règle numérique en retirant les bornes qui les séparent
    (regroupement d'une modalité non significative avec sa voisine, 5.1). Le rattachement du missing
    à une classe est reporté sur la classe fusionnée."""
    from utils.binning import _libelles

    anciennes = regle["bornes"]
    bornes = [b for b in anciennes if not any(np.isclose(b, a) for a in a_retirer)]
    assert len(bornes) == len(anciennes) - len(a_retirer), "borne à retirer introuvable"
    nouvelle = regle | {"bornes": bornes, "methode": f"{regle.get('methode', '')} + regroupement"}
    if regle.get("missing"):
        i = _libelles(anciennes).index(regle["missing"])
        haut = anciennes[i] if i < len(anciennes) else np.inf
        nouvelle["missing"] = _libelles(bornes)[int(np.searchsorted(np.asarray(bornes, dtype=float), haut, side="left"))]
    return nouvelle


# ---------------------------------------------------------------------------
# Optimisation des hyperparamètres de XGBoost (5.2)
# ---------------------------------------------------------------------------

GRILLE_XGB = {
    "max_depth": [2, 3, 4, 5, 6],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "min_child_weight": [1, 5, 10, 20, 50],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.5, 0.7, 0.9, 1.0],
    "reg_lambda": [0.1, 1.0, 5.0, 10.0],
    "gamma": [0.0, 0.5, 1.0],
}


def optimiser_xgb(X: np.ndarray, y, groupes, n_essais: int = 30, n_plis: int = 5, grille: dict | None = None,
                  seed: int = 0) -> tuple[dict, pl.DataFrame]:
    """Recherche aléatoire des hyperparamètres de XGBoost par validation croisée **par client**
    (StratifiedGroupKFold : un client n'est jamais à la fois en estimation et en validation, plis stratifiés
    sur le défaut). Chaque combinaison est jugée sur le Gini moyen des plis de validation, avec arrêt anticipé
    dans chaque pli. À appeler sur l'apprentissage seulement : le test n'intervient pas.
    Renvoie la meilleure combinaison et le tableau de toutes les combinaisons essayées."""
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold

    grille = grille or GRILLE_XGB
    y, groupes = np.asarray(y), np.asarray(groupes)
    rng = np.random.default_rng(seed)
    plis = list(StratifiedGroupKFold(n_splits=n_plis, shuffle=True, random_state=seed).split(X, y, groupes))
    vus, lignes = set(), []
    while len(lignes) < n_essais:
        params = {k: v[rng.integers(len(v))] for k, v in grille.items()}
        cle = tuple(sorted(params.items()))
        if cle in vus:
            continue
        vus.add(cle)
        ginis, arbres = [], []
        for i_est, i_val in plis:
            m = entrainer_xgb(X[i_est], y[i_est], X[i_val], y[i_val], seed=seed, **params)
            ginis.append(2 * roc_auc_score(y[i_val], m.predict_proba(X[i_val])[:, 1]) - 1)
            arbres.append(m.best_iteration + 1)
        lignes.append(params | {"gini_cv": float(np.mean(ginis)), "gini_cv_ecart_type": float(np.std(ginis)),
                                "n_arbres_moyen": float(np.mean(arbres))})
    resultats = pl.DataFrame(lignes).sort("gini_cv", descending=True)
    meilleurs = {k: resultats[k][0] for k in grille}
    return meilleurs, resultats


# ---------------------------------------------------------------------------
# Comparaison des modèles : incertitude et courbes ROC (5.3)
# ---------------------------------------------------------------------------

def gini_bootstrap(y, scores: dict[str, np.ndarray], groupes, reference: str | None = None, n_tirages: int = 1000,
                   niveau: float = 0.95, seed: int = 0) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Intervalles de confiance des Gini par bootstrap **par client** (on tire les clients avec remise,
    toutes les lignes d'un client tiré sont reprises). Renvoie :
    - le Gini de chaque modèle et son intervalle de confiance ;
    - l'écart de Gini de chaque modèle avec le modèle de référence et son intervalle : si l'intervalle
      ne contient pas 0, la différence est significative.
    Implémentation par pondération : chaque ligne reçoit le nombre de fois où son client a été tiré."""
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y)
    _, idx_client = np.unique(np.asarray(groupes), return_inverse=True)
    n_clients = idx_client.max() + 1
    rng = np.random.default_rng(seed)
    noms = list(scores)
    tirages = np.empty((n_tirages, len(noms)))
    for b in range(n_tirages):
        poids = np.bincount(rng.integers(0, n_clients, n_clients), minlength=n_clients)[idx_client]
        for j, nom in enumerate(noms):
            tirages[b, j] = 2 * roc_auc_score(y, scores[nom], sample_weight=poids) - 1
    a = (1 - niveau) / 2
    gini = pl.DataFrame({
        "modele": noms,
        "gini": [2 * roc_auc_score(y, scores[n]) - 1 for n in noms],
        "ic_bas": np.quantile(tirages, a, axis=0), "ic_haut": np.quantile(tirages, 1 - a, axis=0),
    })
    reference = reference or noms[-1]
    r = noms.index(reference)
    ecarts = pl.DataFrame([
        {"modele": n, "reference": reference, "ecart_gini": float(gini["gini"][j] - gini["gini"][r]),
         "ic_bas": float(np.quantile(tirages[:, j] - tirages[:, r], a)),
         "ic_haut": float(np.quantile(tirages[:, j] - tirages[:, r], 1 - a))}
        for j, n in enumerate(noms) if j != r
    ]).with_columns(significatif=(pl.col("ic_bas") > 0) | (pl.col("ic_haut") < 0))
    return gini, ecarts


def courbes_roc(y, scores: dict[str, np.ndarray], ax=None):
    """Courbes ROC superposées (taux de défauts détectés en fonction du taux de sains écartés), avec le Gini
    de chaque modèle en légende."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_auc_score, roc_curve

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    for nom, s in scores.items():
        fpr, tpr, _ = roc_curve(y, s)
        ax.plot(fpr, tpr, label=f"{nom} (Gini = {2 * roc_auc_score(y, s) - 1:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="hasard")
    ax.set(xlabel="taux de faux positifs (sains classés risqués)", ylabel="taux de vrais positifs (défauts détectés)",
           title="Courbes ROC — échantillon de test")
    ax.legend(loc="lower right", fontsize=8)
    return ax
