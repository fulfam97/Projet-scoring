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

def rattacher_missing_absent(regles: dict[str, dict], df: pl.DataFrame, target: str) -> tuple[dict[str, dict], list[str]]:
    """Variables sans valeur manquante dans l'apprentissage (`df`) : aucune classe « missing » n'a pu y être
    estimée. Par prudence, un manquant rencontré plus tard (test, OOT, production) est rattaché à la classe
    la plus risquée de la variable. Renvoie les règles complétées et la liste des variables concernées."""
    nouvelles, concernees = dict(regles), []
    for v, r in regles.items():
        if r.get("missing") is None and v in df.columns and df[v].null_count() == 0:
            pire = ordre_risque(df, v, r, target)[-1]
            nouvelles[v] = r | {"missing": pire}
            concernees.append(v)
    return nouvelles, concernees


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


def reexaminer_ecartees(fit: pl.DataFrame, valid: pl.DataFrame, retenues: list[str], candidates: list[str],
                        regles: dict[str, dict], target: str, alpha: float = 0.05, gain_gini_min: float = 0.002,
                        tolerance_coef: float = 0.0) -> tuple[list[str], pl.DataFrame]:
    """Second passage du stepwise (5.5). Le premier passage juge chaque variable une seule fois, face au modèle
    du moment : une variable écartée tôt peut devenir utile une fois les autres présentes. Ici, à chaque tour,
    chaque variable écartée est réessayée face au modèle complet ; parmi celles qui passent les mêmes critères
    (significative, gain de Gini de validation >= gain_gini_min, coefficients cohérents), la meilleure entre.
    Arrêt quand plus aucune ne passe. Renvoie les variables retenues et le journal de chaque tour."""
    from sklearn.metrics import roc_auc_score

    ordres = {v: ordre_risque(fit, v, regles[v], target) for v in retenues + candidates}
    y_fit, y_val = fit[target].to_numpy(), valid[target].to_numpy()

    def evaluer(variables):
        X_fit, _, groupes = matrice_indicatrices(fit, variables, regles, ordres)
        X_val, _, _ = matrice_indicatrices(valid, variables, regles, ordres)
        res = ajuster_logit(X_fit, y_fit)
        return res, X_fit, groupes, 2 * roc_auc_score(y_val, predire(res, X_val)) - 1

    gini0 = evaluer(retenues)[3]
    restantes = [v for v in candidates if len(ordres[v]) > 1]
    journal, tour = [], 0
    while restantes:
        tour += 1
        essais = []
        for v in restantes:
            try:
                res, X, groupes, gini = evaluer(retenues + [v])
            except Exception:                        # matrice singulière, non-convergence
                continue
            p = test_lr_variable(res, X, y_fit, groupes, v)[2]
            coherent = all(coefficients_coherents(res, groupes, tolerance_coef).values())
            essais.append({"tour": tour, "variable": v, "p_lr": p, "gini_valid": gini, "gain_gini": gini - gini0,
                           "coherent": coherent,
                           "eligible": p < alpha and gini - gini0 >= gain_gini_min and coherent})
        eligibles = [e for e in essais if e["eligible"]]
        meilleure = max(eligibles, key=lambda e: e["gain_gini"])["variable"] if eligibles else None
        journal += [e | {"decision": "ajoutée" if e["variable"] == meilleure else ""} for e in essais]
        if meilleure is None:
            break
        retenues = retenues + [meilleure]
        restantes.remove(meilleure)
        gini0 = next(e["gini_valid"] for e in essais if e["variable"] == meilleure)
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


# ---------------------------------------------------------------------------
# Interactions : détection par XGBoost (SHAP), intégration en variables croisées
# ---------------------------------------------------------------------------

def interactions_shap(modele, X: np.ndarray, variables: list[str], n_max: int = 10_000, seed: int = 0) -> pl.DataFrame:
    """Interactions de XGBoost (SHAP interaction values, 10.1) : pour chaque paire de variables, contribution
    absolue moyenne de leur effet conjoint, au-delà de leurs effets séparés. Calcul natif XGBoost sur un
    sous-échantillon de n_max lignes (coût en n x p²)."""
    import xgboost as xgb

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(n_max, len(X)), replace=False)
    inter = modele.get_booster().predict(xgb.DMatrix(X[idx]), pred_interactions=True,
                                         iteration_range=(0, modele.best_iteration + 1))[:, :-1, :-1]
    m = np.abs(inter).mean(axis=0)
    principal = np.diag(m)
    i, j = np.triu_indices(len(variables), k=1)
    return pl.DataFrame({
        "var_a": [variables[k] for k in i], "var_b": [variables[k] for k in j],
        "interaction": 2 * m[i, j],                                   # phi_ab + phi_ba
        "rapport_effet_principal": 2 * m[i, j] / np.minimum(principal[i], principal[j]),
    }).sort("interaction", descending=True)


def nom_croisement(a: str, b: str) -> str:
    return f"{a}_x_{b}"


def croiser(df: pl.DataFrame, a: str, b: str, regles: dict[str, dict]) -> pl.DataFrame:
    """Ajoute la variable croisée a x b : couple (classe de a, classe de b) des découpages finaux."""
    from utils.binning import appliquer_regle

    ca, cb = appliquer_regle(df, a, regles[a])["classe"], appliquer_regle(df, b, regles[b])["classe"]
    return df.with_columns((ca + " | " + cb).alias(nom_croisement(a, b)))


def ajouter_croisements(df: pl.DataFrame, croisements: list[tuple[str, str]], regles: dict[str, dict]) -> pl.DataFrame:
    for a, b in croisements:
        df = croiser(df, a, b, regles)
    return df


def _fusion_ratio(n, d, e, part_min: float, defauts_min: int, ecart_min: float) -> list[list[int]]:
    """Cellules triées par ratio défauts observés / attendus ; fusion de cellules adjacentes jusqu'à respecter
    effectif minimal, défauts minimaux et écart relatif de ratio minimal entre groupes voisins."""
    groupes, total = [[i] for i in range(len(n))], n.sum()
    while len(groupes) > 1:
        ng, dg, eg = (np.array([x[g].sum() for g in groupes]) for x in (n, d, e))
        ratio = (dg + 0.5) / (eg + 0.5)
        petits = np.flatnonzero((ng < part_min * total) | (dg < defauts_min))
        if petits.size:
            k = petits[np.argmin(ng[petits])]
            j = min((v for v in (k - 1, k + 1) if 0 <= v < len(groupes)), key=lambda v: abs(ratio[v] - ratio[k]))
            i = min(k, j)
        else:
            ecarts = ratio[1:] / ratio[:-1] - 1
            if ecarts.min() >= ecart_min:
                break
            i = int(np.argmin(ecarts))
        groupes[i:i + 2] = [groupes[i] + groupes[i + 1]]
    return groupes


def regle_croisement(fit: pl.DataFrame, a: str, b: str, regles: dict[str, dict], target: str, pd_base: np.ndarray,
                     part_min: float = 0.05, defauts_min: int = 20, ecart_min: float = 0.3) -> dict:
    """Découpage de la variable croisée a x b sur fit. Chaque cellule est comparée au modèle sans interaction :
    ratio = défauts observés / défauts attendus (somme de pd_base). Cellules triées par ratio puis fusionnées
    (taille, défauts, écart de 30 %, 4.5-4.6) : le croisement ne capte que le risque que les effets séparés
    de a et b n'expliquent pas. Cellule absente de fit : rattachée par prudence au groupe le plus risqué."""
    nom = nom_croisement(a, b)
    cel = (croiser(fit, a, b, regles).select(nom, target).with_columns(e=pl.Series(pd_base))
           .group_by(nom).agg(n=pl.len(), d=pl.col(target).sum(), e=pl.col("e").sum())
           .with_columns(ratio=(pl.col("d") + 0.5) / (pl.col("e") + 0.5)).sort("ratio"))
    groupes = _fusion_ratio(*(cel[c].to_numpy().astype(float) for c in ("n", "d", "e")), part_min, defauts_min, ecart_min)
    cellules = cel[nom].to_list()
    ordre = [f"G{k + 1}" for k in range(len(groupes))]
    return {"type": "quali", "groupes": {cellules[i]: f"G{k + 1}" for k, g in enumerate(groupes) for i in g},
            "ordre": ordre, "missing": ordre[-1]}


def _ajuster_valid(fit, valid, variables, regles, target, ordres):
    from sklearn.metrics import roc_auc_score

    X_fit, _, groupes = matrice_indicatrices(fit, variables, regles, ordres)
    X_val, _, _ = matrice_indicatrices(valid, variables, regles, ordres)
    res = ajuster_logit(X_fit, fit[target].to_numpy())
    return res, predire(res, X_fit), 2 * roc_auc_score(valid[target].to_numpy(), predire(res, X_val)) - 1, groupes


def tester_croisements(fit: pl.DataFrame, valid: pl.DataFrame, variables: list[str], paires: list[tuple[str, str]],
                       regles: dict[str, dict], target: str, alpha: float = 0.05, gain_gini_min: float = 0.002,
                       **params) -> tuple[list[str], dict[str, dict], dict[str, list[str]], list[tuple[str, str]], pl.DataFrame]:
    """Ajoute au modèle, paire par paire (interaction SHAP décroissante), la variable croisée a x b, en gardant
    a et b. Retenue si significative (rapport de vraisemblance), gain de Gini de validation >= gain_gini_min et
    coefficients cohérents. Chaque croisement est construit par rapport au modèle courant.
    Classes du croisement ordonnées par ratio observé / attendu (référence = ratio le plus bas).
    Renvoie variables, règles complétées, ordres, croisements retenus et journal."""
    from scipy.stats import chi2

    regles = dict(regles)
    ordres = {v: ordre_risque(fit, v, regles[v], target) for v in variables}
    res0, pd0, gini0, _ = _ajuster_valid(fit, valid, variables, regles, target, ordres)
    retenus, journal = [], []
    for a, b in paires:
        nom, ligne = nom_croisement(a, b), {"var_a": a, "var_b": b}
        regle = regle_croisement(fit, a, b, regles, target, pd0, **params)
        ligne["n_classes"] = len(regle["ordre"])
        if len(regle["ordre"]) < 2:
            journal.append(ligne | {"decision": "écartée", "motif": "aucun écart au modèle sans interaction"}); continue
        r, o = regles | {nom: regle}, ordres | {nom: regle["ordre"]}
        f, v = croiser(fit, a, b, regles), croiser(valid, a, b, regles)
        res, pd1, gini, groupes = _ajuster_valid(f, v, variables + [nom], r, target, o)
        p = float(chi2.sf(2 * (res.llf - res0.llf), len(regle["ordre"]) - 1))
        coh = coefficients_coherents(res, groupes)
        ligne |= {"p_lr": p, "gini_valid": gini, "gain_gini": gini - gini0}
        if p >= alpha:
            motif = f"non significatif (p = {p:.3f})"
        elif gini - gini0 < gain_gini_min:
            motif = f"gain de Gini insuffisant ({gini - gini0:+.4f})"
        elif not all(coh.values()):
            motif = "coefficients incohérents : " + ", ".join(k for k, ok in coh.items() if not ok)
        else:
            motif = None
        if motif:
            journal.append(ligne | {"decision": "écartée", "motif": motif}); continue
        journal.append(ligne | {"decision": "retenue", "motif": ""})
        variables, regles, ordres = variables + [nom], r, o
        res0, pd0, gini0, fit, valid = res, pd1, gini, f, v
        retenus.append((a, b))
    return variables, regles, ordres, retenus, pl.DataFrame(journal)


# ---------------------------------------------------------------------------
# Importance par permutation (10.1)
# ---------------------------------------------------------------------------

def importance_permutation(df: pl.DataFrame, variables: list[str], target: str, predire_df,
                           n_repet: int = 10, seed: int = 0) -> pl.DataFrame:
    """Importance par permutation (10.1) : perte de Gini quand les valeurs d'une variable sont mélangées
    (le lien avec la cible est rompu, la distribution conservée). predire_df : fonction df -> score.
    Moyenne et écart-type sur n_repet permutations. S'applique à tout modèle, en points de Gini."""
    from sklearn.metrics import roc_auc_score

    y = df[target].to_numpy()
    gini = lambda s: 2 * roc_auc_score(y, s) - 1
    g0, lignes = gini(predire_df(df)), []
    for v in variables:
        pertes = [g0 - gini(predire_df(df.with_columns(pl.col(v).shuffle(seed=seed + r)))) for r in range(n_repet)]
        lignes.append({"variable": v, "perte_gini": float(np.mean(pertes)), "ecart_type": float(np.std(pertes))})
    return pl.DataFrame(lignes).with_columns(gini_complet=pl.lit(g0)).sort("perte_gini", descending=True)


def _fusion_paire(regle: dict, a: str, b: str) -> dict | None:
    """Fusionne les classes a et b d'une règle (5.1). Manquant : rattaché à l'autre classe. Numérique : seulement
    si les deux classes sont contiguës (None sinon). Qualitative : b rejoint le groupe de a."""
    from utils.binning import _libelles

    if "missing" in (a, b):
        return regle | {"missing": b if a == "missing" else a}
    if regle["type"] == "quali":
        return regle | {"groupes": {m: (a if g == b else g) for m, g in regle["groupes"].items()},
                        "ordre": [g for g in regle["ordre"] if g != b]}
    libelles = _libelles(regle["bornes"])
    i, j = sorted((libelles.index(a), libelles.index(b)))
    if j - i != 1:
        return None
    return retirer_bornes(regle, [regle["bornes"][i]])


def _fusionner_incoherence(regle: dict, ordre: list[str], coefs: np.ndarray) -> dict | None:
    """Une étape de regroupement (5.1) : première paire de classes, dans l'ordre de risque, dont le coefficient
    décroît ; les deux classes sont fusionnées. None si impossible."""
    c = np.concatenate([[0.0], coefs])                # la référence vaut 0
    k = next((i for i in range(len(c) - 1) if c[i + 1] < c[i]), None)
    return None if k is None else _fusion_paire(regle, ordre[k], ordre[k + 1])


def regrouper_non_significatives(df: pl.DataFrame, variables: list[str], regles: dict[str, dict], target: str,
                                 alpha: float = 0.05, max_iter: int = 30) -> tuple[dict[str, dict], pl.DataFrame]:
    """Test par modalité (5.1) appliqué automatiquement : tant qu'une indicatrice n'est pas significative
    (p > alpha), la modalité la moins significative est fusionnée avec sa voisine moins risquée (la référence
    pour la première). Le modèle est réestimé après chaque fusion. Renvoie les règles et le journal."""
    regles, journal, bloquees = dict(regles), [], set()
    for _ in range(max_iter):
        ordres = {v: ordre_risque(df, v, regles[v], target) for v in variables}
        X, _, groupes = matrice_indicatrices(df, variables, regles, ordres)
        p = np.asarray(ajuster_logit(X, df[target]).pvalues)[1:]
        cand = sorted(((p[j], v, k) for v, idx in groupes.items() for k, j in enumerate(idx)
                       if p[j] > alpha and (v, ordres[v][k + 1]) not in bloquees), reverse=True)
        if not cand:
            break
        pv, v, k = cand[0]
        a, b = ordres[v][k], ordres[v][k + 1]
        nouvelle = _fusion_paire(regles[v], a, b)
        if nouvelle is None:
            bloquees.add((v, b))
            journal.append({"variable": v, "modalite": b, "fusionnee_avec": a, "p_value": float(pv), "fusion": False})
            continue
        regles[v] = nouvelle
        journal.append({"variable": v, "modalite": b, "fusionnee_avec": a, "p_value": float(pv), "fusion": True})
    return regles, pl.DataFrame(journal)


def stepwise_guide(fit, valid, retenues, candidates, regles, target, alpha=0.05, gain_gini_min=0.001,
                   max_fusions=5) -> tuple[list[str], dict[str, dict], pl.DataFrame]:
    """Sélection guidée (5.5) : candidates essayées dans l'ordre donné (ex. importance par permutation de
    XGBoost). Coefficients incohérents → les classes inversées sont fusionnées (5.1) et la variable réessayée,
    jusqu'à max_fusions fois. Retenue si significative, cohérente et gain de Gini de validation >= seuil.
    Renvoie les variables, les règles (avec regroupements) et le journal."""
    # ... (code testé ci-dessus, environ 40 lignes)


# ---------------------------------------------------------------------------
# Sélection guidée par le modèle concurrent, avec regroupement automatique (5.1, 5.5)
# ---------------------------------------------------------------------------

def _fusionner_incoherence(regle: dict, ordre: list[str], coefs: np.ndarray) -> dict | None:
    """Une étape de regroupement (5.1) : première paire de classes, dans l'ordre de risque, dont le coefficient
    décroît ; les deux classes sont fusionnées. None si impossible (classes non contiguës d'une numérique)."""
    from utils.binning import _libelles

    c = np.concatenate([[0.0], coefs])                # la référence vaut 0
    k = next((i for i in range(len(c) - 1) if c[i + 1] < c[i]), None)
    if k is None:
        return None
    a, b = ordre[k], ordre[k + 1]
    if "missing" in (a, b):                           # manquant rattaché à la classe voisine
        return regle | {"missing": b if a == "missing" else a}
    if regle["type"] == "quali":
        return regle | {"groupes": {m: (a if g == b else g) for m, g in regle["groupes"].items()},
                        "ordre": [g for g in regle["ordre"] if g != b]}
    libelles = _libelles(regle["bornes"])
    i, j = sorted((libelles.index(a), libelles.index(b)))
    if j - i != 1:
        return None
    return retirer_bornes(regle, [regle["bornes"][i]])


def stepwise_guide(fit: pl.DataFrame, valid: pl.DataFrame, retenues: list[str], candidates: list[str],
                   regles: dict[str, dict], target: str, alpha: float = 0.05, gain_gini_min: float = 0.001,
                   max_fusions: int = 5) -> tuple[list[str], dict[str, dict], pl.DataFrame]:
    """Sélection guidée (5.5) : candidates essayées dans l'ordre donné (ex. importance par permutation de
    XGBoost). Coefficients incohérents → les classes inversées sont fusionnées (5.1) et la variable réessayée,
    jusqu'à max_fusions fois. Retenue si significative, cohérente et gain de Gini de validation >= seuil.
    Renvoie les variables, les règles (avec regroupements) et le journal."""
    from sklearn.metrics import roc_auc_score

    regles = dict(regles)
    y_fit, y_val = fit[target].to_numpy(), valid[target].to_numpy()

    def evaluer(variables, r):
        ordres = {v: ordre_risque(fit, v, r[v], target) for v in variables}
        X_fit, _, groupes = matrice_indicatrices(fit, variables, r, ordres)
        X_val, _, _ = matrice_indicatrices(valid, variables, r, ordres)
        res = ajuster_logit(X_fit, y_fit)
        return res, X_fit, groupes, ordres, 2 * roc_auc_score(y_val, predire(res, X_val)) - 1

    gini0 = evaluer(retenues, regles)[4]
    journal = []
    for v in candidates:
        regle, n_fusions, motif = regles[v], 0, None
        while True:
            res, X, groupes, ordres, gini = evaluer(retenues + [v], regles | {v: regle})
            if not groupes[v]:
                motif = "une seule classe"
                break
            coh = coefficients_coherents(res, groupes)
            if all(coh.values()):
                break
            if not coh[v] and n_fusions < max_fusions:
                nouvelle = _fusionner_incoherence(regle, ordres[v], np.asarray(res.params)[1:][groupes[v]])
                if nouvelle is not None:
                    regle, n_fusions = nouvelle, n_fusions + 1
                    continue
            motif = "coefficients incohérents : " + ", ".join(k for k, ok in coh.items() if not ok)
            break
        ligne = {"variable": v, "n_fusions": n_fusions, "n_classes": len(ordres[v])}
        if motif is None:
            p = test_lr_variable(res, X, y_fit, groupes, v)[2]
            ligne |= {"p_lr": p, "gini_valid": gini, "gain_gini": gini - gini0}
            if p >= alpha:
                motif = f"non significative (p = {p:.3f})"
            elif gini - gini0 < gain_gini_min:
                motif = f"gain de Gini insuffisant ({gini - gini0:+.4f})"
        if motif:
            journal.append(ligne | {"decision": "écartée", "motif": motif})
            continue
        journal.append(ligne | {"decision": "retenue", "motif": ""})
        retenues, regles, gini0 = retenues + [v], regles | {v: regle}, gini
    return retenues, regles, pl.DataFrame(journal)


def logit_final(train: pl.DataFrame, test: pl.DataFrame, variables: list[str], regles: dict[str, dict],
                target: str, id_col: str) -> dict:
    """Réestimation d'une variante de régression sur tout l'apprentissage (erreurs robustes par client) :
    manquants absents du train rattachés à la classe la plus risquée, référence = classe la moins risquée.
    Renvoie le modèle, les matrices, les groupes, les ordres et les règles utilisées."""
    r, _ = rattacher_missing_absent({v: regles[v] for v in variables}, train, target)
    ordres = {v: ordre_risque(train, v, r[v], target) for v in variables}
    X_train, noms, groupes = matrice_indicatrices(train, variables, r, ordres)
    X_test, _, _ = matrice_indicatrices(test, variables, r, ordres)
    res = ajuster_logit(X_train, train[target], groupes_clients=train[id_col])
    return {"res": res, "X_train": X_train, "X_test": X_test, "noms": noms, "groupes": groupes,
            "ordres": ordres, "regles": r, "coherent": all(coefficients_coherents(res, groupes).values())}


# ---------------------------------------------------------------------------
# Sous-segmentation : une grille par sous-population (1.4)
# ---------------------------------------------------------------------------

def _modele_sous_population(fit: pl.DataFrame, valid: pl.DataFrame, candidates: list[str], regles: dict[str, dict],
                            target: str, gain_gini_min: float) -> tuple[list[str], dict[str, dict]]:
    """Sélection guidée (avec regroupement des classes incohérentes) puis test par modalité, sur une sous-population."""
    candidates = [v for v in candidates if fit[v].drop_nulls().n_unique() > 1]
    variables, r, _ = stepwise_guide(fit, valid, [], candidates, regles, target, gain_gini_min=gain_gini_min)
    if variables:
        r, _ = regrouper_non_significatives(fit, variables, r, target)
    return variables, r


def predire_sous_population(df_fit: pl.DataFrame, df_pred: pl.DataFrame, variables: list[str], regles: dict[str, dict],
                             target: str) -> tuple[np.ndarray, object, int]:
    """Logit estimé sur df_fit, prédit sur df_pred ; renvoie aussi le modèle et le nombre de paramètres."""
    r, _ = rattacher_missing_absent({v: regles[v] for v in variables}, df_fit, target)
    ordres = {v: ordre_risque(df_fit, v, r[v], target) for v in variables}
    X_fit, _, _ = matrice_indicatrices(df_fit, variables, r, ordres)
    X_pred, _, _ = matrice_indicatrices(df_pred, variables, r, ordres)
    res = ajuster_logit(X_fit, df_fit[target])
    return predire(res, X_pred), res, X_fit.shape[1] + 1


def comparer_sous_segmentations(fit: pl.DataFrame, valid: pl.DataFrame, candidats: dict[str, pl.Expr],
                                candidates: list[str], regles: dict[str, dict], target: str,
                                reference: tuple[list[str], dict[str, dict]], defauts_min: int = 200,
                                gain_gini_min: float = 0.001) -> tuple[pl.DataFrame, dict]:
    """Pour chaque coupure candidate (expression booléenne), un modèle par sous-population : sélection guidée sur
    l'estimation, prédictions assemblées sur la validation. Comparé au modèle unique `reference` (variables, règles)
    estimé sur les mêmes données : Gini de validation global et AIC (somme des AIC des deux modèles).
    Une coupure dont une sous-population a moins de defauts_min défauts d'estimation n'est pas modélisée.
    Renvoie le tableau de comparaison et, pour chaque coupure, les variables et règles de chaque sous-population."""
    from sklearn.metrics import roc_auc_score

    y_val = valid[target].to_numpy()
    p_ref, res_ref, _ = predire_sous_population(fit, valid, *reference, target)
    lignes = [{"coupure": "modèle unique", "part_a": None, "defauts_a": int(fit[target].sum()), "defauts_b": None,
               "variables_a": len(reference[0]), "variables_b": None,
               "gini_valid": 2 * roc_auc_score(y_val, p_ref) - 1, "aic": float(res_ref.aic)}]
    sous_modeles = {}
    for nom, expr in candidats.items():
        masque_f = fit.select(expr.fill_null(False)).to_series().to_numpy()
        masque_v = valid.select(expr.fill_null(False)).to_series().to_numpy()
        fa, fb, va, vb = fit.filter(masque_f), fit.filter(~masque_f), valid.filter(masque_v), valid.filter(~masque_v)
        ligne = {"coupure": nom, "part_a": float(masque_f.mean()), "defauts_a": int(fa[target].sum()),
                 "defauts_b": int(fb[target].sum())}
        if min(ligne["defauts_a"], ligne["defauts_b"]) < defauts_min:
            lignes.append(ligne | {"motif": f"moins de {defauts_min} défauts dans une sous-population"})
            continue
        (vars_a, r_a), (vars_b, r_b) = (_modele_sous_population(f, v, candidates, regles, target, gain_gini_min)
                                        for f, v in ((fa, va), (fb, vb)))
        p = np.empty(valid.height)
        p_a, res_a, _ = predire_sous_population(fa, va, vars_a, r_a, target)
        p_b, res_b, _ = predire_sous_population(fb, vb, vars_b, r_b, target)
        p[masque_v], p[~masque_v] = p_a, p_b
        sous_modeles[nom] = {"expr": expr, "a": (vars_a, r_a), "b": (vars_b, r_b)}
        lignes.append(ligne | {"variables_a": len(vars_a), "variables_b": len(vars_b),
                               "gini_valid": 2 * roc_auc_score(y_val, p) - 1, "aic": float(res_a.aic + res_b.aic),
                               "gini_valid_a": 2 * roc_auc_score(va[target].to_numpy(), p_a) - 1,
                               "gini_valid_b": 2 * roc_auc_score(vb[target].to_numpy(), p_b) - 1})
    return pl.DataFrame(lignes, infer_schema_length=None), sous_modeles


def logit_sous_segmente(train: pl.DataFrame, test: pl.DataFrame, sous_modele: dict, target: str,
                        id_col: str) -> dict:
    """Réestimation sur tout l'apprentissage des deux modèles d'une sous-segmentation (logit_final sur chaque
    sous-population) ; prédictions de test réassemblées dans l'ordre des lignes."""
    expr = sous_modele["expr"].fill_null(False)
    m_tr, m_te = (d.select(expr).to_series().to_numpy() for d in (train, test))
    finaux = {"a": logit_final(train.filter(m_tr), test.filter(m_te), *sous_modele["a"], target, id_col),
              "b": logit_final(train.filter(~m_tr), test.filter(~m_te), *sous_modele["b"], target, id_col)}
    p = np.empty(test.height)
    p[m_te] = predire(finaux["a"]["res"], finaux["a"]["X_test"])
    p[~m_te] = predire(finaux["b"]["res"], finaux["b"]["X_test"])
    return {"finaux": finaux, "pd_test": p, "masque_test": m_te, "masque_train": m_tr}


def coupures_quantiles(df: pl.DataFrame, variables: list[str], quantiles=(0.1, 0.25, 0.5, 0.75, 0.9),
                       chiffres: int = 2) -> dict[str, pl.Expr]:
    """Coupures candidates « variable > seuil » aux quantiles de chaque variable numérique (seuils arrondis à
    `chiffres` chiffres significatifs, doublons retirés)."""
    from utils.binning import _arrondi

    coupures = {}
    for v in variables:
        if not df.schema[v].is_numeric():
            continue
        x = df[v].drop_nulls().cast(pl.Float64)
        for s in sorted({_arrondi(float(x.quantile(q)), chiffres) for q in quantiles}):
            if x.min() < s < x.max():
                coupures[f"{v} > {s:g}"] = pl.col(v) > s
    return coupures


def coupures_arbre_residus(df: pl.DataFrame, pd_ref: np.ndarray, variables: list[str], target: str,
                           profondeur: int = 2, part_min_feuille: float = 0.03, chiffres: int = 2) -> dict[str, pl.Expr]:
    """Coupures candidates tirées d'un arbre de régression ajusté sur les résidus (défaut observé − PD du modèle
    unique) : ses coupures désignent les zones où le modèle unique se trompe le plus, donc les interactions qu'il
    ne capte pas. Chaque coupure de l'arbre est proposée comme coupure globale « variable > seuil »."""
    from sklearn.tree import DecisionTreeRegressor
    from utils.binning import _arrondi

    num = [v for v in variables if df.schema[v].is_numeric()]
    X = df.select([pl.col(v).cast(pl.Float64) for v in num]).to_numpy()
    r = df[target].to_numpy() - pd_ref
    arbre = DecisionTreeRegressor(max_depth=profondeur, min_samples_leaf=int(part_min_feuille * len(r)), random_state=0)
    arbre.fit(X, r)
    t, coupures = arbre.tree_, {}
    for noeud in range(t.node_count):
        if t.children_left[noeud] != t.children_right[noeud]:          # nœud interne
            v, s = num[t.feature[noeud]], _arrondi(float(t.threshold[noeud]), chiffres)
            coupures[f"{v} > {s:g} (arbre)"] = pl.col(v) > s
    return coupures


def cribler_coupures(fit: pl.DataFrame, valid: pl.DataFrame, candidats: dict[str, pl.Expr],
                     reference: tuple[list[str], dict[str, dict]], target: str, defauts_min: int = 200) -> pl.DataFrame:
    """Criblage rapide des coupures : les variables du modèle unique sont conservées, seuls les coefficients
    diffèrent entre les deux sous-populations (test de Chow : rapport de vraisemblance entre deux modèles et le
    modèle unique, emboîtés). Renvoie défauts, Gini de validation global, AIC et p-value du test."""
    from scipy.stats import chi2
    from sklearn.metrics import roc_auc_score

    y_val = valid[target].to_numpy()
    p_ref, res_ref, k_ref = predire_sous_population(fit, valid, *reference, target)
    gini_ref = 2 * roc_auc_score(y_val, p_ref) - 1
    lignes = []
    for nom, expr in candidats.items():
        m_f = fit.select(expr.fill_null(False)).to_series().to_numpy()
        m_v = valid.select(expr.fill_null(False)).to_series().to_numpy()
        d_a, d_b = int(fit.filter(m_f)[target].sum()), int(fit.filter(~m_f)[target].sum())
        ligne = {"coupure": nom, "part_a": float(m_f.mean()), "defauts_a": d_a, "defauts_b": d_b}
        if min(d_a, d_b) < defauts_min:
            continue
        try:
            p_a, res_a, k_a = predire_sous_population(fit.filter(m_f), valid.filter(m_v), *reference, target)
            p_b, res_b, k_b = predire_sous_population(fit.filter(~m_f), valid.filter(~m_v), *reference, target)
        except Exception:
            continue
        p = np.empty(valid.height); p[m_v], p[~m_v] = p_a, p_b
        lr = 2 * (res_a.llf + res_b.llf - res_ref.llf)
        lignes.append(ligne | {"gini_valid": 2 * roc_auc_score(y_val, p) - 1, "gain_gini": 2 * roc_auc_score(y_val, p) - 1 - gini_ref,
                               "aic": float(res_a.aic + res_b.aic), "delta_aic": float(res_a.aic + res_b.aic - res_ref.aic),
                               "p_chow": float(chi2.sf(lr, max(k_a + k_b - k_ref, 1)))})
    return pl.DataFrame(lignes).sort("delta_aic")


# ---------------------------------------------------------------------------
# Autres modèles concurrents (5.2) : LightGBM et CatBoost
# ---------------------------------------------------------------------------

def entrainer_lgbm(X_fit: np.ndarray, y_fit, X_valid: np.ndarray, y_valid, seed: int = 0, **params):
    """LightGBM avec arrêt anticipé sur la validation (AUC). Croissance par feuille, limitée par un nombre
    minimal d'observations par feuille élevé (peu de défauts)."""
    import lightgbm as lgb

    reglages = dict(n_estimators=3000, learning_rate=0.02, num_leaves=31, min_child_samples=200, subsample=0.8,
                    subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, random_state=seed, n_jobs=-1,
                    verbose=-1) | params
    modele = lgb.LGBMClassifier(**reglages)
    modele.fit(X_fit, np.asarray(y_fit), eval_set=[(X_valid, np.asarray(y_valid))], eval_metric="auc",
               callbacks=[lgb.early_stopping(100, verbose=False)])
    return modele


def matrice_catboost(df: pl.DataFrame, variables: list[str]) -> tuple[np.ndarray, list[int]]:
    """Matrice pour CatBoost : variables numériques en valeurs brutes (manquant = NaN), variables qualitatives
    gardées comme catégories (manquant = 'missing'), encodées par CatBoost lui-même (statistiques ordonnées
    sur la cible, sans fuite). Renvoie la matrice (objet) et les indices des colonnes qualitatives."""
    cols, cat_idx = [], []
    for j, v in enumerate(variables):
        if df.schema[v].is_numeric():
            cols.append(df[v].cast(pl.Float64).fill_null(float("nan")).to_numpy())
        else:
            cols.append(df[v].cast(pl.String).fill_null("missing").to_numpy())
            cat_idx.append(j)
    return np.column_stack(cols).astype(object), cat_idx


def entrainer_catboost(X_fit: np.ndarray, y_fit, X_valid: np.ndarray, y_valid, cat_features: list[int],
                       seed: int = 0, **params):
    """CatBoost (arbres symétriques, traitement natif des variables qualitatives) avec arrêt anticipé sur la
    validation (AUC)."""
    from catboost import CatBoostClassifier, Pool

    reglages = dict(iterations=3000, learning_rate=0.03, depth=6, l2_leaf_reg=3.0, eval_metric="AUC",
                    od_type="Iter", od_wait=100, random_seed=seed, verbose=False, thread_count=-1) | params
    modele = CatBoostClassifier(**reglages)
    modele.fit(Pool(X_fit, np.asarray(y_fit), cat_features=cat_features),
               eval_set=Pool(X_valid, np.asarray(y_valid), cat_features=cat_features), use_best_model=True)
    return modele


# ---------------------------------------------------------------------------
# Dépendance : effet d'une variable dans XGBoost comparé à la grille (10.1)
# ---------------------------------------------------------------------------

def valeurs_shap(modele, X: np.ndarray) -> np.ndarray:
    """Valeurs SHAP de XGBoost par observation et par variable (échelle du log-odds), sans la constante."""
    import xgboost as xgb

    return modele.get_booster().predict(xgb.DMatrix(X), pred_contribs=True,
                                        iteration_range=(0, modele.best_iteration + 1))[:, :-1]


def graphique_dependance(df: pl.DataFrame, var: str, shap_var: np.ndarray, regle: dict | None = None,
                         coef_classes: dict[str, float] | None = None, ax=None, n_tranches: int = 40,
                         quantile_max: float = 0.995):
    """SHAP dependence plot agrégé : la variable est découpée en n_tranches de même effectif ; pour chaque
    tranche, contribution SHAP moyenne de `var` au log-odds de XGBoost (ligne) et ses quartiles (bande).
    Aucune observation individuelle n'est tracée. Si la variable est dans la grille : bornes de son découpage
    (pointillés) et effet moyen de la grille par tranche, c'est-à-dire le coefficient de la classe, centré
    (en rouge), sur la même échelle du log-odds. Les sauts de la courbe bleue sont les seuils vus par XGBoost."""
    import matplotlib.pyplot as plt
    from utils.binning import appliquer_regle

    ax = ax or plt.gca()
    x = df[var].cast(pl.Float64).to_numpy()
    ok = ~np.isnan(x)
    haut = np.quantile(x[ok], quantile_max)
    garde = ok & (x <= haut)
    d = pl.DataFrame({"x": x[garde], "s": shap_var[garde]})
    if coef_classes is not None and regle is not None:
        c = appliquer_regle(df, var, regle)["classe"].replace_strict(coef_classes, default=0.0,
                                                                     return_dtype=pl.Float64).to_numpy()
        d = d.with_columns(c=pl.Series(c[garde] - c[ok].mean()))
    d = d.with_columns(tranche=pl.col("x").rank("ordinal").mul(n_tranches).floordiv(pl.len() + 1))
    aggs = [pl.col("x").median().alias("x"), pl.col("s").mean().alias("moyenne"),
            pl.col("s").quantile(0.25).alias("q1"), pl.col("s").quantile(0.75).alias("q3")]
    if "c" in d.columns:
        aggs.append(pl.col("c").mean().alias("grille"))
    t = d.group_by("tranche").agg(aggs).sort("x")
    ax.fill_between(t["x"], t["q1"], t["q3"], alpha=0.25, label="XGBoost : quartiles")
    ax.plot(t["x"], t["moyenne"], marker=".", label="XGBoost : effet moyen (SHAP)")
    if "grille" in t.columns:
        ax.plot(t["x"], t["grille"], color="red", linewidth=2, label="grille (coefficient centré)")
    if regle is not None and regle["type"] == "num":
        for b in regle["bornes"]:
            if np.nanmin(x) < b <= haut:
                ax.axvline(b, color="grey", linestyle="--", linewidth=1)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set(title=var, xlabel=f"{var} (médiane de chaque tranche)", ylabel="effet sur le log-odds")
    return ax