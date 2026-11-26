"""Chapitre 5 — Modèles de scoring (risk differentiation).

- découpage apprentissage / test par identifiant (cours 1.5) ;
- régression logistique sur dummies, modalité 1 (la moins risquée) en référence (5.1) ;
- tests de validité globaux, par variable et par modalité (5.1) ;
- métriques d'ordonnancement : AUC, Gini, KS, IC de Hanley & McNeil, test de DeLong, lift (5.3) ;
- stepwise par ordre de liaison avec fusion des modalités non conformes (5.5) ;
- modèle concurrent : gradient boosting (5.2).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy import stats


# ---------------------------------------------------------------------------
# 1.5 Découpage apprentissage / test
# ---------------------------------------------------------------------------

def decoupage_apprentissage_test(df: pl.DataFrame, id_col: str, target: str, part_test: float = 0.30,
                                 seed: int = 42, nom: str = "echantillon") -> pl.DataFrame:
    """Ajoute la colonne `nom` ("apprentissage" / "test"). Le tirage se fait par identifiant (cours 1.5.5) :
    toutes les observations d'un même dossier sont du même côté. Stratifié sur la cible du dossier
    (défaut au moins une fois), pour garder le même taux de défaut des deux côtés."""
    from sklearn.model_selection import train_test_split
    ids = df.group_by(id_col).agg(_y=pl.col(target).max()).sort(id_col)
    _, ids_test = train_test_split(ids[id_col].to_numpy(), test_size=part_test,
                                   stratify=ids["_y"].to_numpy(), random_state=seed)
    return df.with_columns(
        pl.when(pl.col(id_col).is_in(pl.Series(ids_test))).then(pl.lit("test"))
        .otherwise(pl.lit("apprentissage")).alias(nom)
    )


def bilan_echantillons(df: pl.DataFrame, target: str, nom: str = "echantillon") -> pl.DataFrame:
    return (df.group_by(nom).agg(n=pl.len(), n_defaut=pl.col(target).sum(), taux_defaut=pl.col(target).mean())
            .with_columns(part=pl.col("n") / pl.col("n").sum()).sort(nom))


# ---------------------------------------------------------------------------
# 5.3 Métriques d'ordonnancement
# ---------------------------------------------------------------------------

def auc_hanley_mcneil(auc: float, n0: int, n1: int) -> float:
    """Écart-type de l'AUC (Hanley & McNeil, 1982)."""
    q1, q2 = auc / (2 - auc), 2 * auc ** 2 / (1 + auc)
    var = (auc * (1 - auc) + (n1 - 1) * (q1 - auc ** 2) + (n0 - 1) * (q2 - auc ** 2)) / (n0 * n1)
    return float(np.sqrt(max(var, 0)))


def ks_stat(y, score) -> float:
    y, score = np.asarray(y), np.asarray(score)
    return float(stats.ks_2samp(score[y == 1], score[y == 0]).statistic)


def performance(y, score, nom: str = "") -> dict:
    """AUC, Gini (= 2 AUC − 1), IC95 % du Gini (Hanley & McNeil), KS. `score` croît avec le risque."""
    from sklearn.metrics import roc_auc_score
    y, score = np.asarray(y), np.asarray(score)
    n1, n0 = int(y.sum()), int(len(y) - y.sum())
    auc = float(roc_auc_score(y, score))
    se = auc_hanley_mcneil(auc, n0, n1)
    return {"modele": nom, "n": len(y), "n_defaut": n1, "auc": auc, "gini": 2 * auc - 1,
            "gini_ic_bas": 2 * (auc - 1.96 * se) - 1, "gini_ic_haut": 2 * (auc + 1.96 * se) - 1,
            "ks": ks_stat(y, score)}


def _midrank(x: np.ndarray) -> np.ndarray:
    return stats.rankdata(x, method="average")


def test_delong(y, score_a, score_b) -> dict:
    """Test de DeLong (1988) de comparaison de deux AUC estimées sur les mêmes individus (cours 5.3.1.4)."""
    y = np.asarray(y)
    a, b = np.asarray(score_a), np.asarray(score_b)
    pos, neg = y == 1, y == 0
    m, n = pos.sum(), neg.sum()
    aucs, v10, v01 = [], [], []
    for s in (a, b):
        r_all = _midrank(s)
        r_pos, r_neg = _midrank(s[pos]), _midrank(s[neg])
        auc = (r_all[pos].sum() - m * (m + 1) / 2) / (m * n)
        aucs.append(auc)
        v10.append((r_all[pos] - r_pos) / n)
        v01.append(1 - (r_all[neg] - r_neg) / m)
    s10, s01 = np.cov(np.vstack(v10)), np.cov(np.vstack(v01))
    cov = s10 / m + s01 / n
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    z = (aucs[0] - aucs[1]) / np.sqrt(var) if var > 0 else 0.0
    return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]), "ecart_gini": float(2 * (aucs[0] - aucs[1])),
            "z": float(z), "pvalue": float(2 * stats.norm.sf(abs(z)))}


def table_lift(y, score, n_classes: int = 10) -> pl.DataFrame:
    """Lift par décile de score, du plus risqué (décile 1) au moins risqué (cours 5.3.5)."""
    d = pl.DataFrame({"y": np.asarray(y), "s": np.asarray(score)}).sort("s", descending=True)
    d = d.with_columns(decile=(pl.int_range(pl.len()) * n_classes // pl.len() + 1))
    tau = d["y"].mean()
    return (d.group_by("decile").agg(n=pl.len(), n_defaut=pl.col("y").sum(), taux_defaut=pl.col("y").mean())
            .sort("decile")
            .with_columns(lift=pl.col("taux_defaut") / tau,
                          part_defauts_captes=pl.col("n_defaut").cum_sum() / pl.col("n_defaut").sum(),
                          lift_cumule=(pl.col("n_defaut").cum_sum() / pl.col("n").cum_sum()) / tau))


def plot_roc_cap(courbes: dict[str, tuple], titre: str = ""):
    """ROC et CAP de plusieurs scores : courbes = {nom: (y, score)}."""
    from sklearn.metrics import roc_curve
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for nom, (y, s) in courbes.items():
        y, s = np.asarray(y), np.asarray(s)
        fpr, tpr, _ = roc_curve(y, s)
        g = performance(y, s)["gini"]
        ax1.plot(fpr, tpr, label=f"{nom} (Gini {g:.1%})")
        ordre = np.argsort(-s)
        ax2.plot(np.arange(1, len(y) + 1) / len(y), np.cumsum(y[ordre]) / y.sum(), label=nom)
    tau = np.asarray(next(iter(courbes.values()))[0]).mean()
    ax1.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax2.plot([0, 1], [0, 1], "k--", lw=0.8, label="aléatoire")
    ax2.plot([0, tau, 1], [0, 1, 1], color="grey", lw=0.8, label="parfait")
    ax1.set(xlabel="Taux de faux positifs (bons)", ylabel="Taux de vrais positifs (défauts)", title="Courbe ROC")
    ax2.set(xlabel="Part de la population (du plus risqué au moins risqué)", ylabel="Part des défauts captés",
            title="Courbe CAP")
    for ax in (ax1, ax2):
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle(titre)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 5.1 Régression logistique sur dummies
# ---------------------------------------------------------------------------

def modalites_observees(df: pl.DataFrame, variables: list[str]) -> dict[str, list[int]]:
    """Classes autres que la référence (classe 1) observées pour chaque variable."""
    return {v: sorted(int(k) for k in df[v].drop_nulls().unique().to_list() if int(k) != 1) for v in variables}


def matrice_dummies(df: pl.DataFrame, variables: list[str], modalites: dict[str, list[int]]) -> pl.DataFrame:
    """Une indicatrice par classe non-référence (modalité 1 = référence, omise)."""
    cols = [(pl.col(v) == k).cast(pl.Float64).alias(f"{v}_{k}") for v in variables for k in modalites[v]]
    return df.select(cols) if cols else pl.DataFrame()


@dataclass
class ModeleLogit:
    variables: list[str]
    modalites: dict[str, list[int]]
    params: np.ndarray
    bse: np.ndarray
    pvalues: np.ndarray
    llf: float
    llnull: float
    n: int
    convergence: bool = True
    noms: list[str] = field(default_factory=list)

    def score_lineaire(self, df: pl.DataFrame) -> np.ndarray:
        X = matrice_dummies(df, self.variables, self.modalites)
        s = np.full(df.height, self.params[0])
        return s + (X.to_numpy() @ self.params[1:] if X.width else 0)

    def proba(self, df: pl.DataFrame) -> np.ndarray:
        return 1 / (1 + np.exp(-self.score_lineaire(df)))

    @property
    def beta0(self) -> float:
        return float(self.params[0])

    def coefficients(self) -> pl.DataFrame:
        """Une ligne par modalité, référence comprise (β = 0) : β, écart-type, p-value de Wald, odds ratio et IC95 %."""
        rows, i = [], 1
        for v in self.variables:
            rows.append({"variable": v, "classe": 1, "beta": 0.0, "se": None, "pvalue": None})
            for k in self.modalites[v]:
                rows.append({"variable": v, "classe": k, "beta": float(self.params[i]), "se": float(self.bse[i]),
                             "pvalue": float(self.pvalues[i])})
                i += 1
        schema = {"variable": pl.String, "classe": pl.Int64, "beta": pl.Float64, "se": pl.Float64, "pvalue": pl.Float64}
        return pl.DataFrame(rows, schema=schema).with_columns(
            odds_ratio=pl.col("beta").exp(),
            or_ic_bas=(pl.col("beta") - 1.96 * pl.col("se")).exp(),
            or_ic_haut=(pl.col("beta") + 1.96 * pl.col("se")).exp(),
        )


def ajuster_logit(df: pl.DataFrame, variables: list[str], target: str,
                  modalites: dict[str, list[int]] | None = None) -> ModeleLogit:
    """Régression logistique par maximum de vraisemblance sur les dummies des variables discrétisées."""
    import statsmodels.api as sm
    modalites = modalites or modalites_observees(df, variables)
    X = matrice_dummies(df, variables, modalites)
    Xn = np.column_stack([np.ones(df.height), X.to_numpy()]) if X.width else np.ones((df.height, 1))
    y = df[target].to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            res = sm.Logit(y, Xn).fit(disp=0, maxiter=200)
            ok = bool(res.mle_retvals.get("converged", True)) and np.all(np.isfinite(res.bse))
        except Exception:
            res = sm.Logit(y, Xn).fit(disp=0, method="bfgs", maxiter=500)
            ok = False
    return ModeleLogit(variables=list(variables), modalites=modalites, params=np.asarray(res.params),
                       bse=np.asarray(res.bse), pvalues=np.asarray(res.pvalues), llf=float(res.llf),
                       llnull=float(res.llnull), n=df.height, convergence=ok, noms=["const", *X.columns])


def vif_variables(df: pl.DataFrame, variables: list[str]) -> dict[str, float]:
    """VIF de chaque variable, calculé sur le numéro de classe (ordinal) : multicolinéarité entre variables du modèle."""
    if len(variables) < 2:
        return {v: 1.0 for v in variables}
    X = df.select(pl.col(variables).cast(pl.Float64)).to_numpy()
    X = (X - X.mean(axis=0)) / np.where(X.std(axis=0) > 0, X.std(axis=0), 1)
    R = np.corrcoef(X, rowvar=False)
    try:
        inv = np.linalg.inv(R)
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(R)
    return {v: float(inv[i, i]) for i, v in enumerate(variables)}


def tests_modele(modele: ModeleLogit, df: pl.DataFrame, target: str, alpha: float = 0.05,
                 seuil_vif: float = 5, test_variables: bool = True) -> tuple[dict, pl.DataFrame, pl.DataFrame]:
    """Tests de validité du cours 5.1.
    - global : LR test du modèle contre le modèle nul ;
    - par variable : LR test de la variable (modèle sans elle), VIF, monotonie des β ;
    - par modalité : Wald (p < alpha), IC95 % de l'odds ratio excluant 1, signe positif (classe plus risquée
      que la référence).
    Renvoie (global, par variable, par modalité)."""
    p = len(modele.params) - 1
    lr = 2 * (modele.llf - modele.llnull)
    glob = {"lr": lr, "ddl": p, "pvalue": float(stats.chi2.sf(lr, p)) if p else None, "convergence": modele.convergence,
            "pseudo_r2": 1 - modele.llf / modele.llnull}
    coefs = modele.coefficients().with_columns(
        significatif=pl.when(pl.col("classe") == 1).then(None).otherwise(pl.col("pvalue") < alpha),
        ic_exclut_1=pl.when(pl.col("classe") == 1).then(None).otherwise(pl.col("or_ic_bas") > 1),
        signe_ok=pl.when(pl.col("classe") == 1).then(None).otherwise(pl.col("beta") > 0),
    )
    vif = vif_variables(df, modele.variables)
    rows = []
    for v in modele.variables:
        b = coefs.filter(pl.col("variable") == v).sort("classe")["beta"].to_numpy()
        lr_v, p_v = None, None
        if test_variables:
            autres = [w for w in modele.variables if w != v]
            m0 = ajuster_logit(df, autres, target, {w: modele.modalites[w] for w in autres})
            lr_v = 2 * (modele.llf - m0.llf)
            p_v = float(stats.chi2.sf(lr_v, len(modele.modalites[v])))
        c = coefs.filter((pl.col("variable") == v) & (pl.col("classe") != 1))
        rows.append({"variable": v, "n_classes": len(b), "lr": lr_v, "pvalue_lr": p_v,
                     "monotone": bool(np.all(np.diff(b) > 0)), "vif": vif[v],
                     "modalites_significatives": bool(c["significatif"].all()),
                     "ic_exclut_1": bool(c["ic_exclut_1"].all()), "signes_ok": bool(c["signe_ok"].all())})
    par_var = pl.DataFrame(rows).with_columns(
        conforme=pl.col("monotone") & pl.col("modalites_significatives") & pl.col("ic_exclut_1")
        & pl.col("signes_ok") & (pl.col("vif") <= seuil_vif)
        & (pl.col("pvalue_lr").is_null() | (pl.col("pvalue_lr") < alpha))
    )
    return glob, par_var, coefs


# ---------------------------------------------------------------------------
# Fusions de classes (retour au chapitre 4)
# ---------------------------------------------------------------------------

def appliquer_fusions(df: pl.DataFrame, fusions: dict[str, dict[int, int]]) -> pl.DataFrame:
    """Renumérote les classes selon fusions = {variable: {ancienne classe: nouvelle classe}}."""
    return df.with_columns(
        pl.col(v).replace_strict({int(a): int(b) for a, b in m.items()}, default=pl.col(v)).cast(pl.Int32).alias(v)
        for v, m in fusions.items() if m
    )


def _fusion_avec_precedente(classes: list[int], k: int) -> dict[int, int]:
    """Fusionne la classe k avec la classe k−1, puis renumérote 1..N."""
    cible = {c: (c if c != k else k - 1) for c in classes}
    nouvelles = sorted(set(cible.values()))
    rang = {c: i + 1 for i, c in enumerate(nouvelles)}
    return {c: rang[cible[c]] for c in classes}


def _composer(m1: dict[int, int], m2: dict[int, int]) -> dict[int, int]:
    return {a: m2.get(b, b) for a, b in m1.items()}


def _classe_a_fusionner(coefs_v: pl.DataFrame, alpha: float) -> int | None:
    """Première classe non conforme d'une variable (rupture de monotonie, signe, non-significativité)."""
    c = coefs_v.sort("classe")
    b, p, lo = c["beta"].to_list(), c["pvalue"].to_list(), c["or_ic_bas"].to_list()
    for i in range(1, len(b)):
        if b[i] <= b[i - 1] or b[i] <= 0:
            return c["classe"][i]
    for i in range(1, len(b)):
        if p[i] is None or p[i] >= alpha or lo[i] is None or lo[i] <= 1:
            return c["classe"][i]
    return None


# ---------------------------------------------------------------------------
# 5.5 Stepwise par ordre de liaison
# ---------------------------------------------------------------------------

@dataclass
class ResultatStepwise:
    variables: list[str]
    fusions: dict[str, dict[int, int]]
    journal: pl.DataFrame
    modele: ModeleLogit


def stepwise_liaison(df_app: pl.DataFrame, candidates: list[str], target: str, df_test: pl.DataFrame | None = None,
                     alpha: float = 0.05, gain_min: float = 0.005, seuil_vif: float = 5, fusion: bool = True,
                     max_variables: int | None = None, fusions_initiales: dict[str, dict[int, int]] | None = None,
                     fusion_retenues: bool = False) -> ResultatStepwise:
    """Construction pas à pas du cours 5.5.
    Les candidates, déjà ordonnées par IV décroissant, sont ajoutées une à une (toutes leurs modalités à la fois).
    Après chaque ajout, tous les tests de 5.1 sont vérifiés sur le modèle complet :
    - si seule la nouvelle variable est non conforme (modalité non significative, IC de l'OR contenant 1,
      rupture de monotonie, signe), sa classe fautive est fusionnée avec la classe précédente (retour au ch. 4)
      et le modèle est réestimé, jusqu'à conformité ou jusqu'à une seule classe (variable écartée) ;
    - si l'ajout rend non conforme une variable déjà retenue, ou dépasse le seuil de VIF, la variable est écartée ;
    - la variable n'est retenue que si son LR test est significatif et si elle apporte un gain de Gini
      d'apprentissage d'au moins gain_min.
    Le Gini de test est suivi à chaque étape pour information (il ne sert pas à la sélection).

    Options :
    - fusions_initiales = {variable: {classe: nouvelle classe}} : regroupements décidés avant le stepwise
      (ex. deux classes aux β quasi identiques), appliqués à l'apprentissage et au test ;
    - fusion_retenues = True : si l'ajout rend non conforme une variable déjà retenue, la classe fautive de
      celle-ci est fusionnée avec sa voisine (au lieu d'écarter la candidate), tant qu'il lui reste 2 classes."""
    retenues: list[str] = []
    fusions: dict[str, dict[int, int]] = {}
    if fusions_initiales:
        fusions = {v: {int(a): int(b) for a, b in m.items()} for v, m in fusions_initiales.items()}
        df_app = appliquer_fusions(df_app, fusions)
    journal = []
    courant = ajuster_logit(df_app, [], target)
    gini_prec = 0.0
    y_app = df_app[target].to_numpy()
    for etape, v in enumerate(candidates, start=1):
        if max_variables and len(retenues) >= max_variables:
            break
        classes = sorted(int(k) for k in df_app[v].drop_nulls().unique().to_list())
        mapping = {c: c for c in classes}
        d = df_app
        motif, n_fusions, m = None, 0, None
        fus_ret: dict[str, dict[int, int]] = {}   # fusions provisoires des variables déjà retenues (numérotation courante)
        while True:
            if len(set(mapping.values())) < 2:
                motif = "une seule classe après fusions"
                break
            d = appliquer_fusions(df_app, {**fus_ret, v: mapping})
            m = ajuster_logit(d, retenues + [v], target)
            if not m.convergence:
                motif = "non convergence (quasi-séparation)"
                break
            _, par_var, coefs = tests_modele(m, d, target, alpha, seuil_vif, test_variables=False)
            pb_autres = par_var.filter((pl.col("variable") != v) & ~(
                pl.col("monotone") & pl.col("modalites_significatives") & pl.col("ic_exclut_1") & pl.col("signes_ok")))
            if pb_autres.height:
                u = pb_autres["variable"][0]
                ku = _classe_a_fusionner(coefs.filter(pl.col("variable") == u), alpha)
                classes_u = sorted(int(c) for c in d[u].drop_nulls().unique().to_list())
                if fusion_retenues and ku is not None and len(classes_u) > 2:
                    base_u = fus_ret.get(u, {c: c for c in sorted(int(c) for c in df_app[u].drop_nulls().unique().to_list())})
                    fus_ret[u] = _composer(base_u, _fusion_avec_precedente(classes_u, ku))
                    n_fusions += 1
                    continue
                motif = f"rend non conforme {', '.join(pb_autres['variable'])}"
                break
            k = _classe_a_fusionner(coefs.filter(pl.col("variable") == v), alpha)
            if k is None:
                break
            if not fusion:
                motif = "modalité non conforme (fusion désactivée)"
                break
            mapping = _composer(mapping, _fusion_avec_precedente(sorted(set(mapping.values())), k))
            n_fusions += 1
        gini_app = gini_test = p_lr = vif_max = None
        if motif is None:
            p_lr = float(stats.chi2.sf(2 * (m.llf - courant.llf), len(m.modalites[v])))
            vif_max = max(vif_variables(d, retenues + [v]).values())
            gini_app = performance(y_app, m.score_lineaire(d))["gini"]
            if p_lr >= alpha:
                motif = f"LR test non significatif (p = {p_lr:.3f})"
            elif vif_max > seuil_vif:
                motif = f"multicolinéarité (VIF max = {vif_max:.1f})"
            elif gini_app - gini_prec < gain_min:
                motif = f"gain de Gini insuffisant (+{gini_app - gini_prec:.2%})"
        # fusions complètes (depuis les classes d'origine) si la candidate est retenue
        fus_new = dict(fusions)
        for u, mu in fus_ret.items():
            fus_new[u] = _composer(fusions.get(u, {c: c for c in mu}), mu)
        fus_new[v] = _composer(fusions[v], mapping) if v in fusions else mapping
        if df_test is not None and m is not None and motif is None:
            dt = appliquer_fusions(df_test, fus_new)
            gini_test = performance(dt[target].to_numpy(), m.score_lineaire(dt))["gini"]
        retenue = motif is None
        journal.append({"etape": etape, "variable": v, "statut": "retenue" if retenue else "écartée",
                        "motif": motif or "", "n_classes_initial": len(classes),
                        "n_classes_final": len(set(mapping.values())), "n_fusions": n_fusions,
                        "pvalue_lr": p_lr, "vif_max": vif_max, "gini_app": gini_app,
                        "gain_gini": None if gini_app is None else gini_app - gini_prec, "gini_test": gini_test})
        if retenue:
            retenues.append(v)
            fusions = {u: mu for u, mu in fus_new.items() if any(a != b for a, b in mu.items())}
            df_app = d
            courant, gini_prec = m, gini_app
    schema = {"etape": pl.Int64, "variable": pl.String, "statut": pl.String, "motif": pl.String,
              "n_classes_initial": pl.Int64, "n_classes_final": pl.Int64, "n_fusions": pl.Int64,
              "pvalue_lr": pl.Float64, "vif_max": pl.Float64, "gini_app": pl.Float64, "gain_gini": pl.Float64,
              "gini_test": pl.Float64}
    return ResultatStepwise(retenues, fusions, pl.DataFrame(journal, schema=schema), courant)


def plot_journal(journal: pl.DataFrame):
    """Gini d'apprentissage et de test au fil des variables retenues (courbe de gain marginal, 5.5)."""
    j = journal.filter(pl.col("statut") == "retenue")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(1, j.height + 1)
    ax.plot(x, j["gini_app"], marker="o", label="apprentissage")
    if j["gini_test"].null_count() < j.height:
        ax.plot(x, j["gini_test"], marker="s", label="test")
    ax.set_xticks(x, j["variable"].to_list(), rotation=45, ha="right", fontsize=8)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set(ylabel="Gini", title="Gini du modèle au fil des ajouts de variables")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_coefficients(coefs: pl.DataFrame, n_cols: int = 3):
    """Odds ratios et IC95 % par classe : la monotonie doit se lire directement (5.1)."""
    variables = coefs["variable"].unique(maintain_order=True).to_list()
    n_rows = int(np.ceil(len(variables) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 3.2 * n_rows), squeeze=False)
    for ax, v in zip(axes.flat, variables):
        c = coefs.filter(pl.col("variable") == v).sort("classe")
        x, orr = c["classe"].to_numpy(), c["odds_ratio"].to_numpy()
        lo = c["or_ic_bas"].fill_null(1).to_numpy()
        hi = c["or_ic_haut"].fill_null(1).to_numpy()
        ax.errorbar(x, orr, yerr=[orr - lo, hi - orr], fmt="o-", capsize=3)
        ax.axhline(1, color="grey", ls="--", lw=0.8)
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_title(v, fontsize=9)
        ax.set_xlabel("classe")
    for ax in list(axes.flat)[len(variables):]:
        ax.axis("off")
    fig.suptitle("Odds ratios par classe (référence = classe 1) et IC95 %")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 5.2 Modèle concurrent : gradient boosting
# ---------------------------------------------------------------------------

def _encoder(df: pl.DataFrame, variables: list[str], codes: dict | None = None):
    """Variables numériques telles quelles (manquants = NaN, gérés nativement) ; qualitatives codées en entiers
    selon les modalités vues en apprentissage (modalité nouvelle = NaN)."""
    codes = codes if codes is not None else {
        v: {m: i for i, m in enumerate(df[v].cast(pl.String).drop_nulls().unique().sort().to_list())}
        for v in variables if not df.schema[v].is_numeric()}
    cols = [pl.col(v).cast(pl.String).replace_strict(codes[v], default=None, return_dtype=pl.Float64).alias(v)
            if v in codes else pl.col(v).cast(pl.Float64).alias(v) for v in variables]
    return df.select(cols).to_numpy(), codes


def gradient_boosting(df_app: pl.DataFrame, variables: list[str], target: str, seed: int = 42, **params):
    """Gradient boosting (HistGradientBoostingClassifier) sur les variables brutes préparées (5.2).
    Régularisation par défaut adaptée à un faible nombre de défauts : arbres peu profonds, feuilles d'au moins
    200 dossiers, arrêt précoce sur un échantillon de validation interne."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    X, codes = _encoder(df_app, variables)
    categ = [v in codes for v in variables]
    defaut = dict(learning_rate=0.05, max_iter=500, max_depth=3, min_samples_leaf=200, l2_regularization=1.0,
                  early_stopping=True, validation_fraction=0.2, n_iter_no_change=30, scoring="roc_auc",
                  random_state=seed)
    gb = HistGradientBoostingClassifier(categorical_features=categ if any(categ) else None, **(defaut | params))
    gb.fit(X, df_app[target].to_numpy())
    gb.codes_, gb.variables_ = codes, list(variables)
    return gb


def predire_gb(gb, df: pl.DataFrame) -> np.ndarray:
    X, _ = _encoder(df, gb.variables_, gb.codes_)
    return gb.predict_proba(X)[:, 1]


# ---------------------------------------------------------------------------
# 5.4 Régularisation : passe exploratoire
# ---------------------------------------------------------------------------

def lasso_exploratoire(df: pl.DataFrame, variables: list[str], target: str, n_c: int = 15, cv: int = 5,
                       seed: int = 42) -> tuple[pl.DataFrame, float]:
    """LASSO logistique sur les dummies (modalité 1 = référence), pénalité choisie par validation croisée sur l'AUC
    (cours 5.4). Outil de dégrossissage seulement : renvoie, par variable, le nombre de modalités au coefficient non
    nul (0 = variable écartée par la pénalisation) et la valeur de C retenue."""
    from sklearn.linear_model import LogisticRegressionCV
    modalites = modalites_observees(df, variables)
    X = matrice_dummies(df, variables, modalites)
    lasso = LogisticRegressionCV(Cs=n_c, cv=cv, penalty="l1", solver="liblinear", scoring="roc_auc",
                                 random_state=seed, max_iter=500)
    lasso.fit(X.to_numpy(), df[target].to_numpy())
    coef = dict(zip(X.columns, lasso.coef_[0]))
    rows = [{"variable": v, "n_modalites": len(modalites[v]),
             "n_non_nuls": sum(abs(coef[f"{v}_{k}"]) > 1e-8 for k in modalites[v])} for v in variables]
    return pl.DataFrame(rows).with_columns(retenue_lasso=pl.col("n_non_nuls") > 0), float(lasso.C_[0])


# ---------------------------------------------------------------------------
# 5.2 bis Diagnostic du modèle concurrent : d'où vient l'écart de Gini ?
# ---------------------------------------------------------------------------

def _plis(y, n_plis: int = 5, seed: int = 0):
    from sklearn.model_selection import StratifiedKFold
    y = np.asarray(y)
    return list(StratifiedKFold(n_plis, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))


def gini_cv_logit(df: pl.DataFrame, variables: list[str], target: str, n_plis: int = 5, seed: int = 0) -> float:
    """Gini de validation croisée (n_plis stratifiés) de la logistique sur les variables discrétisées : chaque dossier
    est noté par un modèle estimé sans lui. Plus stable qu'un seul échantillon de test quand les défauts sont rares."""
    from sklearn.metrics import roc_auc_score
    y = df[target].to_numpy()
    s = np.zeros(len(y))
    for tr, va in _plis(y, n_plis, seed):
        s[va] = ajuster_logit(df[tr], variables, target).score_lineaire(df[va])
    return float(2 * roc_auc_score(y, s) - 1)


def gini_cv_gb(df: pl.DataFrame, variables: list[str], target: str, n_plis: int = 5, seed: int = 0, **params) -> float:
    """Gini de validation croisée du gradient boosting (mêmes plis que gini_cv_logit à seed égal)."""
    from sklearn.metrics import roc_auc_score
    y = df[target].to_numpy()
    p = np.zeros(len(y))
    for tr, va in _plis(y, n_plis, seed):
        p[va] = predire_gb(gradient_boosting(df[tr], variables, target, **params), df[va])
    return float(2 * roc_auc_score(y, p) - 1)


def importance_permutation_gb(gb, df: pl.DataFrame, target: str, n_repeats: int = 5, seed: int = 0) -> pl.DataFrame:
    """Importance par permutation (perte d'AUC quand la variable est mélangée) du gradient boosting."""
    from sklearn.inspection import permutation_importance
    X, _ = _encoder(df, gb.variables_, gb.codes_)
    pi = permutation_importance(gb, X, df[target].to_numpy(), scoring="roc_auc", n_repeats=n_repeats, random_state=seed)
    return pl.DataFrame({"variable": gb.variables_, "importance_auc": pi.importances_mean,
                         "ecart_type": pi.importances_std}).sort("importance_auc", descending=True)


def ajouts_un_a_un(df_app: pl.DataFrame, df_test: pl.DataFrame, variables: list[str], candidates: list[str], target: str,
                   n_plis: int = 5, seed: int = 0, alpha: float = 0.05, seuil_vif: float = 5) -> pl.DataFrame:
    """Ajoute chaque candidate, seule, au modèle `variables` : Gini de validation croisée (gain par rapport au modèle
    de départ), Gini de test pour information, et conformité de toutes les variables aux tests du cours 5.1."""
    from sklearn.metrics import roc_auc_score
    base = gini_cv_logit(df_app, variables, target, n_plis, seed)
    rows = []
    for v in candidates:
        m = ajuster_logit(df_app, variables + [v], target)
        _, tv, _ = tests_modele(m, df_app, target, alpha=alpha, seuil_vif=seuil_vif)
        rows.append({"variable": v, "gini_cv": gini_cv_logit(df_app, variables + [v], target, n_plis, seed),
                     "gini_test": float(2 * roc_auc_score(df_test[target].to_numpy(), m.score_lineaire(df_test)) - 1),
                     "conforme": bool(tv["conforme"].all()),
                     "variables_non_conformes": ", ".join(tv.filter(~pl.col("conforme"))["variable"].to_list())})
    return (pl.DataFrame(rows).with_columns(gain_cv=pl.col("gini_cv") - base)
            .sort("gain_cv", descending=True))
