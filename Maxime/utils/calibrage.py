"""
Quantification du risque (chapitre 8) : calibrage des PD par classe homogène de risque, vérification,
marge de prudence (MoC) et échelle maître. Fonctions indépendantes du segment : elles s'appliquent aux CHR
de tous les segments rassemblées au format commun (segment, score, pd_score, chr).
"""
from __future__ import annotations

import numpy as np
import polars as pl


# ---------------------------------------------------------------------------
# Méthodes de calibrage (8.4, 8.5)
# ---------------------------------------------------------------------------

def pd_par_methode(df: pl.DataFrame, target: str, date: str, score: str = "score", pd_col: str = "pd_score",
                   chr_col: str = "chr", poids_prior: float = 1000.0) -> pl.DataFrame:
    """PD de chaque CHR selon plusieurs méthodes, estimées sur `df` :
    - frequence : taux de défaut observé de la classe ;
    - moyenne_mensuelle : moyenne des taux de défaut mensuels (chaque mois pèse autant : approximation
      d'une moyenne dans le temps, 8.4) ;
    - logistique : recalibration logistique du score (défaut ~ score), PD moyenne de la classe ;
    - isotone : régression isotone (décroissante) du défaut sur le score, PD moyenne de la classe ;
    - bayesien : crédibilité Bêta-binomiale ; la PD du modèle (pd_score moyen de la classe) sert d'a priori
      avec un poids équivalent à `poids_prior` observations : PD = (défauts + poids × PD_modèle) / (n + poids).
      Une classe avec peu de défauts est rapprochée de la PD du modèle."""
    import statsmodels.api as sm
    from sklearn.isotonic import IsotonicRegression

    y, s = df[target].to_numpy().astype(float), df[score].to_numpy().astype(float)
    logit = sm.Logit(y, sm.add_constant(s)).fit(disp=0)
    iso = IsotonicRegression(increasing=False, out_of_bounds="clip").fit(s, y)
    d = df.select(chr_col, target, date, pd_col).with_columns(
        p_logit=pl.Series(np.asarray(logit.predict(sm.add_constant(s)))), p_iso=pl.Series(iso.predict(s)))
    mensuel = d.group_by(chr_col, date).agg(pl.col(target).mean().alias("t")).group_by(chr_col).agg(
        moyenne_mensuelle=pl.col("t").mean())
    t = d.group_by(chr_col).agg(
        n=pl.len(), n_defaut=pl.col(target).sum(), frequence=pl.col(target).mean(),
        pd_modele=pl.col(pd_col).mean(), logistique=pl.col("p_logit").mean(), isotone=pl.col("p_iso").mean(),
    ).join(mensuel, on=chr_col).sort(chr_col)
    return t.with_columns(bayesien=(pl.col("n_defaut") + poids_prior * pl.col("pd_modele")) / (pl.col("n") + poids_prior)
                          ).select(chr_col, "n", "n_defaut", "pd_modele", "frequence", "moyenne_mensuelle",
                                   "logistique", "isotone", "bayesien")


METHODES_CALIBRAGE = ("frequence", "moyenne_mensuelle", "logistique", "isotone", "bayesien")


def appliquer_pd(df: pl.DataFrame, table: pl.DataFrame, colonne: str, chr_col: str = "chr") -> np.ndarray:
    """PD de chaque ligne : PD de sa classe selon la colonne choisie de la table de calibrage."""
    return df[chr_col].replace_strict(dict(zip(table[chr_col], table[colonne])), return_dtype=pl.Float64).to_numpy()


# ---------------------------------------------------------------------------
# Vérification du calibrage (8.6)
# ---------------------------------------------------------------------------

def brier_murphy(y, pd_pred, classes) -> dict:
    """Score de Brier = moyenne de (PD − défaut)² (plus faible = meilleur) et décomposition de Murphy, exacte
    quand la PD est constante par classe : Brier = fiabilité − résolution + incertitude.
    - fiabilité : écart entre PD et taux observé par classe (0 = calibrage parfait) ;
    - résolution : écart des taux observés des classes au taux moyen (élevée = classes bien différenciées) ;
    - incertitude : o(1 − o), propre à l'échantillon, ne dépend pas du modèle."""
    d = pl.DataFrame({"y": np.asarray(y, dtype=float), "p": np.asarray(pd_pred, dtype=float), "c": np.asarray(classes)})
    o, N = d["y"].mean(), d.height
    g = d.group_by("c").agg(n=pl.len(), pk=pl.col("p").mean(), ok=pl.col("y").mean())
    fiab = float((g["n"] * (g["pk"] - g["ok"]) ** 2).sum() / N)
    reso = float((g["n"] * (g["ok"] - o) ** 2).sum() / N)
    return {"brier": float(((d["p"] - d["y"]) ** 2).mean()), "fiabilite": fiab, "resolution": reso,
            "incertitude": float(o * (1 - o))}


def test_jeffreys(n, n_defaut, pd_pred) -> np.ndarray:
    """Test de Jeffreys (validation des PD, usage BCE) : p-value unilatérale = P(PD réelle ≤ PD estimée) sous la loi
    a posteriori Bêta(défauts + ½, n − défauts + ½). Une p-value faible (< 5 %) indique une PD sous-estimée."""
    from scipy.stats import beta

    n, d = np.asarray(n, dtype=float), np.asarray(n_defaut, dtype=float)
    return beta.cdf(np.asarray(pd_pred, dtype=float), d + 0.5, n - d + 0.5)


def verifier_calibrage(df: pl.DataFrame, table: pl.DataFrame, target: str, chr_col: str = "chr",
                       methodes: tuple[str, ...] = METHODES_CALIBRAGE) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Vérification de chaque méthode sur `df` (le test) : Brier et décomposition de Murphy par méthode, et par
    classe le taux observé, la PD de chaque méthode et la p-value du test de Jeffreys."""
    y = df[target].to_numpy()
    globaux = pl.DataFrame([{"methode": m, **brier_murphy(y, appliquer_pd(df, table, m, chr_col), df[chr_col])}
                            for m in methodes])
    obs = df.group_by(chr_col).agg(n=pl.len(), n_defaut=pl.col(target).sum(), taux_observe=pl.col(target).mean())
    par_classe = obs.join(table.select(chr_col, *methodes), on=chr_col).sort(chr_col)
    par_classe = par_classe.with_columns(**{f"p_jeffreys_{m}": pl.Series(test_jeffreys(par_classe["n"], par_classe["n_defaut"], par_classe[m]))
                                            for m in methodes})
    return globaux.sort("brier"), par_classe


def diagramme_calibration(par_classe: pl.DataFrame, methodes: tuple[str, ...], titre: str = "", ax=None):
    """Diagramme de calibration : taux observé par classe (avec intervalle de Wilson à 95 %) contre PD de chaque
    méthode, en échelle logarithmique. Un calibrage parfait place les points sur la diagonale."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    n, d = par_classe["n"].to_numpy(), par_classe["n_defaut"].to_numpy()
    p = par_classe["taux_observe"].to_numpy()
    z = 1.96
    centre = (p + z ** 2 / (2 * n)) / (1 + z ** 2 / n)
    demi = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / (1 + z ** 2 / n)
    for m in methodes:
        x = par_classe[m].to_numpy()
        ax.errorbar(x, p, yerr=[p - np.maximum(centre - demi, 1e-5), centre + demi - p], fmt="o", capsize=3, label=m)
    lim = [max(1e-4, min(p.min(), par_classe.select(methodes).min().min_horizontal()[0]) / 2), min(1, p.max() * 2)]
    ax.plot(lim, lim, linestyle="--", color="grey")
    ax.set(xscale="log", yscale="log", xlabel="PD calibrée", ylabel="taux de défaut observé (IC 95 %)",
           title=titre or "Diagramme de calibration")
    ax.legend(fontsize=8)
    return ax


# ---------------------------------------------------------------------------
# Marge de prudence et PD réglementaire (8.7)
# ---------------------------------------------------------------------------

def moc_a_cycle(df: pl.DataFrame, target: str, date: str) -> dict:
    """MoC de catégorie A liée à l'absence de cycle économique complet (une seule année de défauts) :
    ajout relatif = taux de défaut du mois le plus défavorable / taux moyen − 1, mesuré au niveau du segment."""
    m = df.group_by(date).agg(pl.col(target).mean().alias("t")).sort(date)
    moyen = float(df[target].mean())
    return {"taux_moyen": moyen, "taux_pire_mois": float(m["t"].max()), "moc_a_relative": float(m["t"].max() / moyen - 1),
            "taux_par_mois": dict(zip([str(x) for x in m[date]], m["t"].to_list()))}


def moc_c_jeffreys(n, n_defaut, pd_pred, niveau: float = 0.90) -> np.ndarray:
    """MoC de catégorie C (erreur d'estimation) : écart, s'il est positif, entre le quantile `niveau` de la loi
    a posteriori Bêta(défauts + ½, n − défauts + ½) et la PD estimée."""
    from scipy.stats import beta

    n, d = np.asarray(n, dtype=float), np.asarray(n_defaut, dtype=float)
    return np.maximum(beta.ppf(niveau, d + 0.5, n - d + 0.5) - np.asarray(pd_pred, dtype=float), 0.0)


def pd_reglementaire(table: pl.DataFrame, colonne: str, moc_a_relative: float, niveau_c: float = 0.90,
                     plancher: float = 0.0005, chr_col: str = "chr") -> pl.DataFrame:
    """PD finale par classe : PD calibrée (meilleure estimation) × (1 + MoC A) + MoC C, puis plancher réglementaire
    (0,05 % pour la clientèle de détail) et contrôle de la monotonie (une classe plus risquée ne peut avoir une
    PD inférieure à la précédente)."""
    t = table.sort(chr_col)
    moc_c = moc_c_jeffreys(t["n"], t["n_defaut"], t[colonne], niveau_c)
    t = t.select(chr_col, "n", "n_defaut", pl.col(colonne).alias("pd_estimee")).with_columns(
        moc_a=pl.col("pd_estimee") * moc_a_relative, moc_c=pl.Series(moc_c))
    t = t.with_columns(pd_avant_plancher=pl.col("pd_estimee") + pl.col("moc_a") + pl.col("moc_c"))
    return t.with_columns(pd_finale=pl.max_horizontal(pl.col("pd_avant_plancher"), pl.lit(plancher)).cum_max())


# ---------------------------------------------------------------------------
# Échelle maître (7.5)
# ---------------------------------------------------------------------------

BORNES_ECHELLE = [0.0005, 0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128, 0.256]


def echelle_maitre(bornes: list[float] = BORNES_ECHELLE) -> pl.DataFrame:
    """Grades de l'échelle maître définis par des bornes de PD (doublement d'un grade au suivant)."""
    b = [0.0] + list(bornes) + [1.0]
    return pl.DataFrame({"grade": list(range(1, len(b))), "pd_min": b[:-1], "pd_max": b[1:]})


def affecter_grade(pd_finale, bornes: list[float] = BORNES_ECHELLE) -> np.ndarray:
    """Grade de l'échelle maître de chaque PD : grade g si pd_min < PD ≤ pd_max."""
    return np.searchsorted(np.asarray(bornes), np.asarray(pd_finale, dtype=float), side="left") + 1
