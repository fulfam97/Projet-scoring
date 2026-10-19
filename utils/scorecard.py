"""
Grille de score : traduction des coefficients de la régression logistique en points (chapitre 6).
"""
from __future__ import annotations

import numpy as np
import polars as pl

from utils.binning import appliquer_regle


def _coefficients(res, groupes: dict[str, list[int]], ordres: dict[str, list[str]]) -> pl.DataFrame:
    """Une ligne par variable et par classe : coefficient de la classe (0 pour la classe de référence)."""
    beta = np.asarray(res.params)[1:]
    lignes = []
    for v, idx in groupes.items():
        lignes.append({"variable": v, "classe": ordres[v][0], "coefficient": 0.0, "reference": True})
        for c, j in zip(ordres[v][1:], idx):
            lignes.append({"variable": v, "classe": c, "coefficient": float(beta[j]), "reference": False})
    return pl.DataFrame(lignes)


def parametres_pdo(score_ref: float = 600, odds_ref: float = 50, pdo: float = 20) -> tuple[float, float]:
    """Méthode PDO (6.2) : score = Offset + Factor × ln(cote), avec cote = P(sain) / P(défaut).
    Factor = PDO / ln 2 (la cote double tous les PDO points) ; Offset = score_ref − Factor × ln(odds_ref)
    (une cote de odds_ref contre 1 correspond au score score_ref)."""
    factor = pdo / np.log(2)
    return factor, score_ref - factor * np.log(odds_ref)


def grille_pdo(res, groupes: dict[str, list[int]], ordres: dict[str, list[str]], score_ref: float = 600,
               odds_ref: float = 50, pdo: float = 20, arrondi: bool = True) -> pl.DataFrame:
    """Grille de score par la méthode PDO. ln(cote) = −(β0 + Σ βj) : chaque classe reçoit
    −Factor × β (le risque fait perdre des points) plus une part égale de (Offset − Factor × β0)
    répartie entre les variables. Score élevé = risque faible."""
    factor, offset = parametres_pdo(score_ref, odds_ref, pdo)
    beta0, k = float(np.asarray(res.params)[0]), len(groupes)
    g = _coefficients(res, groupes, ordres).with_columns(
        points=-pl.col("coefficient") * factor + (offset - factor * beta0) / k)
    return g.with_columns(pl.col("points").round(0).cast(pl.Int64)) if arrondi else g


def grille_normalisee(res, groupes: dict[str, list[int]], ordres: dict[str, list[str]], total: float = 1000,
                      arrondi: bool = True) -> pl.DataFrame:
    """Grille par normalisation des coefficients (6.3) : dans chaque variable, la classe la plus risquée
    reçoit 0 point et la moins risquée (la référence) le maximum ; les maxima sont proportionnels à l'étendue
    des coefficients de la variable et somment à `total`. Score de 0 (pire profil) à `total` (meilleur)."""
    g = _coefficients(res, groupes, ordres)
    etendue = g.group_by("variable").agg(beta_max=pl.col("coefficient").max())
    somme = etendue["beta_max"].sum()
    g = g.join(etendue, on="variable").with_columns(
        points=(pl.col("beta_max") - pl.col("coefficient")) / somme * total).drop("beta_max")
    return g.with_columns(pl.col("points").round(0).cast(pl.Int64)) if arrondi else g


def appliquer_grille(df: pl.DataFrame, grille: pl.DataFrame, regles: dict[str, dict]) -> pl.Series:
    """Score de chaque ligne : somme des points des classes de ses variables."""
    total = np.zeros(df.height)
    for v in grille["variable"].unique(maintain_order=True):
        pts = dict(zip(*grille.filter(pl.col("variable") == v).select("classe", "points").to_dict(as_series=False).values()))
        classe = appliquer_regle(df, v, regles[v])["classe"]
        total += classe.replace_strict(pts, return_dtype=pl.Float64).to_numpy()
    return pl.Series("score", total)


def score_vers_pd(score, score_ref: float = 600, odds_ref: float = 50, pdo: float = 20) -> np.ndarray:
    """PD correspondant à un score PDO : cote = exp((score − Offset) / Factor), PD = 1 / (1 + cote)."""
    factor, offset = parametres_pdo(score_ref, odds_ref, pdo)
    return 1 / (1 + np.exp((np.asarray(score, dtype=float) - offset) / factor))


def taux_par_tranche(score, y, n_tranches: int = 10) -> pl.DataFrame:
    """Taux de défaut par tranche de score (quantiles, tranche 1 = scores les plus faibles = plus risqués).
    Le taux doit décroître quand le score augmente ; `part_defauts_cumulee` montre la concentration des défauts."""
    d = pl.DataFrame({"score": np.asarray(score, dtype=float), "y": np.asarray(y)})
    d = d.with_columns(tranche=(pl.col("score").rank("ordinal") * n_tranches / d.height).ceil().cast(pl.Int64))
    t = (d.group_by("tranche").agg(score_min=pl.col("score").min(), score_max=pl.col("score").max(), n=pl.len(),
                                   n_defaut=pl.col("y").sum(), taux_defaut=pl.col("y").mean())
         .sort("tranche"))
    return t.with_columns(part_defauts_cumulee=pl.col("n_defaut").cum_sum() / pl.col("n_defaut").sum())
