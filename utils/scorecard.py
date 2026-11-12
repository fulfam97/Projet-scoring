"""Chapitre 6 — Du modèle au score : traduction en points.

- méthode du PDO (Points to Double the Odds) : Offset / Factor, formule additive par modalité (6.2) ;
- grille par normalisation des coefficients sur 1000 points, contributions CTR et q (6.3) ;
- application de la grille à une base (y compris un futur échantillon Out-of-Time).
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import polars as pl


def offset_factor(points0: float = 600, odds0: float = 50, pdo: float = 20) -> tuple[float, float]:
    """Factor = PDO / ln 2 ; Offset = Points0 − Factor × ln(odds0) (cours 6.2.3)."""
    factor = pdo / np.log(2)
    return points0 - factor * np.log(odds0), factor


def grille_pdo(coefs: pl.DataFrame, beta0: float, points0: float = 600, odds0: float = 50, pdo: float = 20,
               arrondi: bool = False) -> pl.DataFrame:
    """Points par modalité (cours 6.2.4) : (Offset − Factor·β0) / p − Factor·β_jk.
    `coefs` : colonnes variable, classe, beta (référence comprise, β = 0)."""
    offset, factor = offset_factor(points0, odds0, pdo)
    p = coefs["variable"].n_unique()
    pts = (offset - factor * beta0) / p - factor * pl.col("beta")
    return coefs.select("variable", "classe", "beta").with_columns(points=pts.round(0) if arrondi else pts)


def grille_normalisation(coefs: pl.DataFrame, total: float = 1000, arrondi: bool = False) -> pl.DataFrame:
    """Note de la modalité (cours 6.3.2) : SC(j,k) = total × (R_j − β_jk) / Σ_v R_v, avec R_j = max_k β_jk.
    La modalité la plus risquée de chaque variable vaut 0, la moins risquée R_j × total / Σ R."""
    c = coefs.select("variable", "classe", "beta").with_columns(R=pl.col("beta").max().over("variable"))
    somme_r = c.group_by("variable").agg(pl.col("R").first())["R"].sum()
    sc = total * (pl.col("R") - pl.col("beta")) / somme_r
    return c.with_columns(points=sc.round(0) if arrondi else sc).drop("R")


def contributions(grille: pl.DataFrame, parts: pl.DataFrame, total: float = 1000) -> pl.DataFrame:
    """CTR_j = max_k SC(j,k) / (total/100) et q_j = σ_j / Σ σ_v, σ_j écart-type pondéré des notes (cours 6.3.3).
    `parts` : colonnes variable, classe, part (part de la population dans la classe)."""
    g = grille.join(parts, on=["variable", "classe"], how="left").with_columns(pl.col("part").fill_null(0))
    moy = (pl.col("part") * pl.col("points")).sum().over("variable") / pl.col("part").sum().over("variable")
    par_var = (
        g.with_columns(moyenne=moy)
        .group_by("variable", maintain_order=True)
        .agg(note_max=pl.col("points").max(), moyenne=pl.col("moyenne").first(),
             sigma=(((pl.col("points") - pl.col("moyenne")) ** 2 * pl.col("part")).sum() / pl.col("part").sum()).sqrt())
    )
    return par_var.with_columns(CTR=pl.col("note_max") / (total / 100),
                                q=pl.col("sigma") / pl.col("sigma").sum()).sort("q", descending=True)


def parts_classes(df_classes: pl.DataFrame, variables: list[str]) -> pl.DataFrame:
    """Part de la population dans chaque classe de chaque variable."""
    return pl.concat([
        df_classes.group_by(pl.col(v).cast(pl.Int64).alias("classe")).agg(n=pl.len())
        .with_columns(variable=pl.lit(v), part=pl.col("n") / pl.col("n").sum()).select("variable", "classe", "part")
        for v in variables
    ])


def noter(df_classes: pl.DataFrame, grille: pl.DataFrame, nom: str = "score") -> pl.DataFrame:
    """Ajoute le score en points : somme des points des modalités de l'individu. Une classe absente de la grille
    (jamais observée en apprentissage) reçoit les points de la classe la plus risquée de la variable."""
    exprs = []
    for v in grille["variable"].unique(maintain_order=True).to_list():
        g = grille.filter(pl.col("variable") == v)
        pire = float(g.sort("classe")["points"][-1])
        exprs.append(pl.col(v).cast(pl.Int64)
                     .replace_strict(dict(zip(g["classe"].to_list(), g["points"].to_list())), default=pire,
                                     return_dtype=pl.Float64))
    return df_classes.with_columns(pl.sum_horizontal(exprs).alias(nom))


def scorer_nouvelle_base(df_brut: pl.DataFrame, decoupages: dict, fusions: dict, grille: pl.DataFrame,
                         variables_quali: list[str] | None = None, nom: str = "score") -> pl.DataFrame:
    """Applique à une nouvelle base (par exemple l'échantillon Out-of-Time) toute la chaîne apprise en
    apprentissage, sans rien réestimer : manquant des qualitatives → "missing", découpages du ch. 4,
    fusions du ch. 5, grille du ch. 6."""
    from .binning import appliquer_decoupages
    from .modele import appliquer_fusions
    variables = grille["variable"].unique(maintain_order=True).to_list()
    d = df_brut
    if variables_quali:
        d = d.with_columns(pl.col(c).cast(pl.String).fill_null("missing") for c in variables_quali if c in d.columns)
    d = appliquer_decoupages(d, {v: decoupages[v] for v in variables})
    d = appliquer_fusions(d, {v: {int(a): int(b) for a, b in m.items()} for v, m in fusions.items() if v in variables})
    return noter(d, grille, nom)


def plot_grille(grille: pl.DataFrame, titre: str = "", n_cols: int = 3):
    """Points par classe pour chaque variable de la grille."""
    variables = grille["variable"].unique(maintain_order=True).to_list()
    n_rows = int(np.ceil(len(variables) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.3 * n_cols, 2.8 * n_rows), squeeze=False)
    for ax, v in zip(axes.flat, variables):
        g = grille.filter(pl.col("variable") == v).sort("classe")
        ax.bar(g["classe"].cast(pl.String).to_list(), g["points"].to_list(), color="#4C72B0")
        for x, y in zip(range(g.height), g["points"].to_list()):
            ax.text(x, y, f"{y:.0f}", ha="center", va="bottom", fontsize=8)
        ax.set_title(v, fontsize=9)
        ax.set_xlabel("classe")
    for ax in list(axes.flat)[len(variables):]:
        ax.axis("off")
    fig.suptitle(titre)
    fig.tight_layout()
    return fig


def plot_distribution_score(score, y, titre: str = "", bins: int = 40):
    """Distribution du score en points chez les sains et chez les défauts (densités)."""
    score, y = np.asarray(score), np.asarray(y)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(score[y == 0], bins=bins, density=True, alpha=0.6, label="sains")
    ax.hist(score[y == 1], bins=bins, density=True, alpha=0.6, label="défauts")
    ax.set(xlabel="score (points)", ylabel="densité", title=titre)
    ax.legend()
    fig.tight_layout()
    return fig
