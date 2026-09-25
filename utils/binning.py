from __future__ import annotations

import polars as pl

SEUILS_SIDDIQI = [(0.02, "inutile"), (0.1, "faible"), (0.3, "moyen"), (0.5, "fort"), (float("inf"), "suspect")]


def _est_discrete(df: pl.DataFrame, var: str, max_modalites: int) -> bool:
    t = df.schema[var]
    return isinstance(t, (pl.String, pl.Categorical, pl.Enum, pl.Boolean)) or df[var].n_unique() <= max_modalites


def _classes(df: pl.DataFrame, var: str, n_bins: int, max_modalites: int) -> pl.DataFrame:
    """Colonnes 'classe' (libellé) et 'ordre' (tri) ; le missing est une classe à part entière, placée en dernier."""
    if _est_discrete(df, var, max_modalites):
        return df.select(
            pl.col(var).cast(pl.String).fill_null("missing").alias("classe"),
            pl.col(var).rank("dense").cast(pl.Float64).alias("ordre"),
        )
    q = pl.col(var).qcut(n_bins, allow_duplicates=True, include_breaks=True)
    return df.select(q.alias("q")).select(
        pl.col("q").struct.field("category").cast(pl.String).fill_null("missing").alias("classe"),
        pl.col("q").struct.field("breakpoint").alias("ordre"),
    )

def woe_iv_table(df: pl.DataFrame, var: str, target: str, n_bins: int = 10, max_modalites: int = 10) -> pl.DataFrame:
    """Table WOE/IV d'une variable : WOE = ln(%bons / %mauvais), lissage 0,5 pour les classes pures."""
    d = _classes(df, var, n_bins, max_modalites).with_columns(df[target].alias("y"))
    n_bad, n_good = d["y"].sum(), d.height - d["y"].sum()
    return (
        d.group_by("classe")
        .agg(ordre=pl.col("ordre").min(), n=pl.len(), n_defaut=pl.col("y").sum())
        .with_columns(n_sain=pl.col("n") - pl.col("n_defaut"))
        .with_columns(
            taux_defaut=pl.col("n_defaut") / pl.col("n"),
            part_population=pl.col("n") / d.height,
            dist_bad=(pl.col("n_defaut") + 0.5) / (n_bad + 0.5),
            dist_good=(pl.col("n_sain") + 0.5) / (n_good + 0.5),
        )
        .with_columns(woe=(pl.col("dist_good") / pl.col("dist_bad")).log())
        .with_columns(iv=(pl.col("dist_good") - pl.col("dist_bad")) * pl.col("woe"))
        .sort("ordre", nulls_last=True)
        .select("classe", "n", "n_defaut", "taux_defaut", "part_population", "woe", "iv")
    )

def _siddiqi(iv: float) -> str:
    return next(label for seuil, label in SEUILS_SIDDIQI if iv < seuil)


def iv_ranking(df, variables, target, n_bins=10, max_modalites=10):
    """IV de chaque variable avec l'interprétation Siddiqi, triée décroissante."""
    rows = []
    for v in variables:
        t = woe_iv_table(df, v, target, n_bins, max_modalites)
        rows.append({"variable": v, "iv": t["iv"].sum(), "pouvoir": _siddiqi(t["iv"].sum()),
                     "n_classes": t.height, "min_n_classe": t["n"].min()})
    return pl.DataFrame(rows).sort("iv", descending=True)



def iv_by_segment(df: pl.DataFrame, variables: list[str], target: str, segment: str, n_bins: int = 10,
                  max_modalites: int = 10) -> pl.DataFrame:
    """IV et rang de chaque variable dans chaque segment : des classements différents justifient la segmentation."""
    parts = []
    for s in sorted(df[segment].unique().to_list()):
        r = iv_ranking(df.filter(pl.col(segment) == s), variables, target, n_bins, max_modalites)
        parts.append(r.with_row_index("rang", offset=1).select(
            "variable", pl.col("iv").alias(f"iv_{s}"), pl.col("rang").alias(f"rang_{s}")))
    out = parts[0]
    for p in parts[1:]:
        out = out.join(p, on="variable")
    return out.sort(out.columns[1], descending=True)
