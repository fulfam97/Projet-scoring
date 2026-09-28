from __future__ import annotations

import numpy as np
import polars as pl


def _regles_feuilles(tree, noms: list[str]) -> dict[int, str]:
    """Chemin menant à chaque feuille de l'arbre."""
    t = tree.tree_
    regles = {}

    def parcours(noeud: int, conditions: list[str]):
        if t.children_left[noeud] == -1:
            regles[noeud] = " ET ".join(conditions) or "tout"
            return
        var, seuil = noms[t.feature[noeud]], t.threshold[noeud]
        gauche = t.missing_go_to_left[noeud] if hasattr(t, "missing_go_to_left") else True
        na_g, na_d = (" (ou manquant)", "") if gauche else ("", " (ou manquant)")
        parcours(t.children_left[noeud], conditions + [f"{var} <= {seuil:.4g}{na_g}"])
        parcours(t.children_right[noeud], conditions + [f"{var} > {seuil:.4g}{na_d}"])

    parcours(0, [])
    return regles

def tree_segmentation(df: pl.DataFrame, features: list[str], target: str, max_depth: int = 2,
                      min_leaf_share: float = 0.05, seed: int = 0) -> tuple[pl.DataFrame, np.ndarray]:
    """Arbre peu profond sur les variables candidates : chaque feuille = segment candidat (règle, effectif, défauts, taux)."""
    from sklearn.tree import DecisionTreeClassifier

    quali = [c for c in features if isinstance(df.schema[c], (pl.String, pl.Categorical, pl.Enum))]
    X_df = df.select(features).to_dummies(columns=quali) if quali else df.select(features)
    X = X_df.cast(pl.Float64).to_numpy()
    y = df[target].to_numpy()

    tree = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=int(min_leaf_share * len(y)),
                                  random_state=seed).fit(X, y)
    feuille = tree.apply(X)
    regles = _regles_feuilles(tree, X_df.columns)
    feuilles = (
        pl.DataFrame({"feuille": feuille, "y": y})
        .group_by("feuille")
        .agg(n=pl.len(), n_defaut=pl.col("y").sum(), taux_defaut=pl.col("y").mean())
        .with_columns(
            part_population=pl.col("n") / len(y),
            regle=pl.col("feuille").replace_strict(regles, return_dtype=pl.String),
        )
        .sort("taux_defaut", descending=True)
        .select("feuille", "regle", "n", "n_defaut", "taux_defaut", "part_population")
    )
    return feuilles, feuille



# ---------------------------------------------------------------------------
# Variables candidates à la segmentation
# ---------------------------------------------------------------------------

def construire_candidats(df: pl.DataFrame) -> pl.DataFrame:
    """Ajoute les variables candidates et les découpages testés.
    Les compteurs nb_* / engagement* manquants sont lus comme 0 (aucun contrat)."""
    n0 = lambda c: pl.col(c).fill_null(0)
    return df.with_columns(
        type_credit=(
            pl.when(n0("nb_credit_immo") > 0).then(pl.lit("1_immo"))
            .when((n0("nb_credit_perso") + n0("nb_credit_flex") + n0("nb_credit_revolving")) > 0).then(pl.lit("2_conso"))
            .when(n0("nb_credit_MLT") > 0).then(pl.lit("3_MLT"))
            .when(n0("engagement") > 0).then(pl.lit("4_autre_engagement"))
            .otherwise(pl.lit("5_sans_credit"))
        ),
        has_revolving=n0("nb_credit_revolving") > 0,
        has_decouvert=n0("engagement_DAV") > 0,
        joint=pl.col("nb_tiers") > 1,
        anciennete_mois=pl.col("ANCIENNETE_modif"),
        anciennete=pl.col("ANCIENNETE_modif").cut([12, 24, 60], labels=["<1an", "1-2ans", "2-5ans", "5ans+"]),
        age_ans=pl.col("AGE_PP_EN_MOIS") / 12,
        age=(pl.col("AGE_PP_EN_MOIS") / 12).cut([25, 40, 60], labels=["18-25", "25-40", "40-60", "60+"]),
        compte_inactif=n0("CRTAD_IND_0042") == 0,
        historique_defaut=pl.col("DATE_DERNIER_DFO").is_not_null(),
        flux_pro=pl.col("CRTOC_TOK_0010"),
        patrimonial=pl.col("cod_axe_unite") > 1,
    ).with_columns(
        seg_type_credit=pl.col("type_credit").replace_strict({
            "5_sans_credit": "S1_sans_engagement", "4_autre_engagement": "S2_facilites",
            "1_immo": "S3_credit_amort", "2_conso": "S3_credit_amort", "3_MLT": "S3_credit_amort",
        }),
        seg_anciennete=pl.when(pl.col("has_engagement") == 0).then(pl.lit("SA_sans_engagement"))
        .when(pl.col("anciennete_mois") <= 60).then(pl.lit("SB1_recent"))
        .otherwise(pl.lit("SB2_ancien")),
        segment=pl.when(pl.col("has_engagement") == 0).then(pl.lit("SA_sans_engagement"))
        .otherwise(pl.lit("SB_avec_engagement")),
    )


def rank_segmentation_candidates(df: pl.DataFrame, candidates: list[str], target: str) -> pl.DataFrame:
    """Chi2 / V de Cramer de chaque candidate avec la cible, nb de modalités et plus petit nb de défauts par modalité."""
    from .exploration import chi2_association

    rows = []
    for v in candidates:
        g = df.group_by(v).agg(n_defaut=pl.col(target).sum())
        rows.append({**chi2_association(df, v, target),
                     "n_modalites": g.height, "min_n_defaut_modalite": g["n_defaut"].min()})
    return pl.DataFrame(rows).sort("cramers_v", descending=True)


# ---------------------------------------------------------------------------
# Comparaison des moteurs de risque et stabilité
# ---------------------------------------------------------------------------

def comparer_classements(iv_seg: pl.DataFrame, segments: list[str] | None = None, top: int = 15) -> pl.DataFrame:
    """Pour chaque paire de segments (sortie de binning.iv_by_segment) : corrélation de Spearman des IV
    et nombre de variables communes aux deux top N. Classements proches = mêmes moteurs de risque."""
    from scipy.stats import spearmanr

    segments = segments or [c.removeprefix("iv_") for c in iv_seg.columns if c.startswith("iv_")]
    rows = []
    for i, a in enumerate(segments):
        for b in segments[i + 1:]:
            top_a = set(iv_seg.sort(f"iv_{a}", descending=True)["variable"].head(top))
            top_b = set(iv_seg.sort(f"iv_{b}", descending=True)["variable"].head(top))
            rows.append({"paire": f"{a} vs {b}",
                         "corr_rangs": spearmanr(iv_seg[f"iv_{a}"], iv_seg[f"iv_{b}"]).statistic,
                         f"top{top}_communs": len(top_a & top_b)})
    return pl.DataFrame(rows)


def risk_rate_over_time(df: pl.DataFrame, var: str, target: str, date: str) -> pl.DataFrame:
    """Taux de défaut par date (lignes) et modalité (colonnes) : l'ordre des segments doit être stable."""
    out = (
        df.group_by(date, var)
        .agg(taux_defaut=pl.col(target).mean())
        .with_columns(pl.col(var).cast(pl.String).fill_null("missing"))
        .pivot(on=var, index=date, values="taux_defaut")
        .sort(date)
    )
    return out.select(date, *sorted(c for c in out.columns if c != date))


def split_train_test_oot(df: pl.DataFrame, id_col: str,
    date_col: str, target: str, date_oot, segment: str | None = None, test_share: float = 0.2, seed: int = 42) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Découpe la base en 'train', 'test' et 'oot'.
    - 'oot' : observations avec date_col >= date_oot ;
    - 'train' / 'test' : tirage au niveau id_col (aucun dossier des deux côtés),
      stratifié par (segment, target) sur la période hors OOT."""
    rng = np.random.default_rng(seed)
    hors_oot = df.filter(pl.col(date_col) < date_oot)
    ids = (hors_oot.group_by(id_col)
        .agg(_y=pl.col(target).max(),_seg=pl.col(segment).first() if segment else pl.lit("global"))
        .sort(id_col))
    ids = (ids.with_columns(_u=rng.random(ids.height))
        .with_columns(
            split=pl.when(
                pl.col("_u").rank("ordinal").over("_seg", "_y")
                <= (pl.len().over("_seg", "_y") * test_share)
            ).then(pl.lit("test")).otherwise(pl.lit("train"))
        ).select(id_col, "split"))

    out = (df.drop("split", strict=False)
        .join(ids, on=id_col, how="left")
        .with_columns(split=pl.when(pl.col(date_col) >= date_oot)
            .then(pl.lit("oot")).otherwise(pl.col("split"))))

    cles = ([segment] if segment else []) + ["split"]
    bilan = (out.group_by(cles).agg(n=pl.len(), n_id=pl.col(id_col).n_unique(), n_defaut=pl.col(target).sum(),
            taux_defaut=pl.col(target).mean())
        .with_columns(part=pl.col("n") / pl.col("n").sum().over(segment)
            if segment else pl.col("n") / out.height
        ).sort(cles))
    return out, bilan
