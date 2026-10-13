from __future__ import annotations

import numpy as np
import polars as pl
from scipy import stats

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


def variables_non_redondantes(df: pl.DataFrame, iv_table: pl.DataFrame, n: int = 10,
                              seuil: float = 0.8) -> list[str]:
    """Les n variables de plus forte IV, sans garder deux variables corrélées (|Spearman| >= seuil)."""
    from .exploration import paires_redondantes

    candidats = iv_table.sort("iv", descending=True)["variable"].head(4 * n).to_list()
    numeriques = [c for c in candidats if df.schema[c].is_numeric()]
    paires = paires_redondantes(df, numeriques, seuil)
    voisins = {}
    for a, b in zip(paires["var1"], paires["var2"]):
        voisins.setdefault(a, set()).add(b)
        voisins.setdefault(b, set()).add(a)
    retenues = []
    for v in candidats:
        if not voisins.get(v, set()) & set(retenues):
            retenues.append(v)
        if len(retenues) == n:
            break
    return retenues


def test_interactions_segment(df: pl.DataFrame, variables: list[str], target: str, segment: str,
                              n_bins: int = 10) -> tuple[dict, pl.DataFrame]:
    """Test LR du cours : modèle poolé (mêmes pentes, constante par segment) contre modèle avec
    interactions WOE × segment (une pente par segment). Renvoie le test global (LR, AIC, BIC) et,
    pour chaque variable, les pentes par segment et un test de Wald d'égalité des pentes."""
    import statsmodels.api as sm
    from .binning import woe_encode

    X = woe_encode(df, variables, target, n_bins).to_numpy()
    y = df[target].to_numpy()
    segs = sorted(df[segment].unique().to_list())
    S = np.column_stack([(df[segment] == s).to_numpy().astype(float) for s in segs])

    colonnes, noms = [], []
    for k, s in enumerate(segs):
        dans_s = S[:, k] == 1
        for j, v in enumerate(variables):
            if np.std(X[dans_s, j]) > 1e-12:   # variable constante dans le segment : pente non estimable
                colonnes.append(X[:, j] * S[:, k])
                noms.append((v, s))
    poole = sm.Logit(y, np.column_stack([S, X])).fit(disp=False, maxiter=300)
    inter = sm.Logit(y, np.column_stack([S] + colonnes)).fit(disp=False, maxiter=300)

    lr = 2 * (inter.llf - poole.llf)
    ddl = len(colonnes) - X.shape[1]
    global_ = {
        "segmentation": segment, "LR": lr, "ddl": ddl, "pvalue": stats.chi2.sf(lr, ddl),
        "AIC_poole": poole.aic, "AIC_interactions": inter.aic,
        "AIC_prefere": "interactions" if inter.aic < poole.aic else "poolé",
        "BIC_poole": poole.bic, "BIC_interactions": inter.bic,
        "BIC_prefere": "interactions" if inter.bic < poole.bic else "poolé",
    }

    lignes = []
    for v in variables:
        idx = [len(segs) + i for i, (vv, _) in enumerate(noms) if vv == v]
        pentes = {f"pente_{s}": inter.params[len(segs) + i] for i, (vv, s) in enumerate(noms) if vv == v}
        ligne = {"variable": v, **{f"pente_{s}": None for s in segs}, **pentes}
        if len(idx) >= 2:
            R = np.zeros((len(idx) - 1, len(inter.params)))
            for r, i in enumerate(idx[1:]):
                R[r, idx[0]], R[r, i] = 1, -1
            w = inter.wald_test(R, scalar=True)
            ligne |= {"wald": float(w.statistic), "ddl": len(idx) - 1, "pvalue": float(w.pvalue)}
        else:
            ligne |= {"wald": None, "ddl": 0, "pvalue": None}
        lignes.append(ligne)
    return global_, pl.DataFrame(lignes).sort("wald", descending=True, nulls_last=True)

def afdm(df: pl.DataFrame, quanti: list[str], quali: list[str], n_axes: int = 5) -> dict:
    """Analyse factorielle de données mixtes (AFDM) : quantitatives centrées-réduites, qualitatives en
    indicatrices divisées par la racine de leur fréquence puis centrées, et ACP sur l'ensemble.
    Manquants : moyenne pour les quantitatives, modalité « missing » pour les qualitatives.
    Renvoie les coordonnées des individus, la part de variance de chaque axe et, pour chaque variable,
    son lien avec chaque axe (corrélation au carré pour une quantitative, rapport de corrélation η² pour une qualitative)."""
    blocs = []
    for v in quanti:
        x = df[v].cast(pl.Float64).to_numpy()
        x = np.where(np.isnan(x), np.nanmean(x), x)
        blocs.append(((x - x.mean()) / x.std())[:, None])
    for v in quali:
        indic = df.select(pl.col(v).cast(pl.String).fill_null("missing")).to_dummies().to_numpy().astype(float)
        p = indic.mean(axis=0)
        blocs.append(indic / np.sqrt(p) - np.sqrt(p))
    Z = np.hstack(blocs)
    valeurs, vecteurs = np.linalg.eigh(Z.T @ Z / len(Z))
    ordre = np.argsort(valeurs)[::-1][:n_axes]
    valeurs, vecteurs = valeurs[ordre], vecteurs[:, ordre]
    F = Z @ vecteurs
    noms_axes = [f"axe_{k + 1}" for k in range(n_axes)]

    liens = []
    for v in quanti + quali:
        ligne = {"variable": v, "type": "quanti" if v in quanti else "quali"}
        for k, a in enumerate(noms_axes):
            f = F[:, k]
            if v in quanti:
                x = df[v].cast(pl.Float64).to_numpy()
                x = np.where(np.isnan(x), np.nanmean(x), x)
                ligne[a] = np.corrcoef(x, f)[0, 1] ** 2
            else:
                g = df[v].cast(pl.String).fill_null("missing").to_numpy()
                inter = sum((g == m).mean() * (f[g == m].mean() - f.mean()) ** 2 for m in np.unique(g))
                ligne[a] = inter / f.var()
        liens.append(ligne)

    return {
        "coordonnees": pl.DataFrame(F, schema=noms_axes),
        "variance": pl.DataFrame({"axe": noms_axes, "valeur_propre": valeurs,
                                  "part_variance": valeurs / np.trace(Z.T @ Z / len(Z))})
                      .with_columns(part_cumulee=pl.col("part_variance").cum_sum()),
        "liens": pl.DataFrame(liens),
    }


def clusters_kmeans(coordonnees: pl.DataFrame, k_values: tuple[int, ...] = (2, 3, 4), n_axes: int = 4,
                    n_silhouette: int = 10_000, seed: int = 0) -> tuple[pl.DataFrame, dict[int, np.ndarray]]:
    """k-means sur les premiers axes d'une analyse factorielle ; silhouette calculée sur un échantillon."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    X = coordonnees.select(coordonnees.columns[:n_axes]).to_numpy()
    echantillon = np.random.default_rng(seed).choice(len(X), size=min(n_silhouette, len(X)), replace=False)
    resultats, labels = [], {}
    for k in k_values:
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
        labels[k] = km.labels_
        resultats.append({"k": k, "inertie": km.inertia_,
                          "silhouette": silhouette_score(X[echantillon], km.labels_[echantillon])})
    return pl.DataFrame(resultats), labels


def profil_groupes(df: pl.DataFrame, groupe: str, target: str, quanti: list[str], quali: list[str]) -> pl.DataFrame:
    """Portrait de chaque groupe : effectif, défauts, taux, moyenne des quantitatives, part de « oui »
    pour les variables booléennes et part de chaque modalité pour les autres qualitatives."""
    aggs = [pl.len().alias("n"), pl.col(target).sum().alias("n_defaut"), pl.col(target).mean().alias("taux")]
    aggs += [pl.col(v).mean().round(1).alias(v) for v in quanti]
    for v in quali:
        if df.schema[v] == pl.Boolean:
            aggs.append(pl.col(v).mean().round(2).alias(f"part {v}"))
        else:
            for m in df[v].cast(pl.String).drop_nulls().unique().sort().to_list():
                aggs.append((pl.col(v).cast(pl.String) == m).mean().round(2).alias(f"part {v}={m}"))
    return df.group_by(groupe).agg(aggs).sort(groupe)

