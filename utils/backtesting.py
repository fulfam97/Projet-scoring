"""
Application Out-of-Time (OOT) et backtesting (chapitre 9) :
préparation de la base OOT à l'identique du développement, application de la chaîne
réglementaire (grille PDO -> CHR -> PD calibrée / réglementaire et échelle maître),
stabilité des distributions (PSI / CSI) et backtesting du pouvoir discriminant et de la calibration.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from utils import binning, calibrage, chr as chrm, exploration, models, scorecard
from utils.segmentation import construire_candidats

TRONQUES_JOINTS = {
    "RATIO_SOLDE_MOYEN_SUR_EPAR": "RATIO_SOLDE_MOYEN_SUR_EPARGNE",
    "RATIO_NBJDEB_NONAUTO_SUR_NBJ": "RATIO_NBJDEB_NONAUTO_SUR_NBJDEB",
}

MAX_SEUL_JOINTS = ("CODITDBDF_PAR", "CODITDBDF_PRO", "cod_axe_unite")


# ---------------------------------------------------------------------------
# Préparation de la base OOT (périmètre, segmentation, feature engineering)
# ---------------------------------------------------------------------------

def charger_base_brute(source: str | Path, cache: str | Path | None = None,
                       date: str = "datdelhis") -> pl.DataFrame:
    """Charge une base brute (.parquet, .sas7bdat ou .sas7bdat.gz) avec cache parquet optionnel,
    convertit NaN -> null et normalise la colonne date au format pl.Date."""
    source = Path(source)
    if cache is not None and Path(cache).exists():
        df = pl.read_parquet(cache)
    elif source.suffix == ".parquet":
        df = pl.read_parquet(source)
    else:
        import pandas as pd

        df = pl.from_pandas(pd.read_sas(source, encoding="latin1"))
        if cache is not None:
            Path(cache).parent.mkdir(parents=True, exist_ok=True)
            df.write_parquet(cache)

    df = exploration.nan_to_null(df)
    if date in df.columns and df.schema[date] != pl.Date:
        df = df.with_columns(
            pl.date(pl.col(date).cast(pl.Int32) // 100, pl.col(date).cast(pl.Int32) % 100, 1).alias(date)
        )
    return df


def preparer_oot(df: pl.DataFrame, target: str, segment_cible: str = "SB_mono_avec_eng") -> tuple[pl.DataFrame, dict]:
    """Applique à l'OOT le pipeline complet du développement :
    1. filtres de périmètre (défaut à date NBJ_ARR_SS >= 90, mineurs < 18 ans) ;
    2. unification des dossiers joints (unifier_joints) ;
    3. segmentation amont (SM_multi / SA_mono_sans_eng / SB_mono_avec_eng) et filtrage sur `segment_cible` ;
    4. typage, neutralisation des valeurs impossibles et feature engineering (Dfo_max_Mm4_Mm36,
       variables d'arriérés, TX_UTIL_DAV).
    Renvoie la base OOT prête au scoring et un dictionnaire de tables de suivi."""
    age_ans = (
        pl.coalesce("AGE_PP_EN_MOIS", "AGE_PP_EN_MOIS_med") / 12
        if "AGE_PP_EN_MOIS_med" in df.columns
        else pl.col("AGE_PP_EN_MOIS") / 12
    )
    base_perim, suivi_perimetre = exploration.appliquer_filtres(df, {
        "défaut à date (NBJ_ARR_SS >= 90)": pl.col("NBJ_ARR_SS") >= 90,
        "mineurs (< 18 ans)": age_ans < 18,
    }, target)

    base_perim, rapport_joints = exploration.unifier_joints(
        base_perim, TRONQUES_JOINTS, max_seul=MAX_SEUL_JOINTS, garder_min_max=True
    )

    cand = construire_candidats(base_perim)
    multi = pl.col("nb_tiers") > 1
    engage = pl.col("has_engagement") == 1
    cand = cand.with_columns(
        segment=pl.when(multi).then(pl.lit("SM_multi"))
        .when(~engage).then(pl.lit("SA_mono_sans_eng"))
        .otherwise(pl.lit("SB_mono_avec_eng"))
    )

    repartition_segments = (
        cand.group_by("segment")
        .agg(n=pl.len(), n_defaut=pl.col(target).sum(), taux_defaut=pl.col(target).mean())
        .sort("segment")
    )

    oot = cand.select(*[c for c in base_perim.columns if c != "segment"], "segment")
    oot = oot.filter(pl.col("segment") == segment_cible)

    if "CODACVPRO_modif" in oot.columns:
        oot = oot.with_columns(pl.col("CODACVPRO_modif").cast(pl.Int64).cast(pl.String))

    valeurs_impossibles = {
        "AGE_PP_EN_MOIS": pl.col("AGE_PP_EN_MOIS") > 110 * 12,
        "ANCIENNETE_modif": pl.col("ANCIENNETE_modif") > pl.col("AGE_PP_EN_MOIS"),
        "PART_EPARGNE_DANS_AVOIRS": ~pl.col("PART_EPARGNE_DANS_AVOIRS").is_between(0, 1),
    }
    regles_controle = {
        k: expr for k, expr in valeurs_impossibles.items() if k in oot.columns
    }
    if "anciennete_jour_sain_M0" in oot.columns:
        regles_controle["anciennete_jour_sain_M0 < 0"] = pl.col("anciennete_jour_sain_M0") < 0
    impossibles, _ = exploration.controle_groupes(oot, regles_controle, target) if regles_controle else (pl.DataFrame(), pl.DataFrame())
    oot = exploration.neutraliser_valeurs(oot, {k: expr for k, expr in valeurs_impossibles.items() if k in oot.columns})

    if "anciennete_jour_sain_M0" in oot.columns:
        oot = oot.with_columns(pl.col("anciennete_jour_sain_M0").clip(lower_bound=0))

    dfo_cols = [c for c in oot.columns if c.startswith("Dfo_max") and c != "Dfo_max_Mm4_Mm36"]
    if dfo_cols:
        oot = oot.with_columns(Dfo_max_Mm4_Mm36=pl.max_horizontal(dfo_cols))

    oot = exploration.variables_arrieres(oot)
    oot = exploration.taux_utilisation_decouvert(oot)

    suivi = {
        "perimetre": suivi_perimetre,
        "joints": rapport_joints,
        "segments": repartition_segments,
        "impossibles": impossibles,
    }
    return oot, suivi


# ---------------------------------------------------------------------------
# Application de la chaîne réglementaire (Score PDO -> CHR -> PD & Grade)
# ---------------------------------------------------------------------------

def appliquer_chaine_scoring(df: pl.DataFrame, grille: pl.DataFrame, regles_modele: dict[str, dict],
                             regle_chr: dict, pd_finales: pl.DataFrame, segment: str,
                             score_ref: float = 600, odds_ref: float = 50, pdo: float = 20,
                             numeros_chr: dict[str, int] | None = None) -> pl.DataFrame:
    """Applique la chaîne complète du modèle final sur `df` :
    - score PDO (somme des points de la grille) et pd_score (conversion PDO -> probabilité) ;
    - CHR (1 = moins risquée, K = plus risquée) via `regle_chr` ;
    - PD estimée, MoC A/C, PD réglementaire finale et grade sur l'échelle maître via `pd_finales`."""
    score = scorecard.appliquer_grille(df, grille, regles_modele)
    pd_score = pl.Series("pd_score", scorecard.score_vers_pd(score, score_ref, odds_ref, pdo))
    res = df.with_columns(score=score, pd_score=pd_score)

    dec = binning.appliquer_regle(res, "score", regle_chr)
    if numeros_chr is not None:
        chr_col = dec["classe"].replace_strict(numeros_chr, return_dtype=pl.Int64).alias("chr")
    else:
        n_classes = len(regle_chr["bornes"]) + 1
        chr_col = (pl.lit(n_classes) - dec["ordre"]).cast(pl.Int64).alias("chr")

    res = res.with_columns(chr_col, intervalle_score=dec["classe"])
    pd_seg = pd_finales.filter(pl.col("segment") == segment).select(
        "chr", "pd_estimee", "moc_a", "moc_c", "pd_finale", "grade"
    )
    return res.join(pd_seg, on="chr", how="left")


# ---------------------------------------------------------------------------
# Stabilité de la population : PSI et CSI (9.3)
# ---------------------------------------------------------------------------

def _verdict_psi(val: float) -> str:
    if val < 0.10:
        return "stable (< 0,10)"
    if val < 0.25:
        return "à surveiller (0,10 – 0,25)"
    return "instable (≥ 0,25)"


def psi_entre_echantillons(ref: pl.DataFrame, oot: pl.DataFrame, col: str,
                           eps: float = 1e-6) -> tuple[float, pl.DataFrame]:
    """PSI (Population Stability Index) entre une base de référence et l'OOT sur une colonne discrète (ex. chr)."""
    r = ref.group_by(col).agg(n_ref=pl.len()).with_columns(part_ref=pl.col("n_ref") / ref.height)
    o = oot.group_by(col).agg(n_oot=pl.len()).with_columns(part_oot=pl.col("n_oot") / oot.height)
    t = r.join(o, on=col, how="full", coalesce=True).fill_null(0).sort(col)
    t = t.with_columns(
        ecart_part=pl.col("part_oot") - pl.col("part_ref"),
        contribution_psi=(pl.col("part_oot") - pl.col("part_ref"))
                         * ((pl.col("part_oot") + eps) / (pl.col("part_ref") + eps)).log(),
    )
    return float(t["contribution_psi"].sum()), t


def csi_variables(ref: pl.DataFrame, oot: pl.DataFrame, regles_modele: dict[str, dict],
                  eps: float = 1e-6) -> tuple[pl.DataFrame, pl.DataFrame]:
    """CSI (Characteristic Stability Index) entre la référence et l'OOT pour chaque variable du modèle.
    Renvoie la synthèse par variable (triée par CSI décroissant) et le détail par modalité."""
    synthese, details = [], []
    for v, regle in regles_modele.items():
        c_ref = binning.appliquer_regle(ref, v, regle).select(pl.col("classe").alias("modalite"))
        c_oot = binning.appliquer_regle(oot, v, regle).select(pl.col("classe").alias("modalite"))
        val, det = psi_entre_echantillons(c_ref, c_oot, "modalite", eps=eps)
        synthese.append({"variable": v, "csi": val, "statut": _verdict_psi(val)})
        details.append(det.with_columns(variable=pl.lit(v)).select("variable", *det.columns))
    return (
        pl.DataFrame(synthese).sort("csi", descending=True),
        pl.concat(details) if details else pl.DataFrame(),
    )


# ---------------------------------------------------------------------------
# Backtesting du pouvoir discriminant (9.1) et de la calibration (9.2)
# ---------------------------------------------------------------------------

def backtest_discrimination(echantillons: dict[str, pl.DataFrame], target: str,
                            score_col: str = "score", chr_col: str = "chr",
                            id_col: str | None = None, n_bootstrap: int = 1000) -> pl.DataFrame:
    """Compare AUC, Gini (score continu et CHR), perte de Gini et KS entre échantillons (train / test / OOT).
    Si `id_col` est fourni, calcule aussi l'IC à 95 % par bootstrap par client sur le Gini du score."""
    lignes = []
    for nom, df in echantillons.items():
        y = df[target].to_numpy()
        m_score = models.metriques(y, -df[score_col].to_numpy(), nom)
        m_chr = models.metriques(y, df[chr_col].to_numpy(), nom)
        ligne = {
            "echantillon": nom, "n": df.height, "n_defaut": int(y.sum()), "taux_defaut": float(y.mean()),
            "auc_score": m_score["auc"], "gini_score": m_score["gini"], "ks_score": m_score["ks"],
            "gini_chr": m_chr["gini"], "perte_gini_pts": 100.0 * (m_score["gini"] - m_chr["gini"]),
        }
        if id_col is not None:
            ic, _ = models.gini_bootstrap(
                y, {"score": -df[score_col].to_numpy(), "chr": df[chr_col].to_numpy()},
                df[id_col], reference="score", n_tirages=n_bootstrap,
            )
            row_score = ic.filter(pl.col("modele") == "score").row(0, named=True)
            ligne["gini_ic_bas"] = float(row_score["ic_bas"])
            ligne["gini_ic_haut"] = float(row_score["ic_haut"])
        lignes.append(ligne)
    return pl.DataFrame(lignes)


def discrimination_mensuelle(df: pl.DataFrame, target: str, date: str,
                             score_col: str = "score", chr_col: str = "chr") -> pl.DataFrame:
    """Suivi mensuel sur l'OOT : effectif, défauts, taux de défaut, Gini score, Gini CHR et KS."""
    lignes = []
    for d in df[date].unique().sort().to_list():
        sub = df.filter(pl.col(date) == d)
        y = sub[target].to_numpy()
        if y.sum() == 0 or y.sum() == sub.height:
            lignes.append({date: d, "n": sub.height, "n_defaut": int(y.sum()), "taux_defaut": float(y.mean()),
                           "gini_score": float("nan"), "gini_chr": float("nan"), "ks_score": float("nan")})
            continue
        m_s = models.metriques(y, -sub[score_col].to_numpy())
        m_c = models.metriques(y, sub[chr_col].to_numpy())
        lignes.append({date: d, "n": sub.height, "n_defaut": int(y.sum()), "taux_defaut": float(y.mean()),
                       "gini_score": m_s["gini"], "gini_chr": m_c["gini"], "ks_score": m_s["ks"]})
    return pl.DataFrame(lignes)


def backtest_calibration(oot: pl.DataFrame, pd_finales: pl.DataFrame, segment: str, target: str,
                         chr_col: str = "chr") -> tuple[pl.DataFrame, pl.DataFrame]:
    """Backtesting de la calibration sur l'OOT (9.2) :
    - global : score de Brier et décomposition de Murphy pour `pd_estimee` (meilleure estimation)
      et `pd_finale` (PD réglementaire avec MoC A + C et plancher), taux observé vs PD moyenne ;
    - par CHR : effectif, défauts, taux observé + IC Wilson 95 %, `pd_estimee`, `pd_finale`,
      grade, p-values de Jeffreys (sous-estimation si < 5 %)."""
    pd_seg = pd_finales.filter(pl.col("segment") == segment).select(
        chr_col, "pd_estimee", "pd_finale", "grade"
    ).sort(chr_col)
    obs = (oot.group_by(chr_col)
           .agg(n=pl.len(), n_defaut=pl.col(target).sum(), taux_observe=pl.col(target).mean())
           .sort(chr_col))
    par_classe = pd_seg.join(obs, on=chr_col, how="left").fill_null(0)

    bas, haut = chrm._wilson(par_classe["n_defaut"].to_numpy(), np.maximum(par_classe["n"].to_numpy(), 1))
    par_classe = par_classe.with_columns(
        part_population=pl.col("n") / pl.col("n").sum(),
        ic_bas=pl.Series(bas), ic_haut=pl.Series(haut),
        p_jeffreys_estimee=pl.Series(calibrage.test_jeffreys(par_classe["n"], par_classe["n_defaut"], par_classe["pd_estimee"])),
        p_jeffreys_finale=pl.Series(calibrage.test_jeffreys(par_classe["n"], par_classe["n_defaut"], par_classe["pd_finale"])),
    ).select(
        chr_col, "grade", "n", "part_population", "n_defaut", "taux_observe", "ic_bas", "ic_haut",
        "pd_estimee", "p_jeffreys_estimee", "pd_finale", "p_jeffreys_finale"
    )

    y = oot[target].to_numpy()
    globaux = pl.DataFrame([
        {
            "pd": col,
            "taux_observe": float(y.mean()),
            "pd_moyenne": float(oot[col].mean()),
            "ratio_obs_pred": float(y.mean() / oot[col].mean()) if oot[col].mean() > 0 else float("nan"),
            **calibrage.brier_murphy(y, oot[col].to_numpy(), oot[chr_col].to_numpy()),
            "nb_rejets_jeffreys_5pct": int((par_classe[f"p_jeffreys_{col.split('_')[1]}"] < 0.05).sum()),
        }
        for col in ("pd_estimee", "pd_finale")
    ])
    return globaux, par_classe
