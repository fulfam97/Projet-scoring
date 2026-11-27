"""
Consolidation Out-of-Time (OOT) commune aux trois segments (SA, SB, SM) :
- préparation unique du périmètre OOT (exclusions + dossiers joints + segmentation) ;
- scoring autonome de chaque segment avec ses propres artefacts et utilitaires (sans collision d'imports) ;
- rattachement à l'échelle maître unifiée (chapitre 7.5) ;
- backtesting portefeuille (discrimination, perte liée à l'échelle maître, homogénéité inter-segments,
  discrimination résiduelle intra-grade et calibration globale).
"""
from __future__ import annotations

import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

BORNES_ECHELLE = [0.0005, 0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128, 0.256]

TRONQUES_JOINTS = {
    "RATIO_SOLDE_MOYEN_SUR_EPAR": "RATIO_SOLDE_MOYEN_SUR_EPARGNE",
    "RATIO_NBJDEB_NONAUTO_SUR_NBJ": "RATIO_NBJDEB_NONAUTO_SUR_NBJDEB",
}
MAX_SEUL_JOINTS = ("CODITDBDF_PAR", "CODITDBDF_PRO", "cod_axe_unite")

COLONNES_COMMUNES = [
    "gdt", "datdelhis", "ddefaut_ndb", "segment",
    "score", "pd_score", "chr", "pd_estimee", "pd_finale", "grade",
]


# ---------------------------------------------------------------------------
# Utilitaires d'isolation et échelle maître
# ---------------------------------------------------------------------------

@contextmanager
def _avec_utils(racine: Path):
    """Isole temporairement `sys.path` et `sys.modules['utils*']` sur `racine` pour éviter toute
    collision entre `utils/` (racine / SA), `Maxime/utils/` (SB) et `Anis/utils/` (SM)."""
    racine_str = str(Path(racine).resolve())
    saved_path = list(sys.path)
    saved_mods = {k: sys.modules.pop(k) for k in list(sys.modules) if k == "utils" or k.startswith("utils.")}
    sys.path.insert(0, racine_str)
    try:
        yield
    finally:
        for k in list(sys.modules):
            if k == "utils" or k.startswith("utils."):
                sys.modules.pop(k, None)
        sys.modules.update(saved_mods)
        sys.path[:] = saved_path


def _trouver(root: Path, *candidats: str) -> Path:
    for rel in candidats:
        p = root / rel
        if p.exists():
            return p
    raise FileNotFoundError(f"Aucun des fichiers attendus trouvé sous {root.resolve()} : {candidats}")


def echelle_maitre(bornes: list[float] = BORNES_ECHELLE) -> pl.DataFrame:
    """Grades de l'échelle maître (doublement de PD d'un grade au suivant)."""
    b = [0.0] + list(bornes) + [1.0]
    return pl.DataFrame({"grade": list(range(1, len(b))), "pd_min": b[:-1], "pd_max": b[1:]})


def affecter_grade(pd_vals, bornes: list[float] = BORNES_ECHELLE) -> np.ndarray:
    """Grade de l'échelle maître (1..K+1), convention ]pd_min ; pd_max]."""
    return np.searchsorted(np.asarray(bornes), np.asarray(pd_vals, dtype=float), side="left") + 1


def score_pdo_vers_pd(score, score_ref: float = 600, odds_ref: float = 50, pdo: float = 20) -> np.ndarray:
    """Convertit un score PDO (score élevé = risque faible) en probabilité de défaut."""
    factor = pdo / np.log(2)
    offset = score_ref - factor * np.log(odds_ref)
    return 1.0 / (1.0 + np.exp((np.asarray(score, dtype=float) - offset) / factor))


# ---------------------------------------------------------------------------
# Préparation unique de la base OOT (périmètre + dossiers joints + segments)
# ---------------------------------------------------------------------------

def charger_et_segmenter_oot(source: Path, cache: Path, target: str = "ddefaut_ndb",
                             date_col: str = "datdelhis", root: Path = Path(".")) -> tuple[pl.DataFrame, dict]:
    """Charge l'OOT brute une seule fois, applique le périmètre commun (NBJ_ARR_SS < 90, âge >= 18 ans),
    unifie les dossiers joints (avec _min/_max pour SM) et affecte les 3 segments SA / SB / SM."""
    with _avec_utils(root / "Maxime"):
        from utils import exploration
        from utils.backtesting import charger_base_brute
        from utils.segmentation import construire_candidats

        brute = charger_base_brute(source, cache=cache, date=date_col)
        age_cols = [c for c in ("AGE_PP_EN_MOIS", "AGE_PP_EN_MOIS_med") if c in brute.columns]
        age_ans = pl.coalesce(age_cols) / 12
        perim, suivi_perim = exploration.appliquer_filtres(brute, {
            "défaut à date (NBJ_ARR_SS >= 90)": pl.col("NBJ_ARR_SS") >= 90,
            "mineurs (< 18 ans)": age_ans < 18,
        }, target)
        perim, rapport_joints = exploration.unifier_joints(
            perim, TRONQUES_JOINTS, max_seul=MAX_SEUL_JOINTS, garder_min_max=True
        )
        cand = construire_candidats(perim).with_columns(
            segment_lib=pl.when(pl.col("nb_tiers") > 1).then(pl.lit("SM_multi"))
            .when(pl.col("has_engagement") == 0).then(pl.lit("SA_mono_sans_eng"))
            .otherwise(pl.lit("SB_mono_avec_eng"))
        )

    repartition = (
        cand.group_by("segment_lib")
        .agg(n=pl.len(), n_defaut=pl.col(target).sum(), taux_defaut=pl.col(target).mean())
        .with_columns(part=pl.col("n") / pl.col("n").sum())
        .sort("segment_lib")
    )
    return cand, {"perimetre": suivi_perim, "joints": rapport_joints, "segments": repartition}


# ---------------------------------------------------------------------------
# Scoring autonome de chaque segment (SB, SM, SA) -> format commun
# ---------------------------------------------------------------------------

def _standardiser(df: pl.DataFrame, segment: str, id_col: str, date_col: str, target: str) -> pl.DataFrame:
    return df.select(
        pl.col(id_col).alias("gdt"),
        pl.col(date_col).cast(pl.Date).alias("datdelhis"),
        pl.col(target).cast(pl.Int64).alias("ddefaut_ndb"),
        pl.lit(segment).alias("segment"),
        pl.col("score").cast(pl.Float64),
        pl.col("pd_score").cast(pl.Float64),
        pl.col("chr").cast(pl.Int64),
        pl.col("pd_estimee").cast(pl.Float64),
        pl.col("pd_finale").cast(pl.Float64),
        pl.Series("grade", affecter_grade(df["pd_finale"].to_numpy())).cast(pl.Int64),
    )


def scorer_segment_sb(oot_cand: pl.DataFrame, root: Path = Path("."),
                      id_col: str = "gdt", date_col: str = "datdelhis", target: str = "ddefaut_ndb") -> pl.DataFrame:
    """Applique la chaîne finale du segment SB (Maxime) sur la sous-population SB_mono_avec_eng."""
    oot_sb = oot_cand.filter(pl.col("segment_lib") == "SB_mono_avec_eng")
    with _avec_utils(root / "Maxime"):
        from utils import backtesting, binning, exploration

        if "CODACVPRO_modif" in oot_sb.columns:
            oot_sb = oot_sb.with_columns(pl.col("CODACVPRO_modif").cast(pl.Int64).cast(pl.String))
        valeurs_impossibles = {
            "AGE_PP_EN_MOIS": pl.col("AGE_PP_EN_MOIS") > 110 * 12,
            "ANCIENNETE_modif": pl.col("ANCIENNETE_modif") > pl.col("AGE_PP_EN_MOIS"),
            "PART_EPARGNE_DANS_AVOIRS": ~pl.col("PART_EPARGNE_DANS_AVOIRS").is_between(0, 1),
        }
        oot_sb = exploration.neutraliser_valeurs(
            oot_sb, {k: e for k, e in valeurs_impossibles.items() if k in oot_sb.columns}
        )
        if "anciennete_jour_sain_M0" in oot_sb.columns:
            oot_sb = oot_sb.with_columns(pl.col("anciennete_jour_sain_M0").clip(lower_bound=0))
        dfo_cols = [c for c in oot_sb.columns if c.startswith("Dfo_max") and c != "Dfo_max_Mm4_Mm36"]
        if dfo_cols:
            oot_sb = oot_sb.with_columns(Dfo_max_Mm4_Mm36=pl.max_horizontal(dfo_cols))
        oot_sb = exploration.variables_arrieres(oot_sb)
        oot_sb = exploration.taux_utilisation_decouvert(oot_sb)

        regles_modele = binning.charger_regles(
            _trouver(root, "Maxime/outputs/scorecard/regles_modele_SB.json", "outputs/scorecard/regles_modele_SB.json")
        )
        grille = pl.read_excel(
            _trouver(root, "outputs/scorecard/grille_SB.xlsx", "Maxime/outputs/scorecard/grille_SB.xlsx"),
            sheet_name="SB_Grille",
        )
        regle_chr = binning.charger_regles(
            _trouver(root, "Maxime/outputs/chr/regle_chr_SB.json", "outputs/chr/regle_chr_SB.json")
        )["chr"]
        chr_ref = pl.read_excel(
            _trouver(root, "outputs/chr/chr_SB.xlsx", "Maxime/outputs/chr/chr_SB.xlsx"),
            sheet_name="SB_CHR_train",
        )
        numeros_chr = dict(zip(chr_ref["intervalle_score"], chr_ref["chr"]))
        pd_finales = pl.read_parquet(
            _trouver(root, "outputs/quantification/pd_finales.parquet", "Maxime/outputs/quantification/pd_finales.parquet")
        )

        res = backtesting.appliquer_chaine_scoring(
            oot_sb, grille, regles_modele, regle_chr, pd_finales,
            segment="SB", score_ref=600, odds_ref=50, pdo=20, numeros_chr=numeros_chr,
        )
    return _standardiser(res, "SB", id_col, date_col, target)


def _harmoniser_modalites(df: pl.DataFrame, modalites_ref: dict[str, list[str]]) -> pl.DataFrame:
    """Recode les modalités qualitatives ne différant d'une modalité vue en apprentissage que par la casse
    ou des espaces en bordure (identique à Anis/utils/oot.harmoniser_modalites, sans dépendance seaborn)."""
    exprs = []
    for v, ref in modalites_ref.items():
        if v not in df.columns:
            continue
        ref_str = [str(m) for m in ref]
        cle = {m.strip().lower(): m for m in ref_str}
        s = pl.col(v).cast(pl.String).fill_null("missing")
        e = s.str.strip_chars().str.to_lowercase().replace_strict(cle, default=None)
        exprs.append(pl.coalesce(pl.when(s.is_in(ref_str)).then(s), e, s).alias(v))
    return df.with_columns(exprs) if exprs else df


def scorer_segment_sm(oot_cand: pl.DataFrame, root: Path = Path("."), version: str = "_v2",
                      id_col: str = "gdt", date_col: str = "datdelhis", target: str = "ddefaut_ndb") -> pl.DataFrame:
    """Applique la chaîne finale du segment SM comptes joints (Anis) sur SM_multi."""
    oot_sm = oot_cand.filter(pl.col("segment_lib") == "SM_multi")
    anis_dir = root / "Anis"
    decoupages = json.loads((anis_dir / "outputs/binning/decoupages_joint.json").read_text(encoding="utf-8"))
    g = json.loads((anis_dir / f"outputs/scorecard/grille_joint{version}.json").read_text(encoding="utf-8"))
    bornes = json.loads((anis_dir / f"outputs/chr/chr_joint{version}.json").read_text(encoding="utf-8"))["bornes_points"]
    calib = json.loads((anis_dir / f"outputs/calibrage/calibrage_joint{version}.json").read_text(encoding="utf-8"))
    pd_lra = {int(k): float(v["pd_calibree"]) for k, v in calib["pd"].items()}
    pd_fin = {int(k): float(v["pd_finale"]) for k, v in calib["pd"].items()}
    vars_m = g["variables"]
    quali = [v for v in vars_m if decoupages[v]["nature"] == "qualitative"]
    grille = pl.DataFrame(g["grille_pdo"])

    oot_sm = _harmoniser_modalites(oot_sm, {v: list(decoupages[v]["modalites"]) for v in quali})
    with _avec_utils(anis_dir):
        from utils.chr import affecter_chr
        from utils.scorecard import scorer_nouvelle_base

        scored = scorer_nouvelle_base(oot_sm, decoupages, g["fusions"], grille, quali, "score")
        chr_arr = affecter_chr(scored["score"], bornes)
        res = scored.with_columns(
            pd_score=pl.Series(score_pdo_vers_pd(scored["score"])),
            chr=pl.Series(chr_arr).cast(pl.Int64),
            pd_estimee=pl.Series([pd_lra[int(k)] for k in chr_arr]),
            pd_finale=pl.Series([pd_fin[int(k)] for k in chr_arr]),
        )
    return _standardiser(res, "SM", id_col, date_col, target)


def scorer_segment_sa(oot_cand: pl.DataFrame, root: Path = Path("."),
                      base_dev_sa: Path = Path("data/segments/mono_sans_engagement/base_mono_sans_engagement.parquet"),
                      id_col: str = "gdt", date_col: str = "datdelhis", target: str = "ddefaut_ndb") -> pl.DataFrame:
    """Applique la chaîne finale du segment SA mono sans engagement (Cheikna) sur SA_mono_sans_eng,
    en exécutant `Analyse_Mono_Sans_Engagement_decoupe.ipynb` dans un espace de noms isolé."""
    import matplotlib.pyplot as plt
    from IPython.utils.capture import capture_output

    oot_sa = oot_cand.filter(pl.col("segment_lib") == "SA_mono_sans_eng")
    schema_dev = pl.read_parquet_schema(root / base_dev_sa)
    absentes = [c for c in schema_dev if c not in oot_sa.columns]
    oot_sa = oot_sa.with_columns([pl.lit(None).alias(c) for c in absentes])
    oot_sa = oot_sa.select([pl.col(c).cast(t, strict=False) for c, t in schema_dev.items()])

    with _avec_utils(root):
        from utils import exploration as ex
        from utils import feature_engineering as fe

        ns: dict = {"__name__": "__main__"}
        os.environ["SCORING_BASE"] = str(root / base_dev_sa)
        nb_path = root / "Analyse_Mono_Sans_Engagement_decoupe.ipynb"
        cellules = json.loads(nb_path.read_text(encoding="utf-8"))["cells"]
        with capture_output():
            for k, cel in enumerate(cellules):
                if cel["cell_type"] == "code":
                    exec(compile("".join(cel["source"]), f"{nb_path} [cellule {k}]", "exec"), ns)
                    plt.close("all")

        oot_sa = ex.nan_to_null(oot_sa)
        for c, t in oot_sa.schema.items():
            if t == pl.String:
                oot_sa = oot_sa.with_columns(
                    pl.when(pl.col(c).is_in(["NaN", "nan", ""])).then(None).otherwise(pl.col(c)).alias(c)
                )
        for c in ns["COLONNES_CATEGORIELLES_NUMERIQUES"]:
            if c in oot_sa.columns:
                oot_sa = oot_sa.with_columns(pl.col(c).cast(pl.Int64, strict=False).cast(pl.String))
        oot_sa = oot_sa.with_columns(fe.en_date(oot_sa, date_col).alias(ns["DATE_OBS"]))
        oot_sa = ns["creer_variables"](oot_sa)
        for nom, paire in ns["INTERACTIONS"].items():
            oot_sa = fe.appliquer_interaction_arbre(oot_sa, ns["ARBRES"][nom], list(paire), nom)

        score_pdo = ns["score_grille"](oot_sa, "points_pdo")
        pd_lr = ns["predire"](ns["res_final"], oot_sa, ns["VARS_FINALES"])
        oot_s = oot_sa.with_columns(
            pl.Series(ns["SCORE"], score_pdo),
            pl.Series("pd_score", pd_lr),
        )
        oot_s = ns["ajouter_chr"](oot_s)

        pd_lra = ns["pd_lra"]
        pd_regl = dict(zip(ns["pd_reglementaire"]["CHR"].to_list(), ns["pd_reglementaire"]["pd_reglementaire"].to_list()))
        res = oot_s.with_columns(
            pl.col(ns["SCORE"]).alias("score"),
            pl.col("CHR").str.extract(r"(\d+)").cast(pl.Int64).alias("chr"),
            pl.col("CHR").replace_strict(pd_lra, return_dtype=pl.Float64).alias("pd_estimee"),
            pl.col("CHR").replace_strict(pd_regl, return_dtype=pl.Float64).alias("pd_finale"),
        )
    return _standardiser(res, "SA", id_col, date_col, target)


# ---------------------------------------------------------------------------
# Métriques portefeuille, échelle maître, homogénéité inter-segments, calibration
# ---------------------------------------------------------------------------

def _wilson(d, n, z: float = 1.96):
    n_safe = np.maximum(n, 1)
    p = d / n_safe
    c = (p + z ** 2 / (2 * n_safe)) / (1 + z ** 2 / n_safe)
    h = z * np.sqrt(p * (1 - p) / n_safe + z ** 2 / (4 * n_safe ** 2)) / (1 + z ** 2 / n_safe)
    return np.maximum(c - h, 0.0), np.minimum(c + h, 1.0)


def _metriques_score(y, s) -> dict:
    y, s = np.asarray(y, dtype=int), np.asarray(s, dtype=float)
    pi = float(y.mean())
    fpr, tpr, _ = roc_curve(y, s)
    auc = float(roc_auc_score(y, s))
    ap = float(average_precision_score(y, s))
    apn = (ap - pi) / (1 - pi) if pi < 1 else float("nan")
    return {
        "n": int(y.size), "n_defaut": int(y.sum()), "taux_defaut": pi,
        "auc": auc, "gini": 2 * auc - 1, "auc_pr": ap, "auc_pr_norm": apn,
        "lift_pr": ap / pi if pi > 0 else float("nan"),
        "ks": float(np.max(tpr - fpr)),
    }


def synthese_discrimination_portefeuille(df: pl.DataFrame, target: str = "ddefaut_ndb") -> pl.DataFrame:
    """Discrimination OOT par segment (SA, SB, SM) et sur le portefeuille consolidé, pour 3 niveaux :
    - `pd_score` (score continu) ;
    - `pd_finale` (CHR calibrées) ;
    - `grade` (échelle maître unifiée)."""
    lignes = []
    groupes = [(s, df.filter(pl.col("segment") == s)) for s in sorted(df["segment"].unique().to_list())]
    groupes.append(("PORTEFEUILLE", df))
    for nom_seg, sub in groupes:
        y = sub[target].to_numpy()
        for niveau, col in (("1. Score continu (pd_score)", "pd_score"),
                            ("2. CHR calibrée (pd_finale)", "pd_finale"),
                            ("3. Échelle maître (grade)", "grade")):
            m = _metriques_score(y, sub[col].to_numpy())
            lignes.append({"segment": nom_seg, "niveau": niveau, **m})
    return pl.DataFrame(lignes)


def cascade_echelle_maitre(df: pl.DataFrame, target: str = "ddefaut_ndb") -> pl.DataFrame:
    """Perte d'information du portefeuille consolidé entre score continu, CHR calibrées et échelle maître."""
    y = df[target].to_numpy()
    etapes = [
        ("0. Score continu (pd_score)", df["pd_score"].to_numpy()),
        ("1. CHR calibrée (pd_estimee)", df["pd_estimee"].to_numpy()),
        ("2. CHR réglementaire (pd_finale)", df["pd_finale"].to_numpy()),
        ("3. Échelle maître (grade)", df["grade"].to_numpy()),
    ]
    lignes = [{"etape": i, "modele": nom, **_metriques_score(y, s)} for i, (nom, s) in enumerate(etapes)]
    t = pl.DataFrame(lignes)
    g0, pr0, prn0 = t["gini"][0], t["auc_pr"][0], t["auc_pr_norm"][0]
    return t.with_columns(
        delta_gini_pts=100.0 * (pl.col("gini") - pl.col("gini").shift(1)),
        perte_cum_gini_pts=100.0 * (pl.col("gini") - g0),
        delta_auc_pr_pts=100.0 * (pl.col("auc_pr") - pl.col("auc_pr").shift(1)),
        perte_cum_auc_pr_pts=100.0 * (pl.col("auc_pr") - pr0),
        delta_auc_pr_norm_pts=100.0 * (pl.col("auc_pr_norm") - pl.col("auc_pr_norm").shift(1)),
        perte_cum_auc_pr_norm_pts=100.0 * (pl.col("auc_pr_norm") - prn0),
    )


def table_echelle_maitre_oot(df: pl.DataFrame, target: str = "ddefaut_ndb") -> pl.DataFrame:
    """Performance et calibration de l'échelle maître unifiée sur l'OOT (par grade 1..11)."""
    em = echelle_maitre()
    agg = (
        df.group_by("grade")
        .agg(
            n=pl.len(), n_defaut=pl.col(target).sum(), taux_observe=pl.col(target).mean(),
            pd_estimee_moy=pl.col("pd_estimee").mean(), pd_finale_moy=pl.col("pd_finale").mean(),
            n_SA=(pl.col("segment") == "SA").sum(),
            n_SB=(pl.col("segment") == "SB").sum(),
            n_SM=(pl.col("segment") == "SM").sum(),
        )
        .sort("grade")
    )
    t = em.join(agg, on="grade", how="inner")
    n, d = t["n"].to_numpy(), t["n_defaut"].to_numpy()
    bas, haut = _wilson(d, n)
    return t.with_columns(
        part_population=pl.col("n") / pl.col("n").sum(),
        ic_bas=pl.Series(bas), ic_haut=pl.Series(haut),
        p_jeffreys_estimee=pl.Series(stats.beta.cdf(t["pd_estimee_moy"].to_numpy(), d + 0.5, n - d + 0.5)),
        p_jeffreys_finale=pl.Series(stats.beta.cdf(t["pd_finale_moy"].to_numpy(), d + 0.5, n - d + 0.5)),
    ).select(
        "grade", "pd_min", "pd_max", "n", "part_population", "n_SA", "n_SB", "n_SM",
        "n_defaut", "taux_observe", "ic_bas", "ic_haut",
        "pd_estimee_moy", "p_jeffreys_estimee", "pd_finale_moy", "p_jeffreys_finale",
    )


def heterogeneite_grades(table_em: pl.DataFrame) -> pl.DataFrame:
    """Séparation entre grades adjacents de l'échelle maître (rapport de taux, h de Cohen, recouvrement IC, z-test)."""
    lignes = []
    rows = list(table_em.iter_rows(named=True))
    for a, b in zip(rows, rows[1:]):
        p1, p2 = a["taux_observe"], b["taux_observe"]
        pp = (a["n_defaut"] + b["n_defaut"]) / (a["n"] + b["n"])
        denom = np.sqrt(pp * (1 - pp) * (1 / a["n"] + 1 / b["n"])) if 0 < pp < 1 else np.nan
        z = (p2 - p1) / denom if denom and np.isfinite(denom) else np.nan
        lignes.append({
            "paire": f"G{a['grade']} → G{b['grade']}", "taux_1": p1, "taux_2": p2,
            "rapport": p2 / p1 if p1 > 0 else np.inf,
            "h_cohen": float(2 * np.arcsin(np.sqrt(p2)) - 2 * np.arcsin(np.sqrt(p1))),
            "ic_se_recouvrent": bool(a["ic_haut"] >= b["ic_bas"]),
            "p_value": float(stats.norm.sf(z)) if np.isfinite(z) else np.nan,
        })
    return pl.DataFrame(lignes)


def discrimination_intra_grade(df: pl.DataFrame, target: str = "ddefaut_ndb",
                               score_col: str = "pd_score") -> pl.DataFrame:
    """Discrimination résiduelle (AUC-ROC, Gini, AUC-PR brute & normalisée) de `score_col` au sein de chaque grade."""
    y_all = df[target].to_numpy().astype(int)
    s_all = df[score_col].to_numpy().astype(float)
    g_all = df["grade"].to_numpy()
    lignes = []
    for g in sorted(np.unique(g_all)):
        m = g_all == g
        yg, sg = y_all[m], s_all[m]
        n, nd = int(yg.size), int(yg.sum())
        pi = float(yg.mean()) if n else float("nan")
        if 0 < nd < n:
            auc = float(roc_auc_score(yg, sg))
            ap = float(average_precision_score(yg, sg))
            apn = (ap - pi) / (1 - pi) if pi < 1 else float("nan")
            gini = 2 * auc - 1
        else:
            auc = gini = ap = apn = float("nan")
        lignes.append({
            "grade": int(g), "n": n, "n_defaut": nd, "taux_defaut": pi,
            "auc_roc": auc, "gini": gini, "auc_pr": ap, "auc_pr_norm": apn,
        })
    return pl.DataFrame(lignes)


def homogeneite_inter_segments(df: pl.DataFrame, target: str = "ddefaut_ndb",
                               n_min: int = 30) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Vérifie à grade égal de l'échelle maître que les segments SA, SB et SM ont des taux de défaut
    comparables (homogénéité inter-segments, 7.5) : tableau croisé + test du Chi² et h de Cohen max."""
    detail = (
        df.group_by("grade", "segment")
        .agg(n=pl.len(), n_defaut=pl.col(target).sum(), taux_defaut=pl.col(target).mean(),
             pd_finale_moy=pl.col("pd_finale").mean())
        .sort("grade", "segment")
    )
    lignes = []
    for g in sorted(df["grade"].unique().to_list()):
        sub = detail.filter((pl.col("grade") == g) & (pl.col("n") >= n_min))
        if sub.height < 2:
            continue
        taux = sub["taux_defaut"].to_numpy()
        tab = np.column_stack([sub["n_defaut"].to_numpy(), (sub["n"] - sub["n_defaut"]).to_numpy()])
        p_val = float(stats.chi2_contingency(tab)[1]) if tab[:, 0].sum() > 0 else 1.0
        h = float(2 * np.arcsin(np.sqrt(taux.max())) - 2 * np.arcsin(np.sqrt(taux.min())))
        lignes.append({
            "grade": int(g), "segments": ", ".join(sub["segment"].to_list()),
            "taux_min": float(taux.min()), "taux_max": float(taux.max()),
            "rapport_max_min": float(taux.max() / taux.min()) if taux.min() > 0 else float("inf"),
            "h_cohen_max": h, "p_chi2": p_val,
        })
    return detail, pl.DataFrame(lignes)


def calibration_portefeuille(df: pl.DataFrame, target: str = "ddefaut_ndb") -> pl.DataFrame:
    """Synthèse de calibration OOT (observés vs attendus, Brier & décomposition de Murphy sur les couples
    (segment, chr)) par segment et sur le portefeuille consolidé."""
    lignes = []
    groupes = [(s, df.filter(pl.col("segment") == s)) for s in sorted(df["segment"].unique().to_list())]
    groupes.append(("PORTEFEUILLE", df))
    for nom, sub in groupes:
        y = sub[target].to_numpy().astype(float)
        o, N = float(y.mean()), sub.height
        for col in ("pd_estimee", "pd_finale"):
            p = sub[col].to_numpy().astype(float)
            g = sub.group_by("segment", "chr").agg(n=pl.len(), pk=pl.col(col).mean(), ok=pl.col(target).mean())
            fiab = float((g["n"] * (g["pk"] - g["ok"]) ** 2).sum() / N)
            reso = float((g["n"] * (g["ok"] - o) ** 2).sum() / N)
            lignes.append({
                "segment": nom, "pd": col, "n": N, "observes": int(y.sum()),
                "attendus": float(p.sum()), "taux_observe": o, "pd_moyenne": float(p.mean()),
                "ratio_obs_pred": float(o / p.mean()) if p.mean() > 0 else float("nan"),
                "brier": float(np.mean((p - y) ** 2)), "fiabilite": fiab, "resolution": reso,
                "incertitude": float(o * (1 - o)),
            })
    return pl.DataFrame(lignes)


def plot_roc_pr_portefeuille(df: pl.DataFrame, target: str = "ddefaut_ndb", col: str = "pd_score"):
    """Trace côte à côte les courbes ROC et Précision-Rappel de SA, SB, SM et du portefeuille global."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    groupes = [(s, df.filter(pl.col("segment") == s)) for s in sorted(df["segment"].unique().to_list())]
    groupes.append(("PORTEFEUILLE", df))
    for nom, sub in groupes:
        y, s = sub[target].to_numpy(), sub[col].to_numpy()
        fpr, tpr, _ = roc_curve(y, s)
        gini = 2 * roc_auc_score(y, s) - 1
        lw = 2.4 if nom == "PORTEFEUILLE" else 1.5
        axes[0].plot(fpr, tpr, linewidth=lw, label=f"{nom} (Gini = {gini:.3f})")

        prec, rec, _ = precision_recall_curve(y, s)
        ap = average_precision_score(y, s)
        pi = float(y.mean())
        apn = (ap - pi) / (1 - pi) if pi < 1 else float("nan")
        axes[1].plot(rec, prec, linewidth=lw, label=f"{nom} (PR={ap:.3f} | norm={apn:.3f} | π={pi:.2%})")

    axes[0].plot([0, 1], [0, 1], linestyle="--", color="grey", label="hasard")
    axes[0].set(xlabel="taux de faux positifs", ylabel="taux de vrais positifs", title=f"Courbes ROC OOT ({col})")
    axes[0].legend(loc="lower right", fontsize=8); axes[0].grid(alpha=0.25)

    pi_glob = float(df[target].mean())
    axes[1].axhline(pi_glob, linestyle="--", color="grey", label=f"hasard portefeuille ({pi_glob:.2%})")
    axes[1].set(xlabel="rappel", ylabel="précision", title=f"Courbes Précision-Rappel OOT ({col})")
    axes[1].legend(loc="upper right", fontsize=8); axes[1].grid(alpha=0.25)
    fig.tight_layout()
    return fig
