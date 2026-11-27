"""Application du système de notation à l'échantillon Out-of-Time (cours 1.5 et chapitre 9).

Rien n'est réestimé : la base OOT suit le même périmètre que la base de développement (notebook 0), puis les
découpages (ch. 4), fusions (ch. 5), grille (ch. 6), bornes des CHR (ch. 7) et PD (ch. 8) appris sur
l'apprentissage lui sont appliqués tels quels.

- chargement avec cache parquet ;
- périmètre : mêmes exclusions et même unification des dossiers joints qu'au notebook 0 ;
- stabilité de population (9.3) : PSI de la répartition des classes d'une base cible contre une base de référence
  (CSI quand il est calculé variable par variable), et ordre des taux de défaut par classe dans la cible.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from utils.exploration import appliquer_filtres, nan_to_null, unifier_joints
from utils.modele import performance
from utils.segmentation import construire_candidats

# Paramètres du notebook 0 (unification des dossiers joints)
TRONQUES = {
    "RATIO_SOLDE_MOYEN_SUR_EPAR": "RATIO_SOLDE_MOYEN_SUR_EPARGNE",
    "RATIO_NBJDEB_NONAUTO_SUR_NBJ": "RATIO_NBJDEB_NONAUTO_SUR_NBJDEB",
}
MAX_SEUL = ("CODITDBDF_PAR", "CODITDBDF_PRO", "cod_axe_unite")


def charger_sas(source: Path, cache: Path, chunksize: int = 200_000) -> pl.DataFrame:
    """Lit un fichier SAS (éventuellement compressé) une seule fois et le met en cache au format parquet."""
    source, cache = Path(source), Path(cache)
    if not cache.exists():
        if not source.is_file():
            raise FileNotFoundError(f"Fichier source introuvable : {source.resolve()}")
        import pandas as pd
        morceaux = [pl.from_pandas(m) for m in pd.read_sas(source, encoding="latin1", chunksize=chunksize)]
        cache.parent.mkdir(parents=True, exist_ok=True)
        pl.concat(morceaux, how="diagonal_relaxed").write_parquet(cache)
    return nan_to_null(pl.read_parquet(cache))


def perimetre_joint(base: pl.DataFrame, target: str, date_col: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Même périmètre qu'au notebook 0 : date AAAAMM → date, exclusion des défauts à date et des mineurs,
    unification des variables des dossiers joints, variables construites, puis segment comptes joints
    (nb_tiers > 1). Renvoie la base du segment et le suivi des exclusions."""
    if not base.schema[date_col].is_temporal():
        base = base.with_columns(
            pl.date(pl.col(date_col).cast(pl.Int32) // 100, pl.col(date_col).cast(pl.Int32) % 100, 1).alias(date_col))
    age_pp = [c for c in ("AGE_PP_EN_MOIS", "AGE_PP_EN_MOIS_med") if c in base.columns]
    age_ans = pl.coalesce(age_pp) / 12
    d, suivi = appliquer_filtres(base, {
        "défaut à date (NBJ_ARR_SS >= 90)": pl.col("NBJ_ARR_SS") >= 90,
        "mineurs": age_ans < 18,
    }, target)
    d, _ = unifier_joints(d, TRONQUES, max_seul=tuple(c for c in MAX_SEUL))
    d = construire_candidats(d)
    d, suivi_seg = appliquer_filtres(d, {"hors segment comptes joints (nb_tiers ≤ 1)": ~(pl.col("nb_tiers") > 1)},
                                     target)
    return d, pl.concat([suivi, suivi_seg.slice(1)])


def harmoniser_modalites(df: pl.DataFrame, modalites_ref: dict[str, list[str]]) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Recode les modalités qualitatives qui ne diffèrent d'une modalité de référence (vue en apprentissage) que par la
    casse ou des espaces en bordure (ex. « EER » → « eer »). Les autres modalités inconnues restent telles quelles.
    Renvoie la base et, par variable, la part de lignes recodées et la part encore inconnue après recodage."""
    exprs, rapport = [], []
    for v, ref in modalites_ref.items():
        if v not in df.columns:
            continue
        cle = {str(m).strip().lower(): str(m) for m in ref}
        s = pl.col(v).cast(pl.String)
        e = s.str.strip_chars().str.to_lowercase().replace_strict(cle, default=None)
        exprs.append(pl.coalesce(pl.when(s.is_in(list(map(str, ref)))).then(s), e, s).alias(v))
    out = df.with_columns(exprs)
    for v, ref in modalites_ref.items():
        if v in df.columns:
            rapport.append({"variable": v,
                            "part_recodee": float((df[v].cast(pl.String) != out[v]).fill_null(False).mean()),
                            "part_inconnue_avant": float((~df[v].cast(pl.String).is_in(list(map(str, ref)))).mean()),
                            "part_inconnue_apres": float((~out[v].is_in(list(map(str, ref)))).mean())})
    return out, pl.DataFrame(rapport)


def psi(part_ref, part_cible, eps: float = 1e-6) -> float:
    """Population Stability Index : Σ (p_cible − p_ref) · ln(p_cible / p_ref)."""
    a, b = np.asarray(part_ref, dtype=float), np.asarray(part_cible, dtype=float)
    return float(np.sum((b - a) * np.log((b + eps) / (a + eps))))


def lecture_psi(x: float) -> str:
    return "stable" if x < 0.10 else ("à surveiller" if x < 0.25 else "glissement significatif")


def stabilite_classes(ref: pl.DataFrame, cible: pl.DataFrame, variables: list[str], target: str,
                      noms: tuple[str, str] = ("ref", "cible")) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Pour chaque variable en classes (ou la CHR) : part et taux de défaut de chaque classe dans la référence et
    dans la cible, PSI / CSI de la cible contre la référence, et nombre de paires de classes adjacentes dont le taux
    de défaut n'est pas strictement croissant dans la cible. Renvoie (détail par classe, synthèse par variable)."""
    r, c = noms
    details, synth = [], []
    for v in variables:
        agg = lambda d: d.group_by(pl.col(v).cast(pl.Int64).alias("classe")).agg(
            n=pl.len(), n_defaut=pl.col(target).sum())
        t = (agg(ref).join(agg(cible), on="classe", how="full", coalesce=True, suffix=f"_{c}")
             .rename({"n": f"n_{r}", "n_defaut": f"n_defaut_{r}"}).fill_null(0).sort("classe")
             .with_columns(**{f"part_{r}": pl.col(f"n_{r}") / pl.col(f"n_{r}").sum(),
                              f"part_{c}": pl.col(f"n_{c}") / pl.col(f"n_{c}").sum(),
                              f"taux_{r}": pl.col(f"n_defaut_{r}") / pl.col(f"n_{r}"),
                              f"taux_{c}": pl.col(f"n_defaut_{c}") / pl.col(f"n_{c}")}))
        ordre = t["classe"].to_list()          # convention du projet : classe 1 = la moins risquée
        taux_c = dict(zip(t["classe"], t[f"taux_{c}"]))
        paires = list(zip(ordre[:-1], ordre[1:]))
        inversions = sum(1 for a, b in paires if not (taux_c[a] is not None and taux_c[b] is not None
                                                      and taux_c[a] < taux_c[b]))
        indice = psi(t[f"part_{r}"], t[f"part_{c}"])
        details.append(t.select(pl.lit(v).alias("variable"), pl.all()))
        synth.append({"variable": v, "n_classes": t.height, "psi": indice, "lecture": lecture_psi(indice),
                      "paires_adjacentes": len(paires), "inversions_cible": inversions})
    return pl.concat(details, how="diagonal_relaxed"), pl.DataFrame(synth)


def performance_par_groupe(df: pl.DataFrame, target: str, score: str, groupe: str, croissant_risque: bool = False,
                           min_defauts: int = 5) -> pl.DataFrame:
    """Gini, IC et KS du score dans chaque groupe (cohorte mensuelle...) ; groupe ignoré sous min_defauts défauts."""
    rows = []
    for (g,), d in df.group_by(groupe, maintain_order=True):
        if d[target].sum() < min_defauts or d[target].sum() == d.height:
            continue
        s = d[score].to_numpy() * (1 if croissant_risque else -1)
        rows.append({groupe: g, **{k: v for k, v in performance(d[target], s).items() if k != "modele"}})
    return pl.DataFrame(rows).sort(groupe)
