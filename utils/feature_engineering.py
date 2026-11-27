"""Création de variables (feature engineering) — chapitre 2.4 du cours.

Toutes les fonctions `ajouter_*` sont déterministes : aucun paramètre n'est
appris, elles s'appliquent donc à l'identique au train et au test sans fuite.
`feature_interaction_arbre` apprend des seuils : ajustée sur le train
uniquement, puis appliquée au test.

Si les colonnes sources d'une variable sont absentes, la variable n'est pas
créée et un message l'indique (pas d'erreur bloquante).
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier

MOIS = 30.4375  # nombre moyen de jours dans un mois
_ORIGINE_SAS = dt.date(1960, 1, 1)

# Fenêtres d'historique (du plus récent au plus ancien) et milieu de fenêtre en mois
FENETRES = [("Mm1_Mm3", 2), ("Mm4_Mm6", 5), ("Mm7_Mm9", 8), ("Mm10_Mm12", 11),
            ("Mm13_Mm18", 15.5), ("Mm19_Mm24", 21.5), ("Mm25_Mm36", 30.5), ("Mm37_Mm48", 42.5)]


# ---------------------------------------------------------------------------
# Outils
# ---------------------------------------------------------------------------

def _presentes(df: pl.DataFrame, cols: list[str], variable: str) -> bool:
    absentes = [c for c in cols if c not in df.columns]
    if absentes:
        print(f"[feature_engineering] {variable} non créée : colonnes absentes {absentes}")
        return False
    return True


def _num(col: str) -> pl.Expr:
    return pl.col(col).cast(pl.Float64, strict=False)


def _n0(col: str) -> pl.Expr:
    """Compteur / encours manquant lu comme 0 (même hypothèse que la segmentation)."""
    return _num(col).fill_null(0)


def en_date(df: pl.DataFrame, col: str) -> pl.Series:
    """Convertit une colonne de date en type Date, quel que soit son codage :
    Date/Datetime, texte, entier AAAAMM (1er du mois), entier AAAAMMJJ,
    ou nombre de jours depuis le 01/01/1960 (date SAS)."""
    s = df[col]
    if s.dtype == pl.Date:
        return s
    if s.dtype.is_temporal():
        return s.cast(pl.Date)
    if s.dtype == pl.String:
        for fmt in ("%Y-%m-%d", "%Y%m%d", "%Y%m", "%d/%m/%Y"):
            out = s.str.strptime(pl.Date, fmt, strict=False)
            if out.null_count() == s.null_count():
                return out
        raise ValueError(f"{col} : format de date texte non reconnu")
    x = s.cast(pl.Float64)
    v = x.drop_nulls()
    if v.is_empty():
        return pl.Series(col, [None] * len(s), dtype=pl.Date)
    vmax = v.max()
    if 190001 <= v.min() and vmax <= 299912:                        # AAAAMM
        e = x.cast(pl.Int64)
        return pl.select(pl.date(e // 100, e % 100, 1)).to_series().alias(col)
    if 19000101 <= v.min() and vmax <= 29991231:                    # AAAAMMJJ
        e = x.cast(pl.Int64)
        return pl.select(pl.date(e // 10000, (e // 100) % 100, e % 100)).to_series().alias(col)
    if vmax < 200_000:                                              # jours SAS
        return (pl.select(pl.lit(_ORIGINE_SAS) + pl.duration(days=x.cast(pl.Int64)))
                .to_series().cast(pl.Date).alias(col))
    raise ValueError(f"{col} : codage numérique de date non reconnu (max = {vmax})")


def controler_dates_futures(df: pl.DataFrame, date_obs: str, cols: list[str]) -> dict[str, int]:
    """Nombre de lignes où une date d'historique est postérieure à la date d'observation."""
    obs = df[date_obs]
    return {c: int((en_date(df, c) > obs).sum()) for c in cols if c in df.columns}


# ---------------------------------------------------------------------------
# A. Indicateurs simples (litiges, garanties immobilières)
# ---------------------------------------------------------------------------

def ajouter_tops_litige_garantie(df: pl.DataFrame) -> pl.DataFrame:
    """TOP_LITIGE : au moins un litige (manquant conservé comme manquant).
    TOP_GARANTIE_IMMO : au moins un crédit immobilier garanti (compteur manquant = 0)."""
    exprs = []
    if _presentes(df, ["TOTAL_LITIGES_PRO_IND_0209"], "TOP_LITIGE"):
        exprs.append((_num("TOTAL_LITIGES_PRO_IND_0209") > 0).cast(pl.Int8).alias("TOP_LITIGE"))
    garanties = ["nb_credit_immo__crelog", "nb_credit_immo__hypotheque", "nb_credit_immo__autre_OCM",
                 "nb_credit_immo__contrat_vie", "nb_credit_immo__autre_garantie"]
    if _presentes(df, garanties, "TOP_GARANTIE_IMMO"):
        exprs.append((pl.sum_horizontal([_n0(c) for c in garanties]) > 0).cast(pl.Int8).alias("TOP_GARANTIE_IMMO"))
    return df.with_columns(exprs) if exprs else df


# ---------------------------------------------------------------------------
# B. Historique de défaut : récence et durée
# ---------------------------------------------------------------------------

def ajouter_historique_defaut(
    df: pl.DataFrame,
    date_obs: str,
    col_dfo: str = "DATE_DERNIER_DFO",
    col_sain: str = "DATE_SAIN",
) -> pl.DataFrame:
    """Récence du dernier défaut, durée du défaut, ancienneté du retour au sain (en mois).
    Une date postérieure à la date d'observation est neutralisée (sinon fuite).
    Manquant = jamais en défaut : conservé comme modalité à part au binning."""
    if not _presentes(df, [date_obs, col_dfo, col_sain], "historique de défaut"):
        return df
    obs = pl.col(date_obs)
    d = df.with_columns(en_date(df, col_dfo).alias("_dfo"), en_date(df, col_sain).alias("_sain"))
    dfo = pl.when(pl.col("_dfo") <= obs).then(pl.col("_dfo"))
    sain = pl.when(pl.col("_sain") <= obs).then(pl.col("_sain"))
    return d.with_columns(
        TOP_HISTO_DEFAUT=dfo.is_not_null().cast(pl.Int8),
        RECENCE_DFO_MOIS=(obs - dfo).dt.total_days() / MOIS,
        DUREE_DFO_MOIS=(sain - dfo).dt.total_days().clip(lower_bound=0) / MOIS,
        RECENCE_SAIN_MOIS=(obs - sain).dt.total_days() / MOIS,
    ).drop("_dfo", "_sain")


def ajouter_historique_arrieres(df: pl.DataFrame) -> pl.DataFrame:
    """Résumés des fenêtres glissantes d'arriérés et de défaut (M-1 à M-48) :
    nombre de fenêtres touchées, pire arriéré sur 12 mois, récence du dernier arriéré."""
    exprs = []
    dfo = [f"Dfo_max_{f}" for f, _ in FENETRES if f"Dfo_max_{f}" in df.columns]
    arr = [(f"NBJ_ARR_SS_max_{f}", m) for f, m in FENETRES if f"NBJ_ARR_SS_max_{f}" in df.columns]
    if dfo:
        exprs.append(pl.sum_horizontal([(_n0(c) > 0).cast(pl.Int8) for c in dfo]).alias("NB_FENETRES_DFO_48M"))
    else:
        print("[feature_engineering] NB_FENETRES_DFO_48M non créée : fenêtres Dfo_max_* absentes")
    if arr:
        exprs.append(pl.sum_horizontal([(_n0(c) > 0).cast(pl.Int8) for c, _ in arr]).alias("NB_FENETRES_ARR_48M"))
        douze = [c for c, m in arr if m <= 12]
        if douze:
            exprs.append(pl.max_horizontal([_n0(c) for c in douze]).alias("MAX_NBJ_ARR_12M"))
        recence = pl.lit(None, dtype=pl.Float64)
        for c, m in reversed(arr):            # la fenêtre la plus récente l'emporte
            recence = pl.when(_n0(c) > 0).then(pl.lit(float(m))).otherwise(recence)
        exprs.append(recence.alias("RECENCE_ARR_MOIS"))
    else:
        print("[feature_engineering] variables d'arriérés non créées : fenêtres NBJ_ARR_SS_max_* absentes")
    return df.with_columns(exprs) if exprs else df


# ---------------------------------------------------------------------------
# C. Fonctionnement du compte courant
# ---------------------------------------------------------------------------

def ajouter_comportement_compte(df: pl.DataFrame) -> pl.DataFrame:
    """Débit non autorisé, intensité débitrice sur 15 mois, activité, solde minimal."""
    exprs = []
    if "CRTOC_IND_0301" in df.columns:
        exprs.append((_n0("CRTOC_IND_0301") > 0).cast(pl.Int8).alias("TOP_DEBIT_NON_AUTO"))
    elif _presentes(df, ["RATIO_NBJDEB_NONAUTO_SUR_NBJDEB"], "TOP_DEBIT_NON_AUTO"):
        exprs.append((_num("RATIO_NBJDEB_NONAUTO_SUR_NBJDEB") > 0).cast(pl.Int8).alias("TOP_DEBIT_NON_AUTO"))
    if _presentes(df, ["CRTAD_AG_NBJDE_BA"], "PART_JOURS_DEBITEURS_15M"):
        exprs.append((_n0("CRTAD_AG_NBJDE_BA") / (15 * MOIS)).clip(0, 1).alias("PART_JOURS_DEBITEURS_15M"))
    if _presentes(df, ["CRTAD_IND_0042"], "LOG_NB_ECRITURES"):
        exprs.append(_n0("CRTAD_IND_0042").clip(lower_bound=0).log1p().alias("LOG_NB_ECRITURES"))
        exprs.append((_n0("CRTAD_IND_0042") == 0).cast(pl.Int8).alias("TOP_COMPTE_INACTIF"))
    if _presentes(df, ["MINSOLDE_PAR"], "TOP_SOLDE_MIN_NEGATIF"):
        exprs.append((_num("MINSOLDE_PAR") < 0).cast(pl.Int8).alias("TOP_SOLDE_MIN_NEGATIF"))
    return df.with_columns(exprs) if exprs else df


# ---------------------------------------------------------------------------
# D. Flux et patrimoine
# ---------------------------------------------------------------------------

def ajouter_flux_patrimoine(df: pl.DataFrame) -> pl.DataFrame:
    """Sorties / entrées d'argent sur 12 mois, avoirs et couverture des avoirs en mois d'entrées."""
    exprs = []
    deb, cre = "CRTOC_AG_LIGDEB_IND_0015", "CRTOC_AG_LIGCRE_IND_0009"
    if _presentes(df, [deb, cre], "RATIO_DEBIT_CREDIT_12M"):
        exprs.append(pl.when(_num(cre) > 0).then(_num(deb).abs() / _num(cre))
                     .alias("RATIO_DEBIT_CREDIT_12M"))
        exprs.append(_n0(cre).clip(lower_bound=0).log1p().alias("LOG_FLUX_CREDITEURS_12M"))
    if _presentes(df, ["ENCOURS_PAR"], "LOG_AVOIRS"):
        exprs.append(_n0("ENCOURS_PAR").clip(lower_bound=0).log1p().alias("LOG_AVOIRS"))
        exprs.append((_n0("ENCOURS_PAR") <= 0).cast(pl.Int8).alias("TOP_SANS_AVOIRS"))
        if cre in df.columns:
            exprs.append(pl.when(_num(cre) > 0).then(_n0("ENCOURS_PAR") / (_num(cre) / 12))
                         .alias("MOIS_ENTREES_EN_AVOIRS"))
    return df.with_columns(exprs) if exprs else df


# ---------------------------------------------------------------------------
# E. Âge et ancienneté
# ---------------------------------------------------------------------------

def ajouter_age_anciennete(
    df: pl.DataFrame,
    col_age_mois: str = "AGE_PP_EN_MOIS",
    col_anc_mois: str = "ANCIENNETE_modif",
) -> pl.DataFrame:
    """Âge d'entrée en relation et part de la vie adulte passée comme client."""
    if not _presentes(df, [col_age_mois, col_anc_mois], "AGE_ENTREE_ANS / PART_VIE_CLIENT"):
        return df
    age, anc = _num(col_age_mois), _num(col_anc_mois)
    vie_adulte = (age - 18 * 12).clip(lower_bound=1)
    return df.with_columns(
        AGE_ENTREE_ANS=((age - anc) / 12).clip(lower_bound=0),
        PART_VIE_CLIENT=(anc / vie_adulte).clip(0, 1),
    )


# ---------------------------------------------------------------------------
# F. Qualité de l'information (« thin file »)
# ---------------------------------------------------------------------------

def ajouter_nb_manquants(df: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    """Nombre de variables manquantes parmi `cols` : proxy d'un dossier peu renseigné."""
    cols = [c for c in cols if c in df.columns]
    if not cols:
        print("[feature_engineering] NB_MANQUANTS non créée : aucune colonne disponible")
        return df
    return df.with_columns(
        NB_MANQUANTS=pl.sum_horizontal([pl.col(c).is_null().cast(pl.Int16) for c in cols])
    )


# ---------------------------------------------------------------------------
# G. Interaction apprise par arbre (train seulement)
# ---------------------------------------------------------------------------

def _x_arbre(df: pl.DataFrame, variables: list[str]) -> np.ndarray:
    return df.select(pl.col(v).cast(pl.Float64) for v in variables).fill_null(-999).to_numpy()


def ajuster_interaction_arbre(
    df: pl.DataFrame,
    variables: list[str],
    target: str,
    profondeur: int = 2,
    min_part_feuille: float = 0.05,
) -> DecisionTreeClassifier:
    """Arbre peu profond croisant 2-3 variables, ajusté sur l'échantillon d'apprentissage.
    Manquants codés -999 : ils forment leur propre branche."""
    return DecisionTreeClassifier(
        max_depth=profondeur,
        min_samples_leaf=max(int(min_part_feuille * df.height), 1),
        random_state=0,
    ).fit(_x_arbre(df, variables), df[target].to_numpy())


def appliquer_interaction_arbre(
    df: pl.DataFrame, arbre: DecisionTreeClassifier, variables: list[str], nom: str
) -> pl.DataFrame:
    """Ajoute la variable qualitative « feuille de l'arbre » (seuils gelés)."""
    feuilles = arbre.apply(_x_arbre(df, variables))
    return df.with_columns(pl.Series(nom, [f"F{f}" for f in feuilles]))


def feature_interaction_arbre(
    train: pl.DataFrame,
    test: pl.DataFrame,
    variables: list[str],
    target: str,
    nom: str,
    profondeur: int = 2,
    min_part_feuille: float = 0.05,
) -> tuple[pl.DataFrame, pl.DataFrame, DecisionTreeClassifier]:
    """Raccourci : arbre ajusté sur `train`, appliqué à `train` et `test`."""
    arbre = ajuster_interaction_arbre(train, variables, target, profondeur, min_part_feuille)
    return (appliquer_interaction_arbre(train, arbre, variables, nom),
            appliquer_interaction_arbre(test, arbre, variables, nom), arbre)


# ---------------------------------------------------------------------------
# Évaluation : Gini univarié (à calculer sur le TRAIN)
# ---------------------------------------------------------------------------

def gini_univarie(
    df: pl.DataFrame, variables: list[str], target: str, n_quantiles: int = 10
) -> pl.DataFrame:
    """Gini de chaque variable après découpage en quantiles + classe « manquant ».
    Le taux de défaut par classe sert de score : la mesure capte aussi les
    relations non monotones. Gini = 2 * AUC - 1."""
    y = df[target].to_numpy()
    lignes = []
    for v in variables:
        col = pl.col(v)
        if df[v].dtype.is_numeric() and df[v].n_unique() > n_quantiles:
            classe = col.cast(pl.Float64).qcut(n_quantiles, allow_duplicates=True).cast(pl.String)
        else:
            classe = col.cast(pl.String)
        d = df.select(classe.fill_null("MANQUANT").alias("c"), pl.col(target).alias("y"))
        score = d.with_columns(pl.col("y").mean().over("c").alias("s"))["s"].to_numpy()
        lignes.append({
            "variable": v,
            "gini": 2 * roc_auc_score(y, score) - 1 if 0 < y.sum() < len(y) else np.nan,
            "taux_manquant": df[v].null_count() / df.height,
            "n_classes": d["c"].n_unique(),
        })
    return pl.DataFrame(lignes).sort("gini", descending=True, nulls_last=True)
