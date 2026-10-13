"""
Analyse exploratoire
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import polars as pl
import seaborn as sns
from scipy import stats

from .binning import woe_iv_table, _classes

def correlation_woe(df: pl.DataFrame, columns: list[str], target: str, n_bins: int = 10,
                    max_modalites: int = 10) -> pl.DataFrame:
    """Matrice de corrélation de Pearson entre variables encodées en WOE (découpage fin, missing = classe
    à part). L'encodage met quantitatives et qualitatives sur la même échelle : la matrice couvre aussi
    les redondances entre variables de natures différentes et les relations non monotones."""
    n_bad = df[target].sum()
    n_good = df.height - n_bad
    bad, tot = pl.col("y").sum().over("classe"), pl.len().over("classe")
    woe = (((tot - bad + 0.5) / (n_good + 0.5)) / ((bad + 0.5) / (n_bad + 0.5))).log()
    W = np.column_stack([
        _classes(df, c, n_bins, max_modalites).with_columns(df[target].alias("y")).select(woe).to_series().to_numpy()
        for c in columns
    ]).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        R = np.nan_to_num(np.corrcoef(W, rowvar=False))
    np.fill_diagonal(R, 1.0)
    return pl.DataFrame({"variable": columns, **{c: R[:, j] for j, c in enumerate(columns)}})

def _vif(R: np.ndarray) -> np.ndarray:
    """VIF à partir d'une matrice de corrélation : diagonale de son inverse."""
    return np.diag(np.linalg.pinv(R))


def selection_finale(fiche: pl.DataFrame, corr: pl.DataFrame, n_min: int = 45, n_max: int = 55,
                     iv_min: float = 0.02, missing_max: float = 0.95, seuil_corr: float = 0.8,
                     seuil_corr_min: float = 0.5, pas: float = 0.05, seuil_vif: float = 5.0
                     ) -> tuple[pl.DataFrame, dict]:
    """Sélection des variables transmises à la discrétisation, avec un nombre cible entre n_min et n_max.
    fiche : colonnes variable, iv, taux_missing, quasi_constante ; corr : sortie de correlation_woe.
    Étapes, dans l'ordre :
    1. filtres univariés : quasi constante, taux_missing >= missing_max, IV < iv_min ;
    2. redondance : parcours par IV décroissante, une variable est écartée si sa corrélation avec une
       variable déjà retenue atteint le seuil. Tant qu'il reste plus de n_max variables, le seuil est
       abaissé de `pas`, sans descendre sous seuil_corr_min ;
    3. multicolinéarité : tant qu'un VIF dépasse seuil_vif, la variable de plus fort VIF est écartée ;
    4. s'il reste plus de n_max variables, seules les n_max de meilleur IV sont gardées.
    Renvoie (fiche complétée de statut, redondante_avec, corr_max, vif ; bilan de la sélection)."""
    cols = corr["variable"].to_list()
    pos = {c: i for i, c in enumerate(cols)}
    R = corr.select(cols).to_numpy().astype(np.float64)

    # 1. Filtres univariés
    motif = {}
    for r in fiche.iter_rows(named=True):
        if r["quasi_constante"]:
            motif[r["variable"]] = "écartée : quasi constante"
        elif r["taux_missing"] >= missing_max:
            motif[r["variable"]] = "écartée : trop de manquants"
        elif r["iv"] < iv_min:
            motif[r["variable"]] = "écartée : non discriminante"
    eligibles = (fiche.filter(~pl.col("variable").is_in(list(motif)) & pl.col("variable").is_in(cols))
                 .sort("iv", descending=True)["variable"].to_list())

    # 2. Redondance, seuil abaissé tant que la cible n'est pas atteinte
    for seuil in np.arange(seuil_corr, seuil_corr_min - 1e-9, -pas):
        retenues, redondante = [], {}
        for v in eligibles:
            liees = [k for k in retenues if abs(R[pos[v], pos[k]]) >= seuil]
            if liees:
                redondante[v] = max(liees, key=lambda k: abs(R[pos[v], pos[k]]))
            else:
                retenues.append(v)
        if len(retenues) <= n_max:
            break
    n_redondance = len(retenues)

    # 3. Multicolinéarité
    ecartees_vif = []
    while len(retenues) > 1:
        idx = [pos[v] for v in retenues]
        vifs = _vif(R[np.ix_(idx, idx)])
        if vifs.max() <= seuil_vif:
            break
        ecartees_vif.append(retenues.pop(int(vifs.argmax())))

    # 4. Plafond : variables de meilleur IV
    retenues, au_dela = retenues[:n_max], retenues[n_max:]
    idx = [pos[v] for v in retenues]
    vif_final = dict(zip(retenues, _vif(R[np.ix_(idx, idx)]).tolist())) if retenues else {}

    motif |= {v: "écartée : redondante" for v in redondante}
    motif |= {v: "écartée : multicolinéarité (VIF)" for v in ecartees_vif}
    motif |= {v: "écartée : au-delà du nombre cible (IV plus faible)" for v in au_dela}
    corr_max = {v: float(R[pos[v], pos[k]]) for v, k in redondante.items()}

    out = fiche.with_columns(
        statut=pl.col("variable").replace_strict(motif, default="retenue", return_dtype=pl.String),
        redondante_avec=pl.col("variable").replace_strict(redondante, default=None, return_dtype=pl.String),
        corr_max=pl.col("variable").replace_strict(corr_max, default=None, return_dtype=pl.Float64),
        vif=pl.col("variable").replace_strict(vif_final, default=None, return_dtype=pl.Float64),
    ).sort("iv", descending=True)
    bilan = {
        "n_depart": fiche.height, "n_eligibles": len(eligibles), "seuil_corr_retenu": round(float(seuil), 2),
        "n_apres_redondance": n_redondance, "n_ecartees_vif": len(ecartees_vif),
        "n_au_dela_cible": len(au_dela), "n_retenues": len(retenues),
        "cible_atteinte": n_min <= len(retenues) <= n_max,
    }
    return out, bilan


# ---------------------------------------------------------------------------
# Stats descriptives generales 
# ---------------------------------------------------------------------------

def nan_to_null(df: pl.DataFrame) -> pl.DataFrame:
    """Convertit les NaN des colonnes float en null (sinon null_count() ne les voit pas)."""
    return df.with_columns(pl.col(pl.Float32, pl.Float64).fill_nan(None))


def _numeric_cols(df: pl.DataFrame, exclude: list[str] | None) -> list[str]:
    exclude = set(exclude or [])
    return [c for c, t in df.schema.items() if t.is_numeric() and c not in exclude]


def _qualitative_cols(df: pl.DataFrame, exclude: list[str] | None) -> list[str]:
    exclude = set(exclude or [])
    quali = (pl.String, pl.Categorical, pl.Enum, pl.Boolean)
    return [c for c, t in df.schema.items() if isinstance(t, quali) and c not in exclude]

def missing_report(df: pl.DataFrame) -> pl.DataFrame:
    """Taux et volume de valeurs manquantes par colonne, trie decroissant."""
    n = df.height
    report = pl.DataFrame({
        "variable": df.columns,
        "dtype":[str(t) for t in df.dtypes],
        "n_missing": [df[c].null_count() for c in df.columns],
    }).with_columns(
        ((pl.col("n_missing") / n).alias("taux_missing")),
        ((n - pl.col("n_missing")).alias("n_present"))
    )
    return report.sort("taux_missing", descending = True)


def univariate_summary(df: pl.DataFrame, columns: list[str] | None = None, exclude=None) -> pl.DataFrame:
    """Resumé des variables quantitatives : n, missing, moyenne, ecart-type, quantiles, min/max."""
    cols = columns or _numeric_cols(df, exclude)
    rows = []
    for c in cols:
        s = df[c]
        rows.append({
            "variable": c,
            "n": s.len() - s.null_count(),
            "taux_missing": s.null_count() / s.len(),
            "mean": s.mean(),
            "std": s.std(),
            "min": s.min(),
            "p01": s.quantile(0.01),
            "p25": s.quantile(0.25),
            "median": s.median(),
            "p75": s.quantile(0.75),
            "p99": s.quantile(0.99),
            "max": s.max(),
        })
    return pl.DataFrame(rows)


def modality_summary(df: pl.DataFrame, columns: list[str] | None = None, exclude=None) -> pl.DataFrame:
    """Resumé des variables qualitatives : nb modalites, modalite la plus frequente, taux missing."""
    cols = columns or _qualitative_cols(df, exclude)
    rows = []
    for c in cols:
        s = df[c]
        n_valid = s.len() - s.null_count()
        vc = s.drop_nulls().value_counts(sort=True)
        top_row = vc.row(0) if vc.height else (None, 0)
        rows.append({
            "variable": c,
            "n_modalites": s.drop_nulls().n_unique(),
            "taux_missing": s.null_count() / s.len(),
            "modalite_top": None if top_row[0] is None else top_row[0],
            "freq_top": top_row[1] / n_valid if n_valid else None,
        })
    return pl.DataFrame(rows)


def outlier_report(df: pl.DataFrame, columns: list[str] | None = None, method: str = "iqr", k: float = 1.5, exclude=None) -> pl.DataFrame:
    """Detection d'outliers univaries par methode IQR (par defaut) ou z-score."""
    cols = columns or _numeric_cols(df, exclude)
    rows = []
    for c in cols:
        s = df[c].drop_nulls()
        if method == "iqr":
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            low, high = q1 - k * iqr, q3 + k * iqr
            mask = (s < low) | (s > high)
        elif method == "zscore":
            z = (s - s.mean()) / s.std()
            mask = z.abs() > k
            low, high = None, None
        else:
            raise ValueError("method doit etre 'iqr' ou 'zscore'")
        rows.append({
            "variable": c,
            "n_outliers": int(mask.sum()),
            "taux_outliers": mask.sum() / s.len() if s.len() else None,
            "borne_basse": low,
            "borne_haute": high,
        })
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# Taux de risque par modalite / segment candidat
# ---------------------------------------------------------------------------

def risk_rate_by_modality(df: pl.DataFrame, var: str, target: str) -> pl.DataFrame:
    """Taux de defaut, effectif et poids par modalite (missing = modalite a part entiere)."""
    total = df.height
    return (
        df.group_by(var)
        .agg(n=pl.len(), n_defaut=pl.col(target).sum())
        .with_columns(
            (pl.col("n_defaut") / pl.col("n")).alias("taux_defaut"),
            (pl.col("n") / total).alias("part_population"),
        )
        .sort("taux_defaut", descending=True, nulls_last=True)
    )


def chi2_association(df: pl.DataFrame, var: str, target: str) -> dict:
    """Test du Chi2 d'indépendance entre une variable qualitative et la cible + V de Cramer (ch. 3.2)."""
    sub = df.select(pl.col(var).cast(pl.String), pl.col(target)).drop_nulls(target)
    ct = sub.pivot(index=var, on=target, values=target, aggregate_function="len").fill_null(0)
    counts = ct.drop(var).to_numpy()
    chi2, pval, dof, _ = stats.chi2_contingency(counts)
    n = counts.sum()
    r, k = counts.shape
    cramers_v = np.sqrt((chi2 / n) / (min(r - 1, k - 1) or 1))
    return {"variable": var, "chi2": chi2, "pvalue": pval, "dof": dof, "cramers_v": cramers_v}


# ---------------------------------------------------------------------------
# Associations quanti/quanti et multicolinearite 
# ---------------------------------------------------------------------------

def correlation_matrix(df: pl.DataFrame, columns: list[str], method: str = "pearson") -> pl.DataFrame:
    """Matrice de correlation (pearson, spearman), missing exclus paire par paire."""
    if method not in ("pearson", "spearman"):
        raise ValueError("method doit être 'pearson' ou 'spearman'")
    exprs = [pl.corr(c1, c2, method=method).alias(f"{c1}|{c2}") for c1 in columns for c2 in columns]
    values = df.select(exprs).row(0)
    k = len(columns)
    data = {c2: [values[i * k + j] for i in range(k)] for j, c2 in enumerate(columns)}
    return pl.DataFrame({"variable": columns, **data})


def vif_report(df: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    """Variance Inflation Factor par variable (nécessite statsmodels ; conversion numpy locale)."""
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    sub = df.select(columns).drop_nulls()
    X = sub.to_numpy()
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    X = np.column_stack([np.ones(X.shape[0]), X])
    vifs = [variance_inflation_factor(X, i) for i in range(1, X.shape[1])]
    return pl.DataFrame({"variable": columns, "VIF": vifs}).sort("VIF", descending=True)

def appliquer_filtres(df: pl.DataFrame, filtres: dict[str, pl.Expr], target: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Applique les filtres d'exclusion dans l'ordre ; renvoie la base filtrée et le suivi avant/après."""
    def bilan(etape, d):
        return {"etape": etape, "n": d.height, "n_defaut": d[target].sum(), "taux_defaut": d[target].mean()}

    suivi = [bilan("base initiale", df)]
    for nom, exclusion in filtres.items():
        df = df.filter(~exclusion.fill_null(False))
        suivi.append(bilan(f"exclusion : {nom}", df))
    return df, pl.DataFrame(suivi)


def controle_groupes(df: pl.DataFrame, groupes: dict[str, pl.Expr], target: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Volume et taux de défaut de groupes définis par des conditions, et leur recouvrement."""
    volumes = pl.DataFrame([
        df.filter(expr.fill_null(False)).select(
            pl.lit(nom).alias("groupe"),
            pl.len().alias("n"),
            pl.col(target).sum().alias("n_defaut"),
            pl.col(target).mean().alias("taux_defaut"),
        ).row(0, named=True)
        for nom, expr in groupes.items()
    ])
    recouvrement = (
        df.select(**{nom: expr.fill_null(False) for nom, expr in groupes.items()})
        .group_by(list(groupes)).len().sort("len", descending=True)
    )
    return volumes, recouvrement


# ---------------------------------------------------------------------------
# Dossiers à plusieurs tiers (nb_tiers > 1)
# ---------------------------------------------------------------------------

SUFFIXES_JOINTS = ("_min", "_max", "_med")


def unifier_joints(df: pl.DataFrame, tronques: dict[str, str] | None = None,
                   max_seul: tuple[str, ...] = ()) -> tuple[pl.DataFrame, dict]:
    """Remplit chaque variable individuelle par sa version _med (dossiers joints), ou _max pour les codes
    listés dans max_seul ; supprime les colonnes _min/_max/_med traitées.
    tronques : racine tronquée par SAS (32 car.) -> nom de la variable de base."""
    tronques = tronques or {}
    racines = sorted({c.rsplit("_", 1)[0] for c in df.columns if c.endswith(SUFFIXES_JOINTS)})
    base_de = {r: tronques.get(r, r) for r in racines}
    par_med = {r: b for r, b in base_de.items() if f"{r}_med" in df.columns and b in df.columns}
    par_max = {r: base_de[r] for r in max_seul if f"{r}_max" in df.columns}

    exprs = [pl.coalesce(b, f"{r}_med").alias(b) for r, b in par_med.items()]
    exprs += [pl.coalesce([b, f"{r}_max"] if b in df.columns else [f"{r}_max"]).alias(b) for r, b in par_max.items()]
    traitees = par_med | par_max
    a_supprimer = [f"{r}{s}" for r in traitees for s in SUFFIXES_JOINTS if f"{r}{s}" in df.columns]

    rapport = {
        "unifiees_med": len(par_med),
        "unifiees_max": sorted(par_max.values()),
        "non_traitees": [r for r in racines if r not in traitees],
    }
    return df.with_columns(exprs).drop(a_supprimer), rapport


# ---------------------------------------------------------------------------
# Statistiques descriptives par segment
# ---------------------------------------------------------------------------

def constant_report(df: pl.DataFrame, columns: list[str] | None = None, seuil: float = 0.99) -> pl.DataFrame:
    """Part de la valeur la plus fréquente (manquant compris) : une variable quasi constante n'apporte rien."""
    rows = []
    for c in columns or df.columns:
        vc = df[c].value_counts(sort=True)
        rows.append({"variable": c, "n_valeurs": vc.height, "valeur_top": str(vc[c][0]),
                     "part_top": vc["count"][0] / df.height})
    return (pl.DataFrame(rows)
            .with_columns(quasi_constante=pl.col("part_top") >= seuil)
            .sort("part_top", descending=True))


def paires_redondantes(df: pl.DataFrame, columns: list[str], seuil: float = 0.8,
                       n_max: int = 50_000, seed: int = 0) -> pl.DataFrame:
    """Paires de variables numériques avec |Spearman| >= seuil, manquants exclus paire par paire.
    Calcul matriciel (corrélation des rangs globaux sur les lignes communes) sur un échantillon de n_max lignes.
    Les colonnes constantes sont ignorées ; une paire sans variance sur ses lignes communes n'est pas retenue."""
    d = df.select(columns)
    if d.height > n_max:
        d = d.sample(n_max, seed=seed)
    columns = [c for c in columns if d[c].drop_nulls().n_unique() > 1]
    X = d.select(pl.col(columns).cast(pl.Float64).rank("average")).to_numpy().astype(np.float64)
    X = X - np.nanmean(X, axis=0)
    M = (~np.isnan(X)).astype(np.float64)
    Xz = np.nan_to_num(X)
    n = M.T @ M
    sx = Xz.T @ M
    sxx = (Xz ** 2).T @ M
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = Xz.T @ Xz - sx * sx.T / n
        var = sxx - sx ** 2 / n
        var[var <= 1e-9 * sxx] = np.nan
        r = np.clip(cov / np.sqrt(var * var.T), -1.0, 1.0)
    i, j = np.triu_indices(len(columns), k=1)
    rij = r[i, j]
    garde = np.abs(np.nan_to_num(rij)) >= seuil
    noms = np.array(columns)
    return pl.DataFrame({
        "var1": noms[i[garde]], "var2": noms[j[garde]],
        "spearman": rij[garde], "n_commun": n[i, j][garde].astype(int),
    }).sort(pl.col("spearman").abs(), descending=True)


def cote_a_cote(tables: dict[str, pl.DataFrame], cle: str, colonnes: list[str]) -> pl.DataFrame:
    """Juxtapose, pour chaque segment, les colonnes choisies d'une même table (jointure sur la clé)."""
    out = None
    for s, t in tables.items():
        t = t.select(cle, *[pl.col(c).alias(f"{c}_{s}") for c in colonnes])
        out = t if out is None else out.join(t, on=cle, how="full", coalesce=True)
    return out


def preselection_variables(fiche: pl.DataFrame, paires: pl.DataFrame, iv_min: float = 0.02,
                           iv_suspect: float = 0.5, missing_max: float = 0.95, iv_rare: float = 0.1) -> pl.DataFrame:
    """Statut provisoire de chaque variable d'un segment.
    fiche : colonnes variable, taux_missing, quasi_constante, iv ; paires : sortie de paires_redondantes.
    Quasi constante : écartée seulement si son IV reste < iv_rare (une valeur rare mais très
    discriminante, comme un défaut récent, est gardée avec une alerte).
    Redondance : parcours par IV décroissante, une variable est écartée si elle est corrélée
    à une variable déjà retenue (qui devient sa représentante)."""
    constante_inutile = pl.col("quasi_constante") & (pl.col("iv") < iv_rare)
    eligibles = fiche.filter(
        ~constante_inutile & (pl.col("taux_missing") < missing_max) & (pl.col("iv") >= iv_min)
    ).sort("iv", descending=True)["variable"].to_list()
    voisins = {}
    for a, b in zip(paires["var1"], paires["var2"]):
        voisins.setdefault(a, set()).add(b)
        voisins.setdefault(b, set()).add(a)
    retenues, redondante = [], {}
    for v in eligibles:
        representante = next((k for k in retenues if k in voisins.get(v, ())), None)
        if representante is None:
            retenues.append(v)
        else:
            redondante[v] = representante
    return fiche.with_columns(
        redondante_avec=pl.col("variable").replace_strict(redondante, default=None, return_dtype=pl.String)
    ).with_columns(
        statut=pl.when(constante_inutile).then(pl.lit("écartée : quasi constante"))
        .when(pl.col("taux_missing") >= missing_max).then(pl.lit("écartée : trop de manquants"))
        .when(pl.col("iv") < iv_min).then(pl.lit("écartée : non discriminante (IV < 0,02)"))
        .when(pl.col("redondante_avec").is_not_null()).then(pl.lit("écartée : redondante"))
        .when(pl.col("iv") >= iv_suspect).then(pl.lit("candidate : IV > 0,5, vérifier la fuite"))
        .when(pl.col("quasi_constante")).then(pl.lit("candidate : valeur rare, à surveiller"))
        .otherwise(pl.lit("candidate"))
    )


def exporter_excel(feuilles: dict[str, pl.DataFrame], chemin) -> None:
    """Un onglet par table (nom tronqué à 31 caractères, limite Excel)."""
    from pathlib import Path

    import xlsxwriter

    Path(chemin).parent.mkdir(parents=True, exist_ok=True)
    with xlsxwriter.Workbook(chemin) as wb:
        for nom, t in feuilles.items():
            t.with_columns(pl.col(pl.Float32, pl.Float64).fill_nan(None)).write_excel(
                workbook=wb, worksheet=nom[:31], autofit=True)


# ---------------------------------------------------------------------------
# Associations avec la cible et entre variables (ch. 3)
# ---------------------------------------------------------------------------

def _fini(x) -> float | None:
    return None if x is None or not np.isfinite(x) else float(x)


def liaison_quanti_cible(df: pl.DataFrame, columns: list[str], target: str,
                         n_max: int = 50_000, seed: int = 0) -> pl.DataFrame:
    """Pearson (= point-bisérial pour une cible 0/1), Spearman et Kendall de chaque variable quantitative
    avec la cible, manquants exclus. Kendall est calculé sur un échantillon de n_max lignes."""
    ech = df.sample(n_max, seed=seed) if df.height > n_max else df
    rows = []
    for c in columns:
        d = df.select(pl.col(c).cast(pl.Float64), target).drop_nulls()
        x, y = d[c].to_numpy(), d[target].to_numpy()
        row = {"variable": c, "n": d.height, "pearson": None, "spearman": None, "kendall": None, "pvalue_spearman": None}
        if d.height > 2 and np.ptp(x) > 0 and np.ptp(y) > 0:
            rho, p = stats.spearmanr(x, y)
            e = ech.select(pl.col(c).cast(pl.Float64), target).drop_nulls()
            tau = stats.kendalltau(e[c].to_numpy(), e[target].to_numpy())[0] if e.height > 2 else None
            row.update(pearson=_fini(stats.pearsonr(x, y)[0]), spearman=_fini(rho), kendall=_fini(tau),
                       pvalue_spearman=_fini(p))
        rows.append(row)
    schema = {"variable": pl.String, "n": pl.Int64, "pearson": pl.Float64, "spearman": pl.Float64,
              "kendall": pl.Float64, "pvalue_spearman": pl.Float64}
    return pl.DataFrame(rows, schema=schema).sort(pl.col("spearman").abs(), descending=True, nulls_last=True)


def cramers_v_matrix(df: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    """Matrice des V de Cramer entre variables qualitatives (missing = modalité à part entière)."""
    codes = df.select((pl.col(c).cast(pl.String).fill_null("missing").rank("dense") - 1).cast(pl.Int64).alias(c)
                      for c in columns).to_numpy() if columns else np.empty((0, 0), dtype=np.int64)
    k = codes.max(axis=0) + 1 if codes.size else np.array([], dtype=np.int64)
    V = np.eye(len(columns))
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            v = np.nan
            if min(k[i], k[j]) > 1:
                ct = np.bincount(codes[:, i] * k[j] + codes[:, j], minlength=k[i] * k[j]).reshape(k[i], k[j])
                chi2 = stats.chi2_contingency(ct, correction=False)[0]
                v = np.sqrt(chi2 / ct.sum() / (min(k[i], k[j]) - 1))
            V[i, j] = V[j, i] = v
    return pl.DataFrame({"variable": columns, **{c: V[:, j] for j, c in enumerate(columns)}})


def _eta2(df: pl.DataFrame, quanti: list[str], quali: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """eta² = SCE inter / SCE totale de chaque variable quanti selon les modalités de quali (manquants de la
    variable quanti exclus). Renvoie (eta2, n, nombre de groupes) par variable."""
    g = df.filter(pl.col(quali).is_not_null()).group_by(quali).agg(
        *[pl.col(c).cast(pl.Float64).mean().alias(f"m|{c}") for c in quanti],
        *[pl.col(c).cast(pl.Float64).var(ddof=0).alias(f"v|{c}") for c in quanti],
        *[pl.col(c).count().alias(f"n|{c}") for c in quanti],
    )
    N = g.select([f"n|{c}" for c in quanti]).to_numpy().astype(np.float64)
    M = np.nan_to_num(g.select([f"m|{c}" for c in quanti]).to_numpy().astype(np.float64))
    W = np.nan_to_num(g.select([f"v|{c}" for c in quanti]).to_numpy().astype(np.float64))
    n = N.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        inter = (N * (M - (N * M).sum(axis=0) / n) ** 2).sum(axis=0)
        eta2 = inter / (inter + (N * W).sum(axis=0))
    return eta2, n, (N > 0).sum(axis=0)


def anova_report(df: pl.DataFrame, columns: list[str], groupe: str) -> pl.DataFrame:
    """Liaison de chaque variable quantitative avec une variable qualitative (ex. la cible) :
    ANOVA à un facteur (F, p-value), rapport de corrélation eta² et test de Kruskal-Wallis."""
    eta2, n, k = _eta2(df, columns, groupe)
    rows = []
    for i, c in enumerate(columns):
        row = {"variable": c, "n": int(n[i]), "eta2": _fini(eta2[i]), "F": None, "pvalue_anova": None,
               "H_kruskal": None, "pvalue_kruskal": None}
        if row["eta2"] is not None and k[i] > 1 and n[i] > k[i]:
            F = (eta2[i] / (k[i] - 1)) / ((1 - eta2[i]) / (n[i] - k[i])) if eta2[i] < 1 else np.inf
            groupes = [p[c].to_numpy() for p in df.select(c, groupe).drop_nulls().partition_by(groupe)]
            H, p_h = stats.kruskal(*groupes)
            row.update(F=_fini(F), pvalue_anova=_fini(stats.f.sf(F, k[i] - 1, n[i] - k[i])),
                       H_kruskal=_fini(H), pvalue_kruskal=_fini(p_h))
        rows.append(row)
    schema = {"variable": pl.String, "n": pl.Int64, "eta2": pl.Float64, "F": pl.Float64,
              "pvalue_anova": pl.Float64, "H_kruskal": pl.Float64, "pvalue_kruskal": pl.Float64}
    return pl.DataFrame(rows, schema=schema).sort("eta2", descending=True, nulls_last=True)


def eta2_matrix(df: pl.DataFrame, quanti: list[str], quali: list[str]) -> pl.DataFrame:
    """Rapports de corrélation eta² croisés : une ligne par variable quantitative, une colonne par qualitative."""
    return pl.DataFrame({"variable": quanti, **{q: _eta2(df, quanti, q)[0] for q in quali}}).with_columns(
        pl.col(quali).fill_nan(None))


def paires_fortes(mat: pl.DataFrame, seuil: float, nom: str = "valeur") -> pl.DataFrame:
    """Paires de variables d'une matrice carrée (colonne 'variable' + une colonne par variable) dont la
    valeur absolue atteint le seuil."""
    cols = mat["variable"].to_list()
    M = mat.select(cols).to_numpy().astype(np.float64)
    i, j = np.triu_indices(len(cols), k=1)
    garde = np.abs(np.nan_to_num(M[i, j])) >= seuil
    noms = np.array(cols)
    return pl.DataFrame({"var1": noms[i[garde]], "var2": noms[j[garde]], nom: M[i, j][garde]},
                        schema={"var1": pl.String, "var2": pl.String, nom: pl.Float64}
                        ).sort(pl.col(nom).abs(), descending=True)


def completer_mediane(df: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    """Colonnes numériques avec manquants remplacés par la médiane. Réservé aux diagnostics qui exigent une
    matrice complète (VIF, ACP) : ce n'est pas un traitement des données."""
    return df.select(pl.col(c).cast(pl.Float64).fill_null(pl.col(c).cast(pl.Float64).median()) for c in columns)


def acp_exploratoire(df: pl.DataFrame, columns: list[str], n_composantes: int = 5) -> tuple[pl.DataFrame, pl.DataFrame]:
    """ACP normée (diagonalisation de la matrice de corrélation), manquants remplacés par la médiane.
    Les colonnes constantes sont ignorées.
    Renvoie (variance expliquée par axe, corrélations variables-axes pour les n_composantes premiers axes)."""
    X = completer_mediane(df, columns).to_numpy()
    garde = np.nan_to_num(X.std(axis=0)) > 0
    columns, X = [c for c, g in zip(columns, garde) if g], X[:, garde]
    Z = (X - X.mean(axis=0)) / X.std(axis=0)
    val, vec = np.linalg.eigh(Z.T @ Z / Z.shape[0])
    ordre = np.argsort(val)[::-1]
    val, vec = np.clip(val[ordre], 0, None), vec[:, ordre]
    k = min(n_composantes, len(columns))
    variance = pl.DataFrame({
        "axe": [f"F{i + 1}" for i in range(len(columns))],
        "valeur_propre": val,
        "part_variance": val / val.sum(),
        "part_cumulee": np.cumsum(val) / val.sum(),
    })
    correlations = pl.DataFrame({"variable": columns, **{f"F{i + 1}": vec[:, i] * np.sqrt(val[i]) for i in range(k)}})
    return variance, correlations


def monotonie_report(df: pl.DataFrame, columns: list[str], target: str, n_bins: int = 10, max_modalites: int = 10,
                     iv_min: float = 0.02, seuil_rho: float = 0.7) -> pl.DataFrame:
    """Forme de la relation entre chaque variable quantitative et le taux de défaut, sur un découpage fin
    en n_bins classes (classe 'missing' exclue) :
    rho_classes = Spearman entre le rang de la classe et son taux de défaut (±1 = relation monotone),
    n_inversions = nombre de changements de sens du taux de défaut d'une classe à la suivante.
    Une variable est signalée non monotone si elle discrimine (IV >= iv_min) alors que |rho_classes| < seuil_rho."""
    rows = []
    for c in columns:
        t = woe_iv_table(df, c, target, n_bins, max_modalites)
        tdr = t.filter(pl.col("classe") != "missing")["taux_defaut"].to_numpy()
        pas = np.sign(np.diff(tdr))
        pas = pas[pas != 0]
        rho = stats.spearmanr(np.arange(len(tdr)), tdr)[0] if len(tdr) > 2 and np.ptp(tdr) > 0 else None
        rows.append({"variable": c, "iv": t["iv"].sum(), "n_classes": len(tdr), "rho_classes": _fini(rho),
                     "n_inversions": int((pas[1:] != pas[:-1]).sum()),
                     "tdr_min": _fini(tdr.min()) if len(tdr) else None,
                     "tdr_max": _fini(tdr.max()) if len(tdr) else None})
    schema = {"variable": pl.String, "iv": pl.Float64, "n_classes": pl.Int64, "rho_classes": pl.Float64,
              "n_inversions": pl.Int64, "tdr_min": pl.Float64, "tdr_max": pl.Float64}
    return pl.DataFrame(rows, schema=schema).with_columns(
        non_monotone=(pl.col("iv") >= iv_min) & (pl.col("rho_classes").abs() < seuil_rho)
    ).sort("iv", descending=True)


# ---------------------------------------------------------------------------
# Graphiques (ch. 3)
# ---------------------------------------------------------------------------

def plot_matrice(mat: pl.DataFrame, titre: str = "", vmin: float = -1, vmax: float = 1, cmap: str = "RdBu_r",
                 annot: bool | None = None, figsize: tuple[float, float] | None = None) -> plt.Figure:
    """Heatmap d'une matrice d'association (colonne 'variable' + une colonne par variable).
    Corrélations : vmin=-1, cmap divergente ; V de Cramer ou eta² : vmin=0, cmap='Blues'.
    Les valeurs sont écrites dans les cases tant que la matrice reste lisible (15 variables au plus)."""
    lignes, cols = mat["variable"].to_list(), [c for c in mat.columns if c != "variable"]
    M = mat.select(cols).to_numpy().astype(np.float64)
    if annot is None:
        annot = max(len(lignes), len(cols)) <= 15
    if figsize is None:
        figsize = (max(5, 0.45 * len(cols) + 3), max(4, 0.4 * len(lignes) + 2))
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(M, ax=ax, xticklabels=cols, yticklabels=lignes, vmin=vmin, vmax=vmax, cmap=cmap,
                annot=annot, fmt=".2f", annot_kws={"fontsize": 7}, linewidths=0.5, linecolor="white",
                cbar_kws={"shrink": 0.6})
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
    ax.set_title(titre, fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_acp(variance: pl.DataFrame, correlations: pl.DataFrame, axes: tuple[str, str] = ("F1", "F2"),
             n_axes: int = 10, n_labels: int = 15, figsize: tuple[float, float] = (13, 6)) -> plt.Figure:
    """Éboulis des valeurs propres (part de variance par axe et cumul) et cercle des corrélations sur deux axes.
    Seules les n_labels variables les mieux représentées sur le plan sont étiquetées."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    v = variance.head(n_axes)
    x = np.arange(v.height)
    ax1.bar(x, v["part_variance"].to_list(), color="#1f77b4", alpha=0.75, edgecolor="none", label="Part de variance")
    ax1.plot(x, v["part_cumulee"].to_list(), color="#333333", marker="o", linewidth=1.5, label="Cumul")
    for xi, p in zip(x, v["part_variance"].to_list()):
        ax1.text(xi, p + 0.01, f"{p:.0%}", ha="center", fontsize=8, color="#333333")
    ax1.set_xticks(x)
    ax1.set_xticklabels(v["axe"].to_list(), fontsize=8)
    ax1.set_ylim(0, 1.05)
    ax1.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax1.set_ylabel("Variance expliquée")
    ax1.set_title("Éboulis des valeurs propres", fontsize=10, fontweight="bold")
    ax1.grid(axis="x", visible=False)
    ax1.legend(fontsize=8, loc="center right")

    a, b = axes
    xs, ys = correlations[a].to_numpy(), correlations[b].to_numpy()
    noms = correlations["variable"].to_list()
    etiquetees = set(np.argsort(xs ** 2 + ys ** 2)[::-1][:n_labels].tolist())
    ax2.add_patch(plt.Circle((0, 0), 1, fill=False, color="#7f7f7f", linewidth=1))
    ax2.axhline(0, color="#7f7f7f", linewidth=0.6)
    ax2.axvline(0, color="#7f7f7f", linewidth=0.6)
    for i, (xi, yi) in enumerate(zip(xs, ys)):
        garde = i in etiquetees
        ax2.annotate("", xy=(xi, yi), xytext=(0, 0),
                     arrowprops=dict(arrowstyle="->", color="#1f77b4", alpha=0.85 if garde else 0.25, linewidth=1))
        if garde:
            ax2.text(xi * 1.06, yi * 1.06, noms[i], fontsize=7, color="#333333",
                     ha="left" if xi >= 0 else "right", va="center")
    part = dict(zip(variance["axe"].to_list(), variance["part_variance"].to_list()))
    ax2.set_xlim(-1.25, 1.25)
    ax2.set_ylim(-1.25, 1.25)
    ax2.set_aspect("equal")
    ax2.set_xlabel(f"{a} ({part[a]:.1%})")
    ax2.set_ylabel(f"{b} ({part[b]:.1%})")
    ax2.set_title("Cercle des corrélations", fontsize=10, fontweight="bold")
    ax2.grid(visible=False)
    fig.tight_layout()
    return fig


def plot_iv_correlees(iv: pl.DataFrame, paires: pl.DataFrame, top_n: int = 25, max_liees: int = 4,
                      hauteur: float = 7.5) -> plt.Figure:
    """IV des top_n variables les plus discriminantes (une barre par variable, IV décroissante) avec,
    au-dessus de chaque barre, les variables auxquelles elle est corrélée.
    iv : colonnes variable, iv ; paires : colonnes var1, var2 (paires jugées redondantes).
    Au-delà de max_liees variables liées, seules les plus discriminantes sont nommées."""
    voisins: dict[str, set[str]] = {}
    for a, b in zip(paires["var1"], paires["var2"]):
        voisins.setdefault(a, set()).add(b)
        voisins.setdefault(b, set()).add(a)
    iv_de = dict(zip(iv["variable"], iv["iv"]))

    top = iv.sort("iv", descending=True).head(top_n)
    noms, valeurs = top["variable"].to_list(), top["iv"].to_list()
    liees = [sorted(voisins.get(v, ()), key=lambda x: -iv_de.get(x, 0)) for v in noms]
    x = np.arange(len(noms))

    fig, ax = plt.subplots(figsize=(max(8, 0.62 * len(noms) + 2), hauteur))
    couleurs = ["#1f77b4" if l else "#7f7f7f" for l in liees]
    ax.bar(x, valeurs, color=couleurs, alpha=0.8, edgecolor="none", width=0.7)
    for seuil in (0.02, 0.1, 0.3, 0.5):
        ax.axhline(seuil, color="#333333", linestyle=":", linewidth=0.7)
        ax.text(len(noms) - 0.4, seuil, f"{seuil:g}".replace(".", ","), fontsize=7, color="#333333", va="center")

    for xi, v, l in zip(x, valeurs, liees):
        if l:
            texte = l[:max_liees] + ([f"+ {len(l) - max_liees} autres"] if len(l) > max_liees else [])
            ax.annotate("\n".join(texte), xy=(xi, v), xytext=(0, 4), textcoords="offset points",
                        rotation=90, ha="center", va="bottom", fontsize=7, color="#333333", linespacing=1.35)

    ax.set_ylim(0, max(valeurs, default=1) * 2.1)   # marge haute réservée aux noms des variables liées
    ax.set_xlim(-0.6, len(noms) + 0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(noms, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Information Value")
    ax.grid(axis="x", visible=False)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color="#1f77b4", alpha=0.8),
                       plt.Rectangle((0, 0), 1, 1, color="#7f7f7f", alpha=0.8)],
              labels=["corrélée à au moins une variable (noms au-dessus de la barre)", "aucune variable corrélée"],
              fontsize=8, loc="upper right")
    ax.set_title(f"IV et variables corrélées — {len(noms)} variables les plus discriminantes",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_tdr_classes(df: pl.DataFrame, variables: list[str], target: str, n_bins: int = 10, max_modalites: int = 10,
                     n_cols: int = 3, figsize_par_plot: tuple[float, float] = (4.6, 3.2)) -> plt.Figure:
    """Taux de défaut par classe (découpage fin en n_bins classes) pour chaque variable : fait apparaître
    les relations non monotones. La classe 'missing' est tracée à part (losange) ; le pointillé est le
    taux de défaut global."""
    n_cols = max(1, min(n_cols, len(variables)))
    n_rows = int(np.ceil(len(variables) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize_par_plot[0] * n_cols, figsize_par_plot[1] * n_rows),
                             squeeze=False)
    tdr_global = float(df[target].mean())
    for ax, var in zip(axes.flat, variables):
        t = woe_iv_table(df, var, target, n_bins, max_modalites)
        classes = t.filter(pl.col("classe") != "missing")
        x = np.arange(1, classes.height + 1)
        ax.plot(x, classes["taux_defaut"].to_list(), color="#1f77b4", marker="o", linewidth=2, markersize=5)
        manquant = t.filter(pl.col("classe") == "missing")
        ticks, labels = list(x), [str(i) for i in x]
        if manquant.height:
            ax.plot([classes.height + 1.5], manquant["taux_defaut"].to_list(), color="#7f7f7f", marker="D",
                    markersize=6, linestyle="none")
            ticks, labels = ticks + [classes.height + 1.5], labels + ["NA"]
        ax.axhline(tdr_global, color="black", linestyle="--", linewidth=0.8)
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, fontsize=7)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))
        ax.tick_params(axis="y", labelsize=7)
        ax.set_title(f"{var}  (IV = {t['iv'].sum():.2f})", fontsize=9, fontweight="bold")
    for ax in axes.flat[len(variables):]:
        ax.set_visible(False)
    fig.supxlabel("Classe (rang croissant de la variable)", fontsize=9)
    fig.supylabel("Taux de défaut", fontsize=9)
    fig.tight_layout()
    return fig
