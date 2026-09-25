"""
Analyse exploratoire
"""
from __future__ import annotations

import numpy as np
import polars as pl
from scipy import stats


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
            t.write_excel(workbook=wb, worksheet=nom[:31], autofit=True)
