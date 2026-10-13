"""
Stats descriptives — segment comptes joints (nb_tiers > 1)
===========================================================
Toutes les fonctions travaillent sur un polars.DataFrame déjà filtré sur
le segment souhaité (ex. df.filter(pl.col("nb_tiers") > 1)).

Principe : fonctions pures → renvoient des pl.DataFrame ou des figures
matplotlib ; aucune logique métier dans le notebook.
"""
from __future__ import annotations

import math
from typing import Sequence

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import polars as pl
import seaborn as sns
from scipy import stats

# ---------------------------------------------------------------------------
# Palette et style communs
# ---------------------------------------------------------------------------

PALETTE_DEFAUT = "#d62728"   # rouge = défaut
PALETTE_SAIN   = "#1f77b4"   # bleu  = sain
PALETTE_NEUTRE = "#7f7f7f"


def _style_projet() -> None:
    """Applique le style seaborn whitegrid avec police standard."""
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})


_style_projet()


# ---------------------------------------------------------------------------
# 1. Profil global du segment
# ---------------------------------------------------------------------------

def profil_segment(
    df: pl.DataFrame,
    target: str = "defaut",
    id_col: str = "id_client",
    date_col: str | None = None,
) -> pl.DataFrame:
    """
    Indicateurs de haut niveau du segment : volume, taux de défaut global,
    et (si date_col fournie) plage temporelle couverte.

    Paramètres
    ----------
    df       : DataFrame déjà filtré sur le segment.
    target   : nom de la variable cible (0/1).
    id_col   : identifiant unique par observation.
    date_col : colonne de date (optionnel) pour la plage temporelle.

    Retourne
    --------
    pl.DataFrame à une ligne avec les métriques clés.
    """
    n_total   = df.height
    n_defaut  = int(df[target].sum())
    n_sain    = n_total - n_defaut
    tdr       = n_defaut / n_total if n_total else None

    row: dict = {
        "n_total":   n_total,
        "n_defaut":  n_defaut,
        "n_sain":    n_sain,
        "taux_defaut": tdr,
    }

    if date_col and date_col in df.columns:
        row["date_min"] = str(df[date_col].min())
        row["date_max"] = str(df[date_col].max())

    return pl.DataFrame([row])


# ---------------------------------------------------------------------------
# 2. Résumé des variables quantitatives
# ---------------------------------------------------------------------------

def resume_quanti(
    df: pl.DataFrame,
    target: str = "defaut",
    columns: list[str] | None = None,
    exclude: list[str] | None = None,
) -> pl.DataFrame:
    """
    Statistiques descriptives des variables numériques :
    n, taux de manquants, moyenne, écart-type, min, p25, médiane, p75, p99, max,
    plus le taux de défaut moyen observé sur les quartiles extrêmes (Q1 vs Q4).

    Paramètres
    ----------
    df      : DataFrame du segment.
    target  : variable cible (0/1).
    columns : liste explicite de colonnes (None = toutes les numériques).
    exclude : colonnes à ignorer même si numériques.

    Retourne
    --------
    pl.DataFrame une ligne par variable.
    """
    _excl = set(exclude or []) | {target}
    if columns is None:
        columns = [c for c, t in df.schema.items() if t.is_numeric() and c not in _excl]

    rows = []
    for c in columns:
        s       = df[c]
        s_notna = s.drop_nulls()
        n_obs   = len(s_notna)

        # Taux de défaut Q1 vs Q4 (manquants exclus)
        tdr_q1 = tdr_q4 = None
        if n_obs >= 4:
            q25 = float(s_notna.quantile(0.25))
            q75 = float(s_notna.quantile(0.75))
            mask_q1 = (pl.col(c) <= q25) & pl.col(c).is_not_null()
            mask_q4 = (pl.col(c) >= q75) & pl.col(c).is_not_null()
            tdr_q1 = float(df.filter(mask_q1)[target].mean() or 0)
            tdr_q4 = float(df.filter(mask_q4)[target].mean() or 0)

        rows.append({
            "variable":     c,
            "n":            n_obs,
            "taux_missing": s.null_count() / s.len(),
            "mean":         s.mean(),
            "std":          s.std(),
            "min":          s.min(),
            "p25":          s.quantile(0.25),
            "median":       s.median(),
            "p75":          s.quantile(0.75),
            "p99":          s.quantile(0.99),
            "max":          s.max(),
            "tdr_q1":       tdr_q1,
            "tdr_q4":       tdr_q4,
        })

    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Résumé des variables qualitatives
# ---------------------------------------------------------------------------

def resume_quali(
    df: pl.DataFrame,
    target: str = "defaut",
    columns: list[str] | None = None,
    exclude: list[str] | None = None,
    top_n: int = 5,
) -> pl.DataFrame:
    """
    Statistiques descriptives des variables catégorielles :
    nb de modalités, taux de manquants, modalité dominante, fréquence, et
    taux de défaut de chaque modalité (missing = modalité à part entière).

    Paramètres
    ----------
    df      : DataFrame du segment.
    target  : variable cible (0/1).
    columns : liste explicite (None = toutes les quali).
    exclude : colonnes à ignorer.
    top_n   : nb de modalités affichées dans la colonne tdr_par_modalite.

    Retourne
    --------
    pl.DataFrame une ligne par variable avec un résumé compact.
    """
    _quali_types = (pl.String, pl.Categorical, pl.Enum, pl.Boolean)
    _excl = set(exclude or []) | {target}
    if columns is None:
        columns = [c for c, t in df.schema.items() if isinstance(t, _quali_types) and c not in _excl]

    rows = []
    for c in columns:
        s      = df[c].cast(pl.String).fill_null("missing")
        vc     = s.value_counts(sort=True)
        n_tot  = df.height

        # Taux de défaut par modalité (top_n)
        tdr_par_modalite = (
            df.with_columns(pl.col(c).cast(pl.String).fill_null("missing").alias("_m"))
            .group_by("_m")
            .agg(
                n       = pl.len(),
                n_def   = pl.col(target).sum(),
            )
            .with_columns(tdr = pl.col("n_def") / pl.col("n"))
            .sort("n", descending=True)
            .head(top_n)
            .select(
                pl.format("{} ({:.1%})", pl.col("_m"), pl.col("tdr"))
            )["literal"].to_list()
        )

        top_row = vc.row(0) if vc.height else (None, 0)
        rows.append({
            "variable":            c,
            "n_modalites":         s.n_unique(),
            "taux_missing":        df[c].null_count() / n_tot,
            "modalite_top":        top_row[0],
            "freq_top":            top_row[1] / n_tot if n_tot else None,
            "tdr_top_modalites":   " | ".join(tdr_par_modalite),
        })

    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. Table croisée variable × target (taux de défaut par modalité/classe)
# ---------------------------------------------------------------------------

def table_bivariate(
    df: pl.DataFrame,
    var: str,
    target: str = "defaut",
    n_bins: int = 10,
    max_modalites: int = 15,
) -> pl.DataFrame:
    """
    Distribution croisée var × target : n, taux de défaut, poids.
    Pour les variables continues : discrétisation en n_bins quantiles.
    Le missing est traité comme modalité à part entière.

    Retourne
    --------
    pl.DataFrame trié par classe (numérique : croissant des breakpoints ;
    catégorielle : fréquence décroissante).
    """
    dtype = df.schema[var]
    est_quali = isinstance(dtype, (pl.String, pl.Categorical, pl.Enum, pl.Boolean)) \
                or df[var].n_unique() <= max_modalites

    if est_quali:
        grp = df.with_columns(
            pl.col(var).cast(pl.String).fill_null("missing").alias("_classe")
        )
        ordre_col = (
            grp.group_by("_classe").agg(pl.len().alias("_freq"))
            .sort("_freq", descending=True)
            .with_row_index("_ordre")
        )
        grp = grp.join(ordre_col.select("_classe", "_ordre"), on="_classe")
    else:
        cuts = df.select(
            pl.col(var).qcut(n_bins, allow_duplicates=True, include_breaks=True).alias("_q")
        ).select(
            pl.col("_q").struct.field("category").cast(pl.String).fill_null("missing").alias("_classe"),
            pl.col("_q").struct.field("breakpoint").alias("_ordre"),
        )
        grp = df.with_columns(cuts)

    total = df.height
    return (
        grp.group_by("_classe")
        .agg(
            ordre    = pl.col("_ordre").min(),
            n        = pl.len(),
            n_defaut = pl.col(target).sum(),
        )
        .with_columns(
            taux_defaut     = pl.col("n_defaut") / pl.col("n"),
            part_population = pl.col("n") / total,
        )
        .sort("ordre", nulls_last=True)
        .select("_classe", "n", "n_defaut", "taux_defaut", "part_population")
        .rename({"_classe": "classe"})
    )


# ---------------------------------------------------------------------------
# 5. Graphiques
# ---------------------------------------------------------------------------

def plot_profil_target(
    df: pl.DataFrame,
    target: str = "defaut",
    titre: str = "Segment : Comptes joints",
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """
    Camembert Sain / Défaut avec annotation du taux de défaut global.
    """
    n_def  = int(df[target].sum())
    n_sain = df.height - n_def
    tdr    = n_def / df.height

    fig, ax_ = (plt.subplots(figsize=(4, 4)) if ax is None else (ax.figure, ax))
    wedges, texts, autotexts = ax_.pie(
        [n_sain, n_def],
        labels=["Sain", "Défaut"],
        colors=[PALETTE_SAIN, PALETTE_DEFAUT],
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    for t in autotexts:
        t.set_fontsize(10)
    ax_.set_title(f"{titre}\nN={df.height:,}  —  TDR={tdr:.2%}", fontsize=11)
    return fig


def plot_quanti(
    df: pl.DataFrame,
    var: str,
    target: str = "defaut",
    titre: str | None = None,
    bins: int = 40,
    figsize: tuple[int, int] = (10, 4),
) -> plt.Figure:
    """
    Pour une variable continue :
      - histogramme superposé Sain vs Défaut (densité normalisée)
      - boxplot côte à côte Sain / Défaut
    Les valeurs nulles sont exclues de l'affichage.
    """
    sub = df.select(var, target).drop_nulls(var)
    sain    = sub.filter(pl.col(target) == 0)[var].to_numpy()
    defaut  = sub.filter(pl.col(target) == 1)[var].to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    titre_ = titre or var

    # --- histogramme ---
    ax = axes[0]
    common = dict(bins=bins, density=True, alpha=0.55, edgecolor="none")
    ax.hist(sain,   label="Sain",   color=PALETTE_SAIN,   **common)
    ax.hist(defaut, label="Défaut", color=PALETTE_DEFAUT, **common)
    ax.set_title(f"{titre_} — distribution", fontsize=10)
    ax.set_xlabel(var); ax.set_ylabel("Densité")
    ax.legend(fontsize=8)

    # --- boxplot ---
    ax2 = axes[1]
    ax2.boxplot(
        [sain, defaut],
        tick_labels=["Sain", "Défaut"],
        patch_artist=True,
        boxprops=dict(facecolor="white"),
        medianprops=dict(color="black", linewidth=2),
        flierprops=dict(marker=".", markersize=2, alpha=0.3),
        notch=False,
        widths=0.5,
    )
    ax2.set_facecolor("#f9f9f9")
    # Colorier le fond de chaque boîte
    for patch, color in zip(ax2.patches, [PALETTE_SAIN, PALETTE_DEFAUT]):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
    ax2.set_title(f"{titre_} — boxplot", fontsize=10)
    ax2.set_ylabel(var)

    fig.suptitle(titre_, fontsize=11, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


def plot_quali(
    df: pl.DataFrame,
    var: str,
    target: str = "defaut",
    titre: str | None = None,
    top_n: int = 20,
    figsize: tuple[int, int] = (10, 4),
) -> plt.Figure:
    """
    Pour une variable catégorielle :
      - barplot horizontal du taux de défaut par modalité (triées par TDR décroissant)
      - annotation : n observations et TDR global en ligne de référence.
    Le missing est affiché comme modalité distincte.
    """
    tdr_global = float(df[target].mean())
    stats = (
        df.with_columns(pl.col(var).cast(pl.String).fill_null("missing").alias("_m"))
        .group_by("_m")
        .agg(n=pl.len(), n_def=pl.col(target).sum())
        .with_columns(tdr=pl.col("n_def") / pl.col("n"))
        .sort("tdr", descending=True)
        .head(top_n)
    )

    labels  = stats["_m"].to_list()
    tdrs    = stats["tdr"].to_list()
    ns      = stats["n"].to_list()
    colors  = [PALETTE_DEFAUT if t >= tdr_global else PALETTE_SAIN for t in tdrs]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(labels[::-1], tdrs[::-1], color=colors[::-1], alpha=0.75, edgecolor="none")
    ax.axvline(tdr_global, color="black", linestyle="--", linewidth=1, label=f"TDR global ({tdr_global:.2%})")

    # Annotation n=
    for bar, n in zip(bars, ns[::-1]):
        ax.text(
            bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
            f"n={n:,}", va="center", fontsize=7.5, color=PALETTE_NEUTRE,
        )

    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))
    ax.set_xlabel("Taux de défaut")
    ax.set_title(titre or f"{var} — taux de défaut par modalité", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_missing_heatmap(
    df: pl.DataFrame,
    columns: list[str] | None = None,
    exclude: list[str] | None = None,
    seuil_affichage: float = 0.0,
    figsize: tuple[int, int] = (12, 6),
) -> plt.Figure:
    """
    Heatmap des taux de valeurs manquantes par variable.
    Seules les colonnes dont le taux de missing > seuil_affichage sont tracées.

    Paramètres
    ----------
    columns            : colonnes à inclure (None = toutes).
    exclude            : colonnes à exclure.
    seuil_affichage    : taux minimum pour qu'une colonne apparaisse.
    """
    _excl = set(exclude or [])
    cols = columns or [c for c in df.columns if c not in _excl]
    n    = df.height

    missing = {c: df[c].null_count() / n for c in cols}
    missing = {c: v for c, v in missing.items() if v > seuil_affichage}

    if not missing:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "Aucun manquant détecté", ha="center", va="center", fontsize=12)
        ax.axis("off")
        return fig

    df_miss = pl.DataFrame({"variable": list(missing.keys()), "taux_missing": list(missing.values())}).sort("taux_missing", descending=True)

    mat = df_miss["taux_missing"].to_numpy().reshape(1, -1)
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        mat,
        ax=ax,
        xticklabels=df_miss["variable"].to_list(),
        yticklabels=["% missing"],
        annot=True,
        fmt=".1%",
        cmap="YlOrRd",
        vmin=0, vmax=1,
        linewidths=0.5,
        cbar_kws={"shrink": 0.4},
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax.set_title("Taux de valeurs manquantes — segment comptes joints", fontsize=11)
    fig.tight_layout()
    return fig


def plot_taux_defaut_temporel(
    df: pl.DataFrame,
    date_col: str,
    target: str = "defaut",
    freq: str = "1q",
    figsize: tuple[int, int] = (11, 4),
) -> plt.Figure:
    """
    Taux de défaut et volume d'observations par période (trimestre par défaut).
    Utile pour détecter des dérives temporelles ou des fenêtres atypiques.

    Paramètres
    ----------
    date_col : colonne de date (type Date ou String YYYY-MM-DD).
    freq     : fréquence de regroupement polars ('1q', '1mo', '1y', …).
    """
    tmp = df.with_columns(
        pl.col(date_col).cast(pl.Date).dt.truncate(freq).alias("_periode")
    ).group_by("_periode").agg(
        n   = pl.len(),
        tdr = pl.col(target).mean(),
    ).sort("_periode")

    periodes = tmp["_periode"].to_list()
    tdrs     = tmp["tdr"].to_list()
    ns       = tmp["n"].to_list()

    fig, ax1 = plt.subplots(figsize=figsize)
    ax2 = ax1.twinx()

    ax1.bar(range(len(periodes)), ns, color=PALETTE_NEUTRE, alpha=0.35, label="Volume (n)")
    ax2.plot(range(len(periodes)), tdrs, color=PALETTE_DEFAUT, marker="o", linewidth=1.8, label="TDR")
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))

    ax1.set_xticks(range(len(periodes)))
    ax1.set_xticklabels([str(p) for p in periodes], rotation=45, ha="right", fontsize=8)
    ax1.set_ylabel("Nombre d'observations", fontsize=9)
    ax2.set_ylabel("Taux de défaut", fontsize=9, color=PALETTE_DEFAUT)
    ax2.tick_params(axis="y", labelcolor=PALETTE_DEFAUT)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")
    ax1.set_title("Taux de défaut et volume par période — segment comptes joints", fontsize=11)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 6. Rapport multi-variables (grille de graphiques)
# ---------------------------------------------------------------------------

def grille_quanti(
    df: pl.DataFrame,
    variables: list[str],
    target: str = "defaut",
    n_cols: int = 2,
    bins: int = 35,
    figsize_par_plot: tuple[int, int] = (10, 3),
) -> plt.Figure:
    """
    Grille de graphiques quanti (histogramme + boxplot) pour une liste de variables.
    Chaque ligne contient un plot_quanti() complet.
    """
    n_rows = math.ceil(len(variables) / n_cols) * 1   # 1 ligne complète par variable
    fig = plt.figure(figsize=(figsize_par_plot[0] * n_cols, figsize_par_plot[1] * len(variables)))
    for idx, var in enumerate(variables):
        ax_hist = fig.add_subplot(len(variables), 2, 2 * idx + 1)
        ax_box  = fig.add_subplot(len(variables), 2, 2 * idx + 2)
        sub = df.select(var, target).drop_nulls(var)
        sain   = sub.filter(pl.col(target) == 0)[var].to_numpy()
        defaut = sub.filter(pl.col(target) == 1)[var].to_numpy()
        common = dict(bins=bins, density=True, alpha=0.55, edgecolor="none")
        ax_hist.hist(sain,   color=PALETTE_SAIN,   label="Sain",   **common)
        ax_hist.hist(defaut, color=PALETTE_DEFAUT, label="Défaut", **common)
        ax_hist.set_title(f"{var}", fontsize=9, fontweight="bold")
        ax_hist.legend(fontsize=7)
        ax_box.boxplot(
            [sain, defaut], tick_labels=["Sain", "Défaut"],
            patch_artist=True, widths=0.5,
            medianprops=dict(color="black", linewidth=2),
            flierprops=dict(marker=".", markersize=2, alpha=0.3),
        )
        for patch, color in zip(ax_box.patches, [PALETTE_SAIN, PALETTE_DEFAUT]):
            patch.set_facecolor(color); patch.set_alpha(0.55)
        ax_box.set_title(f"{var} — boxplot", fontsize=9)
    fig.tight_layout()
    return fig


def grille_quali(
    df: pl.DataFrame,
    variables: list[str],
    target: str = "defaut",
    top_n: int = 15,
    n_cols: int = 2,
    figsize_col: int = 6,
) -> plt.Figure:
    """
    Grille de barplots horizontaux (taux de défaut par modalité) pour une liste
    de variables qualitatives.
    """
    n_rows = math.ceil(len(variables) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize_col * n_cols, 3.5 * n_rows))
    axes_flat = np.array(axes).flatten() if n_rows * n_cols > 1 else [axes]

    tdr_global = float(df[target].mean())

    for ax, var in zip(axes_flat, variables):
        stats = (
            df.with_columns(pl.col(var).cast(pl.String).fill_null("missing").alias("_m"))
            .group_by("_m")
            .agg(n=pl.len(), n_def=pl.col(target).sum())
            .with_columns(tdr=pl.col("n_def") / pl.col("n"))
            .sort("tdr", descending=True)
            .head(top_n)
        )
        labels = stats["_m"].to_list()
        tdrs   = stats["tdr"].to_list()
        colors = [PALETTE_DEFAUT if t >= tdr_global else PALETTE_SAIN for t in tdrs]
        ax.barh(labels[::-1], tdrs[::-1], color=colors[::-1], alpha=0.75, edgecolor="none")
        ax.axvline(tdr_global, color="black", linestyle="--", linewidth=0.8)
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))
        ax.set_title(var, fontsize=9, fontweight="bold")

    # Masquer les axes inutilisés
    for ax in axes_flat[len(variables):]:
        ax.set_visible(False)

    fig.suptitle("Taux de défaut par modalité — variables qualitatives", fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig



# ---------------------------------------------------------------------------
# 7. Préparation des données (ch. 2)
# ---------------------------------------------------------------------------

_QUALI_TYPES = (pl.String, pl.Categorical, pl.Enum, pl.Boolean)


def table_missing(
    df: pl.DataFrame,
    target: str = "defaut",
    columns: list[str] | None = None,
    exclude: list[str] | None = None,
) -> pl.DataFrame:
    """
    Valeurs manquantes par variable et lien avec le défaut : taux de défaut
    des manquants vs des renseignés, écart, et p-value du Chi² d'indépendance
    entre l'indicatrice « manquant » et la cible.
    Un écart significatif = missing informatif, à conserver comme modalité.

    Paramètres
    ----------
    df      : DataFrame du segment.
    target  : variable cible (0/1).
    columns : liste explicite de colonnes (None = toutes).
    exclude : colonnes à ignorer.

    Retourne
    --------
    pl.DataFrame une ligne par variable, trié par taux de manquants décroissant.
    """
    _excl = set(exclude or []) | {target}
    cols  = columns or [c for c in df.columns if c not in _excl]
    n_tot = df.height
    n_def = int(df[target].sum())

    agg = df.select(
        *[pl.col(c).is_null().sum().alias(f"n|{c}") for c in cols],
        *[pl.col(target).filter(pl.col(c).is_null()).sum().alias(f"d|{c}") for c in cols],
    ).row(0, named=True) if cols else {}

    rows = []
    for c in cols:
        n_miss = int(agg[f"n|{c}"])
        d_miss = int(agg[f"d|{c}"] or 0)
        n_rens = n_tot - n_miss
        d_rens = n_def - d_miss

        pvalue = None
        if n_miss and n_rens and 0 < n_def < n_tot:
            pvalue = float(stats.chi2_contingency(
                [[d_miss, n_miss - d_miss], [d_rens, n_rens - d_rens]]
            )[1])

        rows.append({
            "variable":      c,
            "dtype":         str(df.schema[c]),
            "n_missing":     n_miss,
            "taux_missing":  n_miss / n_tot if n_tot else None,
            "tdr_missing":   d_miss / n_miss if n_miss else None,
            "tdr_renseigne": d_rens / n_rens if n_rens else None,
            "pvalue":        pvalue,
        })

    schema = {
        "variable": pl.String, "dtype": pl.String, "n_missing": pl.Int64,
        "taux_missing": pl.Float64, "tdr_missing": pl.Float64,
        "tdr_renseigne": pl.Float64, "pvalue": pl.Float64,
    }
    return (
        pl.DataFrame(rows, schema=schema)
        .with_columns(ecart_tdr = pl.col("tdr_missing") - pl.col("tdr_renseigne"))
        .select("variable", "dtype", "n_missing", "taux_missing",
                "tdr_missing", "tdr_renseigne", "ecart_tdr", "pvalue")
        .sort("taux_missing", descending=True)
    )


def plot_missing_tdr(
    tab_missing: pl.DataFrame,
    top_n: int = 20,
    n_min: int = 30,
    figsize: tuple[int, int] = (10, 6),
) -> plt.Figure:
    """
    Taux de défaut des manquants vs des renseignés, pour les top_n variables
    où l'écart est le plus marqué (en valeur absolue).

    Paramètres
    ----------
    tab_missing : sortie de table_missing().
    top_n       : nombre de variables affichées.
    n_min       : nombre minimal de manquants pour qu'une variable soit tracée.
    """
    tab = (
        tab_missing
        .filter((pl.col("n_missing") >= n_min) & pl.col("ecart_tdr").is_not_null())
        .sort(pl.col("ecart_tdr").abs(), descending=True)
        .head(top_n)
    )

    if tab.is_empty():
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "Aucune variable à comparer", ha="center", va="center", fontsize=12)
        ax.axis("off")
        return fig

    labels = tab["variable"].to_list()[::-1]
    y      = np.arange(len(labels))
    h      = 0.38

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(y + h / 2, tab["tdr_missing"].to_list()[::-1],   height=h, color=PALETTE_DEFAUT, alpha=0.75, label="Manquant")
    ax.barh(y - h / 2, tab["tdr_renseigne"].to_list()[::-1], height=h, color=PALETTE_SAIN,   alpha=0.75, label="Renseigné")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))
    ax.set_xlabel("Taux de défaut")
    ax.set_title("Taux de défaut : manquants vs renseignés", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def missing_en_modalite(
    df: pl.DataFrame,
    columns: list[str] | None = None,
    label: str = "missing",
) -> pl.DataFrame:
    """
    Variables qualitatives : le manquant (null ou chaîne vide) devient une
    modalité explicite `label`. Les colonnes traitées sont converties en String.
    Les variables quantitatives ne sont pas concernées : leur manquant reste
    null et formera une classe à part lors de la discrétisation.

    Paramètres
    ----------
    columns : colonnes à traiter (None = toutes les qualitatives).
    label   : libellé de la modalité « manquant ».
    """
    if columns is None:
        columns = [c for c, t in df.schema.items() if isinstance(t, _QUALI_TYPES)]

    exprs = []
    for c in columns:
        s = pl.col(c).cast(pl.String)
        exprs.append(
            pl.when(s.is_null() | (s.str.strip_chars() == "")).then(pl.lit(label)).otherwise(s).alias(c)
        )
    return df.with_columns(exprs)


def valeurs_aberrantes(
    df: pl.DataFrame,
    plages: dict[str, tuple[float | None, float | None]],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Valeurs aberrantes (impossibles métier) : toute valeur hors de la plage
    admissible est remplacée par null, donc traitée ensuite comme un manquant.

    Paramètres
    ----------
    plages : {variable: (min admissible, max admissible)} ; None = pas de borne.

    Retourne
    --------
    (df corrigé, rapport par variable : bornes, n et taux de valeurs aberrantes).
    """
    rows, exprs = [], []
    for c, (bas, haut) in plages.items():
        hors = pl.lit(False)
        if bas is not None:
            hors = hors | (pl.col(c) < bas)
        if haut is not None:
            hors = hors | (pl.col(c) > haut)
        hors = hors.fill_null(False)

        n_hors = int(df.select(hors.sum()).item())
        n_obs  = df.height - df[c].null_count()
        rows.append({
            "variable":        c,
            "min_admissible":  bas,
            "max_admissible":  haut,
            "n_aberrantes":    n_hors,
            "taux_aberrantes": n_hors / n_obs if n_obs else None,
        })
        exprs.append(pl.when(hors).then(None).otherwise(pl.col(c)).alias(c))

    schema = {
        "variable": pl.String, "min_admissible": pl.Float64, "max_admissible": pl.Float64,
        "n_aberrantes": pl.Int64, "taux_aberrantes": pl.Float64,
    }
    return df.with_columns(exprs), pl.DataFrame(rows, schema=schema)


def winsoriser(
    df: pl.DataFrame,
    columns: list[str],
    p_bas: float = 0.01,
    p_haut: float = 0.99,
    bornes: pl.DataFrame | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Winsorisation : les valeurs extrêmes sont ramenées aux quantiles p_bas et
    p_haut (les manquants restent manquants). Une variable dont les deux bornes
    coïncident (ex. comptage presque toujours nul) n'est pas modifiée, sinon
    elle deviendrait constante.

    Paramètres
    ----------
    columns : variables quantitatives à traiter.
    p_bas   : quantile de la borne basse.
    p_haut  : quantile de la borne haute.
    bornes  : bornes déjà estimées (sortie d'un appel précédent, typiquement
              sur l'échantillon d'apprentissage) à réappliquer telles quelles.

    Retourne
    --------
    (df winsorisé, bornes et volumes écrêtés par variable).
    """
    if bornes is not None:
        connues = {r["variable"]: (r["borne_basse"], r["borne_haute"]) for r in bornes.iter_rows(named=True)}
    else:
        connues = {}

    rows, exprs = [], []
    for c in columns:
        s = df[c].drop_nulls()
        if c in connues:
            bas, haut = connues[c]
        elif s.len():
            bas, haut = s.quantile(p_bas), s.quantile(p_haut)
        else:
            bas = haut = None

        applique = bas is not None and haut is not None and bas < haut
        n_bas  = int((s < bas).sum()) if applique else 0
        n_haut = int((s > haut).sum()) if applique else 0
        rows.append({
            "variable":     c,
            "borne_basse":  bas,
            "borne_haute":  haut,
            "n_bas":        n_bas,
            "n_haut":       n_haut,
            "taux_ecrete":  (n_bas + n_haut) / s.len() if s.len() else None,
            "winsorisee":   applique,
        })
        if applique:
            if df.schema[c].is_integer():
                bas, haut = int(bas), int(haut)
            exprs.append(pl.col(c).clip(bas, haut).alias(c))

    schema = {
        "variable": pl.String, "borne_basse": pl.Float64, "borne_haute": pl.Float64,
        "n_bas": pl.Int64, "n_haut": pl.Int64, "taux_ecrete": pl.Float64, "winsorisee": pl.Boolean,
    }
    return df.with_columns(exprs), pl.DataFrame(rows, schema=schema)


def ajouter_ratios(
    df: pl.DataFrame,
    ratios: dict[str, tuple[str, str]],
) -> pl.DataFrame:
    """
    Ajoute des variables de type ratio numérateur / dénominateur.
    Le ratio est null si le dénominateur est nul ou manquant, ou si le
    numérateur est manquant : il rejoint alors la classe « manquant ».

    Paramètres
    ----------
    ratios : {nom de la nouvelle variable: (numérateur, dénominateur)}.
    """
    return df.with_columns([
        pl.when(pl.col(den) != 0).then(pl.col(num) / pl.col(den)).otherwise(None).alias(nom)
        for nom, (num, den) in ratios.items()
    ])


def reperer_variables(
    df: pl.DataFrame,
    motifs: dict[str, list[str]],
) -> pl.DataFrame:
    """
    Repère les colonnes dont le nom contient l'un des motifs fournis
    (recherche insensible à la casse). Sert de premier filtre pour les
    variables interdites ; ne remplace pas la relecture du dictionnaire
    de données.

    Paramètres
    ----------
    motifs : {famille: [fragments de nom à rechercher]}.

    Retourne
    --------
    pl.DataFrame : variable, famille, motif trouvé, dtype.
    """
    rows = []
    for c in df.columns:
        for famille, fragments in motifs.items():
            trouve = next((f for f in fragments if f.lower() in c.lower()), None)
            if trouve is not None:
                rows.append({"variable": c, "famille": famille, "motif": trouve, "dtype": str(df.schema[c])})

    schema = {"variable": pl.String, "famille": pl.String, "motif": pl.String, "dtype": pl.String}
    return pl.DataFrame(rows, schema=schema)

def plot_taux_missing(
    df: pl.DataFrame,
    columns: list[str] | None = None,
    exclude: list[str] | None = None,
    seuil_affichage: float = 0.0,
    top_n: int = 30,
    largeur: float = 9,
) -> plt.Figure:
    """
    Barres horizontales du taux de valeurs manquantes par variable, triées par
    taux décroissant, avec le taux et le nombre de manquants en bout de barre.
    Seules les colonnes dont le taux de missing > seuil_affichage sont tracées,
    dans la limite des top_n plus manquantes.

    Paramètres
    ----------
    columns         : colonnes à inclure (None = toutes).
    exclude         : colonnes à exclure.
    seuil_affichage : taux minimum pour qu'une colonne apparaisse.
    top_n           : nombre maximal de variables affichées.
    largeur         : largeur de la figure (la hauteur s'adapte au nombre de barres).
    """
    _excl = set(exclude or [])
    cols  = columns or [c for c in df.columns if c not in _excl]
    n     = df.height

    tab = pl.DataFrame(
        {"variable": cols, "n_missing": [df[c].null_count() for c in cols]},
        schema={"variable": pl.String, "n_missing": pl.Int64},
    ).with_columns(
        taux_missing = pl.col("n_missing") / n
    ).filter(pl.col("taux_missing") > seuil_affichage).sort("taux_missing", descending=True)

    if tab.is_empty():
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "Aucun manquant détecté", ha="center", va="center", fontsize=12)
        ax.axis("off")
        return fig

    n_concernees = tab.height
    tab    = tab.head(top_n)
    labels = tab["variable"].to_list()[::-1]
    taux   = tab["taux_missing"].to_list()[::-1]
    ns     = tab["n_missing"].to_list()[::-1]

    fig, ax = plt.subplots(figsize=(largeur, max(2.5, 0.32 * len(labels) + 1.2)))
    bars = ax.barh(labels, taux, color=PALETTE_SAIN, alpha=0.75, edgecolor="none", height=0.7)

    for bar, t, k in zip(bars, taux, ns):
        ax.text(
            bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{t:.1%}  (n={k:,})".replace(",", " "), va="center", fontsize=8, color="#333333",
        )

    ax.set_xlim(0, 1.22)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_xlabel("Taux de valeurs manquantes")
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", alpha=0.4)
    ax.set_axisbelow(True)
    ax.margins(y=0.01)

    titre = "Taux de valeurs manquantes par variable"
    if n_concernees > top_n:
        titre += f" — {top_n} plus manquantes sur {n_concernees}"
    ax.set_title(titre, fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig

def bilan_variables(
    df: pl.DataFrame,
    df_prep: pl.DataFrame,
    exclues: dict[str, str] | None = None,
    creees: Sequence[str] = (),
    techniques: Sequence[str] = (),
    remarques: dict[str, list[str]] | None = None,
) -> pl.DataFrame:
    """
    Bilan de la préparation, une ligne par variable : statut (gardée, exclue,
    technique), motif d'exclusion, origine (base ou créée) et traitements subis.

    Paramètres
    ----------
    df         : base avant préparation.
    df_prep    : base après préparation.
    exclues    : {variable exclue: motif}. Une colonne absente de df_prep sans
                 motif fourni est notée « exclue » avec le motif « non précisé ».
    creees     : variables ajoutées pendant la préparation.
    techniques : colonnes hors du champ des variables explicatives
                 (identifiant, cible, date, poids...).
    remarques  : {variable: [traitements ou points d'attention]}.

    Retourne
    --------
    pl.DataFrame trié par statut puis par nom de variable.
    """
    exclues, remarques = exclues or {}, remarques or {}
    _tech, _creees = set(techniques), set(creees)

    rows = []
    for c in list(df.columns) + [c for c in df_prep.columns if c not in df.columns]:
        # taux de manquants mesuré avant préparation (après, le manquant des qualitatives est une modalité)
        source = df if c in df.columns else df_prep
        if c in df_prep.columns:
            statut, motif = ("technique", "hors variables explicatives") if c in _tech else ("gardée", None)
        else:
            statut, motif = "exclue", exclues.get(c, "non précisé")
        rows.append({
            "variable":     c,
            "statut":       statut,
            "motif":        motif,
            "origine":      "créée" if c in _creees or c not in df.columns else "base",
            "dtype":        str(source.schema[c]),
            "taux_missing": source[c].null_count() / source.height if source.height else None,
            "remarques":    " ; ".join(remarques.get(c, [])) or None,
        })

    schema = {
        "variable": pl.String, "statut": pl.String, "motif": pl.String, "origine": pl.String,
        "dtype": pl.String, "taux_missing": pl.Float64, "remarques": pl.String,
    }
    return pl.DataFrame(rows, schema=schema).sort("statut", "variable")