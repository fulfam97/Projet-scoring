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
        labels=["Sain", "Défaut"],
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
            [sain, defaut], labels=["Sain", "Défaut"],
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

