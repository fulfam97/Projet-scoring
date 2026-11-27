"""Mise en classe du score : Classes Homogènes de Risque (CHR), chapitre 7 du cours.

- 7.1 construction : découpage fin du score en quantiles (bornes prises sur les valeurs observées, un palier
  de score n'est jamais coupé), puis fusion progressive de classes adjacentes ;
- 7.2 stabilité en risque et en volume par cohorte (réutilise `binning.stabilite_decoupages`) ;
- 7.3 homogénéité intra-CHR selon une variable exogène : test de Welch et h de Cohen ;
- 7.4 hétérogénéité inter-CHR : écart relatif R_k et son IC par la méthode delta ;
- 7.6 perte d'information : Gini, KS et AUC PR du score contre ceux des CHR.

Convention : CHR 1 = la moins risquée, CHR K = la plus risquée ; un score en points élevé = peu risqué.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import polars as pl
from scipy import stats

from utils.binning import bornes_fines
from utils.modele import performance


# ---------------------------------------------------------------------------
# Affectation et statistiques par classe
# ---------------------------------------------------------------------------

def affecter_chr(points, bornes: list[float]) -> np.ndarray:
    """CHR à partir des bornes croissantes du score : points > bornes[-1] → CHR 1 ; points ≤ bornes[0] → CHR K.
    Classes de points ]b(j-1) ; b(j)], comme au chapitre 4."""
    p = np.asarray(points, dtype=float)
    return len(bornes) + 1 - np.searchsorted(np.asarray(bornes, dtype=float), p, side="left")


def ecart_relatif_ic(n_a: int, d_a: int, n_b: int, d_b: int, alpha: float = 0.05) -> tuple[float, float, float]:
    """R = τ_b / τ_a − 1 (b plus risquée que a) et son IC par la méthode delta (cours 7.2.1) :
    Var(R) ≈ (τ_b/τ_a)² [ (1−τ_b)/(N_b τ_b) + (1−τ_a)/(N_a τ_a) ]. Renvoie (R, borne basse, borne haute)."""
    ta, tb = d_a / n_a, d_b / n_b
    if ta == 0 or tb == 0:
        return (float("inf") if ta == 0 and tb > 0 else float("nan")), float("nan"), float("nan")
    r = tb / ta - 1
    sigma = (tb / ta) * np.sqrt((1 - tb) / (n_b * tb) + (1 - ta) / (n_a * ta))
    z = stats.norm.ppf(1 - alpha / 2)
    return float(r), float(r - z * sigma), float(r + z * sigma)


def table_chr(df: pl.DataFrame, chr_col: str, target: str, seuil: float = 0.30, alpha: float = 0.05,
              points: str | None = None) -> pl.DataFrame:
    """Effectif, part, défauts, taux de défaut par CHR, écart relatif R_k avec la CHR précédente (moins risquée)
    et son IC (7.4). Avec `points`, ajoute les bornes de score observées de chaque classe."""
    agg = [pl.len().alias("n"), pl.col(target).sum().alias("n_defaut")]
    if points:
        agg += [pl.col(points).min().alias("points_min"), pl.col(points).max().alias("points_max")]
    t = (df.group_by(chr_col).agg(agg).sort(chr_col)
         .with_columns(part=pl.col("n") / pl.col("n").sum(), taux_defaut=pl.col("n_defaut") / pl.col("n")))
    r, bas, haut = [None], [None], [None]
    rows = t.select("n", "n_defaut").rows()
    for (na, da), (nb, db) in zip(rows[:-1], rows[1:]):
        x = ecart_relatif_ic(na, da, nb, db, alpha)
        r.append(x[0]); bas.append(x[1]); haut.append(x[2])
    t = t.with_columns(R=pl.Series(r, dtype=pl.Float64), R_ic_bas=pl.Series(bas, dtype=pl.Float64),
                       R_ic_haut=pl.Series(haut, dtype=pl.Float64))
    return t.with_columns(
        ecart_ok=pl.when(pl.col("R").is_null()).then(None).otherwise(pl.col("R") >= seuil),
        ecart_ic_ok=pl.when(pl.col("R").is_null()).then(None).otherwise(pl.col("R_ic_bas").fill_nan(-1) >= seuil),
    ).rename({chr_col: "CHR"})


# ---------------------------------------------------------------------------
# 7.1 Construction : découpage fin puis fusion de classes adjacentes
# ---------------------------------------------------------------------------

def _stats_chaine(chaine: list[dict]) -> list[float]:
    return [g["d"] / g["n"] for g in chaine]


def construire_chr(points, y, n_fin: int = 20, seuil: float = 0.30, critere: str = "ponctuel",
                   min_defauts: int = 5, min_part: float = 0.02, alpha: float = 0.05,
                   max_classes: int | None = None, max_part: float | None = None) -> tuple[list[float], pl.DataFrame]:
    """Démarche du cours 7.1.3 sur l'apprentissage :
    1. découpage fin en n_fin quantiles du score (bornes sur les valeurs observées : aucun palier coupé) ;
    2. fusions successives de deux classes adjacentes, une à la fois, par ordre de priorité :
       a. classe trop petite (moins de min_part de l'effectif ou de min_defauts défauts) : elle rejoint sa voisine
          de taux de défaut le plus proche ;
       b. rupture de monotonie (une classe moins bien notée n'est pas plus risquée) ;
       c. écart relatif R_k < seuil (critere="ponctuel") ou borne basse de son IC < seuil (critere="ic") ;
       d. plus de max_classes classes.
       En b à d, la paire fusionnée est celle de plus faible écart relatif.
    max_part : aucune fusion ne peut créer une CHR de plus de max_part de l'effectif (contrainte de concentration) ;
    les règles qui ne peuvent plus être satisfaites sans la violer restent non satisfaites (visibles dans table_chr).
    Renvoie les bornes croissantes du score et le journal des fusions."""
    pts, y = np.asarray(points, dtype=float), np.asarray(y, dtype=int)
    fines = bornes_fines(pl.Series(pts), pl.Series(y), methode="quantiles", n_bins=n_fin, max_modalites=n_fin)
    # chaîne ordonnée de la classe la moins risquée (points élevés) à la plus risquée
    limites = [-np.inf] + list(fines) + [np.inf]
    chaine = []
    for lo, hi in zip(limites[:-1], limites[1:]):
        m = (pts > lo) & (pts <= hi)
        chaine.append({"lo": lo, "hi": hi, "n": int(m.sum()), "d": int(y[m].sum())})
    chaine = [g for g in chaine if g["n"] > 0][::-1]
    n_tot = len(pts)
    journal = []

    def ecart(a, b):
        return ecart_relatif_ic(a["n"], a["d"], b["n"], b["d"], alpha)

    def fusion_permise(k):
        return max_part is None or chaine[k]["n"] + chaine[k + 1]["n"] <= max_part * n_tot

    while len(chaine) > 1:
        tdr = _stats_chaine(chaine)
        petites = [i for i, g in enumerate(chaine) if (g["n"] < min_part * n_tot or g["d"] < min_defauts)
                   and any(0 <= k < len(chaine) - 1 and fusion_permise(k) for k in (i - 1, i))]
        paires = [k for k in range(len(chaine) - 1) if fusion_permise(k)]
        r = {k: ecart(chaine[k], chaine[k + 1]) for k in paires}
        cle = {k: (r[k][0] if critere == "ponctuel" else r[k][1]) for k in paires}
        cle = {k: (-np.inf if np.isnan(v) else v) for k, v in cle.items()}
        if petites:
            i = min(petites, key=lambda k: chaine[k]["n"])
            voisins = [k for k in (i - 1, i + 1) if 0 <= k < len(chaine) and fusion_permise(min(i, k))]
            j = min(voisins, key=lambda k: abs(tdr[k] - tdr[i]))
            motif = "effectif ou nombre de défauts insuffisant"
        else:
            rompues = [k for k in paires if tdr[k + 1] <= tdr[k]]
            faibles = [k for k in paires if cle[k] < seuil]
            trop = paires if max_classes and len(chaine) > max_classes else []
            if rompues:
                cand, motif = rompues, "rupture de monotonie"
            elif faibles:
                cand, motif = faibles, f"écart relatif < {seuil:.0%}" + (" (borne basse de l'IC)" if critere == "ic" else "")
            elif trop:
                cand, motif = trop, f"plus de {max_classes} classes"
            else:
                break
            i = min(cand, key=lambda k: cle[k])
            j = i + 1
        i, j = min(i, j), max(i, j)
        a, b = chaine[i], chaine[j]
        journal.append({"n_classes_avant": len(chaine), "motif": motif,
                        "points_fusionnes": f"]{b['lo']:.0f} ; {a['hi']:.0f}]",
                        "taux_a": a["d"] / a["n"], "taux_b": b["d"] / b["n"]})
        chaine[i:j + 1] = [{"lo": b["lo"], "hi": a["hi"], "n": a["n"] + b["n"], "d": a["d"] + b["d"]}]
    bornes = sorted(g["lo"] for g in chaine if np.isfinite(g["lo"]))
    schema = {"n_classes_avant": pl.Int64, "motif": pl.String, "points_fusionnes": pl.String,
              "taux_a": pl.Float64, "taux_b": pl.Float64}
    return bornes, pl.DataFrame(journal, schema=schema)


def chr_optbinning(points, y, monotonic_trend: str = "descending", min_bin_size: float = 0.02,
                   min_bin_n_event: int = 5, max_n_bins: int | None = None, max_n_prebins: int = 20,
                   solver: str = "cp", **params) -> list[float]:
    """Découpage du score par optbinning (OptimalBinning : pré-découpage par arbre puis optimisation sous contraintes
    de l'IV). Taux de défaut décroissant avec les points, mêmes contraintes d'effectif que construire_chr.
    optbinning coupe en x < s ; les bornes sont ramenées à la plus grande valeur observée sous s, pour la
    convention ]b(j-1) ; b(j)] d'affecter_chr (aucun palier coupé)."""
    from optbinning import OptimalBinning
    pts, y = np.asarray(points, dtype=float), np.asarray(y, dtype=int)
    ob = OptimalBinning(dtype="numerical", solver=solver, monotonic_trend=monotonic_trend,
                        min_bin_size=min_bin_size, min_bin_n_event=min_bin_n_event, max_n_bins=max_n_bins,
                        max_n_prebins=max_n_prebins, **params)
    ob.fit(pts, y)
    valeurs = np.unique(pts)
    bornes = []
    for s in ob.splits:
        sous = valeurs[valeurs < s]
        if len(sous):
            bornes.append(float(sous[-1]))
    return sorted(set(bornes))


def comparer_decoupages(df_app: pl.DataFrame, df_test: pl.DataFrame, decoupages: dict[str, list[float]], target: str,
                        points: str = "points", cohorte: str = "cohorte", seuil: float = 0.30,
                        alpha: float = 0.05) -> pl.DataFrame:
    """Synthèse comparative de plusieurs jeux de bornes : nombre de CHR, parts, défauts et taux par CHR, critères 7.2
    et 7.4 (écart ponctuel et IC en apprentissage, ordre sur le test et dans les cohortes) et perte de Gini (7.6)."""
    from utils.binning import stabilite_decoupages
    rows = []
    for nom, b in decoupages.items():
        a = df_app.with_columns(CHR=pl.Series(affecter_chr(df_app[points], b)))
        t = df_test.with_columns(CHR=pl.Series(affecter_chr(df_test[points], b)))
        ta, tt = table_chr(a, "CHR", target, seuil, alpha), table_chr(t, "CHR", target, seuil, alpha)
        _, st = stabilite_decoupages(a, ["CHR"], target, cohorte, seuil_ecart=seuil)
        pa = perte_information(a[target], a[points], a["CHR"])
        pt = perte_information(t[target], t[points], t["CHR"])
        rows.append({
            "decoupage": nom, "n_chr": len(b) + 1, "bornes": " · ".join(f"{x:.0f}" for x in b),
            "parts_app": " · ".join(f"{x:.1%}" for x in ta["part"]),
            "defauts_app": " · ".join(f"{x:.0f}" for x in ta["n_defaut"]),
            "taux_app": " · ".join(f"{x:.2%}" for x in ta["taux_defaut"]),
            "ecart_30_app": bool(ta["ecart_ok"].drop_nulls().all()),
            "ic_30_app": bool(ta["ecart_ic_ok"].drop_nulls().all()),
            "ordre_test": bool((tt["taux_defaut"].diff().drop_nulls() > 0).all()) and tt.height == ta.height,
            "cohortes_ordonnees": float(st["part_groupes_ordonnes"][0]),
            "gini_chr_app": float(pa["gini"][1]), "gini_chr_test": float(pt["gini"][1]),
            "delta_gini_test": float(pt["delta_gini"][1]),
        })
    return pl.DataFrame(rows)


def libelles_chr(bornes: list[float]) -> dict[int, str]:
    """Libellé en points de chaque CHR (1 = la moins risquée)."""
    k = len(bornes) + 1
    lim = [-np.inf] + list(bornes) + [np.inf]
    out = {}
    for c in range(1, k + 1):
        lo, hi = lim[k - c], lim[k - c + 1]
        out[c] = (f"> {lo:.0f}" if not np.isfinite(hi) else f"≤ {hi:.0f}" if not np.isfinite(lo)
                  else f"]{lo:.0f} ; {hi:.0f}]")
    return out


# ---------------------------------------------------------------------------
# 7.3 Homogénéité intra-CHR selon une variable exogène
# ---------------------------------------------------------------------------

def welch_proportions(n_a: int, d_a: int, n_b: int, d_b: int) -> tuple[float, float, float]:
    """Test de Welch sur deux proportions (cours 7.3.2) : t, degrés de liberté de Welch-Satterthwaite, p-value."""
    ta, tb = d_a / n_a, d_b / n_b
    va, vb = ta * (1 - ta) / n_a, tb * (1 - tb) / n_b
    if va + vb == 0:
        return 0.0, float("nan"), 1.0
    t = (ta - tb) / np.sqrt(va + vb)
    ddl = (va + vb) ** 2 / ((va ** 2 / (n_a - 1) if n_a > 1 else 0) + (vb ** 2 / (n_b - 1) if n_b > 1 else 0))
    return float(t), float(ddl), float(2 * stats.t.sf(abs(t), ddl))


def h_cohen(ta: float, tb: float) -> float:
    """h de Cohen pour deux proportions (cours 7.3.3) : 2 arcsin √τa − 2 arcsin √τb."""
    return float(2 * np.arcsin(np.sqrt(ta)) - 2 * np.arcsin(np.sqrt(tb)))


def _lecture_h(h: float) -> str:
    h = abs(h)
    return "négligeable" if h < 0.2 else "faible" if h < 0.5 else "moyenne" if h < 0.8 else "forte"


def homogeneite_exogene(df: pl.DataFrame, chr_col: str, target: str, exogene: str, alpha: float = 0.05,
                        seuil_h: float = 0.2, min_effectif: int = 100) -> pl.DataFrame:
    """Pour chaque CHR et chaque modalité a de la variable exogène (comparée au reste de la CHR, noté b) :
    taux de défaut, test de Welch et h de Cohen. La CHR est jugée hétérogène seulement si l'écart est à la fois
    significatif (p < alpha) et non négligeable (|h| ≥ seuil_h), à cause de la forte puissance du test (7.3.2).
    Les sous-groupes de moins de min_effectif dossiers ne sont pas testés."""
    d = df.select(pl.col(chr_col).alias("CHR"), pl.col(exogene).cast(pl.String).fill_null("manquant").alias("m"),
                  pl.col(target).alias("y"))
    tot = d.group_by("CHR").agg(N=pl.len(), D=pl.col("y").sum())
    g = d.group_by("CHR", "m").agg(n_a=pl.len(), d_a=pl.col("y").sum()).join(tot, on="CHR")
    if g["m"].n_unique() == 2:   # variable binaire : une seule comparaison par CHR
        g = g.filter(pl.col("m") == sorted(g["m"].unique().to_list())[0])
    rows = []
    for r in g.sort("CHR", "m").iter_rows(named=True):
        n_b, d_b = r["N"] - r["n_a"], r["D"] - r["d_a"]
        base = {"variable": exogene, "CHR": r["CHR"], "modalite": r["m"], "n_a": r["n_a"], "n_b": n_b,
                "taux_a": r["d_a"] / r["n_a"], "taux_b": d_b / n_b if n_b else None}
        if r["n_a"] < min_effectif or n_b < min_effectif:
            rows.append({**base, "t": None, "pvalue": None, "h": None, "taille_effet": None, "conclusion": "effectif insuffisant"})
            continue
        t, _, p = welch_proportions(r["n_a"], r["d_a"], n_b, d_b)
        h = h_cohen(base["taux_a"], base["taux_b"])
        concl = "hétérogène" if (p < alpha and abs(h) >= seuil_h) else (
            "écart significatif mais négligeable" if p < alpha else "homogène")
        rows.append({**base, "t": t, "pvalue": p, "h": h, "taille_effet": _lecture_h(h), "conclusion": concl})
    schema = {"variable": pl.String, "CHR": pl.Int64, "modalite": pl.String, "n_a": pl.Int64, "n_b": pl.Int64,
              "taux_a": pl.Float64, "taux_b": pl.Float64, "t": pl.Float64, "pvalue": pl.Float64, "h": pl.Float64,
              "taille_effet": pl.String, "conclusion": pl.String}
    return pl.DataFrame(rows, schema=schema)


def binariser_mediane(df: pl.DataFrame, var: str, mediane: float) -> pl.Expr:
    """Variable exogène continue → deux groupes autour de la médiane de l'apprentissage."""
    return (pl.when(pl.col(var).is_null()).then(pl.lit("manquant"))
            .when(pl.col(var) <= mediane).then(pl.lit(f"≤ {mediane:g}"))
            .otherwise(pl.lit(f"> {mediane:g}")).alias(var))


# ---------------------------------------------------------------------------
# 7.6 Perte d'information
# ---------------------------------------------------------------------------

def auc_pr(y, score) -> float:
    """Aire sous la courbe précision-rappel (average precision) ; référence = taux de défaut."""
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(np.asarray(y), np.asarray(score, dtype=float)))


def perte_information(y, points, chr_, nom: str = "") -> pl.DataFrame:
    """Gini, AUC, KS et AUC PR du score en points contre ceux des CHR ; ΔGini = Gini score − Gini CHR (7.6)."""
    y = np.asarray(y)
    s, c = -np.asarray(points, dtype=float), np.asarray(chr_, dtype=float)   # score croissant avec le risque
    rows = []
    for lib, x in (("score en points", s), ("CHR", c)):
        p = performance(y, x, f"{lib} — {nom}" if nom else lib)
        rows.append({**p, "auc_pr": auc_pr(y, x), "taux_defaut": float(y.mean())})
    t = pl.DataFrame(rows)
    return t.with_columns(delta_gini=pl.col("gini").first() - pl.col("gini"))


# ---------------------------------------------------------------------------
# Graphiques
# ---------------------------------------------------------------------------

def plot_chr(tables: dict[str, pl.DataFrame], titre: str = ""):
    """Part de population (barres) et taux de défaut (lignes) par CHR, pour un ou plusieurs échantillons."""
    fig, ax = plt.subplots(figsize=(8, 4))
    noms = list(tables)
    largeur = 0.8 / len(noms)
    ax2 = ax.twinx()
    couleurs = ["#4c72b0", "#dd8452", "#55a868"]
    for i, nom in enumerate(noms):
        t = tables[nom].sort("CHR")
        x = t["CHR"].to_numpy()
        ax.bar(x + (i - (len(noms) - 1) / 2) * largeur, t["part"].to_list(), width=largeur, alpha=0.35,
               color=couleurs[i % 3])
        ax2.plot(x, t["taux_defaut"].to_list(), marker="o", color=couleurs[i % 3], label=nom, linewidth=2)
    ax.set_xlabel("CHR (1 = la moins risquée)")
    ax.set_ylabel("Part de population (barres)")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax2.set_ylabel("Taux de défaut (lignes)")
    ax2.set_yscale("log")
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=2))
    ax2.legend(fontsize=8)
    ax.set_xticks(t["CHR"].to_list())
    ax.set_title(titre, fontsize=10, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_bornes_score(points, y, bornes: list[float], titre: str = ""):
    """Distribution du score en points (sains et défauts) et position des bornes des CHR (7.1.2 : paliers)."""
    pts, y = np.asarray(points, dtype=float), np.asarray(y)
    fig, ax = plt.subplots(figsize=(9, 3.6))
    valeurs, n = np.unique(pts, return_counts=True)
    ax.vlines(valeurs, 0, n, color="#9aa9c3", linewidth=1, label="effectif par valeur de score (paliers)")
    ax.set_yscale("log")
    for b in bornes:
        ax.axvline(b + 0.5, color="#d62728", linestyle="--", linewidth=1)
    ax2 = ax.twinx()
    vd, nd = np.unique(pts[y == 1], return_counts=True)
    ax2.scatter(vd, nd, color="#d62728", s=8, label="défauts")
    ax2.set_ylabel("Défauts", color="#d62728")
    ax.set_xlabel("Score en points (grille PDO)")
    ax.set_ylabel("Effectif (échelle log)")
    ax.legend(loc="upper left", fontsize=7)
    ax.set_title(titre, fontsize=10, fontweight="bold")
    fig.tight_layout()
    return fig
