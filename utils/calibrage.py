"""Du score à la probabilité : calibrage des CHR (risk quantification), chapitre 8 du cours.

- 8.2 fréquence observée par CHR et son incertitude : intervalles de Wald et de Wilson ;
- 8.3 PD Point-in-Time par cohorte, absence de croisement, modèle à un facteur (Vasicek / ASRF) ;
- 8.4 PD Through-the-Cycle par Long Run Average (moyenne des taux des cohortes) ;
- 8.5 autres méthodes : régression logistique sur le score, régression isotone, crédibilité (bayésien) ;
- 8.6 vérification : diagramme de calibration, Brier Score et décomposition de Murphy, Hosmer-Lemeshow,
  test binomial par CHR (backtesting 9.2) ;
- 8.7 PD réglementaire : Margin of Conservatism A (statistique), B (limites connues), C (jugement expert).

Convention : CHR 1 = la moins risquée ; un score en points élevé = peu risqué.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import polars as pl
from scipy import stats
from scipy.optimize import brentq, minimize_scalar


# ---------------------------------------------------------------------------
# 8.2 Fréquence observée et intervalles de confiance
# ---------------------------------------------------------------------------

def ic_proportion(d, n, alpha: float = 0.05, methode: str = "wilson") -> tuple[np.ndarray, np.ndarray]:
    """IC bilatéral de niveau 1 − alpha d'une proportion d/n (cours 8.2).
    methode : "wald" (approximation normale, peut sortir de [0, 1]), "wilson" (inversion du test en p, recommandé à
    PD faible) ou "exact" (Clopper-Pearson, par la loi bêta)."""
    d, n = np.asarray(d, dtype=float), np.asarray(n, dtype=float)
    p = d / n
    z = stats.norm.ppf(1 - alpha / 2)
    if methode == "wald":
        demi = z * np.sqrt(p * (1 - p) / n)
        return p - demi, p + demi
    if methode == "wilson":
        centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
        demi = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
        return centre - demi, centre + demi
    if methode == "exact":
        bas = np.where(d > 0, stats.beta.ppf(alpha / 2, d, n - d + 1), 0.0)
        haut = np.where(d < n, stats.beta.ppf(1 - alpha / 2, d + 1, n - d), 1.0)
        return bas, haut
    raise ValueError(f"méthode inconnue : {methode}")


def table_frequence(df: pl.DataFrame, chr_col: str, target: str, alpha: float = 0.05) -> pl.DataFrame:
    """PD naïve par CHR (taux observé n1,k / nk) avec ses IC de Wald et de Wilson (8.2)."""
    t = (df.group_by(chr_col).agg(n=pl.len(), n_defaut=pl.col(target).sum()).sort(chr_col)
         .with_columns(part=pl.col("n") / pl.col("n").sum(), pd_naive=pl.col("n_defaut") / pl.col("n")))
    n, d = t["n"].to_numpy(), t["n_defaut"].to_numpy()
    wa_b, wa_h = ic_proportion(d, n, alpha, "wald")
    wi_b, wi_h = ic_proportion(d, n, alpha, "wilson")
    return t.with_columns(wald_bas=pl.Series(wa_b), wald_haut=pl.Series(wa_h),
                          wilson_bas=pl.Series(wi_b), wilson_haut=pl.Series(wi_h)).rename({chr_col: "CHR"})


# ---------------------------------------------------------------------------
# 8.3 PD Point-in-Time par cohorte
# ---------------------------------------------------------------------------

def taux_cohortes(df: pl.DataFrame, chr_col: str, target: str, cohorte: str) -> pl.DataFrame:
    """Table longue CHR × cohorte : effectif, défauts, taux observé (PD PIT de la cohorte)."""
    return (df.group_by(chr_col, cohorte).agg(n=pl.len(), n_defaut=pl.col(target).sum())
            .with_columns(taux=pl.col("n_defaut") / pl.col("n")).rename({chr_col: "CHR"}).sort(cohorte, "CHR"))


def croisements(taux_long: pl.DataFrame, cohorte: str) -> pl.DataFrame:
    """Par cohorte : l'ordre TD_1 ≤ … ≤ TD_K est-il respecté ? Nombre de paires adjacentes inversées
    (taux strictement décroissant). Une égalité (deux CHR sans défaut) n'est pas comptée comme un croisement."""
    return (taux_long.sort(cohorte, "CHR").group_by(cohorte, maintain_order=True)
            .agg(n_inversions=(pl.col("taux").diff() < 0).sum(), n_defaut=pl.col("n_defaut").sum())
            .with_columns(ordre_respecte=pl.col("n_inversions") == 0))


def rho_bale_retail(pd_) -> np.ndarray:
    """Corrélation d'actif réglementaire « autre clientèle de détail » (CRR art. 154) :
    ρ = 0,03 · (1 − e^(−35 PD)) / (1 − e^(−35)) + 0,16 · [1 − (1 − e^(−35 PD)) / (1 − e^(−35))]."""
    pd_ = np.asarray(pd_, dtype=float)
    w = (1 - np.exp(-35 * pd_)) / (1 - np.exp(-35))
    return 0.03 * w + 0.16 * (1 - w)


def pd_pit(pd_ttc, rho, z) -> np.ndarray:
    """Formule centrale du modèle ASRF (8.3.3) : PD_PIT(Z) = Φ((Φ⁻¹(PD_TTC) − √ρ Z) / √(1 − ρ))."""
    pd_ttc, rho = np.asarray(pd_ttc, dtype=float), np.asarray(rho, dtype=float)
    return stats.norm.cdf((stats.norm.ppf(pd_ttc) - np.sqrt(rho) * z) / np.sqrt(1 - rho))


def pd_choc_conjoncture(pd_, rho, delta_z: float) -> np.ndarray:
    """PD d'une classe si la conjoncture était plus dégradée de delta_z écarts-types du facteur Z que la période
    observée (ASRF, 8.3.3) : la PD observée étant une PD PIT à un état Z0, la PD PIT en Z0 − delta_z vaut
    Φ(Φ⁻¹(PD) + √ρ · delta_z / √(1 − ρ))."""
    pd_, rho = np.asarray(pd_, dtype=float), np.asarray(rho, dtype=float)
    return stats.norm.cdf(stats.norm.ppf(pd_) + np.sqrt(rho) * delta_z / np.sqrt(1 - rho))


def estimer_z(taux_long: pl.DataFrame, pd_ttc: dict[int, float], rho: dict[int, float], cohorte: str) -> pl.DataFrame:
    """Facteur systémique Z_t de chaque cohorte (8.3.3), estimé par maximum de vraisemblance binomiale sur
    toutes les CHR à la fois : défauts_k,t ~ B(n_k,t, PD_PIT_k(Z_t)). Plus robuste qu'une inversion classe par classe
    quand certaines CHR ont 0 défaut dans le mois. Z < 0 : conjoncture plus dégradée que la moyenne de l'historique."""
    rows = []
    for (c,), g in taux_long.group_by(cohorte, maintain_order=True):
        k, n, d = g["CHR"].to_list(), g["n"].to_numpy(), g["n_defaut"].to_numpy()
        pt, rh = np.array([pd_ttc[i] for i in k]), np.array([rho[i] for i in k])

        def nll(z):
            p = np.clip(pd_pit(pt, rh, z), 1e-12, 1 - 1e-12)
            return -float(np.sum(d * np.log(p) + (n - d) * np.log(1 - p)))
        res = minimize_scalar(nll, bounds=(-5, 5), method="bounded")
        rows.append({cohorte: c, "z": float(res.x), "n_defaut": int(d.sum()),
                     "attendus_ttc": float(np.sum(n * pt))})
    return pl.DataFrame(rows).sort(cohorte)


# ---------------------------------------------------------------------------
# 8.4 Long Run Average
# ---------------------------------------------------------------------------

def pd_lra(taux_long: pl.DataFrame) -> pl.DataFrame:
    """PD LRA par CHR = moyenne simple des taux des T cohortes (8.4), comparée au taux poolé (toutes cohortes
    regroupées) ; dispersion des taux mensuels (min, max, écart-type)."""
    return (taux_long.group_by("CHR").agg(
        n_cohortes=pl.len(), n=pl.col("n").sum(), n_defaut=pl.col("n_defaut").sum(),
        pd_lra=pl.col("taux").mean(), taux_min=pl.col("taux").min(), taux_max=pl.col("taux").max(),
        ecart_type=pl.col("taux").std())
        .with_columns(pd_poolee=pl.col("n_defaut") / pl.col("n")).sort("CHR"))


# ---------------------------------------------------------------------------
# 8.5 Autres méthodes de calibrage
# ---------------------------------------------------------------------------

def calibrer_logistique(points, y):
    """Régression logistique de y sur le score en points (8.5) : PD(s) = 1 / (1 + e^−(a + b·s)).
    Renvoie une fonction points → PD et les coefficients (a, b)."""
    from sklearn.linear_model import LogisticRegression
    x = np.asarray(points, dtype=float).reshape(-1, 1)
    m = LogisticRegression(C=1e6, max_iter=1000).fit(x, np.asarray(y, dtype=int))
    a, b = float(m.intercept_[0]), float(m.coef_[0, 0])
    return (lambda s: 1 / (1 + np.exp(-(a + b * np.asarray(s, dtype=float))))), (a, b)


def calibrer_isotone(points, y):
    """Régression isotone (8.5) : PD décroissante par morceaux du score, sans hypothèse de forme.
    Renvoie une fonction points → PD."""
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(increasing=False, out_of_bounds="clip", y_min=0, y_max=1)
    iso.fit(np.asarray(points, dtype=float), np.asarray(y, dtype=float))
    return lambda s: iso.predict(np.asarray(s, dtype=float))


def pd_credibilite(n, d, prior: float, m: float) -> tuple[np.ndarray, np.ndarray]:
    """Approche bayésienne / crédibilité (8.5) : p_bayes = w · p_k + (1 − w) · p̄ avec w = n_k / (n_k + m).
    C'est la moyenne a posteriori d'un a priori Bêta de moyenne p̄ et de « taille » m dossiers.
    Renvoie (p_bayes, w)."""
    n, d = np.asarray(n, dtype=float), np.asarray(d, dtype=float)
    w = n / (n + m)
    return w * d / n + (1 - w) * prior, w


def pd_moyenne_par_chr(chr_, pd_individuelle) -> dict[int, float]:
    """PD d'une CHR = moyenne des PD individuelles de ses dossiers (pour les méthodes qui calibrent le score)."""
    t = pl.DataFrame({"CHR": np.asarray(chr_), "p": np.asarray(pd_individuelle, dtype=float)})
    return dict(t.group_by("CHR").agg(pl.col("p").mean()).sort("CHR").iter_rows())


# ---------------------------------------------------------------------------
# 8.6 Vérifier le calibrage
# ---------------------------------------------------------------------------

def brier(y, p) -> float:
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    return float(np.mean((y - p) ** 2))


def log_loss(y, p) -> float:
    y, p = np.asarray(y, dtype=float), np.clip(np.asarray(p, dtype=float), 1e-15, 1 - 1e-15)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def murphy(chr_, y, p) -> dict:
    """Décomposition de Murphy (8.6.2) : Brier = incertitude − résolution + calibration, la PD étant constante dans
    chaque CHR. Renvoie aussi le Brier direct, pour contrôle."""
    t = pl.DataFrame({"k": np.asarray(chr_), "y": np.asarray(y, dtype=float), "p": np.asarray(p, dtype=float)})
    n, y_bar = t.height, float(t["y"].mean())
    g = t.group_by("k").agg(n=pl.len(), yk=pl.col("y").mean(), pk=pl.col("p").first())
    incertitude = y_bar * (1 - y_bar)
    resolution = float((g["n"] * (g["yk"] - y_bar) ** 2).sum() / n)
    calibration = float((g["n"] * (g["yk"] - g["pk"]) ** 2).sum() / n)
    return {"incertitude": incertitude, "resolution": resolution, "calibration": calibration,
            "brier_reconstruit": incertitude - resolution + calibration, "brier": brier(t["y"], t["p"])}


def hosmer_lemeshow(n, d, p, ddl: int | None = None) -> dict:
    """Test de Hosmer-Lemeshow sur G groupes (8.6.3) : HL = Σ (O − E)² / (E (1 − E/n)) ~ χ²(G − 2) sous H0.
    Sur un échantillon indépendant de celui du calibrage, ddl = G est aussi d'usage (option `ddl`)."""
    n, d, p = (np.asarray(x, dtype=float) for x in (n, d, p))
    e = n * p
    hl = float(np.sum((d - e) ** 2 / (e * (1 - p))))
    ddl = len(n) - 2 if ddl is None else ddl
    return {"hl": hl, "ddl": ddl, "pvalue": float(stats.chi2.sf(hl, ddl))}


def table_verification(df: pl.DataFrame, chr_col: str, target: str, pd_map: dict[int, float],
                       alpha: float = 0.05) -> pl.DataFrame:
    """Par CHR : PD prédite, taux observé et son IC de Wilson, défauts attendus, et test binomial (backtesting 9.2) :
    p-value unilatérale P(D ≥ d | PD prédite) (sous-estimation du risque) et bilatérale."""
    t = (df.group_by(chr_col).agg(n=pl.len(), n_defaut=pl.col(target).sum()).sort(chr_col).rename({chr_col: "CHR"})
         .with_columns(pd_predite=pl.col("CHR").replace_strict(pd_map, return_dtype=pl.Float64)))
    n, d, p = t["n"].to_numpy(), t["n_defaut"].to_numpy().astype(int), t["pd_predite"].to_numpy()
    bas, haut = ic_proportion(d, n, alpha, "wilson")
    p_sup = stats.binom.sf(d - 1, n, p)
    p_bil = [stats.binomtest(int(di), int(ni), float(pi)).pvalue for di, ni, pi in zip(d, n, p)]
    return t.with_columns(
        taux_observe=pl.col("n_defaut") / pl.col("n"), attendus=pl.col("n") * pl.col("pd_predite"),
        obs_wilson_bas=pl.Series(bas), obs_wilson_haut=pl.Series(haut),
        pvalue_sous_estimation=pl.Series(p_sup), pvalue_bilaterale=pl.Series(p_bil, dtype=pl.Float64),
    ).with_columns(pd_dans_ic=(pl.col("pd_predite") >= pl.col("obs_wilson_bas"))
                   & (pl.col("pd_predite") <= pl.col("obs_wilson_haut")))


def synthese_calibrage(df: pl.DataFrame, chr_col: str, target: str, methodes: dict[str, dict[int, float]],
                       nom: str = "") -> pl.DataFrame:
    """Pour chaque jeu de PD par CHR : Brier, log-loss, décomposition de Murphy, Hosmer-Lemeshow (ddl = G),
    défauts attendus contre observés."""
    chr_, y = df[chr_col].to_numpy(), df[target].to_numpy()
    rows = []
    for meth, pd_map in methodes.items():
        p = np.array([pd_map[k] for k in chr_])
        tv = table_verification(df, chr_col, target, pd_map)
        hl = hosmer_lemeshow(tv["n"], tv["n_defaut"], tv["pd_predite"], ddl=tv.height)
        rows.append({"echantillon": nom, "methode": meth, **murphy(chr_, y, p), "log_loss": log_loss(y, p),
                     "attendus": float(p.sum()), "observes": int(y.sum()), "hl": hl["hl"], "hl_ddl": hl["ddl"],
                     "hl_pvalue": hl["pvalue"], "n_chr_hors_ic": int((~tv["pd_dans_ic"]).sum())})
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# 8.7 Margin of Conservatism et PD réglementaire
# ---------------------------------------------------------------------------

def table_moc(n, d, pd_calibree, rho, alpha_a: float = 0.05, delta_z: float = 0.0, moc_c_relative: float = 0.0,
              plancher: float = 0.0) -> pl.DataFrame:
    """PD réglementaire par CHR (8.7) : PD_reg = PD + MoC_A + MoC_B + MoC_C (somme, pratique la plus courante).

    - MoC A (statistique) : borne haute de l'IC de Wilson bilatéral à 1 − alpha_a sur le taux poolé, moins le taux
      poolé (8.2 / 8.7.2) ;
    - MoC B (couverture insuffisante du cycle, 8.4) : écart entre la PD sous une conjoncture plus dégradée de
      delta_z écarts-types du facteur systémique (modèle ASRF, `pd_choc_conjoncture`) et la PD calibrée ;
    - MoC C (jugement expert) : moc_c_relative × PD calibrée.

    Le plancher (PD minimale réglementaire) s'applique après la MoC ; il n'en fait pas partie."""
    n, d, pd_c, rho = (np.asarray(x, dtype=float) for x in (n, d, pd_calibree, rho))
    _, haut = ic_proportion(d, n, alpha_a, "wilson")
    moc_a = np.maximum(haut - d / n, 0.0)
    moc_b = np.maximum(pd_choc_conjoncture(pd_c, rho, delta_z) - pd_c, 0.0)
    moc_c = moc_c_relative * pd_c
    total = moc_a + moc_b + moc_c
    return pl.DataFrame({
        "CHR": np.arange(1, len(n) + 1), "pd_calibree": pd_c, "moc_a": moc_a, "moc_b": moc_b, "moc_c": moc_c,
        "moc_totale": total, "pd_reglementaire": pd_c + total,
        "pd_finale": np.maximum(pd_c + total, plancher), "plancher_actif": pd_c + total < plancher,
    })


def appliquer_pd(points, bornes: list[float], pd_map: dict[int, float]) -> np.ndarray:
    """Mapping score → CHR → PD (8.5.1), pour noter une nouvelle base (test, OOT) sans rien réestimer."""
    from utils.chr import affecter_chr
    k = affecter_chr(points, bornes)
    return np.array([pd_map[int(i)] for i in k])


# ---------------------------------------------------------------------------
# 7.5 Échelle maître
# ---------------------------------------------------------------------------

BORNES_ECHELLE = [0.0005, 0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128, 0.256]


def echelle_maitre(bornes: list[float] = BORNES_ECHELLE) -> pl.DataFrame:
    """Grades de l'échelle maître définis par des bornes de PD (doublement d'un grade au suivant)."""
    b = [0.0] + list(bornes) + [1.0]
    return pl.DataFrame({"grade": list(range(1, len(b))), "pd_min": b[:-1], "pd_max": b[1:]})


def affecter_grade(pd_, bornes: list[float] = BORNES_ECHELLE) -> np.ndarray:
    """Grade de l'échelle maître de chaque PD, convention ]pd_min ; pd_max]."""
    return np.searchsorted(np.asarray(bornes), np.asarray(pd_, dtype=float), side="left") + 1


# ---------------------------------------------------------------------------
# Graphiques
# ---------------------------------------------------------------------------

def plot_taux_cohortes(taux_long: pl.DataFrame, cohorte: str, pd_ref: dict[int, float] | None = None, titre: str = ""):
    """Trajectoires des PD PIT par CHR au fil des cohortes (échelle log), avec la PD LRA en pointillés (8.3.1)."""
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for (k,), g in taux_long.sort("CHR", cohorte).group_by("CHR", maintain_order=True):
        x, tx = g[cohorte].to_list(), g["taux"].to_numpy()
        tx_plot = np.where(tx > 0, tx, np.nan)      # 0 défaut : pas de point sur l'échelle log
        ligne, = ax.plot(x, tx_plot, marker="o", label=f"CHR {k}")
        if pd_ref is not None:
            ax.axhline(pd_ref[k], color=ligne.get_color(), ls=":", lw=1)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.2%}"))
    ax.set_ylabel("taux de défaut observé (log)")
    ax.set_title(titre or "PD PIT par CHR et par cohorte (pointillés : PD LRA)")
    ax.legend(fontsize=8, ncols=2)
    fig.tight_layout()
    return fig


def plot_diagramme_calibration(tables: dict[str, pl.DataFrame], titre: str = ""):
    """Diagramme de calibration (8.6.1) : PD prédite contre taux observé par CHR, avec l'IC de Wilson du taux
    observé ; échelle log-log (PD de 0,01 % à quelques %). Un point au-dessus de la diagonale : risque sous-estimé.
    Une CHR sans défaut observé est placée sur l'axe bas."""
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    tout = pl.concat([t.select("pd_predite", "obs_wilson_haut") for t in tables.values()])
    lo = max(min(float(tout["pd_predite"].min()) / 3, 1e-4), 1e-5)
    hi = float(max(tout["pd_predite"].max(), tout["obs_wilson_haut"].max())) * 1.5
    for i, (nom, t) in enumerate(tables.items()):
        x = t["pd_predite"].to_numpy()
        y = np.maximum(t["taux_observe"].to_numpy(), lo)
        yb = np.maximum(t["obs_wilson_bas"].to_numpy(), lo)
        yh = t["obs_wilson_haut"].to_numpy()
        dx = 1 + 0.06 * (i - (len(tables) - 1) / 2)
        couleur = plt.cm.tab10(i)
        ax.errorbar(x * dx, y, yerr=[y - yb, yh - y], fmt="none", capsize=3, alpha=0.7, ecolor=couleur)
        ax.scatter(x * dx, y, s=20 + 120 * t["n"].to_numpy() / t["n"].max(), color=couleur, label=nom, zorder=3)
        for k, xi, yi in zip(t["CHR"], x, y):
            if i == 0:
                ax.annotate(f"{k}", (xi, yi), textcoords="offset points", xytext=(6, -10), fontsize=8)
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="calibrage parfait")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo, hi); ax.set_ylim(lo * 0.9, hi)
    fmt = mticker.FuncFormatter(lambda v, _: f"{v:.2%}")
    ax.xaxis.set_major_formatter(fmt); ax.yaxis.set_major_formatter(fmt)
    ax.set_xlabel("PD prédite"); ax.set_ylabel("taux de défaut observé (IC de Wilson 95 %)")
    ax.set_title(titre or "Diagramme de calibration")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_pd_reglementaire(moc: pl.DataFrame, titre: str = ""):
    """PD calibrée et MoC A, B, C empilées par CHR (8.7.3), échelle log."""
    fig, ax = plt.subplots(figsize=(7.5, 4))
    k = moc["CHR"].to_numpy()
    base = moc["pd_calibree"].to_numpy()
    ax.bar(k, base, label="PD calibrée (LRA)", color="#4C72B0")
    bas = base.copy()
    for col, lab, c in (("moc_a", "MoC A", "#DD8452"), ("moc_b", "MoC B", "#55A868"), ("moc_c", "MoC C", "#C44E52")):
        v = moc[col].to_numpy()
        ax.bar(k, v, bottom=bas, label=lab, color=c)
        bas = bas + v
    ax.scatter(k, moc["pd_finale"].to_numpy(), marker="_", s=400, color="k", label="PD finale (après plancher)", zorder=3)
    ax.set_yscale("log")
    ax.set_ylim(base.min() / 2, moc["pd_finale"].max() * 1.6)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.2%}"))
    ax.set_xticks(k, [f"CHR {i}" for i in k])
    ax.set_title(titre or "Du calibrage à la PD réglementaire")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig
