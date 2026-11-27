"""
Classes homogènes de risque (chapitre 7) : construction à partir du score d'un segment, contrôles
de stabilité, d'hétérogénéité entre classes et d'homogénéité interne. Fonctions indépendantes du segment :
elles s'appliquent telles quelles aux CHR de SA, SB et SM.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from utils.binning import appliquer_regle, bornes_quantiles, psi_mensuel, regrouper, stabilite_regle


# ---------------------------------------------------------------------------
# Construction des classes (7.1)
# ---------------------------------------------------------------------------

def chr_par_regroupement(df: pl.DataFrame, score: str, target: str, date: str, n_fin: int = 30,
                         part_min: float = 0.05, defauts_min: int = 20, ecart_min: float = 0.3,
                         bornes_fines: list[float] | None = None) -> dict:
    """CHR par fusion progressive : découpage fin du score (quantiles, ou bornes fournies, par exemple celles
    de Jenks), puis fusion des classes voisines qui ne respectent pas les règles (effectif, défauts, monotonie,
    écart de risque ≥ ecart_min chaque mois). Renvoie une règle de découpage du score (format du binning)."""
    bornes = bornes_fines if bornes_fines is not None else bornes_quantiles(df, score, n_fin)
    fine = {"type": "num", "bornes": sorted(bornes), "missing": None}
    methode = "jenks + regroupement" if bornes_fines is not None else "quantiles + regroupement"
    return regrouper(df, score, target, fine, date, part_min, defauts_min, ecart_min, monotone=True) | {"methode": methode}


def _jenks(valeurs: np.ndarray, poids: np.ndarray, k: int) -> tuple[list[int], float]:
    """Partition optimale de Fisher-Jenks de valeurs triées pondérées en k groupes (programmation dynamique) :
    minimise la somme des variances intra-groupes. Renvoie les indices de début des groupes 2..k et la SSW."""
    m = len(valeurs)
    W, S1, S2 = (np.concatenate([[0], np.cumsum(x)]) for x in (poids, poids * valeurs, poids * valeurs ** 2))

    def ssw(i, j):                                  # groupe des valeurs i..j-1
        w = W[j] - W[i]
        return S2[j] - S2[i] - (S1[j] - S1[i]) ** 2 / w if w > 0 else 0.0

    cout = np.full((k + 1, m + 1), np.inf); cout[0, 0] = 0.0
    arg = np.zeros((k + 1, m + 1), dtype=int)
    for g in range(1, k + 1):
        for j in range(g, m + 1):
            meilleurs = [(cout[g - 1, i] + ssw(i, j), i) for i in range(g - 1, j)]
            cout[g, j], arg[g, j] = min(meilleurs)
    debuts, j = [], m
    for g in range(k, 0, -1):
        i = arg[g, j]; debuts.append(i); j = i
    return sorted(debuts)[1:], float(cout[k, m])


def chr_jenks(df: pl.DataFrame, score: str, pd_col: str, k: int) -> tuple[dict, float]:
    """CHR par l'algorithme de Jenks (natural breaks) appliqué à la PD prédite : les coupures minimisent la
    variance de la PD à l'intérieur des classes. Renvoie la règle de découpage du score et le GVF
    (goodness of variance fit = 1 − variance intra / variance totale, proche de 1 = classes homogènes)."""
    t = df.group_by(score).agg(pd=pl.col(pd_col).mean(), n=pl.len()).sort(score)
    v, w, s = t["pd"].to_numpy(), t["n"].to_numpy().astype(float), t[score].to_numpy()
    debuts, sswk = _jenks(v, w, k)
    sst = float((w * (v - np.average(v, weights=w)) ** 2).sum())
    bornes = [float((s[i - 1] + s[i]) / 2) for i in debuts]          # coupure entre deux valeurs de score observées
    return {"type": "num", "bornes": bornes, "missing": None, "methode": f"jenks k={k}"}, 1 - sswk / sst


# ---------------------------------------------------------------------------
# Description et contrôles (7.2 à 7.4, 7.6)
# ---------------------------------------------------------------------------

def _wilson(n_defaut, n, z: float = 1.96):
    p = n_defaut / n
    centre = (p + z ** 2 / (2 * n)) / (1 + z ** 2 / n)
    demi = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / (1 + z ** 2 / n)
    return centre - demi, centre + demi


def table_chr(df: pl.DataFrame, score: str, target: str, regle: dict, pd_col: str | None = None) -> pl.DataFrame:
    """Une ligne par CHR, numérotées de 1 (la moins risquée) à K : bornes de score, effectif, défauts,
    taux de défaut et intervalle de Wilson à 95 %, part de population, PD prédite moyenne si fournie."""
    d = appliquer_regle(df, score, regle).with_columns(df[target].alias("y"))
    if pd_col:
        d = d.with_columns(df[pd_col].alias("pd"))
    aggs = [pl.len().alias("n"), pl.col("y").sum().alias("n_defaut"), pl.col("y").mean().alias("taux_defaut")]
    if pd_col:
        aggs.append(pl.col("pd").mean().alias("pd_predite_moyenne"))
    t = d.group_by("classe", "ordre").agg(aggs).sort("taux_defaut")
    bas, haut = _wilson(t["n_defaut"].to_numpy(), t["n"].to_numpy())
    return t.with_columns(chr=pl.int_range(1, t.height + 1), part_population=pl.col("n") / pl.col("n").sum(),
                          ic_bas=pl.Series(bas), ic_haut=pl.Series(haut)).select(
        "chr", pl.col("classe").alias("intervalle_score"), "n", "part_population", "n_defaut", "taux_defaut",
        "ic_bas", "ic_haut", *(["pd_predite_moyenne"] if pd_col else []))


def heterogeneite_chr(table: pl.DataFrame) -> pl.DataFrame:
    """Séparation des CHR voisines (7.4). Avec de grands effectifs, une p-value seule est peu informative ;
    on rapporte donc aussi des mesures d'ampleur :
    - rapport des taux de défaut (règle des 30 %) ;
    - h de Cohen = 2 arcsin √p2 − 2 arcsin √p1 (taille d'effet pour des proportions). Attention : pour des taux
      de défaut de l'ordre du pour cent, h reste toujours faible (< 0,2) même quand le risque double ; les seuils
      usuels de Cohen (0,2 / 0,5 / 0,8) ne s'appliquent pas aux événements rares, d'où le rapport et le log odds ratio ;
    - log odds ratio entre classes voisines (taille d'effet adaptée aux événements rares) ;
    - recouvrement des intervalles de Wilson (oui = classes mal séparées) ;
    - test de différence de deux proportions (z, unilatéral)."""
    from scipy.stats import norm

    lignes = []
    for a, b in zip(table.iter_rows(named=True), list(table.iter_rows(named=True))[1:]):
        p1, p2 = a["taux_defaut"], b["taux_defaut"]
        pp = (a["n_defaut"] + b["n_defaut"]) / (a["n"] + b["n"])
        z = (p2 - p1) / np.sqrt(pp * (1 - pp) * (1 / a["n"] + 1 / b["n"]))
        lignes.append({"paire": f"{a['chr']} → {b['chr']}", "taux_1": p1, "taux_2": p2,
                       "rapport": p2 / p1 if p1 > 0 else np.inf,
                       "h_cohen": 2 * np.arcsin(np.sqrt(p2)) - 2 * np.arcsin(np.sqrt(p1)),
                       "log_odds_ratio": float(np.log(p2 / (1 - p2)) - np.log(p1 / (1 - p1))) if 0 < p1 < 1 and 0 < p2 < 1 else np.nan,
                       "ic_se_recouvrent": bool(a["ic_haut"] >= b["ic_bas"]), "p_value": float(norm.sf(z))})
    return pl.DataFrame(lignes)


def homogeneite_chr(df: pl.DataFrame, score: str, target: str, regle: dict, variable: str,
                    regle_variable: dict | None = None) -> pl.DataFrame:
    """Homogénéité interne (7.3) : dans chaque CHR, taux de défaut selon les modalités d'une variable extérieure
    (non utilisée par le modèle, ou le mois). Pour chaque classe : écart maximal entre modalités en h de Cohen
    et test du Chi² d'indépendance. Une classe homogène a un h faible (< 0,2) entre ses sous-groupes."""
    from scipy.stats import chi2_contingency

    classes = table_chr(df, score, target, regle).select("chr", "intervalle_score")
    sous = (appliquer_regle(df, variable, regle_variable)["classe"] if regle_variable
            else df[variable].cast(pl.String).fill_null("missing"))
    d = appliquer_regle(df, score, regle).with_columns(df[target].alias("y"), sous.alias("sous_groupe"))
    lignes = []
    for c, lib in classes.iter_rows():
        t = d.filter(pl.col("classe") == lib).group_by("sous_groupe").agg(n=pl.len(), nd=pl.col("y").sum()).filter(pl.col("n") >= 30)
        if t.height < 2:
            continue
        taux = (t["nd"] / t["n"]).to_numpy()
        tab = np.column_stack([t["nd"].to_numpy(), (t["n"] - t["nd"]).to_numpy()])
        p = chi2_contingency(tab)[1] if tab[:, 0].sum() > 0 else 1.0
        h = 2 * np.arcsin(np.sqrt(taux.max())) - 2 * np.arcsin(np.sqrt(taux.min()))
        lignes.append({"chr": c, "variable": variable, "n_sous_groupes": t.height, "taux_min": float(taux.min()),
                       "taux_max": float(taux.max()), "h_cohen_max": float(h), "p_chi2": float(p)})
    return pl.DataFrame(lignes)


def stabilite_chr(df: pl.DataFrame, score: str, target: str, date: str, regle: dict) -> tuple[pl.DataFrame, float]:
    """Stabilité dans le temps (7.2) : taux de défaut de chaque CHR par période et écart minimal avec la classe
    voisine (négatif = croisement de classes sur au moins une période) ; PSI maximal de la répartition des
    classes entre périodes (stabilité en volume)."""
    return stabilite_regle(df, score, target, date, regle), psi_mensuel(df, score, regle, date)


def perte_information(df: pl.DataFrame, score: str, target: str, regle: dict) -> dict:
    """Perte d'information de la mise en classes (7.6) : Gini et AUC-PR du score continu contre ceux des CHR
    (chaque client reçoit le taux de défaut de sa classe)."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    y = df[target].to_numpy()
    t = table_chr(df, score, target, regle)
    taux = appliquer_regle(df, score, regle)["classe"].replace_strict(
        dict(zip(t["intervalle_score"], t["taux_defaut"])), return_dtype=pl.Float64).to_numpy()
    s_neg = -df[score].to_numpy()
    pi = float(y.mean())
    g_score = 2 * roc_auc_score(y, s_neg) - 1
    g_chr = 2 * roc_auc_score(y, taux) - 1
    pr_score = float(average_precision_score(y, s_neg))
    pr_chr = float(average_precision_score(y, taux))
    prn_score = (pr_score - pi) / (1 - pi) if pi < 1 else float("nan")
    prn_chr = (pr_chr - pi) / (1 - pi) if pi < 1 else float("nan")
    return {
        "taux_defaut": pi,
        "gini_score": float(g_score), "gini_chr": float(g_chr), "perte_points": float(100 * (g_score - g_chr)),
        "auc_pr_score": pr_score, "auc_pr_chr": pr_chr, "perte_auc_pr_pts": float(100 * (pr_score - pr_chr)),
        "auc_pr_norm_score": prn_score, "auc_pr_norm_chr": prn_chr,
    }


# ---------------------------------------------------------------------------
# Construction par découpage optimal (optbinning) et analyse des « faux bons »
# ---------------------------------------------------------------------------

def chr_optbinning(df: pl.DataFrame, score: str, target: str, part_min: float = 0.05, defauts_min: int = 20,
                   max_n_bins: int | None = None, max_n_prebins: int = 30) -> dict:
    """CHR par découpage optimal (optbinning, programmation par contraintes) : bornes du score qui maximisent
    l'IV sous contraintes de monotonie (risque décroissant avec le score), d'effectif et de défauts minimaux.
    Les coupures sont placées au milieu de deux valeurs de score observées (convention ]a ; b] du binning)."""
    from optbinning import OptimalBinning

    x, y = df[score].to_numpy().astype(float), df[target].to_numpy().astype(int)
    ob = OptimalBinning(name=score, dtype="numerical", solver="cp", monotonic_trend="descending",
                        min_bin_size=part_min, min_bin_n_event=defauts_min,
                        max_n_bins=max_n_bins, max_n_prebins=max_n_prebins)
    ob.fit(x, y)
    valeurs = np.unique(x)
    bornes = []
    for s in ob.splits:                               # optbinning : [s ; ...[ ; ici : ]... ; borne]
        i = np.searchsorted(valeurs, s, side="left")
        bornes.append(float((valeurs[i - 1] + valeurs[i]) / 2) if 0 < i < len(valeurs) else float(s))
    nom = "optbinning" + (f" max={max_n_bins}" if max_n_bins else "") + (f" défauts≥{defauts_min}" if defauts_min != 20 else "")
    return {"type": "num", "bornes": sorted(set(bornes)), "missing": None, "methode": nom}


def regles_chr_respectees(df: pl.DataFrame, score: str, target: str, date: str, regle: dict,
                          part_min: float = 0.05, defauts_min: int = 20, ecart_min: float = 0.3) -> bool:
    """Contrôle des règles de construction (7.1-7.2) : chaque classe a au moins part_min de la population et
    defauts_min défauts, et deux classes voisines diffèrent d'au moins ecart_min en risque chaque mois."""
    t = table_chr(df, score, target, regle)
    stab = stabilite_regle(df, score, target, date, regle)["ecart_min_suivante"].drop_nulls().drop_nans()
    return bool((t["part_population"] >= part_min).all() and (t["n_defaut"] >= defauts_min).all()
                and (stab.len() == 0 or stab.min() >= ecart_min))


def profil_faux_bons(df: pl.DataFrame, target: str, chr_col: str, classes_bonnes: list[int],
                     regles: dict[str, dict], dans_modele: list[str]) -> pl.DataFrame:
    """Défauts classés dans les CHR les moins risquées (« faux bons ») : pour chaque variable, IV calculée
    au sein de ces CHR (défauts contre sains) et taux de défaut de sa classe la plus risquée. Une variable
    hors modèle avec une IV élevée signale un risk driver que le score ne capte pas pour ces clients."""
    from utils.binning import table_regle

    bons = df.filter(pl.col(chr_col).is_in(classes_bonnes))
    lignes = []
    for v, r in regles.items():
        if v not in bons.columns:
            continue
        t = table_regle(bons, v, target, r)
        pire = t.sort("taux_defaut", descending=True).row(0, named=True)
        lignes.append({"variable": v, "dans_modele": v in dans_modele, "iv_dans_bonnes_chr": float(t["iv"].sum()),
                       "classe_la_plus_risquee": pire["classe"], "taux_defaut_classe": pire["taux_defaut"],
                       "part_population_classe": pire["part_population"], "n_defaut_classe": pire["n_defaut"]})
    return pl.DataFrame(lignes).sort("iv_dans_bonnes_chr", descending=True)


def discrimination_intra_chr(df: pl.DataFrame, target: str, chr_col: str,
                             scores: dict[str, np.ndarray]) -> pl.DataFrame:
    """Discrimination résiduelle à l'intérieur de chaque CHR (homogénéité interne, 7.3) :
    pour chaque classe et chaque score fourni (convention : valeur élevée = risque élevé,
    donc -score_pdo pour le score PDO, ou probabilité prédite pour logit / XGBoost),
    calcule effectif, défauts, taux de défaut, AUC-ROC, Gini, AUC-PR et AUC-PR normalisée.
    Une classe parfaitement homogène vis-à-vis du score présente une AUC proche de 0,50 (Gini ≈ 0)."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    y_all = df[target].to_numpy().astype(int)
    c_all = df[chr_col].to_numpy()
    lignes = []
    for k in sorted(np.unique(c_all)):
        masque = c_all == k
        yk = y_all[masque]
        n, nd = int(yk.size), int(yk.sum())
        pi = float(yk.mean()) if n > 0 else float("nan")
        for nom, s in scores.items():
            sk = np.asarray(s, dtype=float)[masque]
            if 0 < nd < n:
                auc = float(roc_auc_score(yk, sk))
                ap = float(average_precision_score(yk, sk))
                apn = (ap - pi) / (1 - pi) if pi < 1 else float("nan")
                gini = 2 * auc - 1
            else:
                auc = gini = ap = apn = float("nan")
            lignes.append({
                "chr": int(k), "score": nom, "n": n, "n_defaut": nd, "taux_defaut": pi,
                "auc_roc": auc, "gini": gini, "auc_pr": ap, "auc_pr_norm": apn,
            })
    return pl.DataFrame(lignes).sort(["chr", "score"])
