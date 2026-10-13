from __future__ import annotations
import json
import numpy as np
import polars as pl

SEUILS_SIDDIQI = [(0.02, "inutile"), (0.1, "faible"), (0.3, "moyen"), (0.5, "fort"), (float("inf"), "suspect")]


def _est_discrete(df: pl.DataFrame, var: str, max_modalites: int) -> bool:
    t = df.schema[var]
    return isinstance(t, (pl.String, pl.Categorical, pl.Enum, pl.Boolean)) or df[var].n_unique() <= max_modalites


def _classes(df: pl.DataFrame, var: str, n_bins: int, max_modalites: int) -> pl.DataFrame:
    """Colonnes 'classe' (libellé) et 'ordre' (tri) ; le missing est une classe à part entière, placée en dernier."""
    if _est_discrete(df, var, max_modalites):
        return df.select(
            pl.col(var).cast(pl.String).fill_null("missing").alias("classe"),
            pl.col(var).rank("dense").cast(pl.Float64).alias("ordre"),
        )
    q = pl.col(var).qcut(n_bins, allow_duplicates=True, include_breaks=True)
    return df.select(q.alias("q")).select(
        pl.col("q").struct.field("category").cast(pl.String).fill_null("missing").alias("classe"),
        pl.col("q").struct.field("breakpoint").alias("ordre"),
    )

def woe_encode(df: pl.DataFrame, variables: list[str], target: str, n_bins: int = 10,
               max_modalites: int = 10) -> pl.DataFrame:
    """Remplace chaque variable par la WOE de sa classe (découpage fin, missing = classe à part)."""
    cols = []
    for v in variables:
        table = woe_iv_table(df, v, target, n_bins, max_modalites).select("classe", "woe")
        classes = _classes(df, v, n_bins, max_modalites).select("classe")
        cols.append(classes.join(table, on="classe", how="left", maintain_order="left")["woe"].alias(v))
    return pl.DataFrame(cols)


def woe_iv_table(df: pl.DataFrame, var: str, target: str, n_bins: int = 10, max_modalites: int = 10) -> pl.DataFrame:
    """Table WOE/IV d'une variable : WOE = ln(%bons / %mauvais), lissage 0,5 pour les classes pures."""
    d = _classes(df, var, n_bins, max_modalites).with_columns(df[target].alias("y"))
    n_bad, n_good = d["y"].sum(), d.height - d["y"].sum()
    return (
        d.group_by("classe")
        .agg(ordre=pl.col("ordre").min(), n=pl.len(), n_defaut=pl.col("y").sum())
        .with_columns(n_sain=pl.col("n") - pl.col("n_defaut"))
        .with_columns(
            taux_defaut=pl.col("n_defaut") / pl.col("n"),
            part_population=pl.col("n") / d.height,
            dist_bad=(pl.col("n_defaut") + 0.5) / (n_bad + 0.5),
            dist_good=(pl.col("n_sain") + 0.5) / (n_good + 0.5),
        )
        .with_columns(woe=(pl.col("dist_good") / pl.col("dist_bad")).log())
        .with_columns(iv=(pl.col("dist_good") - pl.col("dist_bad")) * pl.col("woe"))
        .sort("ordre", nulls_last=True)
        .select("classe", "n", "n_defaut", "taux_defaut", "part_population", "woe", "iv")
    )

def _siddiqi(iv: float) -> str:
    return next(label for seuil, label in SEUILS_SIDDIQI if iv < seuil)


def iv_ranking(df, variables, target, n_bins=10, max_modalites=10):
    """IV de chaque variable avec l'interprétation Siddiqi, triée décroissante."""
    rows = []
    for v in variables:
        t = woe_iv_table(df, v, target, n_bins, max_modalites)
        rows.append({"variable": v, "iv": t["iv"].sum(), "pouvoir": _siddiqi(t["iv"].sum()),
                     "n_classes": t.height, "min_n_classe": t["n"].min()})
    return pl.DataFrame(rows).sort("iv", descending=True)



def iv_by_segment(df: pl.DataFrame, variables: list[str], target: str, segment: str, n_bins: int = 10,
                  max_modalites: int = 10) -> pl.DataFrame:
    """IV et rang de chaque variable dans chaque segment : des classements différents justifient la segmentation."""
    parts = []
    for s in sorted(df[segment].unique().to_list()):
        r = iv_ranking(df.filter(pl.col(segment) == s), variables, target, n_bins, max_modalites)
        parts.append(r.with_row_index("rang", offset=1).select(
            "variable", pl.col("iv").alias(f"iv_{s}"), pl.col("rang").alias(f"rang_{s}")))
    out = parts[0]
    for p in parts[1:]:
        out = out.join(p, on="variable")
    return out.sort(out.columns[1], descending=True)


# --- Règles de découpage : application et table WOE/IV -----------------------

def _fmt(x: float) -> str:
    return ("-inf" if x < 0 else "+inf") if np.isinf(x) else f"{x:.4g}"


def _libelles(bornes: list[float]) -> list[str]:
    bords = [-np.inf, *bornes, np.inf]
    return [f"]{_fmt(a)} ; {_fmt(b)}]" for a, b in zip(bords[:-1], bords[1:])]


def appliquer_regle(df: pl.DataFrame, var: str, regle: dict) -> pl.DataFrame:
    """Colonnes 'classe' (libellé) et 'ordre' ; une modalité inconnue de la règle est traitée comme missing."""
    if regle["type"] == "num":
        x = df[var].cast(pl.Float64).to_numpy()
        idx = np.searchsorted(np.asarray(regle["bornes"], dtype=float), x, side="left").astype(float)
        idx[np.isnan(x)] = np.nan
        libelles = _libelles(regle["bornes"])
        classe = pl.Series(idx).fill_nan(None).cast(pl.Int64).replace_strict(
            dict(enumerate(libelles)), default=None, return_dtype=pl.String)
        ordre_lib = libelles
    else:
        classe = df[var].cast(pl.String).replace_strict(regle["groupes"], default=None, return_dtype=pl.String)
        ordre_lib = regle["ordre"]
    classe = classe.fill_null(regle.get("missing") or "missing")
    rang = {c: i for i, c in enumerate(ordre_lib)}
    return pl.DataFrame({"classe": classe}).with_columns(
        ordre=pl.col("classe").replace_strict(rang, default=len(rang), return_dtype=pl.Int64))


def _table_woe(classes: pl.DataFrame, y: pl.Series) -> pl.DataFrame:
    """Table WOE/IV à partir des colonnes classe/ordre (mêmes formules que woe_iv_table)."""
    d = classes.with_columns(y.alias("y"))
    n_bad, n_good = d["y"].sum(), d.height - d["y"].sum()
    return (
        d.group_by("classe")
        .agg(ordre=pl.col("ordre").min(), n=pl.len(), n_defaut=pl.col("y").sum())
        .with_columns(n_sain=pl.col("n") - pl.col("n_defaut"))
        .with_columns(
            taux_defaut=pl.col("n_defaut") / pl.col("n"),
            part_population=pl.col("n") / d.height,
            dist_bad=(pl.col("n_defaut") + 0.5) / (n_bad + 0.5),
            dist_good=(pl.col("n_sain") + 0.5) / (n_good + 0.5),
        )
        .with_columns(woe=(pl.col("dist_good") / pl.col("dist_bad")).log())
        .with_columns(iv=(pl.col("dist_good") - pl.col("dist_bad")) * pl.col("woe"))
        .sort("ordre")
        .select("classe", "n", "n_defaut", "taux_defaut", "part_population", "woe", "iv")
    )


def table_regle(df: pl.DataFrame, var: str, target: str, regle: dict) -> pl.DataFrame:
    """Table WOE/IV d'une variable découpée selon sa règle ; pour une qualitative, liste des modalités par groupe."""
    t = _table_woe(appliquer_regle(df, var, regle), df[target])
    if regle["type"] == "quali":
        compo = {}
        for m, g in regle["groupes"].items():
            compo.setdefault(g, []).append(m)
        t = t.with_columns(modalites=pl.col("classe").replace_strict(
            {g: ", ".join(sorted(ms)) for g, ms in compo.items()}, default=None, return_dtype=pl.String))
    return t


# --- Découpage fin (4.2) : quantiles, arbre univarié, modalités brutes --------

def bornes_quantiles(df: pl.DataFrame, var: str, n_bins: int = 20) -> list[float]:
    """Bornes aux quantiles (doublons supprimés : une variable à peu de valeurs a moins de classes)."""
    x = df[var].drop_nulls().cast(pl.Float64)
    bornes = sorted({float(x.quantile(k / n_bins, "lower")) for k in range(1, n_bins)})
    return [b for b in bornes if b < x.max()]


def bornes_arbre(df: pl.DataFrame, var: str, target: str, max_classes: int = 8, part_min: float = 0.05) -> list[float]:
    """Bornes d'un arbre de décision univarié (missing exclu, traité comme classe à part)."""
    from sklearn.tree import DecisionTreeClassifier

    d = df.select(var, target).drop_nulls()
    arbre = DecisionTreeClassifier(max_leaf_nodes=max_classes, min_samples_leaf=max(1, int(part_min * df.height)),
                                   random_state=0)
    arbre.fit(d[var].cast(pl.Float64).to_numpy().reshape(-1, 1), d[target].to_numpy())
    t = arbre.tree_
    return sorted(float(s) for s, g in zip(t.threshold, t.children_left) if g != -1)


def regle_fine(df: pl.DataFrame, var: str, max_modalites: int = 10, n_bins: int = 20) -> dict:
    """Découpage fin de départ : modalités brutes (qualitative ou peu de valeurs), sinon quantiles."""
    if df.schema[var].is_numeric():
        if df[var].n_unique() <= max_modalites:
            valeurs = sorted(df[var].drop_nulls().unique().cast(pl.Float64).to_list())
            return {"type": "num", "bornes": valeurs[:-1], "missing": None}
        return {"type": "num", "bornes": bornes_quantiles(df, var, n_bins), "missing": None}
    modalites = df[var].drop_nulls().cast(pl.String).unique().sort().to_list()
    return {"type": "quali", "groupes": {m: m for m in modalites}, "ordre": modalites, "missing": None}


# --- Du découpage fin au découpage final (4.4 à 4.7) --------------------------

def _fusion(n: np.ndarray, nd: np.ndarray, part_min: float, defauts_min: int, ecart_min: float,
            monotone: bool) -> list[list[int]]:
    """Fusionne des classes adjacentes jusqu'à respecter effectif minimal, défauts minimaux,
    monotonie et écart relatif de risque minimal. n, nd : effectifs et défauts (classes x périodes) ;
    l'écart retenu pour une paire est le pire entre global et chaque période (4.6).
    Renvoie les groupes (indices des classes fines)."""
    groupes = [[i] for i in range(len(n))]
    total = n.sum()
    while len(groupes) > 1:
        ng = np.array([n[g].sum(axis=0) for g in groupes])            # classes x périodes
        dg = np.array([nd[g].sum(axis=0) for g in groupes])
        n_tot, d_tot = ng.sum(axis=1), dg.sum(axis=1)
        taux = (d_tot + 0.5) / (n_tot + 1)
        taux_p = (dg + 0.5) / (ng + 1)

        if monotone:
            sens = 1 if np.polyfit(np.arange(len(taux)), taux, 1, w=np.sqrt(n_tot))[0] >= 0 else -1
        else:
            sens = np.where(np.diff(taux) >= 0, 1, -1)                # sens propre à chaque paire
        ecarts = np.minimum(_ecarts(taux, sens), _ecarts(taux_p, sens).min(axis=1))
        petits = np.flatnonzero((n_tot < part_min * total) | (d_tot < defauts_min))
        if petits.size:
            k = petits[np.argmin(n_tot[petits])]
            j = min((v for v in (k - 1, k + 1) if 0 <= v < len(groupes)), key=lambda v: abs(taux[v] - taux[k]))
            i = min(k, j)
        elif ecarts.min() < ecart_min:
            i = int(np.argmin(ecarts))
        else:
            break
        groupes[i:i + 2] = [groupes[i] + groupes[i + 1]]
    return groupes


def regrouper(df: pl.DataFrame, var: str, target: str, regle: dict, date: str | None = None, part_min: float = 0.05,
              defauts_min: int = 20, ecart_min: float = 0.3, monotone: bool = True) -> dict:
    """Règle finale à partir d'une règle fine. Numérique : fusion de classes adjacentes (monotonie imposée
    sauf monotone=False, ex. relation en U justifiée métier). Qualitative : modalités triées par taux de
    défaut puis fusionnées. Missing : classe à part s'il respecte part_min/defauts_min, sinon rattaché
    à la classe de taux le plus proche. Avec date : l'écart de 30 % est exigé sur chaque période (4.6)."""
    t = table_regle(df, var, target, regle)
    miss = t.filter(pl.col("classe") == "missing")
    t = t.filter(pl.col("classe") != "missing")
    if regle["type"] == "quali":
        t = t.sort("taux_defaut")
    if date is None:
        n, nd = t["n"].to_numpy()[:, None], t["n_defaut"].to_numpy()[:, None]
    else:
        p = (appliquer_regle(df, var, regle).with_columns(df[target].alias("y"), df[date].alias("periode"))
             .group_by("classe", "periode").agg(n=pl.len(), nd=pl.col("y").sum()))
        grille = t.select("classe").join(p.select("periode").unique().sort("periode"), how="cross")
        p = grille.join(p, on=["classe", "periode"], how="left", maintain_order="left").fill_null(0)
        k = t.height
        n, nd = p["n"].to_numpy().reshape(k, -1), p["nd"].to_numpy().reshape(k, -1)
    groupes = _fusion(n, nd, part_min, defauts_min, ecart_min, monotone or regle["type"] == "quali")
    if regle["type"] == "num":
        fins = regle["bornes"]
        nouvelle = {"type": "num", "bornes": [fins[g[-1]] for g in groupes[:-1]], "missing": None}
    else:
        classes = t["classe"].to_list()
        groupe_de = {c: f"G{k + 1}" for k, g in enumerate(groupes) for c in (classes[i] for i in g)}
        nouvelle = {"type": "quali", "groupes": {m: groupe_de[c] for m, c in regle["groupes"].items() if c in groupe_de},
                    "ordre": [f"G{k + 1}" for k in range(len(groupes))], "missing": None}
    if miss.height and (miss["part_population"][0] < part_min or miss["n_defaut"][0] < defauts_min):
        finale = table_regle(df, var, target, nouvelle).filter(pl.col("classe") != "missing")
        proche = (finale["taux_defaut"] - miss["taux_defaut"][0]).abs().arg_min()
        nouvelle["missing"] = finale["classe"][proche]
    return nouvelle


def est_monotone(table: pl.DataFrame) -> bool:
    """Taux de défaut monotone sur les classes ordonnées (hors missing)."""
    taux = table.filter(pl.col("classe") != "missing")["taux_defaut"].to_numpy()
    d = np.diff(taux)
    return bool((d >= 0).all() or (d <= 0).all())


def stabilite_regle(df: pl.DataFrame, var: str, target: str, date: str, regle: dict) -> pl.DataFrame:
    """Taux de défaut par classe et par période (4.6). Pour chaque classe : écart relatif minimal avec la
    classe suivante sur l'ensemble des périodes (négatif = inversion sur au moins une période)."""
    d = appliquer_regle(df, var, regle).with_columns(df[target].alias("y"), df[date].alias("periode"))
    long = d.group_by("classe", "ordre", "periode").agg(taux=pl.col("y").mean(), n=pl.len())
    large = long.pivot(on="periode", index=["classe", "ordre"], values="taux", sort_columns=True).sort("ordre")
    periodes = [c for c in large.columns if c not in ("classe", "ordre")]
    hors_miss = large.filter(pl.col("classe") != "missing")
    taux_global = table_regle(df, var, target, regle).filter(pl.col("classe") != "missing")["taux_defaut"].to_numpy()
    
    m = hors_miss.select(periodes).to_numpy().astype(float)
    ecart = np.full(hors_miss.height, np.nan)
    if hors_miss.height > 1:
        ecart[:-1] = np.nanmin(_ecarts(m, np.where(np.diff(taux_global) >= 0, 1, -1)), axis=1)
    return large.join(hors_miss.select("classe").with_columns(ecart_min_suivante=pl.Series(ecart)),
                      on="classe", how="left").rename({p: f"taux_{p}" for p in periodes}).drop("ordre")


def meme_ordre(ref: pl.DataFrame, autre: pl.DataFrame) -> bool:
    """Les classes adjacentes (hors missing) gardent le même sens de risque d'une table à l'autre (ex. train / test)."""
    sens = [np.sign(np.diff(t.filter(pl.col("classe") != "missing")["taux_defaut"].to_numpy()))
            for t in (ref, autre)]
    return sens[0].shape == sens[1].shape and bool((sens[0] == sens[1]).all())


def bornes_xgb(df: pl.DataFrame, var: str, target: str, sens: int | None = None, part_min: float = 0.05,
               n_arbres: int = 50, seed: int = 0) -> list[float]:
    """Bornes d'un XGBoost univarié (défaut ~ var) : points où sa prédiction en escalier change de valeur.
    sens : +1 / -1 = monotonie imposée, 0 = libre, None = signe de la corrélation de Spearman.
    Missing exclu, traité comme classe à part."""
    import xgboost as xgb
    from scipy.stats import spearmanr

    d = df.select(var, target).drop_nulls()
    x, y = d[var].cast(pl.Float64).to_numpy(), d[target].to_numpy()
    if sens is None:
        sens = int(np.sign(spearmanr(x, y).statistic))
    modele = xgb.XGBClassifier(n_estimators=n_arbres, max_depth=1, learning_rate=0.3, max_bin=64,
                               min_child_weight=part_min * len(y) * y.mean(), monotone_constraints=(sens,),
                               random_state=seed)
    modele.fit(x.reshape(-1, 1), y)
    valeurs = np.unique(x)
    pred = modele.predict_proba(valeurs.reshape(-1, 1))[:, 1]
    return valeurs[np.flatnonzero(np.abs(np.diff(pred)) > 1e-9)].tolist()   # classes ]b_i ; b_i+1]


# --- Comparaison des méthodes de découpage fin et choix (4.7) -----------------

METHODES = ("quantiles", "arbre", "xgb")


def regle_par_methode(df: pl.DataFrame, var: str, target: str, methode: str, date: str | None = None,
                      n_bins: int = 20, part_min: float = 0.05, defauts_min: int = 20, ecart_min: float = 0.3,
                      monotone: bool = True) -> dict:
    """Découpage fin selon la méthode ('quantiles', 'arbre', 'xgb', 'modalites') puis regroupement selon
    les règles du cours. 'modalites' : qualitative ou numérique à peu de valeurs (une classe par valeur)."""
    if methode == "modalites":
        fine = regle_fine(df, var, max_modalites=10 ** 9)
    else:
        bornes = {
            "quantiles": lambda: bornes_quantiles(df, var, n_bins),
            "arbre": lambda: bornes_arbre(df, var, target, max_classes=n_bins, part_min=part_min / 2),
            "xgb": lambda: bornes_xgb(df, var, target, sens=None if monotone else 0, part_min=part_min / 2),
        }[methode]()
        fine = {"type": "num", "bornes": bornes, "missing": None}
    return regrouper(df, var, target, fine, date, part_min, defauts_min, ecart_min, monotone)


def gini_univarie(df: pl.DataFrame, var: str, target: str, regle: dict, ref: pl.DataFrame | None = None) -> float:
    """Gini (= Somers' D) de la variable découpée : 2 AUC - 1 avec la WOE comme score (WOE calculées sur ref)."""
    from sklearn.metrics import roc_auc_score

    woe = woe_encode_regles(df, {var: regle}, target, ref)[var].fill_null(0).to_numpy()
    y = df[target].to_numpy()
    return float(2 * roc_auc_score(y, -woe) - 1) if 0 < y.sum() < len(y) else float("nan")


def psi_mensuel(df: pl.DataFrame, var: str, regle: dict, date: str) -> float:
    """Stabilité en volume (4.6) : PSI maximal, sur les périodes, entre la répartition des classes de la période
    et la répartition globale. Repères usuels : < 0,1 stable ; 0,1 à 0,25 à surveiller ; > 0,25 instable."""
    d = appliquer_regle(df, var, regle).with_columns(df[date].alias("periode"))
    glob = d.group_by("classe").agg(p_glob=pl.len() / d.height)
    per = d.group_by("periode", "classe").len().with_columns(p=pl.col("len") / pl.col("len").sum().over("periode"))
    grille = per.select("periode").unique().join(glob, how="cross")
    t = grille.join(per.select("periode", "classe", "p"), on=["periode", "classe"], how="left").fill_null(0)
    eps = 1e-6
    psi = t.group_by("periode").agg(
        psi=((pl.col("p") - pl.col("p_glob")) * ((pl.col("p") + eps) / (pl.col("p_glob") + eps)).log()).sum())
    return float(psi["psi"].max())


def evaluer_regle(train: pl.DataFrame, test: pl.DataFrame, var: str, target: str, date: str, regle: dict,
                  ecart_min: float = 0.3, psi_max: float = 0.1) -> dict:
    """Indicateurs d'une règle : IV et Gini (train / test), nombre de classes, monotonie, ordre conservé sur
    le test, stabilité temporelle en risque (écart ≥ ecart_min chaque période) et en volume (PSI < psi_max)."""
    t_train, t_test = table_regle(train, var, target, regle), table_regle(test, var, target, regle)
    ecart = stabilite_regle(train, var, target, date, regle)["ecart_min_suivante"].drop_nulls().drop_nans()
    psi = psi_mensuel(train, var, regle, date)
    return {
        "variable": var, "methode": regle.get("methode"), "n_classes": t_train.height,
        "iv_train": t_train["iv"].sum(), "iv_test": t_test["iv"].sum(),
        "gini_train": gini_univarie(train, var, target, regle),
        "gini_test": gini_univarie(test, var, target, regle, ref=train),
        "monotone": est_monotone(t_train), "ordre_conserve_test": meme_ordre(t_train, t_test),
        "ecart_min_mois": ecart.min() if ecart.len() else None, "psi_max_mois": psi,
        "stable": bool(ecart.ge(ecart_min).all()) and psi < psi_max,
    }


def comparer_methodes(train: pl.DataFrame, test: pl.DataFrame, var: str, target: str, date: str,
                      max_modalites: int = 10, monotone: bool = True, psi_max: float = 0.1,
                      **params) -> tuple[dict[str, dict], pl.DataFrame]:
    """Une règle finale par méthode de découpage fin, construite sur le train, évaluée sur train et test
    (evaluer_regle)."""
    discrete = not train.schema[var].is_numeric() or train[var].n_unique() <= max_modalites
    regles, lignes = {}, []
    for m in ("modalites",) if discrete else METHODES:
        r = regle_par_methode(train, var, target, m, date, monotone=monotone, **params) | {"methode": m}
        regles[m] = r
        lignes.append(evaluer_regle(train, test, var, target, date, r, params.get("ecart_min", 0.3), psi_max))
    return regles, pl.DataFrame(lignes)


def choisir_methode(comparaison: pl.DataFrame, tolerance: float = 0.05) -> str:
    """Méthode retenue pour une variable : parmi les règles stables et dont l'ordre des classes se
    conserve sur le test (à défaut toutes), celles dont l'IV test est à moins de `tolerance` (relatif) de la meilleure, puis la plus
    parcimonieuse (moins de classes), puis la meilleure IV test."""
    ok = comparaison.filter(pl.col("stable") & pl.col("ordre_conserve_test"))
    c = ok if ok.height else comparaison
    proches = c.filter(pl.col("iv_test") >= (1 - tolerance) * c["iv_test"].max())
    return proches.sort(["n_classes", "iv_test"], descending=[False, True])["methode"][0]


def binning_auto(train: pl.DataFrame, test: pl.DataFrame, variables: list[str], target: str, date: str,
                 non_monotones: tuple[str, ...] = (), tolerance: float = 0.05, max_modalites: int = 10,
                 params_variables: dict[str, dict] | None = None, psi_max: float = 0.1,
                 **params) -> tuple[dict[str, dict], pl.DataFrame, pl.DataFrame]:
    """Pour chaque variable : les 3 méthodes de découpage fin + regroupement, puis choix (choisir_methode).
    params : n_bins, part_min, defauts_min, ecart_min (transmis à regle_par_methode) ;
    params_variables : surcharges par variable (ex. part_min plus bas pour une modalité rare à risque fort, 4.5) ;
    psi_max : seuil de stabilité en volume (PSI mensuel).
    Renvoie les règles retenues, la synthèse (une ligne par variable) et la comparaison complète."""
    regles, comparaisons, retenues = {}, [], []
    for v in variables:
        p = params | (params_variables or {}).get(v, {})
        r, comp = comparer_methodes(train, test, v, target, date, max_modalites, v not in non_monotones, psi_max, **p)
        m = choisir_methode(comp, tolerance)
        regles[v] = r[m]
        comparaisons.append(comp.with_columns(retenue=pl.col("methode") == m))
        retenues.append(comp.filter(pl.col("methode") == m))
    synthese = pl.concat(retenues).with_columns(
        pouvoir=pl.col("iv_test").map_elements(_siddiqi, return_dtype=pl.String)).sort("iv_test", descending=True)
    return regles, synthese, pl.concat(comparaisons)

def _arrondi(x: float, chiffres: int) -> float:
    if x == 0 or not np.isfinite(x):
        return x
    return float(round(x, chiffres - 1 - int(np.floor(np.log10(abs(x))))))


def arrondir_bornes(regle: dict, chiffres: int = 2) -> dict:
    """Bornes numériques arrondies à `chiffres` chiffres significatifs, pour un découpage lisible (4.7) ;
    le rattachement du missing à une classe est reporté sur la classe arrondie correspondante."""
    if regle["type"] != "num" or not regle["bornes"]:
        return regle
    anciennes = regle["bornes"]
    bornes = sorted({_arrondi(b, chiffres) for b in anciennes})
    nouvelle = regle | {"bornes": bornes}
    if regle.get("missing"):
        i = _libelles(anciennes).index(regle["missing"])
        haut = anciennes[i] if i < len(anciennes) else np.inf
        j = int(np.searchsorted(np.asarray(bornes, dtype=float), _arrondi(haut, chiffres), side="left"))
        nouvelle["missing"] = _libelles(bornes)[j]
    return nouvelle


def arrondir_regles(train: pl.DataFrame, test: pl.DataFrame, regles: dict[str, dict], target: str, date: str,
                    chiffres: int = 2, tolerance: float = 0.05, ecart_min: float = 0.3,
                    psi_max: float = 0.1) -> tuple[dict[str, dict], pl.DataFrame, pl.DataFrame]:
    """Arrondit les bornes de chaque règle et ne garde l'arrondi que s'il ne dégrade pas le découpage :
    toujours stable, ordre conservé sur le test, IV test à moins de `tolerance` (relatif) de l'IV d'origine.
    Renvoie les règles finales, leur synthèse et le bilan de l'arrondi variable par variable."""
    finales, lignes, bilan = {}, [], []
    for v, r in regles.items():
        avant = evaluer_regle(train, test, v, target, date, r, ecart_min, psi_max)
        ra = arrondir_bornes(r, chiffres)
        apres = avant if ra == r else evaluer_regle(train, test, v, target, date, ra, ecart_min, psi_max)
        garde = (ra != r and apres["stable"] and apres["ordre_conserve_test"]
                 and apres["iv_test"] >= (1 - tolerance) * avant["iv_test"])
        finales[v] = ra if garde else r
        lignes.append(apres if garde else avant)
        bilan.append({"variable": v, "bornes_origine": str(r.get("bornes")), "bornes_arrondies": str(ra.get("bornes")),
                      "iv_test_origine": avant["iv_test"], "iv_test_arrondi": apres["iv_test"],
                      "stable_arrondi": apres["stable"], "arrondi_retenu": garde})
    synthese = pl.DataFrame(lignes).with_columns(
        pouvoir=pl.col("iv_test").map_elements(_siddiqi, return_dtype=pl.String)).sort("iv_test", descending=True)
    return finales, synthese, pl.DataFrame(bilan)


LIBELLES_METHODES = {"quantiles": "Quantiles", "arbre": "Arbre", "xgb": "XGBoost",
                     "modalites": "Modalités brutes"}


def presenter_synthese(synthese: pl.DataFrame, selection: pl.DataFrame | None = None,
                       iv_suspect: float = 0.5) -> pl.DataFrame:
    """Synthèse du binning en clair : une ligne par variable, colonnes renommées, verdict explicite.
    Avec la sortie de selection_non_redondante, le verdict intègre le contrôle des redondances."""
    if selection is not None:
        synthese = synthese.join(selection.select("variable", "redondante_avec"), on="variable", how="left")
    else:
        synthese = synthese.with_columns(redondante_avec=pl.lit(None, dtype=pl.String))
    valide = "retenue" if selection is not None else "découpage valide"
    verdict = (
        pl.when(pl.col("n_classes") == 1).then(pl.lit("écartée : une seule classe"))
        .when(~pl.col("stable")).then(pl.lit("à revoir : instable dans le temps"))
        .when(~pl.col("ordre_conserve_test")).then(pl.lit("à revoir : ordre des classes non conservé sur le test"))
        .when(pl.col("redondante_avec").is_not_null())
        .then(pl.lit("écartée : redondante avec ") + pl.col("redondante_avec"))
        .when(pl.col("iv_test") > iv_suspect).then(pl.lit(f"{valide} : IV élevée, vérifier la fuite"))
        .otherwise(pl.lit(valide))
    )
    return synthese.sort("iv_test", descending=True).select(
        pl.col("variable").alias("Variable"),
        pl.col("methode").replace_strict(LIBELLES_METHODES, default=pl.col("methode")).alias("Méthode"),
        pl.col("n_classes").alias("Classes"),
        pl.col("iv_train").round(2).alias("IV train"),
        pl.col("iv_test").round(2).alias("IV test"),
        pl.col("gini_test").round(2).alias("Gini test"),
        pl.col("pouvoir").alias("Pouvoir (Siddiqi)"),
        pl.col("psi_max_mois").round(3).alias("PSI max / mois"),
        pl.col("stable").replace_strict({True: "oui", False: "non"}, return_dtype=pl.String).alias("Stable / mois"),
        pl.col("ordre_conserve_test").replace_strict({True: "oui", False: "non"}, return_dtype=pl.String)
        .alias("Ordre conservé test"),
        verdict.alias("Verdict"),
    )


def tables_woe(df: pl.DataFrame, regles: dict[str, dict], target: str) -> pl.DataFrame:
    """Tables WOE/IV de toutes les variables empilées (export Excel / dossier de validation)."""
    return pl.concat([table_regle(df, v, target, r).select(pl.lit(v).alias("variable"), pl.all())
                      for v, r in regles.items()], how="diagonal_relaxed")


def graphique_regle(df: pl.DataFrame, var: str, target: str, date: str, regle: dict, ax=None):
    """Taux de défaut par classe, une courbe par période, et volume par classe en barres (4.6)."""
    import matplotlib.pyplot as plt

    d = appliquer_regle(df, var, regle).with_columns(df[target].alias("y"), df[date].alias("periode"))
    classes = d.select("classe", "ordre").unique().sort("ordre")["classe"].to_list()
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    vol = d.group_by("classe").len()
    part = [vol.filter(pl.col("classe") == c)["len"][0] / d.height for c in classes]
    ax2 = ax.twinx()
    ax2.bar(range(len(classes)), part, color="lightgrey", alpha=0.6)
    ax2.set_ylabel("part de la population")
    for p in sorted(d["periode"].unique().to_list()):
        t = d.filter(pl.col("periode") == p).group_by("classe").agg(pl.col("y").mean())
        taux = [t.filter(pl.col("classe") == c)["y"].first() for c in classes]
        ax.plot(range(len(classes)), taux, marker="o", label=str(p))
    ax.set_zorder(ax2.get_zorder() + 1); ax.patch.set_visible(False)
    ax.set_xticks(range(len(classes)), classes, rotation=30, ha="right")
    ax.set_ylabel("taux de défaut"); ax.set_title(f"{var} ({regle.get('methode', '')})"); ax.legend(fontsize=8)
    return ax

def classes_regles(df: pl.DataFrame, regles: dict[str, dict]) -> pl.DataFrame:
    """Une colonne par variable : libellé de sa classe selon la règle retenue."""
    return pl.DataFrame([appliquer_regle(df, v, r)["classe"].alias(v) for v, r in regles.items()])


def _v_cramer(a: pl.Series, b: pl.Series) -> float:
    """V de Cramér entre deux variables qualitatives (3.2)."""
    from scipy.stats import chi2_contingency

    t = pl.DataFrame({"a": a, "b": b}).group_by("a", "b").len().pivot(on="b", index="a", values="len").fill_null(0)
    m = t.drop("a").to_numpy()
    if min(m.shape) < 2:
        return 0.0
    chi2 = chi2_contingency(m, correction=False)[0]
    return float(np.sqrt(chi2 / (m.sum() * (min(m.shape) - 1))))


def redondances_classes(df: pl.DataFrame, regles: dict[str, dict], seuil: float = 0.7) -> pl.DataFrame:
    """Paires de variables découpées dont le V de Cramér dépasse le seuil, triées par V décroissant."""
    c = classes_regles(df, regles)
    variables = [v for v in c.columns if c[v].n_unique() > 1]
    lignes = [{"var1": a, "var2": b, "v_cramer": _v_cramer(c[a], c[b])}
              for i, a in enumerate(variables) for b in variables[i + 1:]]
    return pl.DataFrame(lignes, schema={"var1": pl.String, "var2": pl.String, "v_cramer": pl.Float64}).filter(
        pl.col("v_cramer") >= seuil).sort("v_cramer", descending=True)


def selection_non_redondante(synthese: pl.DataFrame, paires: pl.DataFrame, iv_min: float = 0.02) -> pl.DataFrame:
    """Statut de chaque variable après binning : parcours par IV test décroissante, une variable est écartée
    si elle est trop liée (V de Cramér) à une variable déjà retenue, qui devient sa représentante."""
    voisins = {}
    for a, b in zip(paires["var1"], paires["var2"]):
        voisins.setdefault(a, set()).add(b)
        voisins.setdefault(b, set()).add(a)
    retenues, redondante = [], {}
    for v in synthese.filter(pl.col("iv_test") >= iv_min).sort("iv_test", descending=True)["variable"]:
        representante = next((k for k in retenues if k in voisins.get(v, ())), None)
        if representante is None:
            retenues.append(v)
        else:
            redondante[v] = representante
    return synthese.select("variable", "methode", "n_classes", "iv_test", "gini_test").with_columns(
        redondante_avec=pl.col("variable").replace_strict(redondante, default=None, return_dtype=pl.String)
    ).with_columns(
        statut=pl.when(pl.col("iv_test") < iv_min).then(pl.lit("écartée : non discriminante après binning"))
        .when(pl.col("redondante_avec").is_not_null()).then(pl.lit("écartée : redondante"))
        .otherwise(pl.lit("retenue"))
    ).sort("iv_test", descending=True)


def woe_encode_regles(df: pl.DataFrame, regles: dict[str, dict], target: str, ref: pl.DataFrame | None = None) -> pl.DataFrame:
    """Remplace chaque variable par la WOE de sa classe ; WOE calculées sur ref (le train) si fourni."""
    ref = df if ref is None else ref
    cols = []
    for v, r in regles.items():
        woe = table_regle(ref, v, target, r).select("classe", "woe")
        cols.append(appliquer_regle(df, v, r).join(woe, on="classe", how="left", maintain_order="left")["woe"].alias(v))
    return pl.DataFrame(cols)


def sauver_regles(regles: dict[str, dict], chemin) -> None:
    from pathlib import Path

    Path(chemin).parent.mkdir(parents=True, exist_ok=True)
    Path(chemin).write_text(json.dumps(regles, indent=2, ensure_ascii=False))


def charger_regles(chemin) -> dict[str, dict]:
    from pathlib import Path

    return json.loads(Path(chemin).read_text())

def _ecarts(taux: np.ndarray, sens) -> np.ndarray:
    """Écart relatif de risque entre classes adjacentes (lignes), signé : négatif = inversion du sens attendu.
    sens : +1 / -1 pour toutes les paires, ou un sens par paire (relation non monotone)."""
    sens = np.reshape(sens, (-1,) + (1,) * (taux.ndim - 1))
    return sens * (taux[1:] - taux[:-1]) / np.minimum(taux[1:], taux[:-1])

