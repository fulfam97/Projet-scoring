from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import polars as pl
from scipy import stats

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

    
def comparer_woe_groupes(df: pl.DataFrame, variables: list[str], target: str, groupe: str,
                         n_bins: int = 10, max_modalites: int = 10, min_n: int = 500) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Mêmes classes pour tous (découpage fin sur l'ensemble), WOE calculée séparément dans chaque groupe.
    Une variable est bien alignée si ses WOE sont proches d'un groupe à l'autre.
    Renvoie (detail par classe et groupe, synthèse par variable : écart max de WOE sur les classes d'au moins min_n dossiers)."""
    details = []
    for v in variables:
        d = _classes(df, v, n_bins, max_modalites).with_columns(df[target].alias("y"), df[groupe].cast(pl.String).alias("groupe"))
        t = (
            d.group_by("groupe", "classe")
            .agg(ordre=pl.col("ordre").min(), n=pl.len(), n_defaut=pl.col("y").sum())
            .with_columns(n_sain=pl.col("n") - pl.col("n_defaut"))
            .with_columns(
                dist_bad=(pl.col("n_defaut") + 0.5) / (pl.col("n_defaut").sum().over("groupe") + 0.5),
                dist_good=(pl.col("n_sain") + 0.5) / (pl.col("n_sain").sum().over("groupe") + 0.5),
            )
            .with_columns(woe=(pl.col("dist_good") / pl.col("dist_bad")).log(),
                          taux_defaut=pl.col("n_defaut") / pl.col("n"))
            .select(pl.lit(v).alias("variable"), "groupe", "classe", "ordre", "n", "n_defaut", "taux_defaut", "woe")
        )
        details.append(t)
    detail = pl.concat(details).sort("variable", "ordre", "groupe", nulls_last=True)

    synthese = (
        detail.filter(pl.col("n") >= min_n)
        .group_by("variable", "classe")
        .agg(n_groupes=pl.len(), ecart_woe=pl.col("woe").max() - pl.col("woe").min())
        .filter(pl.col("n_groupes") > 1)
        .group_by("variable")
        .agg(ecart_woe_max=pl.col("ecart_woe").max(), ecart_woe_moyen=pl.col("ecart_woe").mean(),
             n_classes_comparees=pl.len())
        .sort("ecart_woe_max", descending=True)
    )
    return detail, synthese


# ---------------------------------------------------------------------------
# Chapitre 4 : du découpage fin au découpage final
# ---------------------------------------------------------------------------

def bornes_fines(x: pl.Series, y: pl.Series, methode: str = "quantiles", n_bins: int = 20,
                 max_modalites: int = 15, min_part: float = 0.02) -> list[float]:
    """Bornes intérieures du découpage fin d'une variable numérique : classes ]b(k-1) ; b(k)], manquants exclus.
    Variable à au plus max_modalites valeurs distinctes : une classe par valeur. Sinon quantiles (n_bins classes)
    ou arbre univarié (n_bins feuilles au plus, chacune d'au moins min_part des dossiers renseignés)."""
    d = pl.DataFrame({"x": x.cast(pl.Float64), "y": y}).drop_nulls("x")
    valeurs = d["x"].unique().sort().to_numpy()
    if len(valeurs) <= max_modalites:
        return valeurs[:-1].tolist()
    if methode == "quantiles":
        # bornes prises parmi les valeurs observées : aucune classe fine n'est vide
        q = np.unique(np.quantile(d["x"].to_numpy(), np.linspace(0, 1, n_bins + 1)[1:-1], method="higher"))
        return q[q < valeurs[-1]].tolist()
    if methode == "arbre":
        from sklearn.tree import DecisionTreeClassifier
        arbre = DecisionTreeClassifier(max_leaf_nodes=n_bins, min_samples_leaf=min_part, random_state=0)
        arbre.fit(d["x"].to_numpy().reshape(-1, 1), d["y"].to_numpy())
        t = arbre.tree_
        return sorted(t.threshold[t.feature >= 0].tolist())
    raise ValueError(f"méthode inconnue : {methode}")


def _tdr(g: dict) -> float:
    return g["n_defaut"] / g["n"]


def _ecart_relatif(a: dict, b: dict) -> float:
    """Écart relatif de taux de défaut |ta - tb| / min(ta, tb) (règle des 30 %, cours 4.6)."""
    ta, tb = _tdr(a), _tdr(b)
    if min(ta, tb) == 0:
        return 0.0 if ta == tb else float("inf")
    return abs(ta - tb) / min(ta, tb)


def _chi2(a: dict, b: dict) -> tuple[float, float]:
    """Chi² d'indépendance du tableau 2 x 2 (classe x cible) de deux classes."""
    tab = np.array([[a["n_defaut"], a["n"] - a["n_defaut"]], [b["n_defaut"], b["n"] - b["n_defaut"]]])
    if (tab.sum(axis=0) == 0).any():
        return 0.0, 1.0
    chi2, p = stats.chi2_contingency(tab, correction=False)[:2]
    return float(chi2), float(p)


def _unir(a: dict, b: dict) -> dict:
    return {"atomes": a["atomes"] + b["atomes"], "n": a["n"] + b["n"], "n_defaut": a["n_defaut"] + b["n_defaut"]}


def _fusionner_chaine(chaine: list[dict], ordonnee: bool, sens: int, n_min: int, min_defauts: int,
                      seuil_ecart: float, alpha: float | None, max_classes: int | None) -> list[dict]:
    """Fusion progressive de classes adjacentes (cours 4.7), une fusion à la fois, par ordre de priorité :
    1. effectif insuffisant : la plus petite classe rejoint son voisin de taux de défaut le plus proche ;
    2. rupture de monotonie (si sens != 0) ;
    3. écart relatif de risque < seuil_ecart, ou différence non significative au Chi² (ChiMerge) ;
    4. nombre de classes > max_classes.
    En 2 à 4, la paire fusionnée est celle de plus faible Chi² (la moins distincte).
    Variable non ordonnée (qualitative) : les classes sont triées par taux de défaut après chaque fusion."""
    chaine = list(chaine)
    while len(chaine) > 1:
        if not ordonnee:
            chaine.sort(key=_tdr)
        petites = [i for i, g in enumerate(chaine) if g["n"] < n_min or g["n_defaut"] < min_defauts]
        if petites:
            i = min(petites, key=lambda k: chaine[k]["n"])
            j = min((k for k in (i - 1, i + 1) if 0 <= k < len(chaine)),
                    key=lambda k: abs(_tdr(chaine[k]) - _tdr(chaine[i])))
        else:
            paires = list(range(len(chaine) - 1))
            chi2 = {k: _chi2(chaine[k], chaine[k + 1]) for k in paires}
            rompues = [k for k in paires if sens and sens * (_tdr(chaine[k + 1]) - _tdr(chaine[k])) <= 0]
            faibles = [k for k in paires if _ecart_relatif(chaine[k], chaine[k + 1]) < seuil_ecart
                       or (alpha is not None and chi2[k][1] > alpha)]
            trop = paires if max_classes and len(chaine) > max_classes else []
            candidates = rompues or faibles or trop
            if not candidates:
                break
            i = min(candidates, key=lambda k: chi2[k][0])
            j = i + 1
        i, j = min(i, j), max(i, j)
        chaine[i:j + 1] = [_unir(chaine[i], chaine[j])]
    return chaine


def decouper_variable(df: pl.DataFrame, var: str, target: str, methode: str = "quantiles", n_bins: int = 20,
                      max_modalites: int = 15, min_part: float = 0.05, min_defauts: int = 50,
                      seuil_ecart: float = 0.30, alpha: float | None = 0.05, max_classes: int | None = None,
                      sens: int | None = None) -> dict:
    """Découpage final d'une variable, du découpage fin à la fusion progressive (cours 4.7).
    Numérique : fusion d'intervalles adjacents, sens de la monotonie imposé = signe du Spearman avec la cible
    (±1 pour le forcer). Avec sens=0 (relation non monotone justifiée métier, cours 4.4), les intervalles
    adjacents de même risque sont d'abord fusionnés, puis les classes consécutives en risque (deux queues d'un U
    peuvent alors former une seule classe).
    Qualitative : modalités ordonnées par taux de défaut puis regroupées (monotone par construction).
    Le missing est une classe à part ; il n'est fusionné, avec la classe de taux le plus proche, que s'il est
    trop petit ou s'il ne s'en distingue pas (écart < seuil_ecart ou Chi² non significatif) — cours 4.5.
    Classes numérotées de 1 (moins risquée) à N (plus risquée), missing compris (cours 4.6).
    Une valeur non vue en apprentissage (manquant absent, modalité nouvelle) va dans la classe la plus risquée."""
    numerique = df.schema[var].is_numeric()
    n_min = int(np.ceil(min_part * df.height))

    if numerique:
        bornes = bornes_fines(df[var], df[target], methode, n_bins, max_modalites)
        atome = pl.col(var).cut(bornes, labels=[str(k) for k in range(len(bornes) + 1)]).cast(pl.String)
        if sens is None:
            d = df.select(var, target).drop_nulls()
            ok = d.height > 2 and d[var].n_unique() > 1 and d[target].n_unique() > 1
            sens = int(np.sign(stats.spearmanr(d[var].to_numpy(), d[target].to_numpy())[0])) if ok else 0
    else:
        atome = pl.col(var).cast(pl.String)
        sens = 1
    stats_atomes = (
        df.select(atome.fill_null("missing").alias("atome"), pl.col(target).alias("y"))
        .group_by("atome").agg(n=pl.len(), n_defaut=pl.col("y").sum())
    )
    groupes = {r["atome"]: {"atomes": [r["atome"]], "n": r["n"], "n_defaut": r["n_defaut"]}
               for r in stats_atomes.iter_rows(named=True)}
    manquant = groupes.pop("missing", None)
    chaine = (sorted(groupes.values(), key=lambda g: int(g["atomes"][0])) if numerique
              else sorted(groupes.values(), key=_tdr))
    n_classes_fin = len(chaine) + (manquant is not None)
    regles = dict(n_min=n_min, min_defauts=min_defauts, seuil_ecart=seuil_ecart, alpha=alpha)

    chaine = _fusionner_chaine(chaine, ordonnee=numerique, sens=sens,
                               max_classes=None if numerique and sens == 0 else max_classes, **regles)
    if numerique and sens == 0:
        chaine = _fusionner_chaine(chaine, ordonnee=False, sens=1, max_classes=max_classes, **regles)
    missing_isole = manquant is not None
    if manquant is not None and chaine:
        k = min(range(len(chaine)), key=lambda i: abs(_tdr(chaine[i]) - _tdr(manquant)))
        trop_petit = manquant["n"] < n_min or manquant["n_defaut"] < min_defauts
        proche = _ecart_relatif(manquant, chaine[k]) < seuil_ecart or (
            alpha is not None and _chi2(manquant, chaine[k])[1] > alpha)
        if trop_petit or proche:
            chaine[k] = _unir(chaine[k], manquant)
            missing_isole = False
    finales = sorted(chaine + ([manquant] if missing_isole else []), key=_tdr)

    classe = {a: c for c, g in enumerate(finales, start=1) for a in g["atomes"]}
    plus_risquee = len(finales)
    decoupage = {"variable": var, "nature": "numerique" if numerique else "qualitative", "sens": sens,
                 "n_classes_fin": n_classes_fin, "missing_isole": missing_isole,
                 "classe_missing": classe.pop("missing", plus_risquee)}
    if numerique:
        # Intervalles fins consécutifs de même classe réunis ; un intervalle vide prend la classe du précédent
        par_atome, courant = [], None
        for k in range(len(bornes) + 1):
            courant = classe.get(str(k), courant)
            par_atome.append(courant)
        premier = next((c for c in par_atome if c is not None), decoupage["classe_missing"])
        par_atome = [premier if c is None else c for c in par_atome]
        ruptures = [k for k in range(1, len(par_atome)) if par_atome[k] != par_atome[k - 1]]
        decoupage["bornes"] = [bornes[k - 1] for k in ruptures]
        decoupage["classes"] = [par_atome[0]] + [par_atome[k] for k in ruptures]
    else:
        decoupage["modalites"] = classe | {"missing": decoupage["classe_missing"]}
        decoupage["classe_inconnue"] = plus_risquee
    return decoupage


def appliquer_decoupages(df: pl.DataFrame, decoupages: dict[str, dict]) -> pl.DataFrame:
    """Remplace chaque variable découpée par son numéro de classe (1 = moins risquée). Les bornes extrêmes
    étant -inf et +inf, toute valeur, observée ou non en apprentissage, reçoit une classe."""
    exprs = []
    for v, d in decoupages.items():
        if d["nature"] == "numerique":
            intervalle = pl.col(v).cut(d["bornes"], labels=[str(k) for k in range(len(d["bornes"]) + 1)])
            e = (intervalle.cast(pl.String)
                 .replace_strict({str(k): c for k, c in enumerate(d["classes"])}, return_dtype=pl.Int32)
                 .fill_null(d["classe_missing"]))
        else:
            e = (pl.col(v).cast(pl.String).fill_null("missing")
                 .replace_strict(d["modalites"], default=d["classe_inconnue"], return_dtype=pl.Int32))
        exprs.append(e.alias(v))
    return df.with_columns(exprs)


def libelles_classes(decoupage: dict, max_car: int = 60) -> dict[int, str]:
    """Libellé métier de chaque classe : intervalle(s) pour une numérique, liste des modalités sinon."""
    morceaux: dict[int, list[str]] = {}
    if decoupage["nature"] == "numerique":
        b = [-np.inf, *decoupage["bornes"], np.inf]
        for k, c in enumerate(decoupage["classes"]):
            bas, haut = ("-inf" if b[k] == -np.inf else f"{b[k]:.4g}"), ("+inf" if b[k + 1] == np.inf else f"{b[k + 1]:.4g}")
            morceaux.setdefault(c, []).append(f"]{bas} ; {haut}]")
        morceaux.setdefault(decoupage["classe_missing"], []).append("manquant")
    else:
        for m, c in sorted(decoupage["modalites"].items()):
            morceaux.setdefault(c, []).append(m)
    out = {}
    for c, m in morceaux.items():
        texte = " ou ".join(m) if decoupage["nature"] == "numerique" else ", ".join(m)
        out[c] = texte if len(texte) <= max_car else texte[:max_car - 1] + "…"
    return out


def table_decoupage(df_classes: pl.DataFrame, decoupage: dict, target: str) -> pl.DataFrame:
    """Table WOE/IV du découpage final (cours 4.3) : une ligne par classe, avec son libellé et l'écart relatif
    de taux de défaut avec la classe précédente (règle des 30 %, cours 4.6)."""
    v = decoupage["variable"]
    lib = libelles_classes(decoupage)
    return (
        woe_iv_table(df_classes, v, target, max_modalites=10 ** 6)
        .with_columns(classe=pl.col("classe").cast(pl.Int32))
        .with_columns(libelle=pl.col("classe").replace_strict(lib, default="", return_dtype=pl.String),
                      ecart_relatif=pl.col("taux_defaut") / pl.col("taux_defaut").shift(1) - 1)
        .select(pl.lit(v).alias("variable"), "classe", "libelle", "n", "n_defaut", "taux_defaut",
                "part_population", "woe", "iv", "ecart_relatif")
    )


def discretiser(df: pl.DataFrame, variables: list[str], target: str, sens: dict[str, int] | None = None,
                **regles) -> tuple[dict[str, dict], pl.DataFrame, pl.DataFrame]:
    """Découpage final de chaque variable (paramètres de decouper_variable dans `regles` ; `sens` force le sens
    de monotonie de certaines variables, 0 = relation non monotone acceptée).
    Renvoie (découpages, tables WOE/IV par classe, bilan par variable : nombre de classes et IV avant / après
    regroupement)."""
    sens = sens or {}
    max_modalites = regles.get("max_modalites", 15)
    decoupages = {v: decouper_variable(df, v, target, sens=sens.get(v), **regles) for v in variables}
    df_cl = appliquer_decoupages(df, decoupages)
    tables = pl.concat([table_decoupage(df_cl, decoupages[v], target) for v in variables])
    iv_fin = {v: woe_iv_table(df, v, target, regles.get("n_bins", 20), max_modalites)["iv"].sum() for v in variables}
    bilan = (
        tables.group_by("variable", maintain_order=True)
        .agg(n_classes=pl.len(), iv=pl.col("iv").sum(), part_min=pl.col("part_population").min(),
             defauts_min=pl.col("n_defaut").min(), ecart_relatif_min=pl.col("ecart_relatif").min())
        .with_columns(
            nature=pl.col("variable").replace_strict({v: d["nature"] for v, d in decoupages.items()}),
            sens=pl.col("variable").replace_strict({v: d["sens"] for v, d in decoupages.items()}),
            n_classes_fin=pl.col("variable").replace_strict({v: d["n_classes_fin"] for v, d in decoupages.items()}),
            iv_fin=pl.col("variable").replace_strict(iv_fin, return_dtype=pl.Float64),
            missing_isole=pl.col("variable").replace_strict({v: d["missing_isole"] for v, d in decoupages.items()}),
        )
        .with_columns(perte_iv=pl.when(pl.col("iv_fin") > 0).then(1 - pl.col("iv") / pl.col("iv_fin")))
        .select("variable", "nature", "sens", "n_classes_fin", "n_classes", "iv_fin", "iv", "perte_iv",
                "part_min", "defauts_min", "ecart_relatif_min", "missing_isole")
        .sort("iv", descending=True)
    )
    return decoupages, tables, bilan


def stabilite_decoupages(df_classes: pl.DataFrame, variables: list[str], target: str, groupe: str,
                         seuil_ecart: float = 0.30, seuil_psi: float = 0.10) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Stabilité du découpage dans le temps (cours 4.6), `groupe` étant la cohorte (ou l'échantillon) :
    - en risque : écart relatif de taux de défaut entre classes consécutives, moyenné sur les groupes, au moins
      égal à seuil_ecart pour chaque paire ; part des groupes où l'ordre des classes est intégralement respecté ;
    - en volume : PSI de la répartition des classes de chaque groupe par rapport à l'ensemble (formalisé en 9.3),
      au plus seuil_psi.
    Renvoie (détail par variable, groupe et classe ; synthèse par variable)."""
    details = []
    for v in variables:
        details.append(
            df_classes.group_by(pl.col(groupe).cast(pl.String).alias("groupe"), pl.col(v).alias("classe"))
            .agg(n=pl.len(), n_defaut=pl.col(target).sum())
            .with_columns(part=pl.col("n") / pl.col("n").sum().over("groupe"),
                          taux_defaut=pl.col("n_defaut") / pl.col("n"))
            .with_columns(part_ref=pl.col("n").sum().over("classe") / pl.col("n").sum())
            .sort("groupe", "classe")
            .with_columns(ecart_relatif=(pl.col("taux_defaut") / pl.col("taux_defaut").shift(1) - 1).over("groupe"))
            .select(pl.lit(v).alias("variable"), "groupe", "classe", "n", "n_defaut", "part", "part_ref",
                    "taux_defaut", "ecart_relatif")
        )
    detail = pl.concat(details)
    eps = 1e-6
    par_groupe = (
        detail.group_by("variable", "groupe")
        .agg(psi=((pl.col("part") - pl.col("part_ref")) * ((pl.col("part") + eps) / (pl.col("part_ref") + eps)).log()).sum(),
             ordre_respecte=(pl.col("ecart_relatif").drop_nulls() > 0).all())
    )
    synthese = (
        detail.filter(pl.col("ecart_relatif").is_not_null())
        .group_by("variable", "classe").agg(ecart_moyen=pl.col("ecart_relatif").mean())
        .group_by("variable").agg(ecart_moyen_min=pl.col("ecart_moyen").min())
        .join(par_groupe.group_by("variable").agg(part_groupes_ordonnes=pl.col("ordre_respecte").mean(),
                                                   psi_max=pl.col("psi").max()), on="variable", how="right")
        .with_columns(stable_risque=pl.col("ecart_moyen_min").fill_null(float("inf")) >= seuil_ecart,
                      stable_volume=pl.col("psi_max") <= seuil_psi)
        .select("variable", "ecart_moyen_min", "part_groupes_ordonnes", "psi_max", "stable_risque", "stable_volume")
        .sort("variable")
    )
    return detail, synthese


def plot_decoupages(tables: pl.DataFrame, variables: list[str], n_cols: int = 3,
                    figsize_par_plot: tuple[float, float] = (4.8, 3.4)) -> plt.Figure:
    """Pour chaque variable : part de population par classe (barres) et taux de défaut (ligne, axe de droite).
    Un découpage valide montre une ligne strictement croissante de la classe 1 à la classe N."""
    n_cols = max(1, min(n_cols, len(variables)))
    n_rows = int(np.ceil(len(variables) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize_par_plot[0] * n_cols, figsize_par_plot[1] * n_rows),
                             squeeze=False)
    for ax, v in zip(axes.flat, variables):
        t = tables.filter(pl.col("variable") == v).sort("classe")
        x = np.arange(t.height)
        ax.bar(x, t["part_population"].to_list(), color="#c7d4e8")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        ax.tick_params(axis="y", labelsize=7)
        ax2 = ax.twinx()
        ax2.plot(x, t["taux_defaut"].to_list(), color="#d62728", marker="o", linewidth=2, markersize=5)
        ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))
        ax2.tick_params(axis="y", labelsize=7, colors="#d62728")
        ax2.set_ylim(bottom=0)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{c} : {l[:16]}" for c, l in zip(t["classe"], t["libelle"])], fontsize=6,
                           rotation=30, ha="right")
        ax.set_title(f"{v}  (IV = {t['iv'].sum():.2f})", fontsize=9, fontweight="bold")
    for ax in axes.flat[len(variables):]:
        ax.set_visible(False)
    fig.supylabel("Part de population (barres) — taux de défaut (ligne)", fontsize=9)
    fig.tight_layout()
    return fig


def plot_stabilite(detail: pl.DataFrame, variables: list[str], n_cols: int = 3,
                   figsize_par_plot: tuple[float, float] = (4.6, 3.2)) -> plt.Figure:
    """Taux de défaut de chaque classe par groupe (cohorte) : des courbes qui se croisent signalent une
    inversion de l'ordre des classes, donc un découpage instable en risque."""
    n_cols = max(1, min(n_cols, len(variables)))
    n_rows = int(np.ceil(len(variables) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize_par_plot[0] * n_cols, figsize_par_plot[1] * n_rows),
                             squeeze=False)
    cmap = plt.get_cmap("coolwarm")   # bleu = classe la moins risquée, rouge = la plus risquée
    for ax, v in zip(axes.flat, variables):
        d = detail.filter(pl.col("variable") == v)
        groupes = sorted(d["groupe"].unique().to_list())
        n_cl = d["classe"].max()
        for c in range(1, n_cl + 1):
            dc = d.filter(pl.col("classe") == c)
            tdr = dict(zip(dc["groupe"], dc["taux_defaut"]))
            ax.plot(range(len(groupes)), [tdr.get(g) for g in groupes], marker="o", markersize=3,
                    color=cmap((c - 1) / max(1, n_cl - 1)), label=str(c))
        ax.set_xticks(range(len(groupes)))
        ax.set_xticklabels(groupes, fontsize=6, rotation=45, ha="right")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))
        ax.tick_params(axis="y", labelsize=7)
        ax.set_title(v, fontsize=9, fontweight="bold")
        ax.legend(title="classe", fontsize=6, title_fontsize=6, ncol=2)
    for ax in axes.flat[len(variables):]:
        ax.set_visible(False)
    fig.supylabel("Taux de défaut", fontsize=9)
    fig.tight_layout()
    return fig
