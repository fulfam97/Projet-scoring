"""Constats rédigés automatiquement à partir des tableaux produits dans les notebooks (ch. 2 à 4).

Chaque fonction prend les tableaux déjà calculés dans le notebook et renvoie un texte Markdown : les chiffres
et listes de variables sont remplis par le code, l'interprétation métier reste à rédiger dans les trames
(passages entre crochets [ … ]). Afficher le résultat avec `afficher(texte)`.
"""
from __future__ import annotations

import polars as pl


# ---------------------------------------------------------------------------
# Outils de mise en forme
# ---------------------------------------------------------------------------

def afficher(texte: str) -> None:
    """Affiche un texte Markdown dans le notebook (ou l'imprime hors notebook)."""
    try:
        from IPython.display import Markdown, display
        display(Markdown(texte))
    except ImportError:
        print(texte)


def _liste(variables, max_n: int = 8) -> str:
    """`a`, `b`, `c` … (+n autres) ; « aucune » si vide."""
    variables = list(variables)
    if not variables:
        return "aucune"
    texte = ", ".join(f"`{v}`" for v in variables[:max_n])
    return texte + (f" … (+{len(variables) - max_n} autres)" if len(variables) > max_n else "")


def _pct(x, decimales: int = 1) -> str:
    return "—" if x is None else f"{x:.{decimales}%}".replace(".", ",")


def _num(x, decimales: int = 2) -> str:
    return "—" if x is None else f"{x:.{decimales}f}".replace(".", ",")


def _table(lignes: list[tuple], entete: tuple) -> str:
    sep = "|" + "|".join("---" for _ in entete) + "|"
    corps = ["| " + " | ".join(str(c) for c in l) + " |" for l in lignes]
    return "\n".join(["| " + " | ".join(entete) + " |", sep, *corps])


# ---------------------------------------------------------------------------
# Chapitre 2 — Préparation des données (notebook 0c)
# ---------------------------------------------------------------------------

def constat_manquants(tab_missing: pl.DataFrame, alpha: float = 0.05, seuil_quasi_vide: float = 0.95,
                      top: int = 5, n_min: int = 30) -> str:
    """Constat 2.2 : volume des manquants, variables quasi vides, manquants informatifs (sortie de table_missing)."""
    avec = tab_missing.filter(pl.col("n_missing") > 0)
    vides = tab_missing.filter(pl.col("taux_missing") >= seuil_quasi_vide)["variable"].to_list()
    info = (tab_missing.filter((pl.col("pvalue") < alpha) & (pl.col("n_missing") >= n_min))
            .sort(pl.col("ecart_tdr").abs(), descending=True))
    plus_risque = info.filter(pl.col("ecart_tdr") > 0).height

    lignes = [
        f"- **{avec.height}** variables sur {tab_missing.height} comportent au moins un manquant"
        + (f" ; taux de manquants médian parmi elles : **{_pct(avec['taux_missing'].median())}**." if avec.height else "."),
        f"- **{len(vides)}** variables ont au moins {_pct(seuil_quasi_vide, 0)} de manquants : {_liste(vides)}."
        " Elles sont retirées à la présélection.",
        f"- **{info.height}** variables ont un manquant lié au défaut (Chi², p < {_num(alpha)}) : chez "
        f"**{plus_risque}** d'entre elles les dossiers manquants sont *plus* risqués que les renseignés, chez "
        f"**{info.height - plus_risque}** ils sont *moins* risqués.",
    ]
    if info.height:
        lignes.append("\nManquants les plus informatifs :\n")
        lignes.append(_table(
            [(f"`{r['variable']}`", _pct(r["taux_missing"]), _pct(r["tdr_missing"], 2), _pct(r["tdr_renseigne"], 2))
             for r in info.head(top).iter_rows(named=True)],
            ("Variable", "Taux de manquants", "TDR manquants", "TDR renseignés"),
        ))
    return "\n".join(lignes)


def constat_outliers(rapport_outliers: pl.DataFrame, bornes: pl.DataFrame, k: float = 3, top: int = 5) -> str:
    """Constat 2.3 : valeurs extrêmes détectées (outlier_report) et winsorisation appliquée (winsoriser)."""
    touchees = rapport_outliers.filter(pl.col("n_outliers") > 0).sort("taux_outliers", descending=True)
    wins = bornes.filter(pl.col("winsorisee"))
    non_modif = bornes.height - wins.height
    lignes = [
        f"- Règle IQR (k = {_num(k, 1)}) : **{touchees.height}** variables continues sur {rapport_outliers.height} "
        "présentent des valeurs extrêmes"
        + (f" ; les plus touchées : {_liste(touchees['variable'].head(top))}"
           f" (jusqu'à {_pct(touchees['taux_outliers'].max())} des valeurs renseignées)." if touchees.height else "."),
        f"- Winsorisation : **{wins.height}** variables écrêtées, part médiane des valeurs ramenées à la borne : "
        f"**{_pct(wins['taux_ecrete'].median() if wins.height else None)}**. "
        f"**{non_modif}** variables non modifiées (bornes confondues, typiquement un comptage presque toujours nul).",
    ]
    return "\n".join(lignes)


def constat_preselection(preselection: pl.DataFrame, top: int = 10) -> str:
    """Constat de la présélection : effectif par motif, redondances (variable écartée → représentante), alertes."""
    compte = preselection.group_by("statut").len().sort("len", descending=True)
    red = preselection.filter(pl.col("statut") == "écartée : redondante").sort("iv", descending=True)
    alerte = preselection.filter(pl.col("statut").str.contains("fuite"))["variable"].to_list()
    rare = preselection.filter(pl.col("statut").str.contains("rare"))["variable"].to_list()

    lignes = [_table([(r["statut"], r["len"]) for r in compte.iter_rows(named=True)], ("Statut", "Variables")), ""]
    if red.height:
        lignes.append(f"**Redondances** ({red.height}) : variable écartée → variable qui la représente.\n")
        lignes.append(_table(
            [(f"`{r['variable']}`", _num(r["iv"], 3), f"`{r['redondante_avec']}`")
             for r in red.head(top).iter_rows(named=True)],
            ("Écartée", "IV", "Représentée par"),
        ))
        if red.height > top:
            lignes.append(f"\n… et {red.height - top} autres (voir le tableau ci-dessus).")
    lignes.append(f"\n- Alertes IV > 0,5 (soupçon de fuite) : {_liste(alerte)}.")
    if rare:
        lignes.append(f"- Valeurs rares gardées malgré une quasi-constance : {_liste(rare)}.")
    return "\n".join(lignes)


# ---------------------------------------------------------------------------
# Chapitre 3 — Analyse exploratoire (notebook 0d)
# ---------------------------------------------------------------------------

def lecture_acp(variance: pl.DataFrame, correlations: pl.DataFrame, axes: tuple[str, ...] = ("F1", "F2"),
                seuil: float = 0.5, top: int = 6) -> str:
    """Lecture des premiers axes de l'ACP : part de variance et variables les plus corrélées à chaque axe."""
    part = dict(zip(variance["axe"], variance["part_variance"]))
    cumul = dict(zip(variance["axe"], variance["part_cumulee"]))
    lignes = []
    for a in axes:
        if a not in correlations.columns:
            continue
        c = correlations.select("variable", a).sort(pl.col(a).abs(), descending=True)
        pos = c.filter(pl.col(a) >= seuil)["variable"].head(top).to_list()
        neg = c.filter(pl.col(a) <= -seuil)["variable"].head(top).to_list()
        lignes.append(f"- **{a}** ({_pct(part.get(a))} de la variance) — pôle positif : {_liste(pos)} ;"
                      f" pôle négatif : {_liste(neg)} (|r| ≥ {_num(seuil, 1)}).")
    dernier = axes[-1]
    lignes.append(f"- Les axes {' et '.join(axes)} résument **{_pct(cumul.get(dernier))}** de la variance.")
    return "\n".join(lignes)


def _n_eta2_fort(eta2_croise: pl.DataFrame | None, seuil: float) -> int:
    if eta2_croise is None or eta2_croise.width < 2:
        return 0
    return (eta2_croise.unpivot(index="variable", variable_name="qualitative").filter(pl.col("value") >= seuil).height)


def synthese_exploration(paires_quanti: pl.DataFrame, paires_quali: pl.DataFrame, vif: pl.DataFrame,
                         non_monotones: pl.DataFrame, synthese: pl.DataFrame,
                         eta2_croise: pl.DataFrame | None = None, seuil_corr: float = 0.8,
                         seuil_cramer: float = 0.8, seuil_eta2: float = 0.5, seuil_vif: float = 5) -> str:
    """Tableau « Synthèse des décisions » du ch. 3, colonne Constat remplie à partir des résultats."""
    statuts = synthese.group_by("statut").len()
    n = dict(zip(statuts["statut"], statuts["len"]))
    candidates = sum(v for s, v in n.items() if s.startswith("candidate"))
    red = n.get("écartée : redondante", 0)
    autres = sum(v for s, v in n.items() if s.startswith("écartée")) - red
    lignes = [
        ("3.1 Quanti / quanti", f"{paires_quanti.height} paires avec \\|Spearman\\| ≥ {_num(seuil_corr, 1)}",
         "une seule variable conservée par paire (meilleur IV)"),
        ("3.2 Quali / quali", f"{paires_quali.height} paires avec V de Cramer ≥ {_num(seuil_cramer, 1)}",
         "une seule variable conservée par paire (meilleur IV)"),
        ("3.3 Quanti / quali", f"{_n_eta2_fort(eta2_croise, seuil_eta2)} couples avec eta² ≥ {_num(seuil_eta2, 1)}",
         "diagnostic : redondances croisées signalées, sans exclusion automatique"),
        ("3.4 Multicolinéarité", f"{vif.filter(pl.col('VIF') > seuil_vif).height} variables avec VIF > {seuil_vif}, "
         f"{vif.filter(pl.col('VIF') > 10).height} avec VIF > 10", "VIF reporté dans la fiche ; traitement au ch. 5"),
        ("3.5 Non monotones", f"{non_monotones.height} variables discriminantes non monotones : "
         f"{_liste(non_monotones['variable'], 5)}", "forme à confirmer ou corriger au regroupement de classes (ch. 4.4)"),
        ("3.6 Présélection", f"{candidates} candidates ; {red} écartées pour redondance, {autres} pour un autre motif",
         "variables candidates transmises à la discrétisation"),
    ]
    return _table(lignes, ("§", "Constat", "Décision"))


# ---------------------------------------------------------------------------
# Chapitre 4 — Discrétisation (notebook 1a)
# ---------------------------------------------------------------------------

def _col(df: pl.DataFrame, prefixe: str, methode: str) -> str | None:
    return next((c for c in df.columns if c.startswith(prefixe) and c.endswith(methode)), None)


def constat_methodes(comparaison: pl.DataFrame, tolerance: float = 0.10) -> str:
    """Constat 4.2 : comparaison quantiles / arbre univarié (sortie pivotée : n_classes_*, iv_*)."""
    iv_q, iv_a = _col(comparaison, "iv", "quantiles"), _col(comparaison, "iv", "arbre")
    nc_q, nc_a = _col(comparaison, "n_classes", "quantiles"), _col(comparaison, "n_classes", "arbre")
    if None in (iv_q, iv_a, nc_q, nc_a):
        return "_Tableau de comparaison non reconnu._"
    d = comparaison.filter(pl.col(iv_q) > 0).with_columns(ecart=(pl.col(iv_a) - pl.col(iv_q)) / pl.col(iv_q))
    proches = d.filter(pl.col("ecart").abs() <= tolerance).height
    meilleur_arbre = d.filter(pl.col("ecart") > tolerance)["variable"].to_list()
    meme_nb = d.filter(pl.col(nc_q) == pl.col(nc_a)).height
    return "\n".join([
        f"- Sur {d.height} variables : IV final à ±{_pct(tolerance, 0)} près pour **{proches}** ; "
        f"même nombre de classes pour **{meme_nb}**.",
        f"- Écart d'IV médian (arbre − quantiles, en relatif) : **{_pct(d['ecart'].median())}**.",
        f"- Variables où l'arbre fait nettement mieux : {_liste(meilleur_arbre)}.",
    ])


def constat_decoupages(bilan: pl.DataFrame, top: int = 3) -> str:
    """Constat 4.3–4.5 : classes, perte d'IV, exemptions de monotonie, missing isolé (sortie `bilan` de discretiser)."""
    pertes = bilan.sort("perte_iv", descending=True, nulls_last=True).head(top)
    exemptees = bilan.filter((pl.col("nature") == "numerique") & (pl.col("sens") == 0))["variable"].to_list()
    miss = bilan.filter(pl.col("missing_isole"))["variable"].to_list()
    return "\n".join([
        f"- **{bilan.height}** variables découpées, **{_num(bilan['n_classes'].mean(), 1)}** classes en moyenne "
        f"(de {bilan['n_classes'].min()} à {bilan['n_classes'].max()}).",
        f"- Perte d'IV médiane entre découpage fin et final : **{_pct(bilan['perte_iv'].median())}**. Plus fortes pertes : "
        + ", ".join(f"`{r['variable']}` ({_pct(r['perte_iv'])})" for r in pertes.iter_rows(named=True)) + ".",
        f"- Variables numériques sans monotonie imposée (sens = 0) : {_liste(exemptees)}.",
        f"- Missing gardé comme classe à part : **{len(miss)}** variables ({_liste(miss)}).",
    ])


def constat_stabilite(stab: pl.DataFrame, n_cohortes: int) -> str:
    """Constat 4.6 : variables instables en risque ou en volume (sortie `stab` de stabilite_decoupages)."""
    risque = stab.filter(~pl.col("stable_risque"))["variable"].to_list()
    volume = stab.filter(~pl.col("stable_volume"))["variable"].to_list()
    lignes = [
        f"- Mesurée sur **{n_cohortes}** cohortes.",
        f"- Instables en risque (écart moyen < 30 % pour au moins une paire de classes) : **{len(risque)}** — {_liste(risque)}.",
        f"- Instables en volume (PSI > seuil) : **{len(volume)}** — {_liste(volume)}.",
        f"- Part médiane des cohortes où l'ordre des classes est intégralement respecté : "
        f"**{_pct(stab['part_groupes_ordonnes'].median(), 0)}**.",
    ]
    if n_cohortes < 2:
        lignes.append("- ⚠️ Une seule cohorte : la stabilité dans le temps n'est pas mesurable.")
    return "\n".join(lignes)


def synthese_discretisation(comparaison: pl.DataFrame, bilan: pl.DataFrame, fiche: pl.DataFrame) -> str:
    """Tableau « Synthèse des décisions » du ch. 4, colonne Constat remplie à partir des résultats."""
    iv_q, iv_a = _col(comparaison, "iv", "quantiles"), _col(comparaison, "iv", "arbre")
    ecart = (comparaison.filter(pl.col(iv_q) > 0).select(((pl.col(iv_a) - pl.col(iv_q)) / pl.col(iv_q)).median()).item()
             if iv_q and iv_a else None)
    exemptees = bilan.filter((pl.col("nature") == "numerique") & (pl.col("sens") == 0))["variable"].to_list()
    n_miss = bilan.filter(pl.col("missing_isole")).height
    a_revoir = fiche.filter(pl.col("statut").str.starts_with("à revoir")).height
    ecartees = fiche.filter(pl.col("statut").str.starts_with("écartée")).height
    retenues = fiche.filter(pl.col("statut") == "retenue").height
    lignes = [
        ("4.2 Méthode", f"écart d'IV médian arbre / quantiles : {_pct(ecart)}", "quantiles fins + fusion ChiMerge sous contraintes"),
        ("4.3 WoE / IV", f"perte d'IV médiane de {_pct(bilan['perte_iv'].median())} pour "
         f"{_num(bilan['n_classes'].mean(), 1)} classes en moyenne", "perte acceptée au titre de la parcimonie"),
        ("4.4 Monotonie", f"{len(exemptees)} exemptions : {_liste(exemptees, 5)}",
         "monotonie imposée partout" if not exemptees else "exemptions justifiées métier (0d, § 3.5)"),
        ("4.5 Missing", f"isolé pour {n_miss} variables sur {bilan.height}", "fusionné avec la classe de risque la plus proche ailleurs"),
        ("4.6 Stabilité", f"{a_revoir} variables à revoir", "variables à revoir non transmises au ch. 5"),
        ("4.7 Bilan", f"{retenues} retenues, {a_revoir} à revoir, {ecartees} écartées", "variables retenues transmises au ch. 5"),
    ]
    return _table(lignes, ("§", "Constat", "Décision"))
