import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def test_interactions_segmentation(
    df: pl.DataFrame,
    seg_cols: list[str] | str,
    features: list[str],
    target: str,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Test formel (LR test et test de Wald joint) de significativité des interactions (x_j * 1_{segment=g})
    dans le modèle logistique pooled, pour un ou plusieurs critères de segmentation candidats."""
    from scipy.stats import chi2
    import statsmodels.api as sm
    pl.Config.set_tbl_rows(-1)
    pl.Config.set_tbl_cols(-1)
    pl.Config.set_fmt_str_lengths(300)
    if isinstance(seg_cols, str):
        seg_cols = [seg_cols]

    # Préparation de la matrice X standardisée (écrêtage p01-p99 et imputation médiane)
    exprs = []
    for c in features:
        s = pl.col(c).cast(pl.Float64)
        q01, q50, q99 = df[c].cast(pl.Float64).quantile(0.01), df[c].cast(pl.Float64).median(), df[c].cast(pl.Float64).quantile(0.99)
        s_clean = s.clip(q01, q99).fill_null(q50)
        exprs.append(((s_clean - s_clean.mean()) / s_clean.std()).alias(c))
    X_base = df.select(exprs).to_numpy()
    y = df[target].cast(pl.Float64).to_numpy()

    bilans, details = [], []
    for seg in seg_cols:
        modalites = sorted(df[seg].cast(pl.String).fill_null("missing").unique().to_list())
        ref, autres = modalites[0], modalites[1:]
        s_arr = df[seg].cast(pl.String).fill_null("missing").to_numpy()

        # On ne garde que les variables ayant une variance non nulle dans chaque segment
        keep_idx = [
            j for j in range(len(features))
            if all(np.std(X_base[s_arr == m, j]) > 1e-8 for m in modalites)
        ]
        X_k = X_base[:, keep_idx]
        noms_k = [features[j] for j in keep_idx]

        dummies = np.column_stack([(s_arr == m).astype(float) for m in autres])
        X0 = np.column_stack([np.ones(len(y)), dummies, X_k])

        inter_cols, inter_meta = [], []
        for idx_m, m in enumerate(autres):
            d_m = dummies[:, [idx_m]]
            inter_cols.append(X_k * d_m)
            inter_meta.extend((seg, m, ref, v) for v in noms_k)
        X1 = np.column_stack([X0, *inter_cols])

        res0 = sm.Logit(y, X0).fit(disp=False, maxiter=100)
        res1 = sm.Logit(y, X1).fit(disp=False, maxiter=100)

        n_inter = X1.shape[1] - X0.shape[1]
        lr_stat = float(2.0 * (res1.llf - res0.llf))
        lr_pval = float(chi2.sf(lr_stat, df=n_inter))

        # Test de Wald joint : R * beta = 0 sur les coefficients d'interaction
        R = np.zeros((n_inter, X1.shape[1]))
        R[:, X0.shape[1]:] = np.eye(n_inter)
        wald = res1.wald_test(R, scalar=True)
        wald_stat, wald_pval = float(wald.statistic), float(wald.pvalue)

        bilans.append({
            "segmentation": seg,
            "n_segments": len(modalites),
            "n_variables": len(noms_k),
            "dof": n_inter,
            "ll_pooled": float(res0.llf),
            "ll_interactions": float(res1.llf),
            "lr_stat": lr_stat,
            "lr_pvalue": lr_pval,
            "wald_stat": wald_stat,
            "wald_pvalue": wald_pval,
            "aic_pooled": float(res0.aic),
            "aic_interactions": float(res1.aic),
            "bic_pooled": float(res0.bic),
            "bic_interactions": float(res1.bic),
        })

        offset_beta = 1 + len(autres)
        offset_inter = X0.shape[1]
        for k, (seg_nom, m, ref_m, v) in enumerate(inter_meta):
            j_var = k % len(noms_k)
            idx_inter = offset_inter + k
            details.append({
                "segmentation": seg_nom,
                "segment": m,
                "reference": ref_m,
                "variable": v,
                "beta_ref": float(res1.params[offset_beta + j_var]),
                "delta_pente": float(res1.params[idx_inter]),
                "std_err": float(res1.bse[idx_inter]),
                "z_wald": float(res1.tvalues[idx_inter]),
                "pvalue": float(res1.pvalues[idx_inter]),
            })

    return pl.DataFrame(bilans), pl.DataFrame(details).sort("segmentation", "pvalue")


def plot_iv_by_segment_and_family(
    iv_seg: pl.DataFrame,
    seg_a: str = "SA_sans_engagement",
    seg_b: str = "SB_avec_engagement",
    top_n: int = 18,
    save_path: str | None = "outputs/figures/iv_par_famille_segments.png",
):
    """Trace le comparatif d'IV entre deux segments avec étiquetage et coloration par famille métier."""


    # Dictionnaire de règles (préfixes / mots-clés -> famille et couleur)
    REGLES_FAMILLES = [
        ("Historique défaut", "#ff7f0e", ["dfo", "sain", "historique_defaut"]),
        ("Incidents / Découvert", "#d62728", ["nbjde", "depassement", "arr_"]),
        ("Solde & Épargne", "#1f77b4", ["solde", "encours", "epargne"]),
        ("Démographie & Relation", "#2ca02c", ["age", "anciennete", "joint", "tiers", "acvpro", "axe_unite"]),
        ("Engagements / Crédits", "#9467bd", ["credit", "engagement", "revolving"]),
    ]

    def _trouver_famille(var: str) -> tuple[str, str]:
        v_low = var.lower()
        for nom_fam, col, motifs in REGLES_FAMILLES:
            if any(m in v_low for m in motifs):
                return nom_fam, col
        return "Autres", "#7f7f7f"

    col_a, col_b = f"iv_{seg_a}", f"iv_{seg_b}"
    top = (
        iv_seg.filter((pl.col(col_a) >= 0.15) | (pl.col(col_b) >= 0.15))
        .sort(col_a, descending=True)
        .head(top_n)
    )

    noms = top["variable"].to_list()
    val_a = top[col_a].to_list()
    val_b = top[col_b].to_list()

    infos = [_trouver_famille(v) for v in noms]
    familles = [info[0] for info in infos]
    couleurs = [info[1] for info in infos]
    labels_axe = [f"[{fam}]  {v}" for fam, v in zip(familles, noms)]

    fig, ax = plt.subplots(figsize=(12, 8.5), dpi=120)
    y_pos = range(len(noms))
    h = 0.38

    bars_a = ax.barh([y + h / 2 for y in y_pos], val_a, height=h, color=couleurs, alpha=0.9,
                     edgecolor="black", linewidth=0.6, label=seg_a)
    bars_b = ax.barh([y - h / 2 for y in y_pos], val_b, height=h, color=couleurs, alpha=0.35,
                     hatch="//", edgecolor="black", linewidth=0.6, label=seg_b)

    ax.axvline(0.3, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.axvline(0.5, color="darkred", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.text(0.31, len(noms) - 0.5, "Pouvoir fort (> 0,3)", color="dimgray", fontsize=9, style="italic")
    ax.text(0.51, len(noms) - 0.5, "Très fort (> 0,5)", color="darkred", fontsize=9, style="italic")

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels_axe, fontsize=9.5)
    ax.invert_yaxis()
    

    ax.set_xlabel("Information Value (IV)", fontsize=11, fontweight="bold")
    ax.set_title(f"Pouvoir discriminant (IV) par famille : {seg_a} vs {seg_b}", fontsize=13, pad=15)
    ax.grid(axis="x", linestyle=":", alpha=0.6)
    ax.legend(handles=[bars_a, bars_b], loc="lower right", framealpha=0.95, fontsize=10.5)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Graphique sauvegardé dans {save_path}")
    return fig, ax
