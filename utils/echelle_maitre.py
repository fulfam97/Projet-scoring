# ---------------------------------------------------------------------------
# Échelle maître (7.5)
# ---------------------------------------------------------------------------
import polars as pl
import numpy as np
BORNES_ECHELLE = [0.0005, 0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128, 0.256]


def echelle_maitre(bornes: list[float] = BORNES_ECHELLE) -> pl.DataFrame:
    """Grades de l'échelle maître définis par des bornes de PD (doublement d'un grade au suivant)."""
    b = [0.0] + list(bornes) + [1.0]
    return pl.DataFrame({"grade": list(range(1, len(b))), "pd_min": b[:-1], "pd_max": b[1:]})


def affecter_grade(pd_finale, bornes: list[float] = BORNES_ECHELLE) -> np.ndarray:
    """Grade de l'échelle maître de chaque PD : grade g si pd_min < PD ≤ pd_max."""
    return np.searchsorted(np.asarray(bornes), np.asarray(pd_finale, dtype=float), side="left") + 1