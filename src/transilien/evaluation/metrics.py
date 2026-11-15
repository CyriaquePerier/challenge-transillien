"""Métriques : MAE officielle, métriques robustes et MAE par segment."""

from __future__ import annotations

import numpy as np
import pandas as pd


def mae(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    return float(np.mean(np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float))))


def metric_suite(
    y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    abs_err = np.abs(err)
    return {
        "mae": float(abs_err.mean()),
        "medae": float(np.median(abs_err)),
        "q90_ae": float(np.quantile(abs_err, 0.90)),
        "q99_ae": float(np.quantile(abs_err, 0.99)),
        "bias": float(err.mean()),
        "rmse": float(np.sqrt(np.mean(err**2))),
    }


def relative_gain(score: float, reference: float) -> float:
    """Gain relatif (positif = meilleur que la référence)."""
    return float((reference - score) / reference) if reference else float("nan")


def delay_class(y: pd.Series | np.ndarray, bounds: tuple[int, int] = (1, 5)) -> pd.Series:
    """Classe de retard : ``|y| <= b0``, ``b0 < |y| <= b1``, ``|y| > b1``."""
    lo, hi = bounds
    a = np.abs(np.asarray(y, dtype=float))
    labels = np.select([a <= lo, a <= hi], [f"|y|<={lo}", f"{lo}<|y|<={hi}"], default=f"|y|>{hi}")
    return pd.Series(labels, index=getattr(y, "index", None))


def segment_mae(
    frame: pd.DataFrame, by: str | list[str], y_col: str = "y", pred_col: str = "pred"
) -> pd.DataFrame:
    """MAE, effectif et biais par segment, triés par MAE décroissante."""
    tmp = frame.assign(
        _ae=(frame[pred_col] - frame[y_col]).abs(), _err=frame[pred_col] - frame[y_col]
    )
    out = tmp.groupby(by, observed=True).agg(
        mae=("_ae", "mean"), n=("_ae", "size"), bias=("_err", "mean")
    )
    return out.sort_values("mae", ascending=False)


def segment_report(
    frame: pd.DataFrame, delay_bounds: tuple[int, int] = (1, 5)
) -> dict[str, pd.DataFrame]:
    """MAE par gare, jour de semaine, arrêt, semaine et classe de retard.

    ``frame`` doit contenir ``gare``, ``date``, ``arret``, ``y`` et ``pred``.
    """
    f = frame.assign(
        jour_semaine=frame["date"].dt.dayofweek,
        semaine=frame["date"].dt.isocalendar().week.astype(int),
        classe_retard=delay_class(frame["y"], delay_bounds).to_numpy(),
    )
    return {
        "gare": segment_mae(f, "gare"),
        "jour_semaine": segment_mae(f, "jour_semaine").sort_index(),
        "arret": segment_mae(f, "arret").sort_index(),
        "semaine": segment_mae(f, "semaine").sort_index(),
        "classe_retard": segment_mae(f, "classe_retard"),
    }
