"""Écriture de la soumission au format exact de ``y_sample_final.csv``."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from transilien.data.schemas import TARGET
from transilien.submission.checks import check_submission


def build_submission(pred: np.ndarray, x_test: pd.DataFrame) -> pd.DataFrame:
    values = np.asarray(pred, dtype=float)
    series = pd.Series(values, index=x_test.index, name=TARGET)
    if np.all(np.mod(values, 1) == 0):
        series = series.astype("int64")
    frame = series.to_frame()
    frame.index.name = None
    return frame


def submission_filename(model_name: str, cv_mae: float, day: date | None = None) -> str:
    day = day or date.today()
    return f"{day.isoformat()}_{model_name}_cv{cv_mae:.3f}.csv"


def write_submission(
    pred: np.ndarray,
    x_test: pd.DataFrame,
    out_dir: Path,
    model_name: str,
    cv_mae: float,
    max_abs_value: float = 200.0,
    reference_pred: np.ndarray | None = None,
) -> tuple[Path, list[str]]:
    """Vérifie puis écrit la soumission ; renvoie son chemin et les avertissements."""
    submission = build_submission(pred, x_test)
    warnings = check_submission(submission, x_test, max_abs_value, reference_pred)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / submission_filename(model_name, cv_mae)
    submission.to_csv(out)  # première colonne : index sans nom, comme y_sample_final.csv
    return out, warnings
