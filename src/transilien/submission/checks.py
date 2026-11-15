"""Vérifications bloquantes appliquées avant l'écriture d'une soumission."""

from __future__ import annotations

import numpy as np
import pandas as pd

from transilien.data.schemas import TARGET
from transilien.utils import get_logger

log = get_logger("submission")


class SubmissionError(ValueError):
    """La soumission ne respecte pas le format officiel."""


def check_submission(
    submission: pd.DataFrame,
    x_test: pd.DataFrame,
    max_abs_value: float = 200.0,
    reference_pred: np.ndarray | None = None,
) -> list[str]:
    """Lève :class:`SubmissionError` si la soumission est invalide ; renvoie les avertissements."""
    if list(submission.columns) != [TARGET]:
        raise SubmissionError(
            f"Colonnes attendues : ['{TARGET}'], reçues : {list(submission.columns)}"
        )
    if len(submission) != len(x_test):
        raise SubmissionError(f"{len(submission)} lignes au lieu de {len(x_test)}.")
    if not submission.index.equals(x_test.index):
        raise SubmissionError("L'index ne correspond pas à celui de x_test (ordre ou valeurs).")
    values = submission[TARGET]
    if not pd.api.types.is_numeric_dtype(values):
        raise SubmissionError(f"Type non numérique : {values.dtype}")
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise SubmissionError("La soumission contient des NaN ou des infinis.")
    if values.abs().max() > max_abs_value:
        raise SubmissionError(f"Valeur hors plage plausible (|y| > {max_abs_value}).")

    warnings: list[str] = []
    if reference_pred is not None and len(reference_pred):
        ref = np.asarray(reference_pred, dtype=float)
        ref_std, test_std = float(ref.std()), float(values.std())
        scale = ref_std if ref_std > 0 else 1.0
        shift = abs(values.mean() - ref.mean()) / scale
        if shift > 3:
            raise SubmissionError(
                "Distribution très différente des prédictions OOF "
                f"(écart de moyenne = {shift:.1f} σ)."
            )
        std_ratio = test_std / ref_std if ref_std > 0 else (1.0 if test_std == 0 else np.inf)
        if shift > 0.5 or not 0.5 < std_ratio < 2:
            warnings.append(
                f"Distribution test ≠ OOF : moyenne {values.mean():.3f} vs {ref.mean():.3f}, "
                f"écart-type {test_std:.3f} vs {ref_std:.3f}."
            )
    for w in warnings:
        log.warning(w)
    return warnings
