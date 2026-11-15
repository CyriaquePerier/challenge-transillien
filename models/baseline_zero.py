"""Baseline triviale : prédit 0 partout (MAE ≈ 0,894). Le plancher à battre."""

import numpy as np
import pandas as pd

from transilien.models.base import BaseModel
from transilien.models.registry import register


@register
class BaselineZero(BaseModel):
    name = "baseline_zero"
    feature_set = "raw"

    def fit(self, X, y, X_valid=None, y_valid=None):
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(X))
