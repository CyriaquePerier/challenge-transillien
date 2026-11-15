"""Benchmark officiel du challenge : forêt aléatoire classique sur les variables brutes.

Point de départ recommandé pour un nouveau modèle : copier ce fichier, changer ``name``,
puis lancer ``make run MODEL=<nom>``.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from transilien.models.base import BaseModel
from transilien.models.registry import register


@register
class BaselineRF(BaseModel):
    name = "baseline_rf"
    feature_set = "raw"
    default_params = {"n_estimators": 100, "min_samples_leaf": 5, "max_samples": 0.5}

    def fit(self, X, y, X_valid=None, y_valid=None):
        self.model_ = RandomForestRegressor(**self.params, random_state=self.seed, n_jobs=-1)
        self.model_.fit(X, y)
        self.columns_ = list(X.columns)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict(X[self.columns_])

    def feature_importances(self) -> pd.Series:
        return pd.Series(self.model_.feature_importances_, index=self.columns_)

    def search_space(self, trial):
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=50),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 50, log=True),
            "max_features": trial.suggest_float("max_features", 0.3, 1.0),
        }
