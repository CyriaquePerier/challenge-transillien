"""XGBoost en perte absolue avec early stopping sur le fold de validation."""

import numpy as np
import pandas as pd
import xgboost as xgb

from transilien.models.base import BaseModel
from transilien.models.registry import register


@register
class XGBoostV1(BaseModel):
    name = "xgboost_v1"
    feature_set = "lags_enrichis"
    default_params = {"objective": "reg:absoluteerror", "n_estimators": 1000, "learning_rate": 0.05}

    def fit(self, X, y, X_valid=None, y_valid=None):
        params = dict(self.params)
        has_valid = X_valid is not None and y_valid is not None
        if not has_valid:
            params.pop("early_stopping_rounds", None)
        self.model_ = xgb.XGBRegressor(
            **params, random_state=self.seed, n_jobs=-1, tree_method="hist"
        )
        eval_set = [(X_valid, y_valid)] if has_valid else None
        self.model_.fit(X, y, eval_set=eval_set, verbose=False)
        if has_valid:
            history = self.model_.evals_result()["validation_0"]
            self.learning_curve_ = {"valid": list(next(iter(history.values())))}
            best = getattr(self.model_, "best_iteration", None)
            self.best_iteration_ = (best + 1) if best is not None else params["n_estimators"]
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict(X)

    def feature_importances(self) -> pd.Series:
        return pd.Series(self.model_.feature_importances_, index=self.model_.feature_names_in_)

    def refit_params(self, fold_models):
        iters = [m.best_iteration_ for m in fold_models if getattr(m, "best_iteration_", None)]
        return {"n_estimators": int(np.median(iters))} if iters else {}

    def search_space(self, trial):
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 100.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        }
