"""CatBoost en perte MAE ; la gare est traitée nativement comme variable catégorielle."""

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from transilien.models.base import BaseModel
from transilien.models.registry import register

CAT_FEATURES = ["gare_code", "jour_semaine"]


@register
class CatBoostV1(BaseModel):
    name = "catboost_v1"
    feature_set = "base"
    default_params = {"loss_function": "MAE", "iterations": 1000, "learning_rate": 0.08}

    def fit(self, X, y, X_valid=None, y_valid=None):
        params = dict(self.params)
        has_valid = X_valid is not None and y_valid is not None
        if not has_valid:
            params.pop("early_stopping_rounds", None)
        self.cat_features_ = [c for c in CAT_FEATURES if c in X.columns]
        self.model_ = CatBoostRegressor(
            **params,
            random_seed=self.seed,
            verbose=False,
            allow_writing_files=False,
            thread_count=-1,
            cat_features=self.cat_features_,
        )
        self.model_.fit(X, y, eval_set=(X_valid, y_valid) if has_valid else None)
        if has_valid:
            evals = self.model_.get_evals_result()["validation"]
            self.learning_curve_ = {"valid": list(next(iter(evals.values())))}
            self.best_iteration_ = self.model_.get_best_iteration() + 1
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict(X)

    def feature_importances(self) -> pd.Series:
        return pd.Series(self.model_.get_feature_importance(), index=self.model_.feature_names_)

    def refit_params(self, fold_models):
        iters = [m.best_iteration_ for m in fold_models if getattr(m, "best_iteration_", None)]
        return {"iterations": int(np.median(iters))} if iters else {}

    def search_space(self, trial):
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.1, 30.0, log=True),
        }
