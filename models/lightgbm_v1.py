"""LightGBM en perte L1 (alignée sur la MAE) avec early stopping sur le fold de validation."""

import lightgbm as lgb
import numpy as np
import pandas as pd

from transilien.models.base import BaseModel
from transilien.models.registry import register


@register
class LightGBMV1(BaseModel):
    name = "lightgbm_v1"
    feature_set = "lags_enrichis"
    default_params = {"objective": "l1", "n_estimators": 1000, "learning_rate": 0.05}

    def fit(self, X, y, X_valid=None, y_valid=None):
        params = dict(self.params)
        early_stopping = params.pop("early_stopping_rounds", None)
        self.model_ = lgb.LGBMRegressor(**params, random_state=self.seed, verbose=-1, n_jobs=-1)
        callbacks, eval_set = [], None
        if X_valid is not None and y_valid is not None:
            eval_set = [(X_valid, y_valid)]
            callbacks.append(lgb.record_evaluation(evals := {}))
            if early_stopping:
                callbacks.append(lgb.early_stopping(early_stopping, verbose=False))
        self.model_.fit(X, y, eval_set=eval_set, callbacks=callbacks)
        if eval_set is not None:
            self.learning_curve_ = {"valid": list(evals["valid_0"]["l1"])}
            self.best_iteration_ = self.model_.best_iteration_ or params["n_estimators"]
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict(X)

    def feature_importances(self) -> pd.Series:
        booster = self.model_.booster_
        return pd.Series(booster.feature_importance("gain"), index=booster.feature_name())

    def refit_params(self, fold_models):
        iters = [m.best_iteration_ for m in fold_models if getattr(m, "best_iteration_", None)]
        if not iters:
            return {}
        return {"n_estimators": int(np.median(iters))}

    def search_space(self, trial):
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 500, log=True),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
