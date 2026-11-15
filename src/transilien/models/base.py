"""Contrat ``BaseModel`` : la seule interface qu'un modèle utilisateur doit respecter."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar

import joblib
import numpy as np
import pandas as pd


class BaseModel(ABC):
    """Classe mère de tous les modèles de ``models/``.

    Obligatoire :
        * ``name`` (attribut de classe) + décorateur ``@register`` ;
        * ``fit(X, y, X_valid=None, y_valid=None)`` ;
        * ``predict(X) -> np.ndarray`` (prédictions continues, le post-traitement est
          géré par la plateforme).

    Optionnel (détecté automatiquement) :
        * ``feature_set`` : nom d'un fichier ``configs/features/<nom>.yaml`` (défaut : ``base``) ;
        * ``default_params`` : hyperparamètres par défaut, surchargés par
          ``configs/model/<nom>.yaml`` puis par la ligne de commande ;
        * ``search_space(trial) -> dict`` : espace Optuna, active ``transilien tune`` ;
        * ``feature_importances() -> pd.Series`` : importance native des features ;
        * ``refit_params(fold_models) -> dict`` : paramètres à utiliser pour les
          réentraînements (ex. nombre d'arbres trouvé par early stopping en CV) ;
        * ``learning_curve_`` : ``{"valid": [...]}`` renseigné pendant ``fit``.
    """

    name: ClassVar[str]
    feature_set: ClassVar[str | None] = None
    default_params: ClassVar[dict[str, Any]] = {}

    def __init__(self, params: dict[str, Any] | None = None, seed: int = 42) -> None:
        self.params: dict[str, Any] = {**self.default_params, **(params or {})}
        self.seed = seed

    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_valid: pd.DataFrame | None = None,
        y_valid: pd.Series | None = None,
    ) -> BaseModel:
        """Entraîne le modèle. Le jeu de validation sert uniquement à l'early stopping."""

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Renvoie un vecteur de prédictions continues de longueur ``len(X)``."""

    def feature_importances(self) -> pd.Series | None:
        return None

    def refit_params(self, fold_models: list[BaseModel]) -> dict[str, Any]:
        return {}

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path, compress=3)

    @staticmethod
    def load(path: Path) -> BaseModel:
        from transilien.models.registry import discover_models

        discover_models()
        model = joblib.load(path)
        if not isinstance(model, BaseModel):
            raise TypeError(f"{path} ne contient pas un BaseModel.")
        return model

    @classmethod
    def has_search_space(cls) -> bool:
        return callable(getattr(cls, "search_space", None))

    def __repr__(self) -> str:
        return f"{type(self).__name__}(params={self.params}, seed={self.seed})"
