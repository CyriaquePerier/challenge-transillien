"""Contrat des transformers de features et registre associé."""

from __future__ import annotations

import inspect
from typing import Any, ClassVar

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

FEATURE_REGISTRY: dict[str, type[FeatureTransformer]] = {}


class FeatureTransformer(BaseEstimator, TransformerMixin):
    """Transformer compatible scikit-learn produisant de *nouvelles* colonnes.

    * ``stateless = True`` : la sortie ne dépend ni de la cible ni de l'échantillon
      d'entraînement ; elle est calculée une seule fois sur train + test puis mise en cache.
    * ``stateless = False`` : le transformer apprend quelque chose dans ``fit`` (ex. encodage
      cible) ; il est réajusté dans **chaque fold** pour éviter toute fuite.

    ``transform`` reçoit les colonnes brutes enrichies des features sans état déjà calculées
    et renvoie un ``DataFrame`` numérique de même index.
    """

    name: ClassVar[str]
    stateless: ClassVar[bool] = True

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> FeatureTransformer:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # pragma: no cover - abstrait
        raise NotImplementedError

    @classmethod
    def source_hash(cls) -> str:
        """Empreinte du code source : invalide le cache si l'implémentation change."""
        from transilien.utils import stable_hash

        return stable_hash(inspect.getsource(cls))


def register_feature(cls: type[FeatureTransformer]) -> type[FeatureTransformer]:
    if not getattr(cls, "name", None):
        raise ValueError(f"{cls.__name__} doit définir un attribut 'name'.")
    if cls.name in FEATURE_REGISTRY and FEATURE_REGISTRY[cls.name] is not cls:
        raise ValueError(f"Transformer de features déjà enregistré : {cls.name}")
    FEATURE_REGISTRY[cls.name] = cls
    return cls


def build_transformer(name: str, params: dict[str, Any] | None = None) -> FeatureTransformer:
    # Import local : peuple le registre avec les transformers fournis par la plateforme.
    from transilien.features import transformers  # noqa: F401

    if name not in FEATURE_REGISTRY:
        raise KeyError(f"Transformer inconnu : {name}. Disponibles : {sorted(FEATURE_REGISTRY)}")
    return FEATURE_REGISTRY[name](**(params or {}))
