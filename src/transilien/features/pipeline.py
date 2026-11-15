"""Assemblage des transformers d'un jeu de features et cache disque des features sans état."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from transilien.features.base import FeatureTransformer, build_transformer
from transilien.utils import get_logger, stable_hash

log = get_logger("features")


class FeaturePipeline:
    """Applique un jeu de features décrit dans ``configs/features/<nom>.yaml``.

    Usage dans la plateforme :

    1. :meth:`precompute` calcule (ou relit en cache) les features sans état sur train + test ;
    2. pour chaque fold, :meth:`fit` ajuste les transformers avec état sur les lignes
       d'entraînement puis :meth:`transform` produit la matrice finale.
    """

    def __init__(self, spec: list[dict[str, Any]], name: str = "custom") -> None:
        self.name = name
        self.spec = [{"name": s["name"], "params": dict(s.get("params") or {})} for s in spec]
        self.transformers: list[FeatureTransformer] = [
            build_transformer(s["name"], s["params"]) for s in self.spec
        ]

    @classmethod
    def from_config(cls, features_cfg: DictConfig) -> FeaturePipeline:
        spec = OmegaConf.to_container(features_cfg.transformers, resolve=True)
        assert isinstance(spec, list)
        return cls(spec, name=str(features_cfg.name))

    @property
    def stateless(self) -> list[FeatureTransformer]:
        return [t for t in self.transformers if t.stateless]

    @property
    def stateful(self) -> list[FeatureTransformer]:
        return [t for t in self.transformers if not t.stateless]

    def cache_key(self, data_hash: str) -> str:
        stateless_spec = [
            {**s, "code": t.source_hash()}
            for s, t in zip(self.spec, self.transformers, strict=True)
            if t.stateless
        ]
        return stable_hash({"data": data_hash, "spec": stateless_spec})

    def precompute(
        self,
        x_train: pd.DataFrame,
        x_test: pd.DataFrame,
        cache_dir: Path | None = None,
        data_hash: str | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Renvoie ``(base_train, base_test)`` : colonnes brutes + features sans état."""
        cache_file = None
        if cache_dir is not None and data_hash is not None:
            cache_file = cache_dir / f"{self.name}_{self.cache_key(data_hash)}.parquet"

        if cache_file is not None and cache_file.exists():
            log.info("Features lues depuis le cache : %s", cache_file.name)
            feats = pd.read_parquet(cache_file)
        else:
            x_all = pd.concat([x_train, x_test], keys=["train", "test"], names=["split", "id"])
            blocks = [t.transform(x_all) for t in self.stateless]
            feats = pd.concat(blocks, axis=1) if blocks else pd.DataFrame(index=x_all.index)
            _check_unique_columns(feats)
            if cache_file is not None:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                feats.to_parquet(cache_file)
                log.info("Features calculées et mises en cache : %s", cache_file.name)

        self.stateless_columns_ = list(feats.columns)

        def _join(x: pd.DataFrame, split: str) -> pd.DataFrame:
            part = feats.xs(split, level="split")
            new_cols = [c for c in part.columns if c not in x.columns]
            return x.join(part[new_cols])

        return _join(x_train, "train"), _join(x_test, "test")

    def fit(self, base: pd.DataFrame, y: pd.Series) -> FeaturePipeline:
        for t in self.stateful:
            t.fit(base, y)
        return self

    def transform(self, base: pd.DataFrame) -> pd.DataFrame:
        """Matrice numérique finale : features sans état + sorties des transformers avec état.

        Les colonnes brutes non numériques (``train``, ``gare``, ``date``) ne sont jamais
        transmises au modèle : seules les sorties des transformers le sont.
        """
        if not hasattr(self, "stateless_columns_"):
            raise RuntimeError("Appeler precompute() avant transform().")
        blocks = [base[self.stateless_columns_]] + [t.transform(base) for t in self.stateful]
        out = pd.concat(blocks, axis=1)
        _check_unique_columns(out)
        return out


def _check_unique_columns(df: pd.DataFrame) -> None:
    dup = df.columns[df.columns.duplicated()].tolist()
    if dup:
        raise ValueError(f"Colonnes de features dupliquées : {dup}")
