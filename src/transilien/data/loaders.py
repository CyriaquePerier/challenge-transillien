"""Chargement du jeu de données prêt à l'emploi (Parquet validés)."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from omegaconf import DictConfig

from transilien.config import path
from transilien.data.ingest import RawFiles, ingest
from transilien.data.schemas import TARGET
from transilien.utils import stable_hash


@dataclass
class Dataset:
    x_train: pd.DataFrame
    y_train: pd.Series
    x_test: pd.DataFrame
    source_hashes: dict[str, str] = field(default_factory=dict)
    sample_frac: float | None = None

    @property
    def data_hash(self) -> str:
        """Hash des fichiers sources + de l'échantillonnage : identifie les données d'un run."""
        return stable_hash({"sources": self.source_hashes, "sample_frac": self.sample_frac})


def raw_files(cfg: DictConfig) -> RawFiles:
    lake = path(cfg, "data_lake")
    files = cfg.data.files
    return RawFiles(lake / files.x_train, lake / files.y_train, lake / files.x_test)


def load_dataset(cfg: DictConfig) -> Dataset:
    interim = path(cfg, "interim")
    hashes = ingest(raw_files(cfg), interim)
    x_train = pd.read_parquet(interim / "x_train.parquet")
    y_train = pd.read_parquet(interim / "y_train.parquet")[TARGET]
    x_test = pd.read_parquet(interim / "x_test.parquet")

    frac = cfg.data.get("sample_frac")
    if frac:
        # Échantillonnage stratifié par date : la structure temporelle est préservée.
        keep = (
            x_train.groupby("date", group_keys=False)
            .sample(frac=float(frac), random_state=int(cfg.seed))
            .index.sort_values()
        )
        x_train, y_train = x_train.loc[keep], y_train.loc[keep]
    return Dataset(x_train, y_train, x_test, hashes, float(frac) if frac else None)
