"""Ingestion : CSV bruts (immuables) → Parquet typés et validés, avec cache par hash."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from transilien.data.schemas import FEATURE_COLUMNS, LAG_COLUMNS, TARGET, validate_dataset
from transilien.utils import file_sha256, get_logger, read_json, write_json

log = get_logger("ingest")

TABLES = ("x_train", "y_train", "x_test")


@dataclass(frozen=True)
class RawFiles:
    x_train: Path
    y_train: Path
    x_test: Path

    def as_dict(self) -> dict[str, Path]:
        return {"x_train": self.x_train, "y_train": self.y_train, "x_test": self.x_test}


def read_raw_features(path: Path) -> pd.DataFrame:
    """Lit un CSV de features, supprime les colonnes parasites et type les colonnes."""
    df = pd.read_csv(path, index_col=0)
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])
    df.index = df.index.astype("int64")
    df.index.name = "id"
    df = df[FEATURE_COLUMNS].copy()
    df["train"] = df["train"].astype(str)
    df["gare"] = df["gare"].astype(str)
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d").astype("datetime64[ns]")
    df["arret"] = df["arret"].astype("int64")
    df[LAG_COLUMNS] = df[LAG_COLUMNS].astype("float64")
    return df


def read_raw_target(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.astype("int64")
    df.index.name = "id"
    return df[[TARGET]].astype("float64")


def ingest(raw: RawFiles, interim_dir: Path, force: bool = False) -> dict[str, str]:
    """Convertit les CSV en Parquet si nécessaire et renvoie les hash SHA-256 des sources.

    La conversion est rejouée uniquement si un fichier source a changé (hash différent)
    ou si un Parquet manque.
    """
    interim_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = interim_dir / "manifest.json"
    hashes = {name: file_sha256(p) for name, p in raw.as_dict().items()}
    parquet_ok = all((interim_dir / f"{t}.parquet").exists() for t in TABLES)
    if not force and parquet_ok and manifest_path.exists() and read_json(manifest_path) == hashes:
        return hashes

    log.info("Ingestion des CSV bruts → Parquet (%s)", interim_dir)
    x_train = read_raw_features(raw.x_train)
    y_train = read_raw_target(raw.y_train)
    x_test = read_raw_features(raw.x_test)
    validate_dataset(x_train, y_train, x_test)
    x_train.to_parquet(interim_dir / "x_train.parquet")
    y_train.to_parquet(interim_dir / "y_train.parquet")
    x_test.to_parquet(interim_dir / "x_test.parquet")
    write_json(manifest_path, hashes)
    log.info("Ingestion terminée : %d lignes train, %d lignes test", len(x_train), len(x_test))
    return hashes
