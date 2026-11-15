"""Fixtures partagées : configuration isolée (sorties dans un dossier temporaire) et données."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from transilien.config import load_config
from transilien.data.loaders import load_dataset


def make_config(tmp_path: Path, *overrides: str):
    """Configuration dont toutes les sorties vont dans ``tmp_path`` (cache des données partagé)."""
    return load_config(
        [
            f"paths.runs={tmp_path / 'runs'}",
            f"paths.reports={tmp_path / 'reports'}",
            f"paths.submissions={tmp_path / 'submissions'}",
            f"paths.mlruns={tmp_path / 'mlruns'}",
            f"paths.features_cache={tmp_path / 'features'}",
            "tracking.enabled=false",
            "reporting.export_png=false",
            *overrides,
        ]
    )


@pytest.fixture
def cfg_factory(tmp_path):
    return lambda *overrides: make_config(tmp_path, *overrides)


@pytest.fixture(scope="session")
def dataset():
    return load_dataset(load_config())


@pytest.fixture(scope="session")
def sample(dataset):
    """5 000 lignes réelles réparties sur toutes les dates."""
    idx = dataset.x_train.sample(5000, random_state=0).index.sort_values()
    return dataset.x_train.loc[idx], dataset.y_train.loc[idx]


@pytest.fixture
def toy_frame():
    """Petit jeu synthétique au format des données brutes (jours ouvrés uniquement)."""
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2023-04-03", periods=60)
    n = 1200
    frame = (
        pd.DataFrame(
            {
                "train": [f"T{i:05d}" for i in range(n)],
                "gare": rng.choice(["AAA", "BBB", "CCC", "DDD"], n),
                "date": pd.to_datetime(rng.choice(dates, n)).astype("datetime64[ns]"),
                "arret": rng.integers(7, 40, n).astype("int64"),
                **{
                    c: rng.integers(-5, 6, n).astype(float)
                    for c in ["p2q0", "p3q0", "p4q0", "p0q2", "p0q3", "p0q4"]
                },
            }
        )
        .sort_values("date")
        .reset_index(drop=True)
    )
    frame.index.name = "id"
    frame["train"] = frame["train"].astype(str)
    frame["gare"] = frame["gare"].astype(str)
    y = pd.Series(rng.integers(-3, 4, n).astype(float), index=frame.index, name="p0q0")
    return frame, y
