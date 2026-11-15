"""Stockage des runs sur disque (``reports/runs/<run_id>/``) et leaderboard."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from transilien.utils import read_json

METRICS_FILE = "metrics.json"


@dataclass(frozen=True, order=True)
class RunFiles:
    """Chemins des artefacts d'un run."""

    root: Path

    @property
    def run_id(self) -> str:
        return self.root.name

    @property
    def metrics(self) -> Path:
        return self.root / METRICS_FILE

    @property
    def report(self) -> Path:
        return self.root / "index.html"

    @property
    def config(self) -> Path:
        return self.root / "config.yaml"

    @property
    def oof(self) -> Path:
        return self.root / "oof.parquet"

    @property
    def holdout(self) -> Path:
        return self.root / "holdout.parquet"

    @property
    def test(self) -> Path:
        return self.root / "test.parquet"

    @property
    def model(self) -> Path:
        return self.root / "model.joblib"

    @property
    def figures(self) -> Path:
        return self.root / "figures"

    @property
    def segments(self) -> Path:
        return self.root / "segments"

    def load_metrics(self) -> dict[str, Any]:
        return read_json(self.metrics)


def list_runs(runs_dir: Path) -> list[RunFiles]:
    if not runs_dir.exists():
        return []
    return sorted(RunFiles(p) for p in runs_dir.iterdir() if (p / METRICS_FILE).exists())


def resolve_run(runs_dir: Path, ref: str) -> RunFiles:
    """Retrouve un run par identifiant exact, préfixe, ou nom de modèle (dernier run)."""
    runs = list_runs(runs_dir)
    exact = [r for r in runs if r.run_id == ref]
    if exact:
        return exact[0]
    prefixed = [r for r in runs if r.run_id.startswith(ref)]
    if len(prefixed) == 1:
        return prefixed[0]
    by_model = [r for r in runs if r.load_metrics().get("model") == ref]
    if by_model:
        return by_model[-1]
    raise KeyError(f"Run introuvable : {ref}")


def leaderboard(runs_dir: Path) -> pd.DataFrame:
    """Classement des runs par MAE hold-out (puis MAE CV)."""
    rows = []
    for run in list_runs(runs_dir):
        m = run.load_metrics()
        holdout = m.get("holdout") or {}
        rows.append(
            {
                "run_id": run.run_id,
                "model": m.get("model"),
                "features": m.get("feature_set"),
                "cv": m.get("cv_name"),
                "cv_mae": m["cv"]["mae_mean"],
                "cv_std": m["cv"]["mae_std"],
                "holdout_mae": holdout.get("mae"),
                "ecart_cv_holdout": (holdout.get("mae") - m["cv"]["mae_mean"])
                if holdout.get("mae") is not None
                else None,
                "postprocess": m.get("postprocess", {}).get("variant"),
                "sample_frac": m.get("sample_frac"),
                "created_at": m.get("created_at"),
            }
        )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df.sort_values(["holdout_mae", "cv_mae"], na_position="last").reset_index(drop=True)
