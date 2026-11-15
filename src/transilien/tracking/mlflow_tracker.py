"""Suivi MLflow local (base SQLite + artefacts dans ``mlruns/``)."""

from __future__ import annotations

import contextlib
import logging
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from transilien.utils import get_logger

log = get_logger("tracking")


def tracking_uri(mlruns_dir: Path) -> str:
    return f"sqlite:///{(mlruns_dir / 'mlflow.db').as_posix()}"


def _flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in d.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(_flatten(value, f"{name}."))
        else:
            out[name] = value
    return out


class Tracker:
    """Fine surcouche de MLflow ; ne fait rien si le suivi est désactivé."""

    def __init__(self, mlruns_dir: Path, experiment: str, enabled: bool = True) -> None:
        self.enabled = enabled
        self.mlruns_dir = mlruns_dir
        self.experiment = experiment
        self._mlflow: Any = None
        if enabled:
            os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
            import mlflow

            logging.getLogger("mlflow").setLevel(logging.WARNING)
            logging.getLogger("alembic").setLevel(logging.WARNING)
            mlruns_dir.mkdir(parents=True, exist_ok=True)
            mlflow.set_tracking_uri(tracking_uri(mlruns_dir))
            if mlflow.get_experiment_by_name(experiment) is None:
                mlflow.create_experiment(
                    experiment, artifact_location=(mlruns_dir / "artifacts").resolve().as_uri()
                )
            mlflow.set_experiment(experiment)
            self._mlflow = mlflow

    @contextlib.contextmanager
    def start_run(self, run_name: str, nested: bool = False) -> Iterator[str | None]:
        if not self.enabled:
            yield None
            return
        with self._mlflow.start_run(run_name=run_name, nested=nested) as run:
            yield run.info.run_id

    def log_params(self, params: dict[str, Any]) -> None:
        if self.enabled:
            flat = {k: str(v)[:500] for k, v in _flatten(params).items()}
            self._mlflow.log_params(flat)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        if self.enabled:
            clean = {k.replace("/", "_"): float(v) for k, v in metrics.items() if v == v}
            self._mlflow.log_metrics(clean, step=step)

    def set_tags(self, tags: dict[str, Any]) -> None:
        if self.enabled:
            self._mlflow.set_tags({k: str(v) for k, v in tags.items()})

    def log_artifacts(self, directory: Path, max_file_mb: float | None = None) -> list[str]:
        """Envoie le contenu de ``directory`` ; renvoie les fichiers ignorés car trop gros."""
        skipped: list[str] = []
        if not self.enabled:
            return skipped
        for file in sorted(p for p in directory.rglob("*") if p.is_file()):
            rel = file.relative_to(directory)
            if max_file_mb is not None and file.stat().st_size > max_file_mb * 1e6:
                skipped.append(str(rel))
                continue
            parent = rel.parent.as_posix()
            self.log_artifact(file, None if parent == "." else parent)
        if skipped:
            self.set_tags({"artifacts_skipped": ", ".join(skipped)})
            log.info("Artefacts trop volumineux non copiés dans MLflow : %s", skipped)
        return skipped

    def log_artifact(self, file: Path, artifact_path: str | None = None) -> None:
        if self.enabled:
            try:
                self._mlflow.log_artifact(str(file), artifact_path=artifact_path)
            except Exception as exc:
                log.warning("Échec de l'envoi de l'artefact MLflow : %s", exc)
