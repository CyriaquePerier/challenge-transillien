"""Optimisation d'hyperparamètres par Optuna, sur la même CV temporelle que les runs."""

from __future__ import annotations

from typing import Any

import numpy as np
import optuna
from omegaconf import DictConfig, OmegaConf

from transilien.config import path
from transilien.evaluation.splitters import TemporalSplitter, holdout_split
from transilien.models.registry import create_model, get_model_class
from transilien.runner import cross_validate, prepare
from transilien.tracking.mlflow_tracker import Tracker
from transilien.utils import get_logger, set_global_seed

log = get_logger("tuning")


def tune(cfg: DictConfig, n_trials: int, timeout: float | None = None) -> dict[str, Any]:
    """Lance l'étude, écrit les meilleurs paramètres dans ``configs/model/<nom>.yaml``."""
    set_global_seed(int(cfg.seed))
    model_name = str(cfg.model.name)
    model_cls = get_model_class(model_name)
    if not model_cls.has_search_space():
        raise ValueError(f"{model_name} ne définit pas search_space(trial) : tuning impossible.")

    cfg, dataset, pipeline, base_train, _ = prepare(cfg)
    dev_mask, _ = holdout_split(base_train["date"], int(cfg.holdout.weeks))
    base_dev, y_dev = base_train[dev_mask], dataset.y_train[dev_mask]
    splitter = TemporalSplitter.from_config(cfg.cv)
    probe = create_model(cfg.model, int(cfg.seed))

    tracker = Tracker(path(cfg, "mlruns"), str(cfg.tracking.experiment), bool(cfg.tracking.enabled))
    study = optuna.create_study(
        direction="minimize",
        study_name=f"tune_{model_name}",
        sampler=optuna.samplers.TPESampler(seed=int(cfg.seed)),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=int(cfg.tuning.n_startup_trials), n_warmup_steps=1
        ),
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        params = probe.search_space(trial)  # type: ignore[attr-defined]
        scores: list[float] = []

        def on_fold(k: int, score: float) -> None:
            scores.append(score)
            trial.report(float(np.mean(scores)), k)
            if trial.should_prune():
                raise optuna.TrialPruned()

        state = "complete"
        try:
            cv = cross_validate(
                lambda extra: create_model(cfg.model, int(cfg.seed), {**params, **extra}),
                pipeline,
                base_dev,
                y_dev,
                splitter,
                on_fold=on_fold,
                keep_models=False,
            )
            return cv.mae_mean
        except optuna.TrialPruned:
            state = "pruned"
            raise
        finally:
            with tracker.start_run(f"trial_{trial.number}", nested=True):
                tracker.log_params(params)
                tracker.set_tags({"state": state, "trial": trial.number})
                if scores:
                    tracker.log_metrics(
                        {"cv_mae_mean": float(np.mean(scores)), "n_folds": len(scores)}
                    )
            log.info(
                "Essai %d (%s) : MAE %.4f — %s",
                trial.number,
                state,
                float(np.mean(scores)) if scores else float("nan"),
                params,
            )

    with tracker.start_run(f"tune_{model_name}"):
        tracker.set_tags({"model": model_name, "kind": "tuning"})
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
        tracker.log_metrics({"best_cv_mae": study.best_value})
        tracker.log_params({f"best.{k}": v for k, v in study.best_params.items()})

    best = study.best_trial.params
    log.info(
        "Meilleur essai %d : MAE CV %.4f — %s", study.best_trial.number, study.best_value, best
    )
    write_best_params(cfg, model_name, best)
    return best


def write_best_params(cfg: DictConfig, model_name: str, best: dict[str, Any]) -> None:
    target = path(cfg, "model_configs") / f"{model_name}.yaml"
    current = (
        OmegaConf.load(target)
        if target.exists()
        else OmegaConf.create({"name": model_name, "params": {}})
    )
    merged = OmegaConf.merge(current, {"params": best})
    OmegaConf.save(merged, target)
    log.info("Paramètres écrits dans %s", target)
