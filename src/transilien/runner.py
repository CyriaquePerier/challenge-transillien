"""Orchestration d'un run complet : CV temporelle → post-traitement → hold-out → soumission."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from transilien.config import path, resolve_feature_set, to_container
from transilien.data.loaders import Dataset, load_dataset
from transilien.evaluation.metrics import mae, metric_suite, relative_gain, segment_report
from transilien.evaluation.postprocess import PostProcessor, select_postprocessor
from transilien.evaluation.splitters import Fold, TemporalSplitter, holdout_split
from transilien.features.pipeline import FeaturePipeline
from transilien.models.base import BaseModel
from transilien.models.registry import create_model, get_model_class
from transilien.runs import RunFiles, leaderboard
from transilien.submission.writer import write_submission
from transilien.tracking.mlflow_tracker import Tracker
from transilien.utils import get_logger, git_commit, set_global_seed, write_json

log = get_logger("runner")

ModelFactory = Callable[[dict[str, Any]], BaseModel]
CONTEXT_COLS = ["date", "gare", "arret"]


@dataclass
class CVResult:
    oof: pd.Series  # prédictions brutes, NaN hors validation
    fold_ids: pd.Series  # numéro de fold (-1 hors validation)
    fold_scores: list[float]
    fold_info: list[dict[str, Any]]
    models: list[BaseModel] = field(default_factory=list)
    fit_seconds: float = 0.0
    predict_seconds: float = 0.0

    @property
    def mask(self) -> np.ndarray:
        return (self.fold_ids >= 0).to_numpy()

    @property
    def mae_mean(self) -> float:
        return float(np.mean(self.fold_scores))

    @property
    def mae_std(self) -> float:
        return float(np.std(self.fold_scores))


@dataclass
class RunResult:
    run_id: str
    files: RunFiles
    metrics: dict[str, Any]
    mlflow_run_id: str | None = None
    submission: Path | None = None


def fit_predict(
    model_factory: ModelFactory,
    pipeline: FeaturePipeline,
    base_fit: pd.DataFrame,
    y_fit: pd.Series,
    base_eval: list[pd.DataFrame],
    extra_params: dict[str, Any] | None = None,
    valid: tuple[pd.DataFrame, pd.Series] | None = None,
) -> tuple[BaseModel, list[np.ndarray], float, float]:
    """Ajuste features + modèle sur ``base_fit`` puis prédit chaque table de ``base_eval``."""
    pipeline.fit(base_fit, y_fit)
    X_fit = pipeline.transform(base_fit)
    X_valid = y_valid = None
    if valid is not None:
        X_valid, y_valid = pipeline.transform(valid[0]), valid[1]
    model = model_factory(extra_params or {})
    t0 = time.perf_counter()
    model.fit(X_fit, y_fit, X_valid, y_valid)
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    preds = [np.asarray(model.predict(pipeline.transform(b)), dtype=float) for b in base_eval]
    return model, preds, fit_s, time.perf_counter() - t0


def cross_validate(
    model_factory: ModelFactory,
    pipeline: FeaturePipeline,
    base: pd.DataFrame,
    y: pd.Series,
    splitter: TemporalSplitter,
    on_fold: Callable[[int, float], None] | None = None,
    keep_models: bool = True,
) -> CVResult:
    """CV temporelle ; ``on_fold(k, mae)`` permet le pruning Optuna après chaque fold."""
    oof = np.full(len(base), np.nan)
    fold_ids = np.full(len(base), -1)
    scores, info, models = [], [], []
    fit_s = pred_s = 0.0
    for fold in splitter.split(base["date"]):
        b_tr, b_va = base.iloc[fold.train_idx], base.iloc[fold.valid_idx]
        y_tr, y_va = y.iloc[fold.train_idx], y.iloc[fold.valid_idx]
        model, (pred,), f_s, p_s = fit_predict(
            model_factory, pipeline, b_tr, y_tr, [b_va], valid=(b_va, y_va)
        )
        score = mae(y_va, pred)
        oof[fold.valid_idx] = pred
        fold_ids[fold.valid_idx] = fold.index
        scores.append(score)
        info.append(_fold_info(fold, score))
        fit_s, pred_s = fit_s + f_s, pred_s + p_s
        if keep_models:
            models.append(model)
        log.info(
            "  fold %d | train %s→%s | valid %s→%s | MAE %.4f",
            fold.index,
            *(d.date() for d in fold.train_dates),
            *(d.date() for d in fold.valid_dates),
            score,
        )
        if on_fold is not None:
            on_fold(fold.index, score)
    return CVResult(
        pd.Series(oof, index=base.index),
        pd.Series(fold_ids, index=base.index),
        scores,
        info,
        models,
        fit_s,
        pred_s,
    )


def _fold_info(fold: Fold, score: float) -> dict[str, Any]:
    return {
        "fold": fold.index,
        "mae": score,
        "n_train": len(fold.train_idx),
        "n_valid": len(fold.valid_idx),
        "train_start": str(fold.train_dates[0].date()),
        "train_end": str(fold.train_dates[1].date()),
        "valid_start": str(fold.valid_dates[0].date()),
        "valid_end": str(fold.valid_dates[1].date()),
    }


def _model_factory(cfg: DictConfig) -> ModelFactory:
    return lambda extra: create_model(cfg.model, seed=int(cfg.seed), extra_params=extra)


def prepare(
    cfg: DictConfig,
) -> tuple[DictConfig, Dataset, FeaturePipeline, pd.DataFrame, pd.DataFrame]:
    """Résout le jeu de features du modèle, charge les données et précalcule les features."""
    model_cls = get_model_class(str(cfg.model.name))
    cfg = resolve_feature_set(cfg, model_cls.feature_set)
    dataset = load_dataset(cfg)
    pipeline = FeaturePipeline.from_config(cfg.features)
    base_train, base_test = pipeline.precompute(
        dataset.x_train, dataset.x_test, path(cfg, "features_cache"), dataset.data_hash
    )
    return cfg, dataset, pipeline, base_train, base_test


def run_experiment(cfg: DictConfig, run_name: str | None = None) -> RunResult:
    """Exécute un run complet et écrit tous ses artefacts."""
    set_global_seed(int(cfg.seed))
    started = datetime.now()
    model_name = str(cfg.model.name)
    run_id = run_name or f"{started:%Y%m%d-%H%M%S}_{model_name}"
    files = RunFiles(path(cfg, "runs") / run_id)
    files.root.mkdir(parents=True, exist_ok=True)
    log.info("Run %s — modèle %s", run_id, model_name)

    cfg, dataset, pipeline, base_train, base_test = prepare(cfg)
    log.info(
        "Features '%s' : %d colonnes sans état", cfg.features.name, len(pipeline.stateless_columns_)
    )
    factory = _model_factory(cfg)
    y = dataset.y_train
    splitter = TemporalSplitter.from_config(cfg.cv)

    # 1. Hold-out final (jamais utilisé pour la sélection) + CV temporelle sur le reste.
    dev_mask, ho_mask = holdout_split(base_train["date"], int(cfg.holdout.weeks))
    base_dev, y_dev = base_train[dev_mask], y[dev_mask]
    log.info(
        "CV temporelle (%s, %d folds) sur %d lignes", cfg.cv.name, splitter.n_splits, len(base_dev)
    )
    cv = cross_validate(factory, pipeline, base_dev, y_dev, splitter)

    # 2. Post-traitement choisi sur l'OOF.
    oof_y, oof_pred = y_dev[cv.mask].to_numpy(), cv.oof[cv.mask].to_numpy()
    post, post_scores = select_postprocessor(
        oof_y,
        oof_pred,
        y_dev.to_numpy(),
        list(cfg.postprocess.candidates),
        tuple(cfg.postprocess.clip_quantiles),
    )
    oof_final = post(oof_pred)
    fold_of = cv.fold_ids[cv.mask].to_numpy()
    for info in cv.fold_info:
        sel = fold_of == info["fold"]
        info["mae_raw"] = info["mae"]
        info["mae"] = mae(oof_y[sel], oof_final[sel])
    cv_scores = [f["mae"] for f in cv.fold_info]
    log.info(
        "CV MAE %.4f ± %.4f (brut %.4f) ; post-traitement retenu : %s",
        np.mean(cv_scores),
        np.std(cv_scores),
        cv.mae_mean,
        post.variant,
    )

    refit = cv.models[0].refit_params(cv.models) if cv.models else {}
    if refit:
        log.info("Paramètres de réentraînement issus de la CV : %s", refit)

    # 3. Hold-out : réentraînement sur tout le dev, évaluation sur les dernières semaines.
    holdout_metrics: dict[str, float] | None = None
    holdout_frame = None
    holdout_model = None
    if ho_mask.any():
        base_ho, y_ho = base_train[ho_mask], y[ho_mask]
        holdout_model, (ho_raw,), _, _ = fit_predict(
            factory, pipeline, base_dev, y_dev, [base_ho], refit
        )
        ho_pred = post(ho_raw)
        holdout_metrics = {**metric_suite(y_ho, ho_pred), "mae_raw": mae(y_ho, ho_raw)}
        holdout_frame = base_ho[CONTEXT_COLS].assign(y=y_ho, pred_raw=ho_raw, pred=ho_pred)
        holdout_frame.to_parquet(files.holdout)
        log.info(
            "Hold-out MAE %.4f (brut %.4f)", holdout_metrics["mae"], holdout_metrics["mae_raw"]
        )
        X_ho_for_importance = pipeline.transform(base_ho)

    # 4. Modèle final sur tout le train → prédictions test.
    final_model, (test_raw,), final_fit_s, final_pred_s = fit_predict(
        factory, pipeline, base_train, y, [base_test], refit
    )
    test_pred = post(test_raw)
    final_model.save(files.model)
    pd.DataFrame({"pred_raw": test_raw, "pred": test_pred}, index=base_test.index).to_parquet(
        files.test
    )

    oof_frame = base_dev[CONTEXT_COLS].assign(
        y=y_dev, pred_raw=cv.oof, pred=np.nan, fold=cv.fold_ids
    )
    oof_frame.loc[cv.mask, "pred"] = oof_final
    oof_frame.to_parquet(files.oof)

    # 5. Métriques, segments, importances.
    oof_valid = oof_frame[cv.mask]
    segments = segment_report(
        (holdout_frame if holdout_frame is not None else oof_valid),
        tuple(cfg.evaluation.delay_classes),
    )
    files.segments.mkdir(exist_ok=True)
    for seg_name, table in segments.items():
        table.to_csv(files.segments / f"{seg_name}.csv")

    importance_model = holdout_model or final_model
    importances = importance_model.feature_importances()
    importance_kind = "native"
    if importances is None and holdout_model is not None:
        importances = permutation_importance(
            holdout_model,
            X_ho_for_importance,
            y[ho_mask],
            int(cfg.evaluation.permutation_sample),
            int(cfg.seed),
        )
        importance_kind = "permutation"
    if importances is not None:
        importances.sort_values(ascending=False).rename("importance").to_csv(
            files.root / "importances.csv"
        )

    reference_rf = _best_holdout(path(cfg, "runs"), "baseline_rf", exclude=run_id)
    headline = holdout_metrics["mae"] if holdout_metrics else post_scores[post.variant]
    metrics: dict[str, Any] = {
        "run_id": run_id,
        "model": model_name,
        "feature_set": str(cfg.features.name),
        "cv_name": str(cfg.cv.name),
        "created_at": started.isoformat(timespec="seconds"),
        "git_commit": git_commit(Path(cfg.paths.root)),
        "data_hash": dataset.data_hash,
        "source_hashes": dataset.source_hashes,
        "sample_frac": dataset.sample_frac,
        "params": {**create_model(cfg.model, int(cfg.seed)).params},
        "refit_params": refit,
        "n_features": len(pipeline.transform(base_test.head(5)).columns),
        "cv": {
            "mae_mean": float(np.mean(cv_scores)),
            "mae_std": float(np.std(cv_scores)),
            "mae_raw_mean": cv.mae_mean,
            "mae_raw_std": cv.mae_std,
            "oof_mae": post_scores[post.variant],
            "oof_suite": metric_suite(oof_y, oof_final),
            "folds": cv.fold_info,
        },
        "holdout": holdout_metrics,
        "postprocess": {
            "variant": post.variant,
            "low": post.low,
            "high": post.high,
            "oof_scores": post_scores,
        },
        "gains": {
            "vs_baseline_zero": relative_gain(headline, float(cfg.reference.baseline_zero_mae)),
            "vs_baseline_rf": relative_gain(headline, reference_rf) if reference_rf else None,
            "baseline_rf_holdout_mae": reference_rf,
        },
        "timings": {
            "cv_fit_seconds": cv.fit_seconds,
            "cv_predict_seconds": cv.predict_seconds,
            "final_fit_seconds": final_fit_s,
            "final_predict_seconds": final_pred_s,
        },
        "importance_kind": importance_kind if importances is not None else None,
        "learning_curves": [getattr(m, "learning_curve_", None) for m in cv.models],
    }

    # 6. Soumission contrôlée.
    submission = None
    if cfg.submission.enabled:
        submission, warnings = write_submission(
            test_pred,
            dataset.x_test,
            path(cfg, "submissions"),
            model_name,
            post_scores[post.variant],
            float(cfg.submission.max_abs_value),
            oof_final,
        )
        metrics["submission"] = {"path": str(submission), "warnings": warnings}
        log.info("Soumission écrite : %s", submission)

    OmegaConf.save(cfg, files.config)
    write_json(files.metrics, metrics)

    # 7. Rapport HTML + suivi MLflow.
    from transilien.reporting.run_report import build_run_report

    build_run_report(
        files,
        cfg,
        final_model=importance_model,
        X_sample=X_ho_for_importance if holdout_model is not None else None,
    )

    tracker = Tracker(path(cfg, "mlruns"), str(cfg.tracking.experiment), bool(cfg.tracking.enabled))
    with tracker.start_run(run_id) as mlflow_run_id:
        tracker.set_tags(
            {
                "model": model_name,
                "git_commit": metrics["git_commit"],
                "data_hash": dataset.data_hash,
                "run_dir": str(files.root),
            }
        )
        tracker.log_params({"config": to_container(cfg)})
        flat = {
            "cv_mae_mean": metrics["cv"]["mae_mean"],
            "cv_mae_std": metrics["cv"]["mae_std"],
            "cv_mae_raw_mean": cv.mae_mean,
            "oof_mae": post_scores[post.variant],
        }
        if holdout_metrics:
            flat.update({f"holdout_{k}": v for k, v in holdout_metrics.items()})
        flat.update({k: v for k, v in metrics["timings"].items()})
        tracker.log_metrics(flat)
        for f in cv.fold_info:
            tracker.log_metrics({"fold_mae": f["mae"]}, step=f["fold"])
        tracker.log_artifacts(files.root, cfg.tracking.get("max_artifact_mb"))
        if submission:
            tracker.log_artifact(submission, "submission")
    if mlflow_run_id:
        metrics["mlflow_run_id"] = mlflow_run_id
        write_json(files.metrics, metrics)

    log.info("Rapport : %s", files.report)
    return RunResult(run_id, files, metrics, mlflow_run_id, submission)


def permutation_importance(
    model: BaseModel, X: pd.DataFrame, y: pd.Series, sample: int, seed: int
) -> pd.Series:
    """Hausse de MAE quand on permute chaque colonne (sur un échantillon du hold-out)."""
    rng = np.random.default_rng(seed)
    if len(X) > sample:
        idx = rng.choice(len(X), sample, replace=False)
        X, y = X.iloc[idx], y.iloc[idx]
    base_score = mae(y, model.predict(X))
    out = {}
    for col in X.columns:
        shuffled = X.copy()
        shuffled[col] = rng.permutation(shuffled[col].to_numpy())
        out[col] = mae(y, model.predict(shuffled)) - base_score
    return pd.Series(out)


def _best_holdout(runs_dir: Path, model: str, exclude: str) -> float | None:
    board = leaderboard(runs_dir)
    if board.empty:
        return None
    rows = board[
        (board["model"] == model) & (board["run_id"] != exclude) & board["sample_frac"].isna()
    ]
    values = rows["holdout_mae"].dropna()
    return float(values.min()) if len(values) else None


def apply_postprocess(pred: np.ndarray, spec: dict[str, Any]) -> np.ndarray:
    return PostProcessor(spec["variant"], spec["low"], spec["high"])(pred)
