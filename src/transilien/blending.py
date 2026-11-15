"""Ensemble pondéré optimisé sur les prédictions out-of-fold de runs existants."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from scipy.optimize import minimize

from transilien.config import path
from transilien.data.loaders import load_dataset
from transilien.evaluation.metrics import mae, metric_suite, relative_gain
from transilien.evaluation.postprocess import select_postprocessor
from transilien.reporting.html import Page
from transilien.runs import RunFiles
from transilien.submission.writer import write_submission
from transilien.utils import get_logger, write_json

log = get_logger("blending")


def optimize_weights(preds: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Poids positifs de somme 1 minimisant la MAE de ``preds @ w`` (``preds`` : n × k)."""
    k = preds.shape[1]
    result = minimize(
        lambda w: mae(y, preds @ w),
        x0=np.full(k, 1 / k),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * k,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
        options={"maxiter": 500, "ftol": 1e-9},
    )
    weights = np.clip(result.x, 0, None)
    return weights / weights.sum()


def _stack(runs: list[RunFiles], file: str) -> tuple[pd.DataFrame, np.ndarray]:
    frames = [pd.read_parquet(getattr(r, file)) for r in runs]
    ref = frames[0]
    for r, f in zip(runs, frames, strict=True):
        if not f.index.equals(ref.index):
            raise ValueError(
                f"{file} de {r.run_id} non aligné : runs issus de données/CV différentes."
            )
    return ref, np.column_stack([f["pred_raw"].to_numpy() for f in frames])


def blend(cfg: DictConfig, runs: list[RunFiles]) -> RunFiles:
    if len(runs) < 2:
        raise ValueError("Le blending nécessite au moins deux runs.")
    metrics = [r.load_metrics() for r in runs]
    if len({m["cv_name"] for m in metrics}) > 1 or len({m["data_hash"] for m in metrics}) > 1:
        raise ValueError("Les runs doivent partager la même CV et les mêmes données.")

    oof, oof_preds = _stack(runs, "oof")
    valid = (oof["fold"] >= 0).to_numpy()
    y_oof = oof.loc[valid, "y"].to_numpy()
    weights = optimize_weights(oof_preds[valid], y_oof)
    oof_blend = oof_preds @ weights
    post, post_scores = select_postprocessor(
        y_oof,
        oof_blend[valid],
        oof["y"].to_numpy(),
        list(cfg.postprocess.candidates),
        tuple(cfg.postprocess.clip_quantiles),
    )
    folds: list[dict[str, Any]] = []
    fold_scores: list[float] = []
    for _, part in oof[valid].assign(pred=post(oof_blend[valid])).groupby("fold"):
        fold_scores.append(mae(part["y"], part["pred"]))
        folds.append(
            {
                "fold": int(part["fold"].iloc[0]),
                "mae": fold_scores[-1],
                "n_valid": len(part),
                "valid_start": str(part["date"].min().date()),
                "valid_end": str(part["date"].max().date()),
            }
        )

    run_id = f"{datetime.now():%Y%m%d-%H%M%S}_blend"
    files = RunFiles(path(cfg, "runs") / run_id)
    files.root.mkdir(parents=True)
    oof.assign(pred_raw=oof_blend, pred=np.where(valid, post(oof_blend), np.nan)).to_parquet(
        files.oof
    )

    holdout_metrics = None
    if all(r.holdout.exists() for r in runs):
        ho, ho_preds = _stack(runs, "holdout")
        ho_raw = ho_preds @ weights
        ho_pred = post(ho_raw)
        holdout_metrics = {**metric_suite(ho["y"], ho_pred), "mae_raw": mae(ho["y"], ho_raw)}
        ho.assign(pred_raw=ho_raw, pred=ho_pred).to_parquet(files.holdout)

    test, test_preds = _stack(runs, "test")
    test_pred = post(test_preds @ weights)
    test.assign(pred_raw=test_preds @ weights, pred=test_pred).to_parquet(files.test)

    weight_map = {r.run_id: float(w) for r, w in zip(runs, weights, strict=True)}
    log.info("Poids optimisés sur l'OOF : %s", {k: round(v, 3) for k, v in weight_map.items()})
    headline = holdout_metrics["mae"] if holdout_metrics else post_scores[post.variant]
    out: dict[str, Any] = {
        "run_id": run_id,
        "model": "blend",
        "feature_set": "—",
        "cv_name": metrics[0]["cv_name"],
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "git_commit": metrics[0]["git_commit"],
        "data_hash": metrics[0]["data_hash"],
        "sample_frac": metrics[0].get("sample_frac"),
        "params": {"weights": weight_map},
        "blend_of": list(weight_map),
        "cv": {
            "mae_mean": float(np.mean(fold_scores)),
            "mae_std": float(np.std(fold_scores)),
            "oof_mae": post_scores[post.variant],
            "folds": folds,
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
            "vs_baseline_rf": None,
        },
    }
    if cfg.submission.enabled:
        x_test = load_dataset(cfg).x_test
        submission, warnings = write_submission(
            test_pred,
            x_test,
            path(cfg, "submissions"),
            "blend",
            post_scores[post.variant],
            float(cfg.submission.max_abs_value),
            post(oof_blend[valid]),
        )
        out["submission"] = {"path": str(submission), "warnings": warnings}
        log.info("Soumission écrite : %s", submission)
    write_json(files.metrics, out)
    _blend_report(files, out, metrics)
    return files


def _blend_report(files: RunFiles, m: dict[str, Any], sources: list[dict[str, Any]]) -> Path:
    page = Page(f"Blend {m['run_id']}", "Ensemble pondéré optimisé sur les prédictions OOF")
    page.kpis(
        [
            ("MAE CV", f"{m['cv']['mae_mean']:.4f} ± {m['cv']['mae_std']:.4f}"),
            ("MAE hold-out", f"{m['holdout']['mae']:.4f}" if m["holdout"] else "—"),
            ("Post-traitement", m["postprocess"]["variant"]),
        ]
    )
    page.h2("Poids et runs sources")
    page.table(
        pd.DataFrame(
            {
                s["run_id"]: {
                    "modèle": s["model"],
                    "poids": m["params"]["weights"][s["run_id"]],
                    "MAE CV": s["cv"]["mae_mean"],
                    "MAE hold-out": (s.get("holdout") or {}).get("mae"),
                }
                for s in sources
            }
        ).T
    )
    return page.save(files.report)
