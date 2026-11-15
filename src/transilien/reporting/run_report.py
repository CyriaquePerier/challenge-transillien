"""Rapport HTML autonome d'un run : ``reports/runs/<run_id>/index.html``."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from transilien.config import path
from transilien.models.base import BaseModel
from transilien.reporting import figures as F
from transilien.reporting.html import Page
from transilien.runs import RunFiles, leaderboard
from transilien.utils import get_logger, write_json

log = get_logger("reporting")

MAX_EXACT_SHAP_NODES = 200_000


def _fmt(v: Any, digits: int = 4) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:.{digits}f}" if isinstance(v, float) else str(v)


def _pct(v: float | None) -> str:
    return "—" if v is None else f"{v:+.1%}"


def compute_shap(
    model: BaseModel, X: pd.DataFrame, n: int, seed: int
) -> tuple[np.ndarray, pd.DataFrame] | None:
    """Valeurs SHAP pour les modèles arborescents exposant ``model_`` (extra ``explain``)."""
    estimator = getattr(model, "model_", None)
    if estimator is None:
        return None
    try:
        import shap
    except ImportError:
        return None
    sample = X.sample(min(n, len(X)), random_state=seed)
    # SHAP exact est très coûteux sur les forêts profondes (centaines de milliers de nœuds) :
    # au-delà d'un seuil, on utilise l'approximation de Saabas, bien plus rapide.
    n_nodes = sum(t.tree_.node_count for t in getattr(estimator, "estimators_", []))
    approximate = n_nodes > MAX_EXACT_SHAP_NODES
    try:
        values = shap.TreeExplainer(estimator).shap_values(sample, approximate=approximate)
    except Exception as exc:  # modèle non supporté par TreeExplainer
        log.info("SHAP indisponible pour ce modèle : %s", exc)
        return None
    return np.asarray(values), sample


def build_run_report(
    files: RunFiles,
    cfg: DictConfig,
    final_model: BaseModel | None = None,
    X_sample: pd.DataFrame | None = None,
) -> None:
    m = files.load_metrics()
    holdout = m.get("holdout")
    page = Page(
        title=f"Run {m['run_id']}",
        subtitle=(
            f"Modèle {m['model']} · features {m['feature_set']} · CV {m['cv_name']} · "
            f"commit {m['git_commit'][:12]} · données {m['data_hash']} · {m['created_at']}"
            + (f" · échantillon {m['sample_frac']:.0%}" if m.get("sample_frac") else "")
        ),
    )
    page.kpis(
        [
            ("MAE CV (moyenne ± σ)", f"{m['cv']['mae_mean']:.4f} ± {m['cv']['mae_std']:.4f}"),
            ("MAE OOF post-traitée", f"{m['cv']['oof_mae']:.4f}"),
            ("MAE hold-out", _fmt(holdout and holdout["mae"])),
            ("Gain vs prédire 0", _pct(m["gains"]["vs_baseline_zero"])),
            ("Gain vs baseline_rf", _pct(m["gains"]["vs_baseline_rf"])),
            ("Post-traitement", m["postprocess"]["variant"]),
        ]
    )

    _comparison_with_best(page, files, cfg, m)

    page.h2("Métriques")
    rows = {"OOF (CV)": m["cv"]["oof_suite"]}
    if holdout:
        rows["Hold-out"] = {k: v for k, v in holdout.items() if k != "mae_raw"}
    page.table(pd.DataFrame(rows).T)
    page.table(pd.DataFrame({"MAE OOF": m["postprocess"]["oof_scores"]}).T)
    page.note(
        "Le post-traitement est choisi sur l'OOF uniquement ; le hold-out (4 dernières semaines) "
        "n'intervient dans aucune sélection."
    )
    timings = pd.DataFrame([m["timings"]]).rename(index={0: "secondes"})
    page.table(timings, float_format="{:.1f}")

    page.h2("Validation croisée temporelle")
    page.table(pd.DataFrame(m["cv"]["folds"]).set_index("fold"))
    page.figure(
        "fold_mae",
        F.fold_mae_bars(
            m["cv"]["folds"], holdout and holdout["mae"], float(cfg.reference.baseline_zero_mae)
        ),
    )
    curves = F.learning_curves(m.get("learning_curves") or [])
    if curves is not None:
        page.figure("learning_curves", curves)

    oof = pd.read_parquet(files.oof)
    oof = oof[oof["fold"] >= 0]
    frames = {"OOF": oof}
    main = oof
    if files.holdout.exists():
        main = pd.read_parquet(files.holdout)
        frames["hold-out"] = main
    label = "hold-out" if files.holdout.exists() else "OOF"

    page.h2(f"Erreurs ({label})")
    page.raw('<div class="grid2">')
    page.figure("pred_vs_true", F.pred_vs_true(main["y"].to_numpy(), main["pred"].to_numpy()))
    page.figure("residuals", F.residual_hist(main["y"].to_numpy(), main["pred"].to_numpy()))
    page.raw("</div>")
    page.figure("residuals_time", F.residuals_over_time(frames))
    page.figure("heatmap_gare_jour", F.gare_weekday_heatmap(main))

    page.h2(f"MAE par segment ({label})")
    for seg in ("classe_retard", "jour_semaine", "arret", "semaine"):
        table_path = files.segments / f"{seg}.csv"
        if table_path.exists():
            page.table(pd.read_csv(table_path, index_col=0).T)
    gare_path = files.segments / "gare.csv"
    if gare_path.exists():
        page.note("Gares les moins bien prédites :")
        page.table(pd.read_csv(gare_path, index_col=0).head(15))

    imp_path = files.root / "importances.csv"
    if imp_path.exists():
        page.h2("Importance des features")
        imp = pd.read_csv(imp_path, index_col=0)["importance"]
        page.figure("importances", F.importance_bars(imp, m.get("importance_kind") or ""))
    if final_model is not None and X_sample is not None:
        shap_result = compute_shap(
            final_model, X_sample, int(cfg.evaluation.shap_sample), int(cfg.seed)
        )
        if shap_result is not None:
            values, sample = shap_result
            page.figure("shap", F.shap_summary(values, sample))
            shap_mean = pd.Series(np.abs(values).mean(axis=0), index=sample.columns)
            write_json(
                files.root / "shap_mean_abs.json", shap_mean.sort_values(ascending=False).to_dict()
            )

    page.h2("Hyperparamètres")
    params = {**m["params"], **{f"refit.{k}": v for k, v in (m.get("refit_params") or {}).items()}}
    page.table(pd.DataFrame({"valeur": {k: str(v) for k, v in params.items()}}))
    if m.get("submission"):
        page.h2("Soumission")
        page.raw(f"<p><code>{m['submission']['path']}</code></p>")
        for w in m["submission"]["warnings"]:
            page.note(f"⚠ {w}")

    export = bool(cfg.get("reporting", {}).get("export_png", True))
    page.save(files.report, files.figures if export else None)


def _comparison_with_best(page: Page, files: RunFiles, cfg: DictConfig, m: dict[str, Any]) -> None:
    board = leaderboard(path(cfg, "runs"))
    if board.empty:
        return
    same_data = (
        board["sample_frac"].isna()
        if m.get("sample_frac") is None
        else (board["sample_frac"] == m["sample_frac"])
    )
    others = board[(board["run_id"] != m["run_id"]) & same_data]
    if others.empty:
        return
    best = others.iloc[0]
    page.h2("Comparaison avec le meilleur run du leaderboard")
    current_ho = (m.get("holdout") or {}).get("mae")
    rows = pd.DataFrame(
        {
            "ce run": {
                "run": m["run_id"],
                "MAE CV": m["cv"]["mae_mean"],
                "MAE hold-out": current_ho,
            },
            "meilleur run": {
                "run": best["run_id"],
                "MAE CV": best["cv_mae"],
                "MAE hold-out": best["holdout_mae"],
            },
        }
    ).T
    page.table(rows)
    if current_ho is not None and pd.notna(best["holdout_mae"]):
        delta = current_ho - float(best["holdout_mae"])
        verdict = "meilleur" if delta < 0 else "moins bon"
        page.note(
            f"Écart de MAE hold-out : {delta:+.4f} ({verdict}). "
            f"Significativité : transilien compare {m['run_id']} {best['run_id']}"
        )
