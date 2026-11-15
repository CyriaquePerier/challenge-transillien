"""Comparaison de runs : métriques, segments et significativité des écarts sur le hold-out."""

from __future__ import annotations

from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from transilien.evaluation.metrics import segment_report
from transilien.reporting import figures as F
from transilien.reporting.html import Page
from transilien.runs import RunFiles


def paired_significance(
    y: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray, n_boot: int = 2000, seed: int = 0
) -> dict[str, float]:
    """Test de Wilcoxon apparié et IC bootstrap (95 %) de ``MAE(a) − MAE(b)``."""
    err_a, err_b = np.abs(pred_a - y), np.abs(pred_b - y)
    diff = err_a - err_b
    if np.allclose(diff, 0):
        p_value = 1.0
    else:
        p_value = float(stats.wilcoxon(err_a, err_b, zero_method="zsplit").pvalue)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    boots = diff[idx].mean(axis=1)
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return {
        "delta_mae": float(diff.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "wilcoxon_p": p_value,
    }


def _aligned_holdout(runs: list[RunFiles]) -> dict[str, pd.DataFrame]:
    frames = {}
    for r in runs:
        if not r.holdout.exists():
            raise FileNotFoundError(f"Le run {r.run_id} n'a pas de prédictions hold-out.")
        frames[r.run_id] = pd.read_parquet(r.holdout)
    ref = next(iter(frames.values()))
    for run_id, f in frames.items():
        if not f.index.equals(ref.index):
            raise ValueError(
                f"Hold-out non comparable ({run_id}) : données ou hold-out différents."
            )
    return frames


def build_comparison_report(runs: list[RunFiles], out_dir: Path) -> Path:
    frames = _aligned_holdout(runs)
    metrics = {r.run_id: r.load_metrics() for r in runs}
    page = Page(
        "Comparaison de runs", " · ".join(f"{k} ({m['model']})" for k, m in metrics.items())
    )

    page.h2("Métriques")
    summary = pd.DataFrame(
        {
            run_id: {
                "modèle": m["model"],
                "MAE CV": m["cv"]["mae_mean"],
                "σ CV": m["cv"]["mae_std"],
                **{f"hold-out {k}": v for k, v in (m.get("holdout") or {}).items()},
            }
            for run_id, m in metrics.items()
        }
    ).T
    page.table(summary)

    page.h2("Significativité des écarts (hold-out, erreurs appariées)")
    rows = []
    for a, b in combinations(frames, 2):
        y = frames[a]["y"].to_numpy()
        res = paired_significance(y, frames[a]["pred"].to_numpy(), frames[b]["pred"].to_numpy())
        rows.append(
            {
                "A": a,
                "B": b,
                **res,
                "significatif (5 %)": res["wilcoxon_p"] < 0.05
                and not (res["ci_low"] <= 0 <= res["ci_high"]),
            }
        )
    page.table(pd.DataFrame(rows), index=False)
    page.note(
        "delta_mae = MAE(A) − MAE(B) : négatif ⇒ A meilleur. IC bootstrap à 95 % ; "
        "un écart est jugé réel si le test de Wilcoxon est significatif et l'IC exclut 0."
    )

    page.h2("MAE par segment")
    segs = {run_id: segment_report(f) for run_id, f in frames.items()}
    for seg in ("classe_retard", "jour_semaine", "arret", "semaine"):
        page.figure(f"seg_{seg}", F.segment_bars({k: v[seg] for k, v in segs.items()}, seg))
    gare = pd.DataFrame({k: v["gare"]["mae"] for k, v in segs.items()})
    gare = gare.assign(ecart=gare.max(axis=1) - gare.min(axis=1)).sort_values(
        "ecart", ascending=False
    )
    page.note("Gares où les runs diffèrent le plus :")
    page.table(gare.head(20))

    page.figure("residuals_time", F.residuals_over_time(frames))
    out = out_dir / f"{datetime.now():%Y%m%d-%H%M%S}" / "index.html"
    return page.save(out)
