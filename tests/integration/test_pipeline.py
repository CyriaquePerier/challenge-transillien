"""Pipeline complet (ingest → soumission) sur un sous-échantillon des vraies données."""

from __future__ import annotations

import pandas as pd
import pytest

from transilien.blending import blend
from transilien.reporting.compare import build_comparison_report
from transilien.runner import run_experiment
from transilien.runs import leaderboard, list_runs

pytestmark = pytest.mark.integration

SMALL = ["data.sample_frac=0.03", "cv.n_splits=3"]


@pytest.fixture(scope="module")
def two_runs(tmp_path_factory):
    from tests.conftest import make_config

    tmp = tmp_path_factory.mktemp("integration")
    zero = run_experiment(make_config(tmp, "model=baseline_zero", *SMALL))
    lgbm = run_experiment(
        make_config(tmp, "model=lightgbm_v1", "model.params.n_estimators=60", *SMALL)
    )
    return tmp, zero, lgbm


def test_run_artifacts(two_runs):
    _, zero, lgbm = two_runs
    for result in (zero, lgbm):
        f = result.files
        for p in (f.metrics, f.report, f.config, f.oof, f.holdout, f.test, f.model):
            assert p.exists(), p
        m = result.metrics
        assert len(m["cv"]["folds"]) == 3
        assert m["holdout"]["mae"] > 0
        assert m["postprocess"]["variant"] in m["postprocess"]["oof_scores"]
        assert "<html" in f.report.read_text()[:200]


def test_submission_is_compliant(two_runs, dataset):
    _, _, lgbm = two_runs
    sub = pd.read_csv(lgbm.submission, index_col=0)
    assert list(sub.columns) == ["p0q0"]
    assert len(sub) == len(dataset.x_test) == 20657
    assert sub.index.equals(dataset.x_test.index)
    assert sub["p0q0"].notna().all()


def test_model_beats_zero_on_holdout(two_runs):
    _, zero, lgbm = two_runs
    assert lgbm.metrics["holdout"]["mae"] < zero.metrics["holdout"]["mae"]


def test_leaderboard_compare_blend(two_runs):
    from tests.conftest import make_config

    tmp, zero, lgbm = two_runs
    cfg = make_config(tmp, *SMALL)
    board = leaderboard(tmp / "runs")
    assert set(board["run_id"]) >= {zero.run_id, lgbm.run_id}

    report = build_comparison_report([zero.files, lgbm.files], tmp / "compare")
    assert report.exists()

    blended = blend(cfg, [zero.files, lgbm.files])
    m = blended.load_metrics()
    assert abs(sum(m["params"]["weights"].values()) - 1) < 1e-6
    assert (
        m["cv"]["oof_mae"]
        <= min(zero.metrics["cv"]["oof_mae"], lgbm.metrics["cv"]["oof_mae"]) + 1e-6
    )
    assert len(list_runs(tmp / "runs")) == 3
