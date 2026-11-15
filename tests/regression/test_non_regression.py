"""Non-régression : la MAE hold-out de baseline_rf doit rester stable (données complètes).

Lancé en CI sur la branche principale : ``uv run pytest -m regression``.
Après un changement *volontaire* de la validation ou des features ``raw``, mettre à jour
``reference.json`` avec la nouvelle valeur.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from transilien.runner import run_experiment

REFERENCE = json.loads((Path(__file__).parent / "reference.json").read_text())


@pytest.mark.regression
def test_baseline_rf_holdout_mae(cfg_factory):
    cfg = cfg_factory("model=baseline_rf", "submission.enabled=false")
    result = run_experiment(cfg)
    ref = REFERENCE["baseline_rf"]
    assert result.metrics["holdout"]["mae"] == pytest.approx(
        ref["holdout_mae"], abs=ref["tolerance"]
    )
