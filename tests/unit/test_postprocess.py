import numpy as np
import pytest

from transilien.evaluation.postprocess import PostProcessor, select_postprocessor


def test_variants():
    pred = np.array([-10.4, 0.4, 0.6, 9.0])
    assert PostProcessor("raw")(pred).tolist() == pred.tolist()
    assert PostProcessor("round")(pred).tolist() == [-10, 0, 1, 9]
    assert PostProcessor("clip", -5, 5)(pred).tolist() == [-5, 0.4, 0.6, 5]
    assert PostProcessor("clip_round", -5, 5)(pred).tolist() == [-5, 0, 1, 5]


def test_unknown_variant():
    with pytest.raises(ValueError):
        PostProcessor("floor")


def test_selection_prefers_rounding_for_integer_target():
    rng = np.random.default_rng(0)
    y = rng.integers(-3, 4, 2000).astype(float)
    pred = y + rng.uniform(-0.3, 0.3, 2000)
    post, scores = select_postprocessor(y, pred, y, ["raw", "round"], (0.001, 0.999))
    assert post.variant == "round"
    assert scores["round"] == 0.0


def test_selection_tie_keeps_simplest():
    y = np.zeros(10)
    post, _ = select_postprocessor(y, y, y, ["clip_round", "round", "raw"], (0.0, 1.0))
    assert post.variant == "raw"
