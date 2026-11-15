import numpy as np
import pandas as pd
import pytest

from transilien.evaluation.metrics import delay_class, mae, metric_suite, relative_gain, segment_mae


def test_mae_known_values():
    assert mae([0, 0, 0], [1, -1, 2]) == pytest.approx(4 / 3)
    assert mae(np.array([1.0, 2.0]), np.array([1.0, 2.0])) == 0.0


def test_metric_suite_known_values():
    m = metric_suite(np.array([0, 0, 0, 0]), np.array([1, 1, -1, 3]))
    assert m["mae"] == pytest.approx(1.5)
    assert m["medae"] == pytest.approx(1.0)
    assert m["bias"] == pytest.approx(1.0)
    assert m["rmse"] == pytest.approx(np.sqrt(3))


def test_relative_gain():
    assert relative_gain(0.5, 1.0) == pytest.approx(0.5)
    assert relative_gain(1.2, 1.0) == pytest.approx(-0.2)


def test_delay_class_bounds():
    labels = delay_class(np.array([0, 1, -2, 5, 6, -100]))
    assert labels.tolist() == ["|y|<=1", "|y|<=1", "1<|y|<=5", "1<|y|<=5", "|y|>5", "|y|>5"]


def test_segment_mae():
    frame = pd.DataFrame({"g": ["a", "a", "b"], "y": [0, 0, 0], "pred": [1, 3, 0]})
    out = segment_mae(frame, "g")
    assert out.loc["a", "mae"] == 2 and out.loc["b", "mae"] == 0
    assert out.index[0] == "a"  # tri par MAE décroissante
