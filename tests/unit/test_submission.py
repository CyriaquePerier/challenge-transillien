import numpy as np
import pandas as pd
import pytest

from transilien.config import load_config, path
from transilien.submission.checks import SubmissionError, check_submission
from transilien.submission.writer import build_submission, write_submission


@pytest.fixture
def x_test():
    return pd.DataFrame({"a": range(10)}, index=pd.RangeIndex(10, name="id"))


def test_valid_submission(x_test):
    sub = build_submission(np.zeros(10), x_test)
    assert check_submission(sub, x_test, reference_pred=np.zeros(10)) == []
    assert sub["p0q0"].dtype == "int64"  # prédictions entières écrites comme entiers


@pytest.mark.parametrize(
    "pred",
    [np.zeros(9), np.r_[np.zeros(9), np.nan], np.r_[np.zeros(9), np.inf], np.r_[np.zeros(9), 1e4]],
)
def test_invalid_submission(pred, x_test):
    with pytest.raises(SubmissionError):
        sub = build_submission(pred, x_test.iloc[: len(pred)])
        check_submission(sub, x_test)


def test_wrong_index(x_test):
    sub = build_submission(np.zeros(10), x_test)
    sub.index = sub.index[::-1]
    with pytest.raises(SubmissionError):
        check_submission(sub, x_test)


def test_distribution_shift_is_blocking(x_test):
    sub = build_submission(np.full(10, 50.0), x_test)
    with pytest.raises(SubmissionError):
        check_submission(sub, x_test, reference_pred=np.random.default_rng(0).normal(0, 1, 100))


def test_format_matches_official_sample(dataset, tmp_path):
    out, _ = write_submission(np.zeros(len(dataset.x_test)), dataset.x_test, tmp_path, "m", 0.5)
    sample = path(load_config(), "data_lake") / "y_sample_final.csv"
    assert out.read_text().splitlines()[0] == sample.read_text().splitlines()[0]
    written, official = pd.read_csv(out, index_col=0), pd.read_csv(sample, index_col=0)
    assert written.index.equals(official.index) and list(written.columns) == list(official.columns)
    assert out.name.endswith("_m_cv0.500.csv")
