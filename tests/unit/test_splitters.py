import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from transilien.evaluation.splitters import TemporalSplitter, holdout_split, week_start


def _dates(n_days=80, start="2023-04-03"):
    days = pd.bdate_range(start, periods=n_days)
    return pd.Series(np.repeat(days, 3)).astype("datetime64[ns]")


def test_holdout_takes_last_weeks():
    dates = _dates()
    dev, ho = holdout_split(dates, weeks=4)
    assert dev.sum() + ho.sum() == len(dates)
    assert dates[ho].min() > dates[dev].max()
    assert week_start(dates[ho]).nunique() == 4


def test_holdout_zero_weeks():
    dev, ho = holdout_split(_dates(), weeks=0)
    assert dev.all() and not ho.any()


@pytest.mark.parametrize("window,max_weeks", [("expanding", None), ("sliding", 3)])
def test_no_temporal_leakage(window, max_weeks):
    dates = _dates()
    splitter = TemporalSplitter(n_splits=4, window=window, max_train_weeks=max_weeks)
    folds = list(splitter.split(dates))
    assert len(folds) == 4
    for fold in folds:
        train_d, valid_d = dates.iloc[fold.train_idx], dates.iloc[fold.valid_idx]
        # aucune date de validation antérieure ou égale à une date d'entraînement
        assert train_d.max() < valid_d.min()
        # folds groupés par date : aucune date à cheval
        assert not set(train_d) & set(valid_d)
        if window == "sliding":
            assert week_start(train_d).nunique() <= max_weeks


def test_validation_blocks_are_consecutive_and_ordered():
    dates = _dates()
    folds = list(TemporalSplitter(n_splits=3).split(dates))
    starts = [f.valid_dates[0] for f in folds]
    assert starts == sorted(starts)
    # fenêtre croissante : chaque fold voit au moins autant de données que le précédent
    sizes = [len(f.train_idx) for f in folds]
    assert sizes == sorted(sizes)


def test_gap_days():
    dates = _dates()
    for fold in TemporalSplitter(n_splits=2, gap_days=3).split(dates):
        gap = dates.iloc[fold.valid_idx].min() - dates.iloc[fold.train_idx].max()
        assert gap > pd.Timedelta(days=3)


def test_too_few_weeks_raises():
    with pytest.raises(ValueError):
        list(TemporalSplitter(n_splits=50).split(_dates(20)))


@settings(max_examples=40, deadline=None)
@given(
    offsets=st.lists(st.integers(0, 200), min_size=30, max_size=300),
    n_splits=st.integers(1, 4),
)
def test_property_no_leakage(offsets, n_splits):
    dates = pd.Series(pd.Timestamp("2023-04-03") + pd.to_timedelta(offsets, unit="D"))
    if week_start(dates).nunique() < n_splits + 1:
        return
    for fold in TemporalSplitter(n_splits=n_splits).split(dates):
        assert dates.iloc[fold.train_idx].max() < dates.iloc[fold.valid_idx].min()
