import numpy as np
import pandas as pd
import pytest

from transilien.features.base import build_transformer
from transilien.features.calendar import days_to_nearest_holiday, is_school_holiday
from transilien.features.pipeline import FeaturePipeline

SPEC = [
    {"name": "raw_lags"},
    {"name": "structural"},
    {"name": "temporal"},
    {"name": "lag_stats"},
    {
        "name": "target_encoding",
        "params": {"keys": [["gare"], ["gare", "jour_semaine"]], "stats": ["mean", "median"]},
    },
]


@pytest.mark.parametrize("name", ["raw_lags", "structural", "temporal", "lag_stats"])
def test_stateless_shape_and_no_nan(name, toy_frame):
    x, _ = toy_frame
    out = build_transformer(name).transform(x)
    assert len(out) == len(x) and out.index.equals(x.index)
    assert not out.isna().any().any()
    assert all(pd.api.types.is_numeric_dtype(out[c]) for c in out.columns)


def test_pipeline_end_to_end(toy_frame, tmp_path):
    x, y = toy_frame
    train, test = x.iloc[:900], x.iloc[900:]
    pipe = FeaturePipeline(SPEC, name="t")
    b_tr, b_te = pipe.precompute(train, test, tmp_path, "hash")
    pipe.fit(b_tr, y.iloc[:900])
    X_tr, X_te = pipe.transform(b_tr), pipe.transform(b_te)
    assert X_tr.shape[1] == X_te.shape[1]
    assert list(X_tr.columns) == list(X_te.columns)
    assert not X_tr.isna().any().any() and not X_te.isna().any().any()
    assert {"train", "gare", "date"}.isdisjoint(X_tr.columns)


def test_cache_is_reused(toy_frame, tmp_path):
    x, _ = toy_frame
    pipe = FeaturePipeline(SPEC[:4], name="t")
    pipe.precompute(x.iloc[:900], x.iloc[900:], tmp_path, "hash")
    files = list(tmp_path.glob("*.parquet"))
    assert len(files) == 1
    mtime = files[0].stat().st_mtime_ns
    pipe2 = FeaturePipeline(SPEC[:4], name="t")
    pipe2.precompute(x.iloc[:900], x.iloc[900:], tmp_path, "hash")
    assert files[0].stat().st_mtime_ns == mtime
    # une autre empreinte de données invalide le cache
    pipe2.precompute(x.iloc[:900], x.iloc[900:], tmp_path, "autre")
    assert len(list(tmp_path.glob("*.parquet"))) == 2


def test_target_encoding_uses_only_fit_rows(toy_frame):
    """Changer la cible hors du fold d'entraînement ne doit pas modifier l'encodage."""
    x, y = toy_frame
    x = x.assign(jour_semaine=x["date"].dt.dayofweek)
    te = build_transformer("target_encoding", {"keys": [["gare"]], "stats": ["mean"]})
    fit_idx = x.index[:600]
    ref = te.fit(x.loc[fit_idx], y.loc[fit_idx]).transform(x)
    y_alt = y.copy()
    y_alt.loc[x.index[600:]] = 1000.0
    alt = te.fit(x.loc[fit_idx], y_alt.loc[fit_idx]).transform(x)
    pd.testing.assert_frame_equal(ref, alt)


def test_target_encoding_unknown_group_falls_back_to_global(toy_frame):
    x, y = toy_frame
    te = build_transformer("target_encoding", {"keys": [["gare"]], "stats": ["mean"]}).fit(x, y)
    unseen = x.head(3).assign(gare="ZZZ")
    assert np.allclose(te.transform(unseen)["te_gare_mean"], y.mean())


def test_calendar():
    d = pd.Series(pd.to_datetime(["2023-04-24", "2023-06-12", "2023-05-17", "2023-11-02"]))
    assert is_school_holiday(d).tolist() == [True, False, True, True]
    assert days_to_nearest_holiday(d).tolist()[2] == -1  # veille de l'Ascension
