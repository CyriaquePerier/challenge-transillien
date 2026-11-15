"""Tests génériques appliqués à CHAQUE modèle de ``models/`` : aucun test à écrire par modèle."""

from __future__ import annotations

import numpy as np
import pytest

from transilien.config import load_config, resolve_feature_set
from transilien.features.pipeline import FeaturePipeline
from transilien.models.base import BaseModel
from transilien.models.registry import create_model, discover_models

MODELS = sorted(discover_models())
# Limite le nombre d'itérations pour garder les tests rapides (les autres paramètres
# viennent bien de la configuration du modèle).
FAST = {"n_estimators": 30, "iterations": 30}


def _matrices(name, sample):
    x, y = sample
    cfg = load_config([f"model={name}"])
    cfg = resolve_feature_set(cfg, discover_models()[name].feature_set)
    pipe = FeaturePipeline.from_config(cfg.features)
    tr, va = x.iloc[:4000], x.iloc[4000:]
    b_tr, b_va = pipe.precompute(tr, va)
    pipe.fit(b_tr, y.iloc[:4000])
    return cfg, pipe.transform(b_tr), y.iloc[:4000], pipe.transform(b_va), y.iloc[4000:]


def _make(cfg) -> BaseModel:
    model = create_model(cfg.model, seed=0)
    fast = {k: v for k, v in FAST.items() if k in model.params}
    return create_model(cfg.model, seed=0, extra_params=fast)


@pytest.fixture(scope="module", params=MODELS)
def fitted(request, sample):
    cfg, X, y, Xv, yv = _matrices(request.param, sample)
    model = _make(cfg).fit(X, y, Xv, yv)
    return cfg, model, X, y, Xv, yv


def test_instantiates_from_config(fitted):
    cfg, model, *_ = fitted
    assert isinstance(model, BaseModel)
    assert model.name == cfg.model.name


def test_predict_shape_and_finite(fitted):
    _, model, _, _, Xv, _ = fitted
    pred = np.asarray(model.predict(Xv))
    assert pred.shape == (len(Xv),)
    assert np.issubdtype(pred.dtype, np.number)
    assert np.isfinite(pred).all()


def test_fit_without_validation_set(fitted):
    cfg, _, X, y, Xv, _ = fitted
    pred = _make(cfg).fit(X, y).predict(Xv)
    assert len(pred) == len(Xv) and np.isfinite(pred).all()


def test_deterministic_with_fixed_seed(fitted):
    cfg, model, X, y, Xv, yv = fitted
    again = _make(cfg).fit(X, y, Xv, yv)
    np.testing.assert_allclose(model.predict(Xv), again.predict(Xv), rtol=1e-6, atol=1e-6)


def test_serialization_roundtrip(fitted, tmp_path):
    _, model, _, _, Xv, _ = fitted
    model.save(tmp_path / "m.joblib")
    loaded = BaseModel.load(tmp_path / "m.joblib")
    np.testing.assert_allclose(model.predict(Xv), loaded.predict(Xv), rtol=1e-12, atol=1e-12)


def test_optional_hooks(fitted):
    _, model, X, *_ = fitted
    imp = model.feature_importances()
    if imp is not None:
        assert len(imp) == X.shape[1]
    assert isinstance(model.refit_params([model]), dict)
    if model.has_search_space():
        params = model.search_space(_RecordingTrial())
        assert isinstance(params, dict) and params


class _RecordingTrial:
    """Faux essai Optuna renvoyant la borne basse de chaque distribution."""

    def suggest_int(self, name, low, high, **_):
        return low

    def suggest_float(self, name, low, high, **_):
        return low

    def suggest_categorical(self, name, choices):
        return choices[0]
