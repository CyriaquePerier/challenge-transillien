import pytest

from transilien.config import load_config, resolve_feature_set
from transilien.models.base import BaseModel
from transilien.models.registry import MODEL_REGISTRY, discover_models, register


def test_default_config():
    cfg = load_config()
    assert cfg.model.name == "baseline_rf" and cfg.features.name == "base"


def test_model_without_yaml_uses_default(monkeypatch):
    cfg = load_config(["model=modele_sans_yaml"])
    assert cfg.model.name == "modele_sans_yaml"
    assert dict(cfg.model.params) == {}


def test_cli_override():
    cfg = load_config(["model=lightgbm_v1", "model.params.num_leaves=7", "cv.n_splits=2"])
    assert cfg.model.params.num_leaves == 7 and cfg.cv.n_splits == 2


def test_feature_set_resolution():
    cfg = resolve_feature_set(load_config(), "lags_enrichis")
    assert cfg.features.name == "lags_enrichis"
    forced = resolve_feature_set(load_config(["features=raw"]), "lags_enrichis")
    assert forced.features.name == "raw"  # la CLI l'emporte sur le modèle


def test_discovery_finds_platform_models():
    models = discover_models()
    assert {"baseline_zero", "baseline_rf", "lightgbm_v1"} <= set(models)


def test_register_rejects_invalid():
    class NoName(BaseModel):
        def fit(self, X, y, X_valid=None, y_valid=None):
            return self

        def predict(self, X):
            return X

    with pytest.raises(ValueError):
        register(NoName)
    with pytest.raises(TypeError):
        register(object)  # type: ignore[arg-type]
    assert "NoName" not in MODEL_REGISTRY
