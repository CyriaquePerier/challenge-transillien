"""Registre des modèles : ``@register`` + découverte automatique du dossier ``models/``."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from transilien.config import PROJECT_ROOT
from transilien.models.base import BaseModel

MODEL_REGISTRY: dict[str, type[BaseModel]] = {}
USER_MODELS_DIR = PROJECT_ROOT / "models"
_MODULE_PREFIX = "transilien_user_models"


def register(cls: type[BaseModel]) -> type[BaseModel]:
    """Décorateur rendant un modèle visible par la CLI, les tests et le tuning."""
    if not isinstance(cls, type) or not issubclass(cls, BaseModel):
        raise TypeError("@register s'applique à une sous-classe de BaseModel.")
    name = getattr(cls, "name", None)
    if not name:
        raise ValueError(f"{cls.__name__} doit définir un attribut de classe 'name'.")
    existing = MODEL_REGISTRY.get(name)
    if existing is not None and existing.__qualname__ != cls.__qualname__:
        raise ValueError(f"Nom de modèle déjà utilisé : {name} ({existing.__qualname__})")
    MODEL_REGISTRY[name] = cls
    return cls


def discover_models(models_dir: Path = USER_MODELS_DIR) -> dict[str, type[BaseModel]]:
    """Importe chaque ``models/*.py`` (hors fichiers préfixés par ``_``) une seule fois."""
    if _MODULE_PREFIX not in sys.modules:
        # Paquet virtuel : permet à pickle/joblib de retrouver les classes des modèles.
        package = types.ModuleType(_MODULE_PREFIX)
        package.__path__ = [str(models_dir)]
        sys.modules[_MODULE_PREFIX] = package
    for file in sorted(models_dir.glob("*.py")):
        if file.name.startswith("_"):
            continue
        module_name = f"{_MODULE_PREFIX}.{file.stem}"
        if module_name in sys.modules:
            continue
        spec = importlib.util.spec_from_file_location(module_name, file)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            del sys.modules[module_name]
            raise
    return dict(MODEL_REGISTRY)


def get_model_class(name: str) -> type[BaseModel]:
    discover_models()
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Modèle inconnu : {name}. Disponibles : {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name]


def create_model(
    model_cfg: DictConfig, seed: int, extra_params: dict[str, Any] | None = None
) -> BaseModel:
    cls = get_model_class(str(model_cfg.name))
    raw = model_cfg.get("params")
    params = OmegaConf.to_container(raw, resolve=True) if raw is not None else {}
    assert isinstance(params, dict)
    return cls(params={**params, **(extra_params or {})}, seed=seed)  # type: ignore[dict-item]
