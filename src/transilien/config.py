"""Chargement de la configuration Hydra (composition + surcharges CLI)."""

from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf, open_dict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"
DEFAULT_MODEL_CONFIG = "_default"


def _split_override(override: str) -> tuple[str, str] | None:
    if "=" not in override or override.startswith(("+", "~")):
        return None
    key, value = override.split("=", 1)
    return key.strip(), value.strip()


def load_config(overrides: list[str] | None = None, config_dir: Path = CONFIG_DIR) -> DictConfig:
    """Compose ``configs/config.yaml`` avec les surcharges de type Hydra.

    Deux comportements facilitent l'ajout d'un modèle :

    * ``model=<nom>`` fonctionne même sans ``configs/model/<nom>.yaml`` : on utilise alors
      la configuration vide ``_default`` et seul le fichier ``models/<nom>.py`` est requis ;
    * si ``features=...`` n'est pas fourni, le jeu de features déclaré par le modèle
      (attribut ``feature_set``) est appliqué plus tard par :func:`resolve_feature_set`.
    """
    overrides = list(overrides or [])
    model_name: str | None = None
    for i, override in enumerate(overrides):
        parsed = _split_override(override)
        if parsed and parsed[0] == "model":
            model_name = parsed[1]
            if not (config_dir / "model" / f"{model_name}.yaml").exists():
                overrides[i] = f"model={DEFAULT_MODEL_CONFIG}"

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name="config", overrides=overrides)

    with open_dict(cfg):
        if model_name is not None:
            cfg.model.name = model_name
        cfg.features_overridden = any(
            (p := _split_override(o)) is not None and p[0] == "features" for o in overrides
        )
        if not Path(cfg.paths.root).is_absolute():
            cfg.paths.root = str((PROJECT_ROOT / cfg.paths.root).resolve())
    return cfg


def resolve_feature_set(cfg: DictConfig, model_feature_set: str | None) -> DictConfig:
    """Applique le ``feature_set`` du modèle sauf si l'utilisateur l'a surchargé en CLI."""
    if cfg.get("features_overridden") or not model_feature_set:
        return cfg
    if model_feature_set == cfg.features.name:
        return cfg
    path = Path(CONFIG_DIR) / "features" / f"{model_feature_set}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Jeu de features inconnu : {model_feature_set} ({path})")
    with open_dict(cfg):
        cfg.features = OmegaConf.load(path)
    return cfg


def path(cfg: DictConfig, key: str) -> Path:
    """Chemin absolu d'une entrée de ``cfg.paths`` (relatif à ``paths.root``)."""
    p = Path(cfg.paths[key])
    return p if p.is_absolute() else Path(cfg.paths.root) / p


def to_container(cfg: DictConfig) -> dict:
    container = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(container, dict)
    return container
