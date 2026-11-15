"""Utilitaires transverses : graines, hachage, git, journalisation."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

LOGGER_NAME = "transilien"


def get_logger(name: str | None = None) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME if name is None else f"{LOGGER_NAME}.{name}")
    root = logging.getLogger(LOGGER_NAME)
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")
        )
        root.addHandler(handler)
        root.setLevel(os.environ.get("TRANSILIEN_LOG_LEVEL", "INFO"))
        root.propagate = False
    return logger


def set_global_seed(seed: int) -> None:
    """Fixe les graines de ``random``, ``numpy`` et de ``PYTHONHASHSEED``."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(obj: Any, length: int = 12) -> str:
    """Hash déterministe d'un objet sérialisable en JSON (clés triées)."""
    payload = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()[:length]


def git_commit(root: Path) -> str:
    """Hash du commit courant (suffixé ``-dirty`` si l'arbre de travail est modifié)."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()
        return f"{sha}-dirty" if dirty else sha
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=_json_default))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)
