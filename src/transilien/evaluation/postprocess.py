"""Post-traitement des prédictions : arrondi à l'entier et écrêtage appris sur le train."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from transilien.evaluation.metrics import mae

VARIANTS = ("raw", "round", "clip", "clip_round")


@dataclass(frozen=True)
class PostProcessor:
    variant: str = "raw"
    low: float = -np.inf
    high: float = np.inf

    def __post_init__(self) -> None:
        if self.variant not in VARIANTS:
            raise ValueError(f"Variante inconnue : {self.variant}. Disponibles : {VARIANTS}")

    @classmethod
    def fit(
        cls, variant: str, y_train: np.ndarray, quantiles: tuple[float, float]
    ) -> PostProcessor:
        low, high = np.quantile(np.asarray(y_train, dtype=float), quantiles)
        return cls(variant=variant, low=float(low), high=float(high))

    def __call__(self, pred: np.ndarray) -> np.ndarray:
        out = np.asarray(pred, dtype=float)
        if self.variant in ("clip", "clip_round"):
            out = np.clip(out, self.low, self.high)
        if self.variant in ("round", "clip_round"):
            out = np.round(out)
        return out


def select_postprocessor(
    y_true: np.ndarray,
    oof_pred: np.ndarray,
    y_train: np.ndarray,
    candidates: list[str],
    quantiles: tuple[float, float],
) -> tuple[PostProcessor, dict[str, float]]:
    """Choisit la variante minimisant la MAE **out-of-fold** (jamais sur le hold-out).

    En cas d'égalité, la variante la plus simple (ordre de ``VARIANTS``) l'emporte.
    """
    scores: dict[str, float] = {}
    processors: dict[str, PostProcessor] = {}
    for variant in sorted(candidates, key=VARIANTS.index):
        processors[variant] = PostProcessor.fit(variant, y_train, quantiles)
        scores[variant] = mae(y_true, processors[variant](oof_pred))
    best = min(scores, key=lambda v: (scores[v], VARIANTS.index(v)))
    return processors[best], scores
