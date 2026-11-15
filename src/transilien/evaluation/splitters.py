"""Découpages temporels : hold-out final et validation croisée groupée par date."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd
from omegaconf import DictConfig


@dataclass(frozen=True)
class Fold:
    index: int
    train_idx: np.ndarray  # positions (iloc) dans le jeu découpé
    valid_idx: np.ndarray
    train_dates: tuple[pd.Timestamp, pd.Timestamp]
    valid_dates: tuple[pd.Timestamp, pd.Timestamp]


def week_start(dates: pd.Series) -> pd.Series:
    """Lundi de la semaine de chaque date."""
    d = pd.to_datetime(dates)
    return (d - pd.to_timedelta(d.dt.dayofweek, unit="D")).dt.normalize()


def holdout_split(dates: pd.Series, weeks: int) -> tuple[np.ndarray, np.ndarray]:
    """Masques booléens ``(dev, holdout)`` : les ``weeks`` dernières semaines vont au hold-out."""
    weeks_of = week_start(dates)
    unique_weeks = np.sort(weeks_of.unique())
    if weeks <= 0:
        return np.ones(len(dates), dtype=bool), np.zeros(len(dates), dtype=bool)
    if weeks >= len(unique_weeks):
        raise ValueError(f"Hold-out de {weeks} semaines impossible ({len(unique_weeks)} semaines).")
    first_holdout_week = unique_weeks[-weeks]
    holdout = (weeks_of >= first_holdout_week).to_numpy()
    return ~holdout, holdout


class TemporalSplitter:
    """Validation croisée temporelle groupée par semaine.

    Les ``n_splits`` derniers blocs de ``val_weeks`` semaines servent tour à tour de
    validation ; l'entraînement utilise toutes les dates **strictement antérieures**
    (fenêtre croissante) ou seulement les ``max_train_weeks`` dernières (fenêtre glissante).
    ``gap_days`` exclut les jours précédant immédiatement la validation.
    """

    def __init__(
        self,
        n_splits: int = 5,
        val_weeks: int = 1,
        window: str = "expanding",
        max_train_weeks: int | None = None,
        gap_days: int = 0,
    ) -> None:
        if window not in {"expanding", "sliding"}:
            raise ValueError(f"window inconnu : {window}")
        if window == "sliding" and not max_train_weeks:
            raise ValueError("Une fenêtre glissante requiert max_train_weeks.")
        self.n_splits = n_splits
        self.val_weeks = val_weeks
        self.window = window
        self.max_train_weeks = max_train_weeks
        self.gap_days = gap_days

    @classmethod
    def from_config(cls, cv_cfg: DictConfig) -> TemporalSplitter:
        return cls(
            n_splits=int(cv_cfg.n_splits),
            val_weeks=int(cv_cfg.val_weeks),
            window=str(cv_cfg.window),
            max_train_weeks=cv_cfg.get("max_train_weeks"),
            gap_days=int(cv_cfg.get("gap_days") or 0),
        )

    def split(self, dates: pd.Series) -> Iterator[Fold]:
        dates = pd.to_datetime(pd.Series(dates).reset_index(drop=True))
        weeks_of = week_start(dates)
        unique_weeks = np.sort(weeks_of.unique())
        needed = self.n_splits * self.val_weeks + 1
        if len(unique_weeks) < needed:
            raise ValueError(f"{len(unique_weeks)} semaines disponibles, {needed} requises.")

        for k in range(self.n_splits):
            end = len(unique_weeks) - (self.n_splits - 1 - k) * self.val_weeks
            val_weeks = unique_weeks[end - self.val_weeks : end]
            valid_mask = weeks_of.isin(val_weeks).to_numpy()
            first_valid = dates[valid_mask].min()
            train_mask = (dates < first_valid - pd.Timedelta(days=self.gap_days)).to_numpy()
            if self.window == "sliding":
                assert self.max_train_weeks is not None
                start_week = val_weeks[0] - np.timedelta64(7 * self.max_train_weeks, "D")
                train_mask = train_mask & (weeks_of >= start_week).to_numpy()
            train_idx, valid_idx = np.flatnonzero(train_mask), np.flatnonzero(valid_mask)
            yield Fold(
                index=k,
                train_idx=train_idx,
                valid_idx=valid_idx,
                train_dates=(dates[train_mask].min(), dates[train_mask].max()),
                valid_dates=(dates[valid_mask].min(), dates[valid_mask].max()),
            )

    def get_n_splits(self) -> int:
        return self.n_splits
