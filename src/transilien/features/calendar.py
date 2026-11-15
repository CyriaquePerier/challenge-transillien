"""Calendrier Île-de-France (zone C) : vacances scolaires et jours fériés 2023-2024."""

from __future__ import annotations

import pandas as pd

# Périodes de vacances scolaires zone C (premier et dernier jour sans classe).
VACANCES_ZONE_C = [
    ("2023-02-18", "2023-03-05"),
    ("2023-04-22", "2023-05-08"),
    ("2023-05-17", "2023-05-21"),  # pont de l'Ascension
    ("2023-07-08", "2023-09-03"),
    ("2023-10-21", "2023-11-05"),
    ("2023-12-23", "2024-01-07"),
    ("2024-02-10", "2024-02-25"),
]

JOURS_FERIES = pd.to_datetime(
    [
        "2023-01-01",
        "2023-04-10",
        "2023-05-01",
        "2023-05-08",
        "2023-05-18",
        "2023-05-29",
        "2023-07-14",
        "2023-08-15",
        "2023-11-01",
        "2023-11-11",
        "2023-12-25",
        "2024-01-01",
    ]
)


def is_school_holiday(dates: pd.Series) -> pd.Series:
    flag = pd.Series(False, index=dates.index)
    for start, end in VACANCES_ZONE_C:
        flag |= dates.between(pd.Timestamp(start), pd.Timestamp(end))
    return flag


def days_to_nearest_holiday(dates: pd.Series) -> pd.Series:
    """Distance (en jours, signée) au jour férié le plus proche, bornée à ±7 (ponts)."""
    deltas = (
        (dates.to_numpy()[:, None] - JOURS_FERIES.to_numpy()[None, :])
        .astype("timedelta64[D]")
        .astype(int)
    )
    nearest = deltas[range(len(dates)), abs(deltas).argmin(axis=1)]
    return pd.Series(nearest, index=dates.index).clip(-7, 7)
