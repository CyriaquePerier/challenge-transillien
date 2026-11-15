"""Contrats de données (pandera) appliqués à l'entrée de la plateforme."""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa

LAG_COLUMNS = ["p2q0", "p3q0", "p4q0", "p0q2", "p0q3", "p0q4"]
CONTEXT_COLUMNS = ["train", "gare", "date", "arret"]
FEATURE_COLUMNS = CONTEXT_COLUMNS + LAG_COLUMNS
TARGET = "p0q0"
# Au-delà d'une journée (1440 min) une valeur est considérée comme corrompue.
MAX_ABS_MINUTES = 1500


def _is_integer_valued(s: pd.Series) -> pd.Series:
    return (s % 1) == 0


def _is_business_day(s: pd.Series) -> pd.Series:
    return s.dt.dayofweek < 5


def features_schema(gares: set[str] | None = None) -> pa.DataFrameSchema:
    """Schéma de ``x_train`` / ``x_test``. ``gares`` : référentiel autorisé (optionnel)."""
    gare_checks = [pa.Check.str_length(min_value=1)]
    if gares is not None:
        gare_checks.append(pa.Check.isin(sorted(gares), error="gare hors référentiel"))
    lag_column = pa.Column(
        float,
        checks=[
            pa.Check(_is_integer_valued, error="valeur passée non entière"),
            pa.Check.in_range(-MAX_ABS_MINUTES, MAX_ABS_MINUTES),
        ],
        nullable=False,
    )
    return pa.DataFrameSchema(
        {
            "train": pa.Column(str, nullable=False),
            "gare": pa.Column(str, checks=gare_checks, nullable=False),
            "date": pa.Column(
                "datetime64[ns]",
                checks=pa.Check(_is_business_day, error="date hors jours ouvrés"),
                nullable=False,
            ),
            "arret": pa.Column(int, checks=pa.Check.ge(0), nullable=False),
            **{col: lag_column for col in LAG_COLUMNS},
        },
        strict=True,
        ordered=True,
        coerce=False,
    )


target_schema = pa.DataFrameSchema(
    {
        TARGET: pa.Column(
            float,
            checks=[
                pa.Check(_is_integer_valued, error="cible non entière"),
                pa.Check.in_range(-MAX_ABS_MINUTES, MAX_ABS_MINUTES),
            ],
            nullable=False,
        )
    },
    strict=True,
)


def validate_dataset(x_train: pd.DataFrame, y_train: pd.DataFrame, x_test: pd.DataFrame) -> None:
    """Valide les trois tables et leur cohérence. Lève ``pandera.errors.SchemaError``."""
    features_schema().validate(x_train)
    target_schema.validate(y_train)
    if len(x_train) != len(y_train) or not x_train.index.equals(y_train.index):
        raise ValueError("x_train et y_train ne sont pas alignés (nombre de lignes ou index).")
    features_schema(gares=set(x_train["gare"].unique())).validate(x_test)
