"""Transformers de features fournis par la plateforme.

Pour ajouter une famille de features : écrire une classe héritant de ``FeatureTransformer``,
la décorer avec ``@register_feature`` puis la référencer dans ``configs/features/*.yaml``.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd

from transilien.data.schemas import LAG_COLUMNS, TARGET
from transilien.features.base import FeatureTransformer, register_feature
from transilien.features.calendar import days_to_nearest_holiday, is_school_holiday

PREV_TRAINS = ["p2q0", "p3q0", "p4q0"]  # trains précédents, même gare
PREV_STATIONS = ["p0q2", "p0q3", "p0q4"]  # même train, gares précédentes


@register_feature
class RawLags(FeatureTransformer):
    """Les six valeurs passées telles quelles."""

    name = "raw_lags"

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[LAG_COLUMNS].astype("float32")


@register_feature
class Structural(FeatureTransformer):
    """Numéro d'arrêt et encodage de la gare (ordinal sur le référentiel trié)."""

    name = "structural"

    def __init__(self, encode_gare: str = "ordinal") -> None:
        self.encode_gare = encode_gare

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame({"arret": X["arret"].astype("int32")}, index=X.index)
        if self.encode_gare == "ordinal":
            codes = {g: i for i, g in enumerate(sorted(X["gare"].unique()))}
            out["gare_code"] = X["gare"].map(codes).astype("int32")
        elif self.encode_gare != "none":
            raise ValueError(f"encode_gare inconnu : {self.encode_gare}")
        return out


@register_feature
class Temporal(FeatureTransformer):
    """Calendrier : jour de semaine, semaine ISO, mois, vacances scolaires, jours fériés.

    ``columns`` permet de n'en garder qu'une partie : ``semaine`` et ``mois`` ne prennent
    en test (mi-novembre → décembre) que des valeurs jamais vues à l'entraînement.
    """

    name = "temporal"
    ALL_COLUMNS: ClassVar[list[str]] = [
        "jour_semaine",
        "semaine",
        "mois",
        "vacances",
        "delta_ferie",
    ]

    def __init__(self, columns: list[str] | None = None) -> None:
        self.columns = columns

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        d = X["date"]
        out = pd.DataFrame(
            {
                "jour_semaine": d.dt.dayofweek.astype("int8"),
                "semaine": d.dt.isocalendar().week.astype("int8"),
                "mois": d.dt.month.astype("int8"),
                "vacances": is_school_holiday(d).astype("int8"),
                "delta_ferie": days_to_nearest_holiday(d).astype("int8"),
            },
            index=X.index,
        )
        return out[list(self.columns or self.ALL_COLUMNS)]


@register_feature
class LagStats(FeatureTransformer):
    """Agrégats et tendances calculés sur les retards passés."""

    name = "lag_stats"

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = {}
        for prefix, cols in (
            ("trains", PREV_TRAINS),
            ("gares", PREV_STATIONS),
            ("all", LAG_COLUMNS),
        ):
            block = X[cols]
            out[f"{prefix}_mean"] = block.mean(axis=1)
            out[f"{prefix}_median"] = block.median(axis=1)
            out[f"{prefix}_std"] = block.std(axis=1)
            out[f"{prefix}_min"] = block.min(axis=1)
            out[f"{prefix}_max"] = block.max(axis=1)
            out[f"{prefix}_n_neg"] = (block < 0).sum(axis=1)
            out[f"{prefix}_n_pos"] = (block > 0).sum(axis=1)
        out["tendance_gares"] = X["p0q2"] - X["p0q4"]
        out["tendance_trains"] = X["p2q0"] - X["p4q0"]
        out["signe_p0q2"] = np.sign(X["p0q2"])
        out["signe_p2q0"] = np.sign(X["p2q0"])
        out["abs_max"] = X[LAG_COLUMNS].abs().max(axis=1)
        out["croise"] = X["p0q2"] * X["p2q0"]
        return pd.DataFrame(out, index=X.index).astype("float32")


@register_feature
class TargetEncoding(FeatureTransformer):
    """Statistiques de la cible par groupe, lissées vers la statistique globale.

    Réajusté dans chaque fold : seules les lignes d'entraînement du fold sont utilisées.
    """

    name = "target_encoding"
    stateless: ClassVar[bool] = False

    def __init__(
        self,
        keys: list[list[str]] | None = None,
        stats: list[str] | None = None,
        smoothing: float = 50.0,
    ) -> None:
        self.keys = keys
        self.stats = stats
        self.smoothing = smoothing

    def _keys(self) -> list[list[str]]:
        return [list(k) for k in (self.keys or [["gare"]])]

    def _stats(self) -> list[str]:
        return list(self.stats or ["mean"])

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> TargetEncoding:
        if y is None:
            raise ValueError("TargetEncoding nécessite la cible.")
        frame = X.assign(**{TARGET: y.to_numpy()})
        self.global_ = {s: float(frame[TARGET].agg(s)) for s in self._stats()}
        self.tables_ = {}
        for key in self._keys():
            grouped = frame.groupby(key, observed=True)[TARGET].agg([*self._stats(), "count"])
            n = grouped.pop("count")
            for s in self._stats():
                grouped[s] = (n * grouped[s] + self.smoothing * self.global_[s]) / (
                    n + self.smoothing
                )
            self.tables_["__".join(key)] = grouped
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=X.index)
        for key in self._keys():
            table = self.tables_["__".join(key)]
            joined = X[key].join(table, on=key)
            for s in self._stats():
                out[f"te_{'_'.join(key)}_{s}"] = joined[s].fillna(self.global_[s]).astype("float32")
        return out
