"""Rapport d'exploration des données : ``reports/eda/index.html``."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from omegaconf import DictConfig

from transilien.config import path
from transilien.data.loaders import load_dataset
from transilien.data.schemas import LAG_COLUMNS, TARGET
from transilien.reporting import figures as F
from transilien.reporting.html import Page


def build_eda_report(cfg: DictConfig) -> Path:
    ds = load_dataset(cfg)
    x, y, t = ds.x_train, ds.y_train, ds.x_test
    train = x.assign(**{TARGET: y})
    page = Page("Exploration des données — Transilien", f"Données {ds.data_hash}")

    page.kpis(
        [
            ("Lignes train", f"{len(x):,}".replace(",", " ")),
            ("Lignes test", f"{len(t):,}".replace(",", " ")),
            ("Jours train / test", f"{x['date'].nunique()} / {t['date'].nunique()}"),
            ("Gares train / test", f"{x['gare'].nunique()} / {t['gare'].nunique()}"),
            ("MAE de « prédire 0 »", f"{y.abs().mean():.3f}"),
            ("Part de |y| > 5", f"{(y.abs() > 5).mean():.2%}"),
        ]
    )
    page.note(
        f"Train : {x['date'].min().date()} → {x['date'].max().date()} ; "
        f"test : {t['date'].min().date()} → {t['date'].max().date()}. "
        f"Gares du test absentes du train : {len(set(t['gare']) - set(x['gare']))}. "
        f"Trains du test vus en train : {len(set(t['train']) & set(x['train']))}."
    )

    page.h2("Cible")
    page.table(y.describe().to_frame().T)
    page.figure("target_distribution", F.target_distribution(y))

    page.h2("Lien entre la cible et les valeurs passées")
    page.figure("correlations", F.correlation_heatmap(train[[TARGET, *LAG_COLUMNS]]))
    means = {c: train.groupby(train[c].clip(-10, 10))[TARGET].mean() for c in ("p0q2", "p2q0")}
    fig = go.Figure(
        [
            go.Scatter(x=s.index, y=s.to_numpy(), mode="lines+markers", name=c)
            for c, s in means.items()
        ]
    )
    fig.update_layout(
        title="p0q0 moyen selon la valeur passée (écrêtée à ±10)",
        xaxis_title="valeur passée",
        yaxis_title="p0q0 moyen",
    )
    page.figure("target_vs_lag", fig)

    page.h2("Retard moyen par gare, jour et semaine")
    by_gare = train.groupby("gare")[TARGET].agg(["mean", "size"]).sort_values("mean")
    fig = go.Figure(
        go.Bar(x=by_gare.index, y=by_gare["mean"], hovertext=by_gare["size"].astype(str))
    )
    fig.update_layout(title="p0q0 moyen par gare", xaxis_title="gare", yaxis_title="p0q0 moyen")
    page.figure("by_gare", fig)
    by_day = train.groupby(train["date"].dt.dayofweek)[TARGET].agg(["mean", "std"])
    fig = go.Figure(go.Bar(x=[F.JOURS[d] for d in by_day.index], y=by_day["mean"]))
    fig.update_layout(title="p0q0 moyen par jour de semaine", yaxis_title="p0q0 moyen")
    page.figure("by_weekday", fig)
    by_date = train.groupby("date")[TARGET].agg(mean="mean", mae0=lambda s: s.abs().mean())
    fig = go.Figure(
        [
            go.Scatter(x=by_date.index, y=by_date["mean"], name="moyenne"),
            go.Scatter(x=by_date.index, y=by_date["mae0"], name="moyenne de |p0q0|"),
        ]
    )
    fig.update_layout(title="Évolution journalière de la cible", xaxis_title="date")
    page.figure("by_date", fig)

    page.h2("Dérive temporelle et comparaison train / test")
    both = pd.concat([x.assign(jeu="train"), t.assign(jeu="test")])
    drift = both.groupby(["jeu", "date"])[LAG_COLUMNS].apply(lambda d: d.abs().mean()).reset_index()
    fig = go.Figure()
    for col in ("p0q2", "p2q0"):
        for jeu in ("train", "test"):
            d = drift[drift["jeu"] == jeu]
            fig.add_trace(
                go.Scatter(x=d["date"], y=d[col], mode="lines", name=f"|{col}| moyen ({jeu})")
            )
    fig.update_layout(title="Distribution des features dans le temps", xaxis_title="date")
    page.figure("drift", fig)
    compare = pd.DataFrame(
        {
            "train": x[[*LAG_COLUMNS, "arret"]]
            .describe()
            .loc[["mean", "std", "50%", "min", "max"]]
            .stack(),
            "test": t[[*LAG_COLUMNS, "arret"]]
            .describe()
            .loc[["mean", "std", "50%", "min", "max"]]
            .stack(),
        }
    ).unstack(0)
    page.table(pd.DataFrame(compare))
    shares = {
        name: frame["p0q2"].clip(-10, 10).value_counts(normalize=True).sort_index()
        for name, frame in (("train", x), ("test", t))
    }
    fig = go.Figure(
        [go.Bar(x=s.index, y=s.to_numpy(), name=name, opacity=0.6) for name, s in shares.items()]
    )
    fig.update_layout(title="p0q2 : train vs test (proportions)", barmode="overlay")
    page.figure("train_vs_test", fig)

    out = path(cfg, "reports") / "eda" / "index.html"
    return page.save(out, out.parent / "figures")
