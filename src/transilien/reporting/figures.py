"""Fabrique des figures Plotly réutilisées par les rapports."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

JOURS = ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"]


def fold_mae_bars(folds: list[dict], holdout_mae: float | None, baseline: float) -> go.Figure:
    df = pd.DataFrame(folds)
    labels = [f"F{f['fold']} ({f['valid_start']})" for f in folds]
    fig = go.Figure(go.Bar(x=labels, y=df["mae"], name="MAE par fold", text=df["mae"].round(4)))
    if holdout_mae is not None:
        fig.add_hline(
            y=holdout_mae, line_dash="dash", annotation_text=f"hold-out {holdout_mae:.4f}"
        )
    fig.add_hline(
        y=baseline, line_dash="dot", line_color="gray", annotation_text=f"prédire 0 : {baseline}"
    )
    fig.update_layout(title="MAE par fold de validation temporelle", yaxis_title="MAE")
    return fig


def learning_curves(curves: list[dict | None]) -> go.Figure | None:
    valid = [c for c in curves if c and c.get("valid")]
    if not valid:
        return None
    fig = go.Figure()
    for i, c in enumerate(valid):
        fig.add_trace(go.Scatter(y=c["valid"], mode="lines", name=f"fold {i}"))
    fig.update_layout(
        title="Courbes d'apprentissage (MAE de validation)",
        xaxis_title="itération",
        yaxis_title="MAE",
    )
    return fig


def pred_vs_true(y: np.ndarray, pred: np.ndarray) -> go.Figure:
    lo, hi = np.quantile(np.concatenate([y, pred]), [0.001, 0.999])
    lo, hi = float(np.floor(lo)) - 1, float(np.ceil(hi)) + 1
    # Échelle log des effectifs : la masse autour de 0 écrase sinon tout le reste.
    counts, xe, ye = np.histogram2d(
        y, pred, bins=[np.arange(lo, hi + 1, 1), np.arange(lo, hi + 0.5, 0.5)]
    )
    fig = go.Figure(
        go.Heatmap(
            z=np.log10(counts.T + 1),
            x=(xe[:-1] + xe[1:]) / 2,
            y=(ye[:-1] + ye[1:]) / 2,
            colorscale="Blues",
            colorbar={"title": "log10(n+1)"},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[lo, hi],
            y=[lo, hi],
            mode="lines",
            line={"dash": "dash", "color": "gray"},
            name="y = x",
        )
    )
    fig.update_layout(
        title="Prédit vs réel (densité, zone centrale)",
        xaxis_title="réel",
        yaxis_title="prédit",
        xaxis_range=[lo, hi],
        yaxis_range=[lo, hi],
    )
    return fig


def residual_hist(y: np.ndarray, pred: np.ndarray) -> go.Figure:
    # Histogramme pré-agrégé : le HTML ne contient que les effectifs, pas les résidus bruts.
    clipped = np.clip(pred - y, -15, 15)
    counts, edges = np.histogram(clipped, bins=60, range=(-15, 15))
    fig = go.Figure(go.Bar(x=(edges[:-1] + edges[1:]) / 2, y=counts, width=edges[1] - edges[0]))
    fig.update_layout(
        title="Distribution des résidus (prédit − réel, écrêtés à ±15)",
        xaxis_title="résidu",
        yaxis_title="effectif",
        yaxis_type="log",
    )
    return fig


def residuals_over_time(frames: dict[str, pd.DataFrame]) -> go.Figure:
    fig = go.Figure()
    for name, f in frames.items():
        by_day = f.assign(ae=(f["pred"] - f["y"]).abs(), err=f["pred"] - f["y"]).groupby("date")
        agg = by_day.agg(mae=("ae", "mean"), bias=("err", "mean")).reset_index()
        fig.add_trace(
            go.Scatter(x=agg["date"], y=agg["mae"], mode="lines+markers", name=f"MAE {name}")
        )
        fig.add_trace(
            go.Scatter(
                x=agg["date"],
                y=agg["bias"],
                mode="lines",
                name=f"biais {name}",
                line={"dash": "dot"},
            )
        )
    fig.update_layout(
        title="Erreur dans le temps (par jour)", xaxis_title="date", yaxis_title="minutes"
    )
    return fig


def gare_weekday_heatmap(frame: pd.DataFrame, top: int = 40) -> go.Figure:
    f = frame.assign(ae=(frame["pred"] - frame["y"]).abs(), jour=frame["date"].dt.dayofweek)
    busiest = f["gare"].value_counts().head(top).index
    pivot = f[f["gare"].isin(busiest)].pivot_table(
        index="gare", columns="jour", values="ae", aggfunc="mean"
    )
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]
    fig = go.Figure(
        go.Heatmap(
            z=pivot.to_numpy(),
            x=[JOURS[int(c)] for c in pivot.columns],
            y=pivot.index,
            colorscale="Reds",
            colorbar={"title": "MAE"},
        )
    )
    fig.update_layout(
        title=f"MAE par gare × jour de semaine ({top} gares les plus fréquentes)",
        height=max(400, 16 * len(pivot)),
    )
    return fig


def importance_bars(importances: pd.Series, kind: str, top: int = 25) -> go.Figure:
    imp = importances.sort_values(ascending=True).tail(top)
    fig = go.Figure(go.Bar(x=imp.to_numpy(), y=imp.index, orientation="h"))
    fig.update_layout(title=f"Importance des features ({kind})", height=max(400, 22 * len(imp)))
    return fig


def shap_summary(shap_values: np.ndarray, X: pd.DataFrame, top: int = 15) -> go.Figure:
    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:top]
    fig = go.Figure()
    rng = np.random.default_rng(0)
    for rank, j in enumerate(order[::-1]):
        col = X.columns[j]
        values = X[col].to_numpy(dtype=float)
        span = np.nanmax(values) - np.nanmin(values) or 1.0
        fig.add_trace(
            go.Scattergl(
                x=shap_values[:, j],
                y=rank + rng.uniform(-0.3, 0.3, len(values)),
                mode="markers",
                marker={
                    "size": 4,
                    "color": (values - np.nanmin(values)) / span,
                    "colorscale": "RdBu_r",
                    "showscale": rank == 0,
                    "colorbar": {"title": "valeur"},
                },
                name=col,
                showlegend=False,
                hovertext=[f"{col} = {v:g}" for v in values],
            )
        )
    fig.update_layout(
        title="Valeurs SHAP (impact de chaque feature sur la prédiction)",
        yaxis={
            "tickvals": list(range(len(order))),
            "ticktext": [X.columns[j] for j in order[::-1]],
        },
        xaxis_title="contribution SHAP (minutes)",
        height=max(420, 30 * len(order)),
    )
    return fig


def segment_bars(tables: dict[str, pd.DataFrame], segment: str) -> go.Figure:
    fig = go.Figure()
    for name, t in tables.items():
        fig.add_trace(go.Bar(x=t.index.astype(str), y=t["mae"], name=name))
    fig.update_layout(title=f"MAE par {segment}", barmode="group", yaxis_title="MAE")
    return fig


def target_distribution(y: pd.Series) -> go.Figure:
    counts = y.value_counts().sort_index()
    fig = go.Figure(go.Bar(x=counts.index, y=counts.to_numpy()))
    fig.update_layout(
        title="Distribution de p0q0 (échelle log)",
        yaxis_type="log",
        xaxis_title="p0q0",
        yaxis_title="effectif",
    )
    return fig


def correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    corr = df.corr(method="spearman")
    return px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Corrélations de Spearman entre p0q0 et les valeurs passées",
    )
