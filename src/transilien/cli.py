"""Point d'entrée en ligne de commande : ``uv run transilien <commande> [surcharges Hydra]``."""

from __future__ import annotations

from typing import Annotated

import pandas as pd
import typer

from transilien.config import load_config, path

app = typer.Typer(
    help="Plateforme d'expérimentation du challenge Transilien.",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)

Overrides = Annotated[
    list[str] | None,
    typer.Argument(help="Surcharges Hydra, ex. model=lightgbm_v1 model.params.num_leaves=63"),
]


@app.command()
def run(overrides: Overrides = None) -> None:
    """CV temporelle + hold-out + rapport + run MLflow + soumission."""
    from transilien.runner import run_experiment

    result = run_experiment(load_config(overrides))
    m = result.metrics
    typer.echo(f"\nRun {result.run_id}")
    typer.echo(f"  MAE CV       {m['cv']['mae_mean']:.4f} ± {m['cv']['mae_std']:.4f}")
    if m["holdout"]:
        typer.echo(f"  MAE hold-out {m['holdout']['mae']:.4f}")
    typer.echo(f"  Rapport      {result.files.report}")
    if result.submission:
        typer.echo(f"  Soumission   {result.submission}")


@app.command()
def tune(
    overrides: Overrides = None,
    trials: Annotated[int, typer.Option(help="Nombre d'essais Optuna")] = 50,
    timeout: Annotated[float | None, typer.Option(help="Durée maximale (secondes)")] = None,
    final_run: Annotated[bool, typer.Option("--final-run/--no-final-run")] = True,
) -> None:
    """Optimisation Optuna puis run final avec les meilleurs paramètres."""
    from transilien.tuning import tune as run_tuning

    cfg = load_config(overrides)
    best = run_tuning(cfg, trials, timeout)
    typer.echo(f"Meilleurs paramètres : {best}")
    if final_run:
        run(overrides)


@app.command()
def eda(overrides: Overrides = None) -> None:
    """Génère le rapport d'exploration des données (reports/eda/index.html)."""
    from transilien.reporting.eda import build_eda_report

    typer.echo(f"Rapport EDA : {build_eda_report(load_config(overrides))}")


@app.command()
def leaderboard(
    top: Annotated[int, typer.Option(help="Nombre de lignes affichées")] = 20,
    all_runs: Annotated[
        bool, typer.Option("--all", help="Inclure les runs sous-échantillonnés")
    ] = False,
) -> None:
    """Classement des runs (MAE hold-out, MAE CV, écart CV/hold-out)."""
    from transilien.runs import leaderboard as build

    board = build(path(load_config(), "runs"))
    if board.empty:
        typer.echo("Aucun run pour l'instant : make run MODEL=baseline_zero")
        raise typer.Exit()
    if not all_runs:
        board = board[board["sample_frac"].isna()]
    cols = [
        "run_id",
        "model",
        "features",
        "cv_mae",
        "cv_std",
        "holdout_mae",
        "ecart_cv_holdout",
        "postprocess",
    ]
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        typer.echo(board[cols].head(top).to_string(float_format=lambda v: f"{v:.4f}"))


@app.command()
def compare(
    runs: Annotated[list[str], typer.Argument(help="Identifiants (ou noms de modèles) des runs")],
) -> None:
    """Rapport comparatif et test de significativité entre plusieurs runs."""
    from transilien.reporting.compare import build_comparison_report
    from transilien.runs import resolve_run

    cfg = load_config()
    files = [resolve_run(path(cfg, "runs"), r) for r in runs]
    out = build_comparison_report(files, path(cfg, "reports") / "compare")
    typer.echo(f"Rapport de comparaison : {out}")


@app.command()
def blend(
    runs: Annotated[list[str], typer.Argument(help="Identifiants (ou noms de modèles) des runs")],
) -> None:
    """Ensemble pondéré optimisé sur les prédictions OOF, avec soumission."""
    from transilien.blending import blend as run_blend
    from transilien.runs import resolve_run

    cfg = load_config()
    files = run_blend(cfg, [resolve_run(path(cfg, "runs"), r) for r in runs])
    typer.echo(f"Blend {files.run_id} : {files.report}")


@app.command()
def report(run_id: str) -> None:
    """Régénère le rapport HTML d'un run existant (sans SHAP)."""
    from omegaconf import OmegaConf

    from transilien.reporting.run_report import build_run_report
    from transilien.runs import resolve_run

    files = resolve_run(path(load_config(), "runs"), run_id)
    build_run_report(files, OmegaConf.load(files.config))  # type: ignore[arg-type]
    typer.echo(f"Rapport : {files.report}")


@app.command()
def submit(run_id: str) -> None:
    """Réécrit (après vérifications) la soumission d'un run à partir de ses prédictions test."""
    from transilien.data.loaders import load_dataset
    from transilien.runs import resolve_run
    from transilien.submission.writer import write_submission

    cfg = load_config()
    files = resolve_run(path(cfg, "runs"), run_id)
    m = files.load_metrics()
    oof = pd.read_parquet(files.oof)
    out, _ = write_submission(
        pd.read_parquet(files.test)["pred"].to_numpy(),
        load_dataset(cfg).x_test,
        path(cfg, "submissions"),
        m["model"],
        m["cv"]["oof_mae"],
        float(cfg.submission.max_abs_value),
        oof["pred"].dropna().to_numpy(),
    )
    typer.echo(f"Soumission : {out}")


@app.command()
def models() -> None:
    """Liste les modèles découverts dans models/."""
    from transilien.models.registry import discover_models

    cfg = load_config()
    for name, cls in sorted(discover_models().items()):
        has_cfg = "oui" if (path(cfg, "model_configs") / f"{name}.yaml").exists() else "non"
        tunable = "oui" if cls.has_search_space() else "non"
        typer.echo(
            f"{name:<16} features={cls.feature_set or 'base':<14} "
            f"config={has_cfg:<4} tuning={tunable}"
        )


@app.command()
def ingest(
    force: Annotated[bool, typer.Option(help="Reconvertir même si le cache est valide")] = False,
) -> None:
    """Convertit et valide les CSV bruts en Parquet."""
    # Les autres commandes le font automatiquement ; utile pour forcer une reconversion.
    from transilien.data.ingest import ingest as run_ingest
    from transilien.data.loaders import raw_files

    cfg = load_config()
    hashes = run_ingest(raw_files(cfg), path(cfg, "interim"), force=force)
    for name, digest in hashes.items():
        typer.echo(f"{name:<8} {digest}")


@app.command()
def ui(port: int = 5000) -> None:
    """Lance l'interface MLflow sur la base locale."""
    import subprocess

    from transilien.tracking.mlflow_tracker import tracking_uri

    subprocess.run(
        [
            "mlflow",
            "ui",
            "--backend-store-uri",
            tracking_uri(path(load_config(), "mlruns")),
            "--port",
            str(port),
        ],
        check=False,
    )


if __name__ == "__main__":
    app()
