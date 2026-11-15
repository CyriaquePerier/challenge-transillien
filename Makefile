# Raccourcis de la plateforme. Toute commande accepte des surcharges Hydra via ARGS :
#   make run MODEL=lightgbm_v1 ARGS="model.params.num_leaves=127 cv=temporal_last_weeks"

MODEL  ?= baseline_rf
TRIALS ?= 50
RUNS   ?=
ARGS   ?=
UV     := uv run --no-sync

.PHONY: help setup venv-nosync fix-pth eda run tune compare leaderboard blend report submit models test regression lint format ui clean-cache

help:  ## Affiche cette aide
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup:  ## Installe l'environnement (uv) et les hooks pre-commit
	@$(MAKE) --no-print-directory venv-nosync
	uv sync --all-extras
	@$(MAKE) --no-print-directory fix-pth
	$(UV) pre-commit install

# macOS + iCloud Drive (Documents/Bureau synchronisés) : iCloud cache et évince les fichiers
# de .venv, ce qui casse les imports. L'environnement est donc créé directement dans
# .venv.nosync (ignoré par iCloud) et .venv devient un lien symbolique vers lui.
venv-nosync:
	@if [ "$$(uname)" = "Darwin" ] && [ ! -L .venv ]; then \
		rm -rf .venv .venv.nosync && \
		UV_PROJECT_ENVIRONMENT=.venv.nosync uv sync --all-extras -q && \
		ln -s .venv.nosync .venv && echo "Environnement créé dans .venv.nosync (hors iCloud)"; \
	fi

# Synchronise l'environnement puis, sous macOS, retire le drapeau « hidden » que reçoit
# parfois le .pth de l'installation éditable (Python ≥ 3.12 ignore les .pth cachés).
fix-pth:
	@uv sync --all-extras -q
	@if [ "$$(uname)" = "Darwin" ]; then chflags nohidden .venv/lib/python*/site-packages/*.pth 2>/dev/null || true; fi

eda: fix-pth  ## Rapport d'exploration des données
	$(UV) transilien eda $(ARGS)

run: fix-pth  ## CV temporelle + hold-out + rapport + MLflow + soumission (MODEL=...)
	$(UV) transilien run model=$(MODEL) $(ARGS)

tune: fix-pth  ## Optuna puis run final (MODEL=... TRIALS=...)
	$(UV) transilien tune model=$(MODEL) --trials $(TRIALS) $(ARGS)

compare: fix-pth  ## Rapport comparatif + significativité (RUNS="a b")
	$(UV) transilien compare $(RUNS)

leaderboard: fix-pth  ## Classement des runs
	$(UV) transilien leaderboard

blend: fix-pth  ## Ensemble pondéré sur les OOF (RUNS="a b c")
	$(UV) transilien blend $(RUNS)

report: fix-pth  ## Régénère le rapport d'un run (RUNS=<run_id>)
	$(UV) transilien report $(RUNS)

submit: fix-pth  ## Réécrit la soumission d'un run (RUNS=<run_id>)
	$(UV) transilien submit $(RUNS)

models: fix-pth  ## Liste les modèles enregistrés
	$(UV) transilien models

test: fix-pth  ## Tests unitaires, de contrat et d'intégration
	$(UV) pytest

regression: fix-pth  ## Test de non-régression (données complètes, lent)
	$(UV) pytest -m regression

lint:  ## ruff (lint + format) et mypy
	$(UV) ruff check .
	$(UV) ruff format --check .
	$(UV) mypy

format:  ## Corrige automatiquement le style
	$(UV) ruff check --fix .
	$(UV) ruff format .

ui: fix-pth  ## Interface MLflow (http://127.0.0.1:5000)
	$(UV) transilien ui

clean-cache:  ## Supprime les caches régénérables (Parquet, features)
	rm -rf data/interim/* data/features/*
