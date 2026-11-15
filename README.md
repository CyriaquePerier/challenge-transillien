# Challenge Transilien — plateforme d'expérimentation

Plateforme pour le challenge Transilien SNCF Voyageurs (cours ECN-STATSC-2627) : prédire
`p0q0`, l'écart entre le temps d'attente théorique et le temps réalisé d'un train en gare,
à partir des retards observés deux gares en amont. La métrique est la **MAE**.

Le principe : **ajouter un modèle = écrire un seul fichier** dans `models/`. Le reste est
automatique : chargement et validation des données, features, validation croisée
temporelle, post-traitement, métriques, rapport HTML, suivi MLflow, soumission vérifiée
et tests.

Les choix de conception sont détaillés dans [ARCHITECTURE.md](ARCHITECTURE.md), et
l'énoncé du challenge dans [challenge_transilien.md](challenge_transilien.md).

---

## Démarrage rapide

Prérequis : [uv](https://docs.astral.sh/uv/) (`pip install uv` ou
`curl -LsSf https://astral.sh/uv/install.sh | sh`). uv installe lui-même Python 3.12.

```bash
make setup                      # environnement + hooks pre-commit
make run MODEL=baseline_zero    # premier run complet (quelques secondes)
make run MODEL=lightgbm_v1      # un vrai modèle
make leaderboard                # classement des runs
```

Chaque `make run` affiche la MAE CV et la MAE hold-out, puis le chemin du rapport HTML
(`reports/runs/<run_id>/index.html`) et celui de la soumission (`submissions/…csv`), prête
à déposer.

Sans `make`, on peut appeler la CLI directement : `uv run transilien run model=lightgbm_v1`.

---

## Comment fonctionne un run

`make run MODEL=x` enchaîne les étapes suivantes (code dans
[src/transilien/runner.py](src/transilien/runner.py)) :

```
data-lake/*.csv ─▶ 1. ingestion ─▶ 2. contrats ─▶ 3. features ─▶ 4. CV temporelle
                    (Parquet)       (pandera)      (cache)          │
                                                                     ▼
 7. rapport HTML + MLflow ◀─ 6. soumission ◀─ 5. hold-out + modèle final ◀─ post-traitement
```

### 1. Ingestion — [data/ingest.py](src/transilien/data/ingest.py)
Les CSV de `data-lake/` ne sont jamais modifiés. Au premier lancement, ils sont convertis
en Parquet typé dans `data/interim/`, sans la colonne parasite `Unnamed: 0`. Le SHA-256
de chaque fichier source est mémorisé : la conversion n'est rejouée que si un fichier
change. Ce hash est enregistré avec chaque run.

### 2. Contrats de données — [data/schemas.py](src/transilien/data/schemas.py)
Des schémas **pandera** valident à l'entrée : colonnes exactes, types, absence de NaN,
dates en jours ouvrés, valeurs entières, bornes plausibles (±1500 min, une valeur réelle
de −1441 existe dans le train), gares du test présentes dans le train, alignement
`x_train`/`y_train`. Une anomalie bloque le run au lieu de fausser silencieusement les
résultats.

### 3. Features — [features/](src/transilien/features/)
Un jeu de features est un fichier `configs/features/<nom>.yaml` qui liste des
*transformers* :

| Transformer | Contenu | Avec état ? |
|---|---|---|
| `raw_lags` | les 6 valeurs passées `p2q0 … p0q4` | non |
| `structural` | `arret`, code ordinal de la gare | non |
| `temporal` | jour de semaine, vacances scolaires zone C, distance au jour férié (+ `semaine`, `mois` en option) | non |
| `lag_stats` | moyenne / médiane / écart-type / min / max des retards passés (trains précédents, gares précédentes, ensemble), tendances `p0q2 − p0q4`, signes, interactions | non |
| `target_encoding` | statistiques de la cible par gare, gare × arrêt, gare × jour, lissées | **oui** |

* Les transformers **sans état** sont calculés une seule fois sur train + test puis mis en
  cache dans `data/features/`. La clé du cache combine le hash des données, la
  configuration et le code source des transformers : modifier une feature invalide le
  cache automatiquement.
* Les transformers **avec état** (encodage cible) sont réajustés **dans chaque fold**,
  sur les seules lignes d'entraînement, pour éviter toute fuite de la cible.

Jeux fournis : `raw` (benchmark officiel), `base`, `lags_enrichis` (base + encodages
cibles). `semaine` et `mois` sont exclus par défaut, car le test (mi-novembre → décembre)
ne contient que des valeurs jamais vues à l'entraînement.

### 4. Validation temporelle — [evaluation/splitters.py](src/transilien/evaluation/splitters.py)
Le train couvre avril → 10 novembre 2023, le test 13 novembre → 22 décembre. La
validation reproduit cette situation :

1. **Hold-out final** : les 4 dernières semaines du train sont mises de côté. Elles ne
   servent à **aucune** décision (ni hyperparamètres, ni post-traitement) et donnent une
   estimation honnête du score sur le test.
2. **CV temporelle à fenêtre croissante** sur le reste (5 folds par défaut) : on entraîne
   sur les semaines 1…n et on valide sur la semaine n+1. Les folds sont groupés par date,
   et un test vérifie qu'aucune date de validation n'est antérieure ou égale à une date
   d'entraînement.
3. Variante `cv=temporal_last_weeks` : fenêtre glissante de 12 semaines, pour tester
   l'hypothèse d'une dérive temporelle.

Le jeu de validation de chaque fold est transmis au modèle pour l'early stopping. Les
modèles de boosting renvoient ensuite, via `refit_params`, le nombre d'itérations médian
trouvé en CV. Les réentraînements (hold-out, modèle final) l'utilisent sans jamais voir
de données de validation.

### 5. Post-traitement — [evaluation/postprocess.py](src/transilien/evaluation/postprocess.py)
La cible est entière et la métrique est une MAE : arrondir est souvent payant. Quatre
variantes sont évaluées sur les prédictions **out-of-fold** : `raw`, `round`, `clip`
(écrêtage aux quantiles 0,1 % / 99,9 % du train) et `clip_round`. La meilleure est
retenue. À égalité, c'est la plus simple qui l'emporte.

### 6. Hold-out, modèle final et soumission — [submission/](src/transilien/submission/)
Le modèle est réentraîné sur toute la partie hors hold-out, puis évalué sur le hold-out.
Il est ensuite réentraîné une dernière fois sur **tout** le train pour prédire `x_test`.
Avant d'écrire la soumission, des vérifications bloquantes s'appliquent : format exact
de `y_sample_final.csv` (index sans nom + `p0q0`), 20 657 lignes dans l'ordre de
`x_test`, pas de NaN ni d'infini, valeurs plausibles, distribution cohérente avec les
prédictions OOF. Le fichier est nommé `submissions/<date>_<modèle>_cv<mae>.csv`.

### 7. Rapport et suivi
* **Rapport HTML autonome** ([reporting/run_report.py](src/transilien/reporting/run_report.py)),
  lisible hors ligne : indicateurs clés, gains par rapport à « prédire 0 » (0,894) et à
  `baseline_rf`, MAE par fold, courbes d'apprentissage, prédit vs réel, résidus
  (distribution et évolution dans le temps), carte de chaleur gare × jour, MAE par
  segment (classe de retard, jour, arrêt, semaine, gare), importance des features
  (native, sinon par permutation), valeurs SHAP pour les modèles arborescents, et
  comparaison avec le meilleur run du leaderboard.
* **MLflow** ([tracking/](src/transilien/tracking/)) : base locale `mlruns/mlflow.db`
  avec la configuration complète, le commit git, le hash des données, les métriques par
  fold et tous les artefacts du run. Interface : `make ui`.

---

## Résultats de référence

Runs sur les données complètes (CV temporelle à 5 folds, hold-out sur les 4 dernières
semaines, graine 42) :

| Modèle | Features | MAE CV | MAE hold-out | Post-traitement | Gain hold-out vs « prédire 0 » |
|---|---|---|---|---|---|
| `baseline_zero` | `raw` | 0,936 ± 0,089 | 0,871 | aucun | — |
| `baseline_rf` (benchmark officiel) | `raw` | 0,731 ± 0,081 | 0,684 | arrondi + écrêtage | −21 % |
| `lightgbm_v1` | `lags_enrichis` | 0,674 ± 0,072 | **0,638** | arrondi | −27 % |

L'écart entre LightGBM et la RF sur le hold-out (−0,046) est significatif : test de
Wilcoxon apparié p < 10⁻⁴, IC bootstrap à 95 % [0,043 ; 0,049]. L'arrondi apporte à lui
seul environ 0,05 de MAE à la RF.

---

## Ajouter un modèle

1. Copier `models/baseline_rf.py` en `models/mon_modele.py`.
2. Changer `name` et le contenu de `fit` / `predict` :

```python
import numpy as np
from sklearn.linear_model import HuberRegressor

from transilien.models.base import BaseModel
from transilien.models.registry import register


@register
class MonModele(BaseModel):
    name = "mon_modele"
    feature_set = "base"  # optionnel (défaut : base)
    default_params = {"epsilon": 1.35}  # optionnel

    def fit(self, X, y, X_valid=None, y_valid=None):
        self.model_ = HuberRegressor(**self.params).fit(X, y)
        return self

    def predict(self, X) -> np.ndarray:
        return self.model_.predict(X)
```

3. `make run MODEL=mon_modele`.

C'est tout : le modèle est découvert automatiquement, il passe les tests de contrat
(`make test`) sans écrire un seul test, et il apparaît dans le leaderboard.

**Éléments optionnels**, pris en compte s'ils existent :

| Élément | Effet |
|---|---|
| `feature_set = "lags_enrichis"` | jeu de features utilisé (surchargeable avec `features=…` en CLI) |
| `configs/model/<nom>.yaml` | hyperparamètres (prioritaires sur `default_params`) |
| `search_space(self, trial) -> dict` | active `make tune` (Optuna) |
| `feature_importances() -> pd.Series` | importance native dans le rapport (sinon permutation) |
| `refit_params(fold_models) -> dict` | paramètres des réentraînements (ex. nombre d'arbres trouvé par early stopping) |
| attribut `learning_curve_ = {"valid": [...]}` | courbes d'apprentissage dans le rapport |

Le post-traitement (arrondi, écrêtage) est géré par la plateforme : `predict` renvoie des
valeurs continues.

## Ajouter des features

Écrire une classe dans [features/transformers.py](src/transilien/features/transformers.py) :

```python
@register_feature
class MaFeature(FeatureTransformer):
    name = "ma_feature"
    stateless = True  # False si fit() utilise la cible → réajustée dans chaque fold

    def transform(self, X):
        return pd.DataFrame({"retard_cumule": X["p0q2"] + X["p0q3"]}, index=X.index)
```

puis la référencer dans un `configs/features/<jeu>.yaml`. Les tests unitaires de
[tests/unit/test_features.py](tests/unit/test_features.py) servent de modèle pour la
tester.

---

## Configuration

La configuration est composée par **Hydra** à partir de [configs/](configs/) :

```
configs/config.yaml        valeurs par défaut (graine, hold-out, post-traitement, chemins…)
configs/cv/                temporal_expanding.yaml, temporal_last_weeks.yaml
configs/features/          raw.yaml, base.yaml, lags_enrichis.yaml
configs/model/             un YAML d'hyperparamètres par modèle (facultatif)
```

Toute valeur se surcharge en ligne de commande :

```bash
make run MODEL=lightgbm_v1 ARGS="model.params.num_leaves=127 cv=temporal_last_weeks"
make run MODEL=lightgbm_v1 ARGS="features=base holdout.weeks=3"
make run MODEL=lightgbm_v1 ARGS="data.sample_frac=0.1"   # prototypage sur 10 % des lignes
```

L'ordre de priorité des hyperparamètres est le suivant : CLI > `configs/model/<nom>.yaml`
> `default_params` de la classe. Chaque valeur utilisée est enregistrée dans
`metrics.json` et dans MLflow.

---

## Commandes

| Commande | Effet |
|---|---|
| `make setup` | Installe l'environnement et les hooks pre-commit |
| `make eda` | Rapport d'exploration des données → `reports/eda/index.html` |
| `make run MODEL=x` | CV temporelle + hold-out + rapport + run MLflow + soumission |
| `make tune MODEL=x TRIALS=n` | Optuna (même CV, pruning après chaque fold), meilleurs paramètres écrits dans `configs/model/x.yaml`, puis run final |
| `make compare RUNS="a b"` | Rapport comparatif : métriques, MAE par segment, test de Wilcoxon apparié et IC bootstrap sur le hold-out |
| `make leaderboard` | Classement par MAE hold-out, avec l'écart CV/hold-out (sur-apprentissage) |
| `make blend RUNS="a b c"` | Ensemble pondéré (poids ≥ 0, somme 1) optimisé sur les prédictions OOF, évalué sur le hold-out, avec soumission |
| `make report RUNS=<run>` / `make submit RUNS=<run>` | Régénère le rapport / la soumission d'un run existant |
| `make models` | Liste les modèles découverts |
| `make test` / `make regression` | Tests / test de non-régression (données complètes) |
| `make lint` / `make format` | ruff + mypy / correction automatique du style |
| `make ui` | Interface MLflow sur http://127.0.0.1:5000 |

Pour `compare`, `blend`, `report` et `submit`, un run se désigne par son identifiant,
un préfixe unique de celui-ci, ou un nom de modèle (le dernier run de ce modèle est
alors utilisé) : `make compare RUNS="baseline_rf lightgbm_v1"`.

---

## Arborescence

```
├── configs/                 configuration Hydra
├── data-lake/               données brutes (lecture seule, versionnées)
├── data/interim/            Parquet validés (cache régénérable)
├── data/features/           features en cache (clé = hash données + config + code)
├── models/                  ★ zone utilisateur : un fichier = un modèle
├── src/transilien/          cœur de la plateforme
│   ├── cli.py               commandes (Typer)
│   ├── config.py            composition Hydra
│   ├── runner.py            orchestration d'un run
│   ├── tuning.py            Optuna
│   ├── blending.py          ensembles sur les OOF
│   ├── runs.py              stockage des runs et leaderboard
│   ├── data/                ingestion, schémas pandera, chargement
│   ├── features/            transformers, registre, pipeline + cache, calendrier
│   ├── models/              contrat BaseModel, registre @register
│   ├── evaluation/          splitters, métriques, post-traitement
│   ├── reporting/           figures Plotly, rapports run / EDA / comparaison
│   ├── tracking/            surcouche MLflow
│   └── submission/          écriture et vérifications
├── tests/                   unit/, contract/, integration/, regression/
├── reports/                 rapports HTML générés
├── submissions/             CSV prêts à déposer
└── mlruns/                  stockage MLflow local
```

Le dossier d'un run (`reports/runs/<run_id>/`) contient : `index.html` (rapport),
`metrics.json` (toutes les métriques et la traçabilité), `config.yaml` (configuration
résolue), `oof.parquet` / `holdout.parquet` / `test.parquet` (prédictions brutes et
post-traitées, réutilisées par `compare` et `blend`), `model.joblib` (modèle final),
`segments/*.csv`, `importances.csv`.

---

## Tests et qualité

| Niveau | Contenu |
|---|---|
| `tests/unit/` | métriques, splitters (dont un test de propriété *hypothesis* contre les fuites temporelles), post-traitement, features (forme, NaN, encodage cible sans fuite, cache), schémas, soumission, configuration et registre |
| `tests/contract/` | paramétrés sur **tous** les modèles du registre : instanciation depuis la config, `fit`/`predict` sur 5 000 lignes réelles, sortie de bonne longueur et finie, fit sans jeu de validation, déterminisme à graine fixée, sérialisation, hooks optionnels |
| `tests/integration/` | pipeline complet sur 3 % des données : artefacts, soumission conforme, LightGBM meilleur que « prédire 0 », leaderboard, comparaison, blending |
| `tests/regression/` | MAE hold-out de `baseline_rf` sur les données complètes égale à la référence de [reference.json](tests/regression/reference.json), à ±0,005 près |

`make test` lance les trois premiers niveaux (environ 15 s). La CI GitHub Actions
([.github/workflows/ci.yml](.github/workflows/ci.yml)) enchaîne lint → tests →
non-régression (sur `main` seulement). pre-commit lance ruff et mypy avant chaque commit.

Reproductibilité : `uv.lock` fige les versions, la graine globale vient de la
configuration, et chaque run enregistre le commit git (suffixé `-dirty` si l'arbre de
travail est modifié) ainsi que le hash des données. Un [Dockerfile](Dockerfile) minimal
permet de relancer la plateforme sur une autre machine.

---

## Dépannage

* **Dépôt dans un dossier iCloud Drive (`Documents`, `Bureau`) sous macOS** : iCloud
  marque les fichiers de `.venv` comme cachés, voire les évince du disque, et Python ≥ 3.12
  ignore les `.pth` cachés, d'où `ModuleNotFoundError: No module named 'transilien'`.
  `make setup` crée donc l'environnement directement dans `.venv.nosync` (iCloud
  ignore les dossiers suffixés `.nosync`) et `.venv` devient un lien symbolique vers lui.
  Si un `.venv` « classique » existe déjà, il est recréé : ses fichiers ont pu être évincés. Chaque
  cible du Makefile retire aussi le drapeau par précaution (`make fix-pth`).
* **« Export PNG ignoré »** : l'export PNG des figures (Kaleido ≥ 1) a besoin de Chrome.
  Pour l'installer : `uv run plotly_get_chrome`. Pour désactiver l'export :
  `ARGS="reporting.export_png=false"`. Les rapports HTML ne sont pas concernés.
* **Avertissement `VIRTUAL_ENV … does not match`** : un autre environnement (conda…) est
  actif. Sans conséquence, uv utilise `.venv` ; `unset VIRTUAL_ENV` le fait disparaître.
