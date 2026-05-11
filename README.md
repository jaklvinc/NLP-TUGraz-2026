# NLP-TUGraz-2026

Final project for NLP VU (DAT.C309UF), SS26 at TU Graz. The task is to classify text snippets as `Fact` or `Opinion`.

The work is split into two stages:

- **Stage 1** — traditional NLP. No transformers.
- **Stage 2** — LLM-based classification.

Stages live in their own packages (`stage1/`, `stage2/`). Anything they both need (data loading, metrics, MLflow setup) is at the repo root.

## Setup

Requires `uv`. The project pins Python ≥3.14.
`uv sync` will fetch it if you don't have it locally.

```bash
uv sync
```

This project uses **marimo** for visualizations and overviews, a better notebook for python.
Python deps already installed via uv, for visualization install the [marimo extension](https://marketplace.visualstudio.com/items?itemName=marimo-team.vscode-marimo) for VSCode

## Stage 1
### Running an experiment

An "experiment approach" is a registered factory in [stage1/pipelines.py](stage1/pipelines.py). A run does stratified 5-fold CV on a 90% slice of the data, then retrains on the full 90% and evaluates on the 10% holdout. Metrics and model go to MLflow.

```bash
uv run python -m stage1.train                            # default: tfidf_logreg
uv run python -m stage1.train --pipeline tfidf_svm
uv run python -m stage1.train --pipeline tfidf_svm --seed 7
```

CLI flags:

| Flag | Default | What |
|---|---|---|
| `--pipeline` | `tfidf_logreg` | name from `PIPELINES` in [stage1/pipelines.py](stage1/pipelines.py) |
| `--dataset` | `dataset.csv` | labeled CSV with `Content,Classification` columns |
| `--seed` | 42 | controls split + classifier |
| `--n-splits` | 5 | CV folds |
| `--holdout-size` | 0.10 | fraction reserved as internal test set |
| `--run-name` | _pipeline name_ | overrides the MLflow run name |

### Adding a new approach

1. Add a factory to [stage1/pipelines.py](stage1/pipelines.py):
   ```python
   def my_approach(seed: int) -> tuple[Pipeline, dict[str, Any]]:
       pipe = Pipeline([...])
       return pipe, {"features": "...", "clf": "..."}
   ```
2. Register it in `PIPELINES`.
3. `uv run python -m stage1.train --pipeline my_approach`.

`train.py` does not need changes.

### Predicting with a saved model

Grab the run id from the MLflow UI, then:

```bash
uv run python -m stage1.predict \
    --model-uri runs:/<RUN_ID>/model \
    --input path/to/input.csv \
    --output path/to/predictions.csv
```

The input CSV must have a `Content` column. Output is `Content,Classification`.

## Stage 2
tbd.

## Comparing runs

```bash
uv run mlflow ui
```

Sort by `cv_f1_macro_mean`. Filter by `tags.approach` to keep results by method. Use the compare button on two selected runs to diff params and overlay metric histories.

## Layout

```
data.py                 # load + clean + stratified split
evaluation.py           # metrics, confusion matrix, misclassified dump
mlflow_utils.py         # tracking URI + experiment
labels.py               # Label literal, LABELS tuple

stage1/
  pipelines.py          # registry of approaches
  features.py           # feature factories (TF-IDF for now)
  knowledge_base.py     # opinion-marker lexicon (stub)
  train.py              # pipeline-agnostic trainer
  predict.py            # batch prediction from a logged model

stage2/                 # not started

dataset.csv             # 855 labeled rows after cleaning
mlflow.db               # SQLite tracking store (gitignored)
mlruns/                 # MLflow artifacts (gitignored)
```

## Notes on the data

- The CSV has a UTF-8 BOM, so it's read with `encoding='utf-8-sig'`.
- Labels are title-cased on load. The raw file mixes `Fact`/`fact`/`Opinion`/`opinion`/`OPinion`.
- 10 rows have duplicate `Content`. They're dropped before splitting to prevent them leaking across CV folds.

## Submission

The prediction CSVs (`group67_classifications_1.csv`, `group67_classifications_2.csv`) get produced via `stage{1,2}.predict` against the best saved model from each stage.