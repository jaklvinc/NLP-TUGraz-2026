"""Apply a trained Stage 1 model to a CSV of texts and write predictions.

Usage:
    uv run python -m stage1.predict \\
        --model-uri runs:/<RUN_ID>/model \\
        --input path/to/inputs.csv \\
        --output path/to/predictions.csv

The input CSV must contain a 'Content' column. Output CSV has columns
['Content', 'Classification'] in the same order as the input.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd

from data import LABEL_COLUMN, TEXT_COLUMN
from mlflow_utils import configure


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1 batch prediction")
    parser.add_argument(
        "--model-uri",
        required=True,
        help="MLflow model URI, e.g. runs:/<run_id>/model or models:/name/version",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input CSV with a 'Content' column",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output CSV path (parent dirs will be created)",
    )
    args = parser.parse_args()

    configure()

    df = pd.read_csv(args.input, encoding="utf-8-sig")
    if TEXT_COLUMN not in df.columns:
        raise ValueError(
            f"Input CSV missing required column {TEXT_COLUMN!r}; got {list(df.columns)}"
        )

    model = mlflow.sklearn.load_model(args.model_uri)
    preds = model.predict(df[TEXT_COLUMN].tolist())

    out = pd.DataFrame({TEXT_COLUMN: df[TEXT_COLUMN], LABEL_COLUMN: preds})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Wrote {len(out)} predictions to {args.output}")


if __name__ == "__main__":
    main()
