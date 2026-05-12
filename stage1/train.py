"""Train a Stage 1 pipeline (selected by --pipeline) and log everything to MLflow.

Run from the repo root:
    uv run python -m stage1.train                            # default: tfidf_logreg
    uv run python -m stage1.train --pipeline tfidf_svm
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np

from data import (
    DEFAULT_DATASET_PATH,
    DEFAULT_HOLDOUT_SIZE,
    DEFAULT_N_SPLITS,
    DEFAULT_SEED,
    LABEL_COLUMN,
    TEXT_COLUMN,
    holdout_split,
    load_dataset,
    stratified_folds,
)
from evaluation import (
    compute_metrics,
    write_classification_report,
    write_confusion_matrix_png,
    write_misclassified_csv,
)
from mlflow_utils import configure
from stage1.pipelines import PIPELINES


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1 trainer (pipeline registry)")
    parser.add_argument(
        "--pipeline",
        choices=sorted(PIPELINES),
        default="tfidf_logreg",
        help="Which pipeline factory in stage1/pipelines.py to run.",
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--n-splits", type=int, default=DEFAULT_N_SPLITS)
    parser.add_argument("--holdout-size", type=float, default=DEFAULT_HOLDOUT_SIZE)
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="MLflow run name (defaults to the pipeline name).",
    )
    args = parser.parse_args()

    configure()

    factory = PIPELINES[args.pipeline]
    df, stats = load_dataset(args.dataset)
    train_df, holdout_df = holdout_split(df, args.holdout_size, args.seed)

    run_name = args.run_name or args.pipeline
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("stage", "1")
        mlflow.set_tag("approach", args.pipeline)

        # Pipeline-specific params (from the factory) merged with run-level params.
        _, pipe_params = factory(args.seed)
        mlflow.log_params(
            {
                **pipe_params,
                "pipeline": args.pipeline,
                "seed": args.seed,
                "n_splits": args.n_splits,
                "holdout_size": args.holdout_size,
                "dataset_path": str(args.dataset),
                "n_raw_rows": stats.n_raw,
                "n_normalized_labels": stats.n_normalized_labels,
                "n_dropped_duplicates": stats.n_dropped_duplicates,
                "n_after_clean": stats.n_after_clean,
                "n_train": len(train_df),
                "n_holdout": len(holdout_df),
            }
        )
        mlflow.log_dict(stats.label_counts, "label_counts.json")

        # ---- 5-fold CV on the train portion --------------------------------
        fold_metrics: list[dict[str, float]] = []
        oof_pred = np.empty(len(train_df), dtype=object)

        for fold, (tr_idx, va_idx) in enumerate(
            stratified_folds(train_df, args.n_splits, args.seed)
        ):
            X_tr = train_df.iloc[tr_idx][TEXT_COLUMN].tolist()
            y_tr = train_df.iloc[tr_idx][LABEL_COLUMN].tolist()
            X_va = train_df.iloc[va_idx][TEXT_COLUMN].tolist()
            y_va = train_df.iloc[va_idx][LABEL_COLUMN].tolist()

            pipe, _ = factory(args.seed)
            pipe.fit(X_tr, y_tr)
            y_pred = pipe.predict(X_va).tolist()
            oof_pred[va_idx] = y_pred

            m = compute_metrics(y_va, y_pred)
            fold_metrics.append(m)
            for k, v in m.items():
                mlflow.log_metric(f"cv_{k}", v, step=fold)

        for k in fold_metrics[0]:
            vals = [m[k] for m in fold_metrics]
            mlflow.log_metric(f"cv_{k}_mean", float(np.mean(vals)))
            mlflow.log_metric(f"cv_{k}_std", float(np.std(vals)))

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            write_classification_report(
                train_df[LABEL_COLUMN].tolist(),
                oof_pred.tolist(),
                tmp / "cv_classification_report.txt",
            )
            write_confusion_matrix_png(
                train_df[LABEL_COLUMN].tolist(),
                oof_pred.tolist(),
                tmp / "cv_confusion_matrix.png",
            )
            n_miss = write_misclassified_csv(
                train_df.index.tolist(),
                train_df[TEXT_COLUMN].tolist(),
                train_df[LABEL_COLUMN].tolist(),
                oof_pred.tolist(),
                tmp / "cv_misclassified.csv",
            )
            mlflow.log_metric("cv_n_misclassified", n_miss)
            mlflow.log_artifacts(str(tmp), artifact_path="cv")

        # ---- Final model: train on full train_df, evaluate on holdout ------
        final_pipe, _ = factory(args.seed)
        final_pipe.fit(train_df[TEXT_COLUMN].tolist(), train_df[LABEL_COLUMN].tolist())
        holdout_pred = final_pipe.predict(holdout_df[TEXT_COLUMN].tolist()).tolist()

        holdout_metrics = compute_metrics(holdout_df[LABEL_COLUMN].tolist(), holdout_pred)
        for k, v in holdout_metrics.items():
            mlflow.log_metric(f"holdout_{k}", v)

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            write_classification_report(
                holdout_df[LABEL_COLUMN].tolist(),
                holdout_pred,
                tmp / "holdout_classification_report.txt",
            )
            write_confusion_matrix_png(
                holdout_df[LABEL_COLUMN].tolist(),
                holdout_pred,
                tmp / "holdout_confusion_matrix.png",
            )
            write_misclassified_csv(
                holdout_df.index.tolist(),
                holdout_df[TEXT_COLUMN].tolist(),
                holdout_df[LABEL_COLUMN].tolist(),
                holdout_pred,
                tmp / "holdout_misclassified.csv",
            )
            mlflow.log_artifacts(str(tmp), artifact_path="holdout")

        name_parts = [args.pipeline]
        for k, v in pipe_params.items():
            if k not in ("features", "clf", "pipeline"):
                name_parts.append(f"{k}-{int(v) if isinstance(v, float) else v}")
        registered_name = "_".join(name_parts)

        mlflow.sklearn.log_model(
            final_pipe,
            name=registered_name,
            registered_model_name=registered_name
        )

        print(
            json.dumps(
                {
                    "pipeline": args.pipeline,
                    "cv_f1_macro_mean": float(
                        np.mean([m["f1_macro"] for m in fold_metrics])
                    ),
                    "cv_f1_macro_std": float(
                        np.std([m["f1_macro"] for m in fold_metrics])
                    ),
                    "holdout_f1_macro": holdout_metrics["f1_macro"],
                    "holdout_accuracy": holdout_metrics["accuracy"],
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
