# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "marimo",
#     "mlflow",
#     "pandas",
# ]
# ///

import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import mlflow
    import pandas as pd

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    return mlflow, mo, pd


@app.cell
def _(mo):
    mo.md(r"""
    # Cross-run misclassification diff
    """)
    return


@app.cell
def _(mlflow, pd):
    runs = mlflow.search_runs(experiment_names=["fact-vs-opinion"])
    # Most recent run per pipeline name (in case of re-runs).
    latest = (
        runs.sort_values("start_time", ascending=False)
        .drop_duplicates("tags.mlflow.runName")
        .set_index("tags.mlflow.runName")
    )
    run_ids = latest["run_id"].to_dict()

    def load_miss(name):
        uri = f"runs:/{run_ids[name]}/cv/cv_misclassified.csv"
        local = mlflow.artifacts.download_artifacts(uri)
        return pd.read_csv(local).set_index("index")

    logreg_miss = load_miss("tfidf_logreg")
    svm_miss = load_miss("tfidf_svm")
    return logreg_miss, svm_miss


@app.cell
def _(logreg_miss, mo, svm_miss):
    only_logreg_wrong = logreg_miss.index.difference(svm_miss.index)
    only_svm_wrong = svm_miss.index.difference(logreg_miss.index)
    both_wrong = logreg_miss.index.intersection(svm_miss.index)

    summary = mo.md(
        f"""
        | set | count |
        |---|---|
        | only logreg wrong | {len(only_logreg_wrong)} |
        | only SVM wrong    | {len(only_svm_wrong)} |
        | both wrong        | {len(both_wrong)} |
        """
    )
    summary
    return both_wrong, only_logreg_wrong, only_svm_wrong


@app.cell
def _(both_wrong, logreg_miss, mo):
    mo.vstack(
        [
            mo.md("## Wrong in both"),
            logreg_miss.loc[both_wrong].sort_index(),
        ]
    )
    return


@app.cell
def _(logreg_miss, mo, only_logreg_wrong):
    mo.vstack(
        [
            mo.md("## Only logreg wrong"),
            logreg_miss.loc[only_logreg_wrong].sort_index(),
        ]
    )
    return


@app.cell
def _(mo, only_svm_wrong, svm_miss):
    mo.vstack(
        [
            mo.md("## Only SVM wrong"),
            svm_miss.loc[only_svm_wrong].sort_index(),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
