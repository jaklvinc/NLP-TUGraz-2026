# Eval shared across tasks.
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from labels import LABELS


def compute_metrics(y_true: list[str], y_pred: list[str]) -> dict[str, float]:
    per_class = f1_score(y_true, y_pred, labels=list(LABELS), average=None)
    return {
        "f1_macro": float(f1_score(y_true, y_pred, labels=list(LABELS), average="macro")),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_fact": float(per_class[LABELS.index("Fact")]),
        "f1_opinion": float(per_class[LABELS.index("Opinion")]),
    }


def write_classification_report(y_true: list[str], y_pred: list[str], path: Path) -> None:
    report = classification_report(y_true, y_pred, labels=list(LABELS), digits=4)
    path.write_text(report)


def write_confusion_matrix_png(y_true: list[str], y_pred: list[str], path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=list(LABELS))
    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay(cm, display_labels=list(LABELS)).plot(ax=ax, colorbar=False)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def write_misclassified_csv(
    indices: list[int],
    texts: list[str],
    y_true: list[str],
    y_pred: list[str],
    path: Path,
) -> int:
    """Write a CSV of misclassified rows.

    `indices` are stable identifiers from the source DataFrame so misses
    can be correlated across runs.
    """
    df = pd.DataFrame(
        {"index": indices, "true": y_true, "pred": y_pred, "text": texts}
    )
    misses = df[df["true"] != df["pred"]]
    misses.to_csv(path, index=False)
    return len(misses)
