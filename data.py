from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from labels import LABELS

DEFAULT_DATASET_PATH = Path("dataset.csv")
DEFAULT_SEED = 42
DEFAULT_HOLDOUT_SIZE = 0.10
DEFAULT_N_SPLITS = 5

TEXT_COLUMN = "Content"
LABEL_COLUMN = "Classification"


@dataclass(frozen=True)
class DatasetStats:
    n_raw: int
    n_normalized_labels: int
    n_dropped_duplicates: int
    n_after_clean: int
    label_counts: dict[str, int]


def load_dataset(path: Path = DEFAULT_DATASET_PATH) -> tuple[pd.DataFrame, DatasetStats]:
    """Load the labeled dataset, title-case labels, drop duplicate Content rows."""
    raw = pd.read_csv(path, encoding="utf-8-sig")

    if list(raw.columns) != [TEXT_COLUMN, LABEL_COLUMN]:
        raise ValueError(
            f"Expected columns [{TEXT_COLUMN!r}, {LABEL_COLUMN!r}], got {list(raw.columns)}"
        )
    if raw[TEXT_COLUMN].isna().any():
        raise ValueError(f"{TEXT_COLUMN} contains nulls; refusing to proceed")

    normalized = raw[LABEL_COLUMN].str.strip().str.title()
    n_normalized = int((normalized != raw[LABEL_COLUMN]).sum())

    unknown = set(normalized.unique()) - set(LABELS)
    if unknown:
        raise ValueError(f"Unexpected labels after normalization: {sorted(unknown)}")

    clean = raw.assign(**{LABEL_COLUMN: normalized})
    before = len(clean)
    clean = clean.drop_duplicates(subset=[TEXT_COLUMN]).reset_index(drop=True)
    n_dropped = before - len(clean)

    stats = DatasetStats(
        n_raw=len(raw),
        n_normalized_labels=n_normalized,
        n_dropped_duplicates=n_dropped,
        n_after_clean=len(clean),
        label_counts={k: int(v) for k, v in clean[LABEL_COLUMN].value_counts().items()},
    )
    return clean, stats


def holdout_split(
    df: pd.DataFrame,
    holdout_size: float = DEFAULT_HOLDOUT_SIZE,
    seed: int = DEFAULT_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train/holdout split. Holdout stays untouched until final evaluation.

    Indices from `df` are preserved on both halves so they act as stable row
    identifiers across runs / seeds (used for cross-run misclassification diffs).
    """
    return train_test_split(
        df,
        test_size=holdout_size,
        stratify=df[LABEL_COLUMN],
        random_state=seed,
    )


def stratified_folds(
    df: pd.DataFrame,
    n_splits: int = DEFAULT_N_SPLITS,
    seed: int = DEFAULT_SEED,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (train_idx, val_idx) tuples for stratified k-fold over df."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    yield from skf.split(np.zeros(len(df)), df[LABEL_COLUMN])
