from __future__ import annotations

from typing import Any, Callable

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from stage1.features import build_tfidf

PipelineFactory = Callable[[int], tuple[Pipeline, dict[str, Any]]]


def tfidf_logreg(seed: int) -> tuple[Pipeline, dict[str, Any]]:
    pipe = Pipeline(
        steps=[
            ("tfidf", build_tfidf()),
            (
                "clf",
                LogisticRegression(
                    C=1.0,
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
        ]
    )
    params = {
        "features": "tfidf_1_2gram_mindf2",
        "clf": "logreg",
        "C": 1.0,
        "class_weight": "balanced",
    }
    return pipe, params


def tfidf_svm(seed: int) -> tuple[Pipeline, dict[str, Any]]:
    pipe = Pipeline(
        steps=[
            ("tfidf", build_tfidf()),
            (
                "clf",
                LinearSVC(
                    C=1.0,
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
        ]
    )
    params = {
        "features": "tfidf_1_2gram_mindf2",
        "clf": "linear_svc",
        "C": 1.0,
        "class_weight": "balanced",
    }
    return pipe, params


PIPELINES: dict[str, PipelineFactory] = {
    "tfidf_logreg": tfidf_logreg,
    "tfidf_svm": tfidf_svm,
}
