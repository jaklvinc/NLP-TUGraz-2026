from __future__ import annotations

from typing import Any, Callable

import keras
from sklearn.decomposition import TruncatedSVD

import stage1.nn_model
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

def tfidf_nn(seed: int) -> tuple[Pipeline, dict[str, Any]]:
    svd_components = 200
    epochs = 30
    layer_size = 64
    pipe = Pipeline(
        steps=[
            ("tfidf", build_tfidf()),
            ('svd', TruncatedSVD(n_components=svd_components, random_state=seed)),
            (
                "clf",
                stage1.nn_model.DeterministicNN(
                    model=stage1.nn_model.build_one_layer_nn,
                    seed=seed,
                    model_kwargs= {"layer_size": layer_size},
                    fit_kwargs={"epochs": epochs, "verbose": 0 }
                )
            ),
        ]
    )
    params = {
        "features": "tfidf_1_2gram_mindf2",
        "clf": "nn",
        "epochs": epochs,
        "svd_n_components": svd_components,
        "seed": seed,
        "layer_size": layer_size
    }
    return pipe, params

PIPELINES: dict[str, PipelineFactory] = {
    "tfidf_logreg": tfidf_logreg,
    "tfidf_svm": tfidf_svm,
    "tfidf_nn":tfidf_nn,
}
