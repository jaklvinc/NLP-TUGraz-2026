from __future__ import annotations

import mlflow

DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"
DEFAULT_EXPERIMENT = "fact-vs-opinion"


def configure(
    experiment: str = DEFAULT_EXPERIMENT,
    tracking_uri: str = DEFAULT_TRACKING_URI,
) -> None:
    """Set tracking URI and active experiment, call on top of run scripts"""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
