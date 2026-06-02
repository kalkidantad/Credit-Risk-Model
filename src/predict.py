"""Offline inference helpers."""

from __future__ import annotations

import os
from typing import Dict

import mlflow
import pandas as pd


def load_sklearn_model(model_uri: str | None = None):
    uri = model_uri or os.environ.get(
        "MLFLOW_MODEL_URI",
        "models:/credit_risk_classifier/Production",
    )
    return mlflow.sklearn.load_model(uri)


def predict_proba_row(model, features: Dict[str, float]) -> float:
    columns = list(features.keys())
    X = pd.DataFrame([features], columns=columns)
    return float(model.predict_proba(X)[0, 1])
