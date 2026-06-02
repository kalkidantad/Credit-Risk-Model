"""FastAPI application for credit risk scoring."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import List

import pandas as pd
from fastapi import FastAPI, HTTPException

from src.api.pydantic_models import HealthResponse, PredictRequest, PredictResponse
from src.predict import load_sklearn_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Bati Bank Credit Risk Scoring", version="1.0.0")

_model = None
_feature_columns: List[str] = []
_model_uri: str | None = None


def _load_feature_columns() -> List[str]:
    path = os.environ.get(
        "FEATURE_COLUMNS_PATH",
        str(Path("data/processed/feature_columns.json")),
    )
    p = Path(path)
    if not p.exists():
        logger.warning("Feature list not found at %s; validation disabled.", p)
        return []
    with p.open(encoding="utf-8") as f:
        return list(json.load(f))


@app.on_event("startup")
def startup_event() -> None:
    global _model, _feature_columns, _model_uri
    _model_uri = os.environ.get("MLFLOW_MODEL_URI", "models:/credit_risk_classifier/Production")
    _model = load_sklearn_model(_model_uri)
    _feature_columns = _load_feature_columns()
    logger.info("Loaded model from %s (%d features expected)", _model_uri, len(_feature_columns))


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse()


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if _feature_columns:
        missing = [c for c in _feature_columns if c not in req.features]
        extra = [c for c in req.features if c not in _feature_columns]
        if missing or extra:
            raise HTTPException(
                status_code=422,
                detail={"missing": missing, "unexpected": extra},
            )
        ordered = {c: float(req.features[c]) for c in _feature_columns}
    else:
        ordered = {k: float(v) for k, v in req.features.items()}
    X = pd.DataFrame([ordered])
    proba = float(_model.predict_proba(X)[0, 1])
    return PredictResponse(risk_probability=proba, model_uri=_model_uri)
