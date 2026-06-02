"""Request and response schemas for the scoring API."""

from __future__ import annotations

from typing import Dict, Optional

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Feature vector for one customer (post feature-engineering pipeline)."""

    features: Dict[str, float] = Field(
        ...,
        description="Map of feature name to numeric value matching training columns.",
    )


class PredictResponse(BaseModel):
    risk_probability: float = Field(..., ge=0.0, le=1.0)
    model_uri: Optional[str] = Field(None, description="Resolved MLflow model URI for auditing.")


class HealthResponse(BaseModel):
    status: str = "ok"
