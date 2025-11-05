"""Response payload models exported by the API."""

from __future__ import annotations

from datetime import datetime
from typing import List

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(..., example="ok")
    detail: str = Field(..., example="ready")


class LivenessPayload(BaseModel):
    is_live: bool
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str
    timestamp: datetime


class ValidationResponse(BaseModel):
    user_id: str
    is_live: bool
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str
    attempts: int = Field(ge=0)
    samples: List[LivenessPayload]
