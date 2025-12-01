"""Response payload models exported by the API."""

from __future__ import annotations

from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field("ok")
    detail: str = Field("ready")


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


class EmbeddingsSimilarityPayload(BaseModel):
    similarity: float = Field(..., ge=0.0, le=1.0)
    status: Literal["approved", "not approved"]


class ModelSimilarityPayload(BaseModel):
    similarity: float = Field(..., ge=0.0, le=1.0)
    status: Literal["approved", "not approved"]
    same_person: bool
    explanation: str


class FaceAPISimilarityPayload(BaseModel):
    similarity: float = Field(..., ge=0.0, le=1.0)
    status: Literal["approved", "not approved"]
    is_identical: bool
    confidence: float = Field(..., ge=0.0, le=1.0)
    reason: str


class ImageSimilarityResponse(BaseModel):
    similarity: float = Field(ge=0.0, le=1.0)
    status: Literal["approved", "not approved"]
    strategies: List[Literal["embeddings", "model", "face_api"]] = Field(
        default_factory=list,
        description="Strategies executed when computing this response",
    )
    embeddings: Optional[EmbeddingsSimilarityPayload] = None
    model: Optional[ModelSimilarityPayload] = None
    face_api: Optional[FaceAPISimilarityPayload] = None
