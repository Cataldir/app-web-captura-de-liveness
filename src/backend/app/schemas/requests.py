"""Request payload models."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field
from pydantic import HttpUrl


class ValidationRequest(BaseModel):
    """Represents a validation request posted by the frontend."""

    user_id: str = Field(..., description="Identifier of the person being validated")
    samples: List[str] = Field(
        default_factory=list, description="Base64 encoded media samples"
    )
    metadata: Optional[dict[str, str]] = Field(
        default=None,
        description="Additional attributes about the capture session",
    )


class ImageSimilarityRequest(BaseModel):
    """Payload for requesting an image similarity evaluation."""

    first_image_url: HttpUrl = Field(
        ..., description="URL containing the reference image"
    )
    second_image_url: HttpUrl = Field(
        ..., description="URL containing the comparison image"
    )
