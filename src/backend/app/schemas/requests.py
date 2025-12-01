"""Request payload models."""

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field
from pydantic import HttpUrl


class SimilarityStrategy(str, Enum):
    """Enumerates the supported similarity analysis strategies."""

    EMBEDDINGS = "embeddings"
    MODEL = "model"
    FACE_API = "face_api"


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
    strategies: Optional[List[SimilarityStrategy]] = Field(
        default=None,
        description=(
            "Subset of strategies to execute. When omitted all available strategies are evaluated."
        ),
    )


class ImageSimilarityBase64Request(BaseModel):
    """Payload for requesting an image similarity evaluation from base64 sources."""

    first_image: str = Field(..., description="Base64 encoded reference image")
    second_image: str = Field(..., description="Base64 encoded comparison image")
    strategies: Optional[List[SimilarityStrategy]] = Field(
        default=None,
        description=(
            "Subset of strategies to execute. Defaults to embeddings and model when omitted."
        ),
    )


ImageSimilarityRequest.model_rebuild()
ImageSimilarityBase64Request.model_rebuild()
