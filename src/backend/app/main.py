"""FastAPI application exposing liveness capabilities."""

from __future__ import annotations

import asyncio
import base64
from typing import Annotated, Literal, cast

import httpx
from fastapi import Depends, FastAPI, HTTPException, WebSocket, WebSocketDisconnect

from app.liveness.engine import LivenessEngine, create_detector_from_env
from app.schemas import (
    EmbeddingsSimilarityPayload,
    FaceAPISimilarityPayload,
    HealthResponse,
    ImageSimilarityRequest,
    ImageSimilarityResponse,
    ModelSimilarityPayload,
    ValidationRequest,
    ValidationResponse,
)
from app.services.embeddings_service import EmbeddingsService
from app.services.faceapi_service import FaceAPIService
from app.services.model_service import ModelService
from app.services.liveness_service import LivenessService

app = FastAPI(title="Liveness API", version="0.1.0")


def get_liveness_service() -> LivenessService:
    engine = LivenessEngine(detector=create_detector_from_env())
    return LivenessService(engine=engine)


def get_embeddings_service() -> EmbeddingsService:
    return EmbeddingsService()


def get_model_service() -> ModelService:
    return ModelService()


def get_faceapi_service() -> FaceAPIService:
    return FaceAPIService()


async def _download_image(client: httpx.AsyncClient, url: str) -> bytes:
    response = await client.get(url)
    response.raise_for_status()
    if not response.content:
        raise ValueError("Image endpoint returned an empty payload")
    return response.content


async def _load_images(first_url: str, second_url: str) -> tuple[bytes, bytes]:
    async with httpx.AsyncClient(timeout=10.0) as client:
        return await asyncio.gather(
            _download_image(client, first_url),
            _download_image(client, second_url),
        )


@app.get("/", response_model=HealthResponse)
async def health_probe() -> HealthResponse:
    return HealthResponse(status="ok", detail="ready")


@app.post("/validate", response_model=ValidationResponse)
async def validate_payload(
    payload: ValidationRequest,
    service: Annotated[LivenessService, Depends(get_liveness_service)],
) -> ValidationResponse:
    try:
        return service.validate_batch(payload)
    except (ValueError, base64.binascii.Error) as exc:  # type: ignore[attr-defined]
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.websocket("/ws/liveness")
async def liveness_socket(
    websocket: WebSocket,
    service: Annotated[LivenessService, Depends(get_liveness_service)],
) -> None:
    await websocket.accept()
    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                service.reset_session()
                break
            payload = message.get("bytes") or message.get("text")
            if payload is None:
                continue

            frame = payload if isinstance(payload, bytes) else payload.encode()
            result = service.evaluate_stream(frame)
            await websocket.send_json(
                {
                    "is_live": result.is_live,
                    "confidence": result.confidence,
                    "reason": result.reason,
                    "timestamp": result.timestamp.isoformat(),
                }
            )
    except WebSocketDisconnect:
        service.reset_session()
    except Exception as exc:  # pragma: no cover - logged in production
        await websocket.close(code=1011)
        raise exc


@app.post("/images/similarity", response_model=ImageSimilarityResponse)
async def compare_images(
    payload: ImageSimilarityRequest,
    embeddings_service: Annotated[EmbeddingsService, Depends(get_embeddings_service)],
    model_service: Annotated[ModelService, Depends(get_model_service)],
    face_service: Annotated[FaceAPIService, Depends(get_faceapi_service)],
) -> ImageSimilarityResponse:
    try:
        first_image, second_image = await _load_images(
            str(payload.first_image_url),
            str(payload.second_image_url),
        )
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=502, detail="Unable to fetch image resources"
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        embeddings_result = embeddings_service.evaluate_pair(first_image, second_image)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    try:
        model_result = model_service.compare_images(first_image, second_image)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    try:
        face_result = face_service.compare_from_urls(
            str(payload.first_image_url),
            str(payload.second_image_url),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    embeddings_status = cast(Literal["approved", "not approved"], embeddings_result.status)
    model_status = cast(Literal["approved", "not approved"], model_result.status)
    face_status = cast(Literal["approved", "not approved"], face_result.status)

    return ImageSimilarityResponse(
        similarity=embeddings_result.similarity,
        status=embeddings_status,
        embeddings=EmbeddingsSimilarityPayload(
            similarity=embeddings_result.similarity,
            status=embeddings_status,
        ),
        model=ModelSimilarityPayload(
            similarity=model_result.similarity,
            status=model_status,
            same_person=model_result.same_person,
            explanation=model_result.explanation,
        ),
        face_api=FaceAPISimilarityPayload(
            similarity=face_result.similarity,
            status=face_status,
            is_identical=face_result.is_identical,
            confidence=face_result.confidence,
            reason=face_result.reason,
        ),
    )
