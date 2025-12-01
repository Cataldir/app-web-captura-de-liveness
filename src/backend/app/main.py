"""FastAPI application exposing liveness capabilities."""

import asyncio
import base64
from typing import Annotated, Literal, Sequence, cast

import httpx
from fastapi import Body, Depends, FastAPI, HTTPException, WebSocket, WebSocketDisconnect

from app.liveness.engine import LivenessEngine, create_detector_from_env
from app.schemas import (
    EmbeddingsSimilarityPayload,
    FaceAPISimilarityPayload,
    HealthResponse,
    ImageSimilarityBase64Request,
    ImageSimilarityRequest,
    ImageSimilarityResponse,
    ModelSimilarityPayload,
    SimilarityStrategy,
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


def get_faceapi_service_optional() -> FaceAPIService | None:
    """Returns a Face API service instance when configuration is available."""

    try:
        return get_faceapi_service()
    except RuntimeError:
        return None


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


def _normalize_strategies(strategies: Sequence[SimilarityStrategy]) -> list[SimilarityStrategy]:
    """Removes duplicates while preserving declaration order."""

    seen: set[SimilarityStrategy] = set()
    ordered: list[SimilarityStrategy] = []
    for strategy in strategies:
        if strategy in seen:
            continue
        seen.add(strategy)
        ordered.append(strategy)
    return ordered


def _evaluate_similarity_strategies(
    *,
    strategies: Sequence[SimilarityStrategy],
    first_image: bytes,
    second_image: bytes,
    first_image_url: str | None,
    second_image_url: str | None,
    embeddings_service: EmbeddingsService,
    model_service: ModelService,
    face_service: FaceAPIService | None,
) -> ImageSimilarityResponse:
    normalized = _normalize_strategies(strategies)
    if not normalized:
        raise HTTPException(status_code=400, detail="At least one strategy must be provided")

    executed: list[SimilarityStrategy] = []
    similarity_values: list[float] = []
    status_values: list[Literal["approved", "not approved"]] = []

    embeddings_payload: EmbeddingsSimilarityPayload | None = None
    model_payload: ModelSimilarityPayload | None = None
    face_payload: FaceAPISimilarityPayload | None = None

    if SimilarityStrategy.EMBEDDINGS in normalized:
        try:
            embeddings_result = embeddings_service.evaluate_pair(first_image, second_image)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

        embeddings_status = cast(
            Literal["approved", "not approved"], embeddings_result.status
        )
        embeddings_payload = EmbeddingsSimilarityPayload(
            similarity=embeddings_result.similarity,
            status=embeddings_status,
        )
        executed.append(SimilarityStrategy.EMBEDDINGS)
        similarity_values.append(embeddings_result.similarity)
        status_values.append(embeddings_status)

    if SimilarityStrategy.MODEL in normalized:
        try:
            model_result = model_service.compare_images(first_image, second_image)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

        model_status = cast(Literal["approved", "not approved"], model_result.status)
        model_payload = ModelSimilarityPayload(
            similarity=model_result.similarity,
            status=model_status,
            same_person=model_result.same_person,
            explanation=model_result.explanation,
        )
        executed.append(SimilarityStrategy.MODEL)
        similarity_values.append(model_result.similarity)
        status_values.append(model_status)

    if SimilarityStrategy.FACE_API in normalized:
        if not first_image_url or not second_image_url:
            raise HTTPException(
                status_code=400,
                detail="Face API strategy requires accessible image URLs",
            )
        if face_service is None:
            raise HTTPException(
                status_code=503,
                detail="Face API service is not configured",
            )
        try:
            face_result = face_service.compare_from_urls(first_image_url, second_image_url)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

        face_status = cast(Literal["approved", "not approved"], face_result.status)
        face_payload = FaceAPISimilarityPayload(
            similarity=face_result.similarity,
            status=face_status,
            is_identical=face_result.is_identical,
            confidence=face_result.confidence,
            reason=face_result.reason,
        )
        executed.append(SimilarityStrategy.FACE_API)
        similarity_values.append(face_result.similarity)
        status_values.append(face_status)

    if not similarity_values:
        raise HTTPException(
            status_code=400,
            detail="No strategies were executed; review the request payload.",
        )

    overall_similarity = sum(similarity_values) / len(similarity_values)
    overall_status: Literal["approved", "not approved"] = (
        "approved" if all(status == "approved" for status in status_values) else "not approved"
    )

    # The aggregated response lists the strategies executed so the frontend can render matching rows.
    return ImageSimilarityResponse(
        similarity=overall_similarity,
        status=overall_status,
        strategies=[strategy.value for strategy in executed],
        embeddings=embeddings_payload,
        model=model_payload,
        face_api=face_payload,
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
    payload: ImageSimilarityRequest = Body(...),
    embeddings_service: EmbeddingsService = Depends(get_embeddings_service),
    model_service: ModelService = Depends(get_model_service),
    face_service: FaceAPIService = Depends(get_faceapi_service),
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

    strategies: Sequence[SimilarityStrategy] = payload.strategies or (
        SimilarityStrategy.EMBEDDINGS,
        SimilarityStrategy.MODEL,
        SimilarityStrategy.FACE_API,
    )

    return _evaluate_similarity_strategies(
        strategies=strategies,
        first_image=first_image,
        second_image=second_image,
        first_image_url=str(payload.first_image_url),
        second_image_url=str(payload.second_image_url),
        embeddings_service=embeddings_service,
        model_service=model_service,
        face_service=face_service,
    )


@app.post("/images/similarity/base64", response_model=ImageSimilarityResponse)
async def compare_images_base64(
    payload: ImageSimilarityBase64Request = Body(...),
    embeddings_service: EmbeddingsService = Depends(get_embeddings_service),
    model_service: ModelService = Depends(get_model_service),
    face_service: FaceAPIService | None = Depends(get_faceapi_service_optional),
) -> ImageSimilarityResponse:
    try:
        first_image = base64.b64decode(payload.first_image, validate=True)
        second_image = base64.b64decode(payload.second_image, validate=True)
    except (ValueError, base64.binascii.Error) as exc:  # type: ignore[attr-defined]
        raise HTTPException(status_code=400, detail="Invalid base64 image payload") from exc

    strategies: Sequence[SimilarityStrategy] = payload.strategies or (
        SimilarityStrategy.EMBEDDINGS,
        SimilarityStrategy.MODEL,
    )

    return _evaluate_similarity_strategies(
        strategies=strategies,
        first_image=first_image,
        second_image=second_image,
        first_image_url=None,
        second_image_url=None,
        embeddings_service=embeddings_service,
        model_service=model_service,
        face_service=face_service,
    )
