"""FastAPI application exposing liveness capabilities."""

from __future__ import annotations

import base64
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, WebSocket, WebSocketDisconnect

from app.liveness.engine import LivenessEngine, create_detector_from_env
from app.schemas import HealthResponse, ValidationRequest, ValidationResponse
from app.services.liveness_service import LivenessService

app = FastAPI(title="Liveness API", version="0.1.0")


def get_liveness_service() -> LivenessService:
    engine = LivenessEngine(detector=create_detector_from_env())
    return LivenessService(engine=engine)


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
