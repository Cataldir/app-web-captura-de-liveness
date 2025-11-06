"""Service orchestration for liveness validation flows."""

from __future__ import annotations

import base64
from statistics import mean
from typing import List, Optional

from app.liveness.engine import LivenessEngine, LivenessResult
from app.schemas import ValidationRequest, ValidationResponse


class LivenessService:
    """Encapsulates liveness evaluation logic for both streaming and batch flows."""

    def __init__(self, engine: Optional[LivenessEngine] = None) -> None:
        self._engine = engine or LivenessEngine()

    def evaluate_stream(self, frame: bytes) -> LivenessResult:
        return self._engine.evaluate(frame)

    def validate_batch(self, request: ValidationRequest) -> ValidationResponse:
        results: List[LivenessResult] = []
        for encoded in request.samples:
            results.append(self._engine.evaluate(self._decode_sample(encoded)))

        if not results:
            aggregate_is_live = False
            aggregate_confidence = 0.0
            reason = "No samples provided"
        else:
            aggregate_is_live = sum(result.is_live for result in results) >= (
                len(results) / 2
            )
            aggregate_confidence = round(
                mean(result.confidence for result in results),
                3,
            )
            reason = (
                "Majority indicates liveness"
                if aggregate_is_live
                else "Majority indicates spoof"
            )

        return ValidationResponse(
            user_id=request.user_id,
            is_live=aggregate_is_live,
            confidence=aggregate_confidence,
            reason=reason,
            attempts=len(results),
            samples=[self._serialize_result(result) for result in results],
        )

    def reset_session(self) -> None:
        self._engine.reset()

    @staticmethod
    def _decode_sample(encoded: str) -> bytes:
        return base64.b64decode(encoded, validate=True)

    @staticmethod
    def _serialize_result(result: LivenessResult) -> dict[str, object]:
        return {
            "is_live": result.is_live,
            "confidence": result.confidence,
            "reason": result.reason,
            "timestamp": result.timestamp.isoformat(),
        }
