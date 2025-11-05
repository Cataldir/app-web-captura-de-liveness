from __future__ import annotations

import base64

from app.schemas import ValidationRequest
from app.services.liveness_service import LivenessService


def test_validate_batch_without_samples_returns_negative() -> None:
    service = LivenessService()
    request = ValidationRequest(user_id="user-123", samples=[])
    response = service.validate_batch(request)
    assert response.is_live is False
    assert response.confidence == 0.0
    assert response.reason == "No samples provided"


def test_validate_batch_with_samples_returns_structured_response() -> None:
    service = LivenessService()
    payload = base64.b64encode(b"frame").decode()
    request = ValidationRequest(user_id="user-456", samples=[payload, payload])
    response = service.validate_batch(request)
    assert response.user_id == "user-456"
    assert response.attempts == 2
    assert len(response.samples) == 2
