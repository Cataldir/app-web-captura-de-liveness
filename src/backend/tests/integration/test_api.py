from __future__ import annotations

import base64

from fastapi.testclient import TestClient


def test_health_probe(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"


def test_validate_endpoint_returns_expected_schema(client: TestClient) -> None:
    sample = base64.b64encode(b"frame").decode()
    response = client.post(
        "/validate",
        json={
            "user_id": "user-789",
            "samples": [sample],
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["user_id"] == "user-789"
    assert "confidence" in payload


def test_websocket_stream_returns_liveness_updates(client: TestClient) -> None:
    with client.websocket_connect("/ws/liveness") as websocket:
        websocket.send_text("test-frame")
        message = websocket.receive_json()
        assert "is_live" in message
        assert "confidence" in message
