from __future__ import annotations

import base64

import httpx
import numpy as np
import pytest  # type: ignore[import]
from fastapi.testclient import TestClient

from app.services.embeddings_service import EmbeddingsService
from app.services.faceapi_service import FaceAPIResult, FaceAPIService
from app.services.model_service import ModelComparisonResult, ModelService


def test_health_probe(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200, response.text
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
@pytest.mark.parametrize(
    "first_url,second_url,expected_status",
    [
        ("https://assets.example/white.png", "https://assets.example/white.png", "approved"),
        ("https://assets.example/white.png", "https://assets.example/black.png", "not approved"),
    ],
)
def test_image_similarity_endpoint(
    monkeypatch: pytest.MonkeyPatch,
    client: TestClient,
    first_url: str,
    second_url: str,
    expected_status: str,
) -> None:
    image_map = {
        "https://assets.example/white.png": b"WHITE",
        "https://assets.example/black.png": b"BLACK",
    }

    embeddings_map = {
        image_map["https://assets.example/white.png"]: np.array([1.0, 0.0], dtype=np.float32),
        image_map["https://assets.example/black.png"]: np.array([0.0, 1.0], dtype=np.float32),
    }

    async def mock_get(
        self: httpx.AsyncClient,
        url: str,
        *args: object,
        **kwargs: object,
    ) -> httpx.Response:
        content = image_map.get(url)
        if content is None:
            return httpx.Response(
                status_code=404,
                request=httpx.Request("GET", url),
            )
        return httpx.Response(
            status_code=200,
            content=content,
            request=httpx.Request("GET", url),
        )

    monkeypatch.setattr(httpx.AsyncClient, "get", mock_get)

    def fake_generate(
        _self: EmbeddingsService,
        payload: bytes,
    ) -> np.ndarray:
        return embeddings_map[payload]

    monkeypatch.setattr(EmbeddingsService, "_generate_embedding", fake_generate)

    def fake_model_compare(
        _self: ModelService,
        _first: bytes,
        _second: bytes,
    ) -> ModelComparisonResult:
        same_person = expected_status == "approved"
        explanation = "Images match" if same_person else "Faces differ"
        similarity_value = 0.99 if same_person else 0.15
        return ModelComparisonResult(
            similarity=similarity_value,
            status=expected_status,
            same_person=same_person,
            explanation=explanation,
        )

    def fake_face_compare(
        _self: FaceAPIService,
        _reference_url: str,
        _candidate_url: str,
    ) -> FaceAPIResult:
        is_identical = expected_status == "approved"
        confidence = 0.985 if is_identical else 0.2
        return FaceAPIResult(
            similarity=confidence,
            status=expected_status,
            is_identical=is_identical,
            confidence=confidence,
            reason="Verification simulated",
        )

    monkeypatch.setattr(ModelService, "compare_images", fake_model_compare)
    monkeypatch.setattr(FaceAPIService, "compare_from_urls", fake_face_compare)

    response = client.post(
        "/images/similarity",
        json={
            "first_image_url": first_url,
            "second_image_url": second_url,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == expected_status
    assert payload["embeddings"]["status"] == expected_status
    assert payload["model"]["status"] == expected_status
    assert payload["face_api"]["status"] == expected_status
    assert payload["strategies"] == ["embeddings", "model", "face_api"]

    expected_embedding_similarity = 1.0 if expected_status == "approved" else 0.0
    expected_model_similarity = 0.99 if expected_status == "approved" else 0.15
    expected_face_similarity = 0.985 if expected_status == "approved" else 0.2
    expected_similarity = (
        expected_embedding_similarity
        + expected_model_similarity
        + expected_face_similarity
    ) / 3

    assert payload["similarity"] == pytest.approx(expected_similarity, rel=1e-6)


@pytest.mark.parametrize(
    "first_payload,second_payload,expected_status",
    [
        (b"WHITE", b"WHITE", "approved"),
        (b"WHITE", b"BLACK", "not approved"),
    ],
)
def test_image_similarity_base64_endpoint(
    monkeypatch: pytest.MonkeyPatch,
    client: TestClient,
    first_payload: bytes,
    second_payload: bytes,
    expected_status: str,
) -> None:
    embeddings_map = {
        b"WHITE": np.array([1.0, 0.0], dtype=np.float32),
        b"BLACK": np.array([0.0, 1.0], dtype=np.float32),
    }

    def fake_generate(
        _self: EmbeddingsService,
        payload: bytes,
    ) -> np.ndarray:
        return embeddings_map[payload]

    monkeypatch.setattr(EmbeddingsService, "_generate_embedding", fake_generate)

    def fake_model_compare(
        _self: ModelService,
        _first: bytes,
        _second: bytes,
    ) -> ModelComparisonResult:
        same_person = expected_status == "approved"
        explanation = "Images match" if same_person else "Faces differ"
        similarity_value = 0.99 if same_person else 0.15
        return ModelComparisonResult(
            similarity=similarity_value,
            status=expected_status,
            same_person=same_person,
            explanation=explanation,
        )

    monkeypatch.setattr(ModelService, "compare_images", fake_model_compare)

    response = client.post(
        "/images/similarity/base64",
        json={
            "first_image": base64.b64encode(first_payload).decode(),
            "second_image": base64.b64encode(second_payload).decode(),
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == expected_status
    assert payload["strategies"] == ["embeddings", "model"]
    assert payload["face_api"] is None

    expected_embedding_similarity = 1.0 if expected_status == "approved" else 0.0
    expected_model_similarity = 0.99 if expected_status == "approved" else 0.15
    expected_similarity = (expected_embedding_similarity + expected_model_similarity) / 2
    assert payload["similarity"] == pytest.approx(expected_similarity, rel=1e-6)


def test_image_similarity_base64_requires_urls_for_face_api(client: TestClient) -> None:
    response = client.post(
        "/images/similarity/base64",
        json={
            "first_image": base64.b64encode(b"WHITE").decode(),
            "second_image": base64.b64encode(b"BLACK").decode(),
            "strategies": ["face_api"],
        },
    )

    assert response.status_code == 400, response.text
    assert "Face API strategy requires accessible image URLs" in response.text
