from __future__ import annotations

import numpy as np
import pytest  # type: ignore[import]

from app.services.embeddings_service import EmbeddingsService


def test_service_transitions_between_states(monkeypatch: pytest.MonkeyPatch) -> None:
    embeddings_map = {
        b"reference": np.array([1.0, 0.0], dtype=np.float32),
        b"identical": np.array([1.0, 0.0], dtype=np.float32),
        b"different": np.array([0.0, 1.0], dtype=np.float32),
    }

    def fake_generate(_self: EmbeddingsService, payload: bytes) -> np.ndarray:
        return embeddings_map[payload]

    monkeypatch.setattr(EmbeddingsService, "_generate_embedding", fake_generate)

    service = EmbeddingsService(threshold=0.99)

    first_result = service.evaluate_pair(b"reference", b"identical")
    assert first_result.status == "approved"
    assert first_result.similarity == pytest.approx(1.0, abs=1e-6)

    second_result = service.evaluate_pair(b"reference", b"different")
    assert second_result.status == "not approved"
    assert second_result.similarity < 0.99


def test_service_rejects_invalid_payloads(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_generate(
        _self: EmbeddingsService,
        _payload: bytes,
    ) -> np.ndarray:  # pragma: no cover - defensive
        return np.array([1.0], dtype=np.float32)

    monkeypatch.setattr(EmbeddingsService, "_generate_embedding", fake_generate)

    service = EmbeddingsService(threshold=0.99)
    with pytest.raises(ValueError):
        service.evaluate_pair(b"", b"image")

    with pytest.raises(ValueError):
        service.evaluate_pair(b"image", b"")
