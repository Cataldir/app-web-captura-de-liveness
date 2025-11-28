from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest  # type: ignore[import]

from app.services.model_service import ModelService


class _FakeCompletions:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = json.dumps(payload)

    def create(self, *args: object, **kwargs: object) -> SimpleNamespace:  # noqa: D401
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self._payload))]
        )


class _FakeClient:
    def __init__(self, payload: dict[str, object]) -> None:
        self.chat = SimpleNamespace(completions=_FakeCompletions(payload))


def test_model_service_state_transitions(monkeypatch: pytest.MonkeyPatch) -> None:
    initial_payload = {
        "similarity": 0.995,
        "same_person": True,
        "explanation": "Faces share identical traits.",
    }

    followup_payload = {
        "similarity": 0.4,
        "same_person": False,
        "explanation": "Distinct facial structure detected.",
    }

    client = _FakeClient(initial_payload)
    service = ModelService(threshold=0.99, client=client)

    result = service.compare_images(b"first", b"second")
    assert result.status == "approved"
    assert np.isclose(result.similarity, 0.995)
    assert result.same_person is True
    assert "identical" in result.explanation

    client.chat.completions = _FakeCompletions(followup_payload)
    result = service.compare_images(b"first", b"second")
    assert result.status == "not approved"
    assert result.same_person is False


def test_model_service_handles_invalid_images() -> None:
    service = ModelService(threshold=0.99, client=_FakeClient({
        "similarity": 0.995,
        "same_person": True,
        "explanation": "",
    }))

    with pytest.raises(ValueError):
        service.compare_images(b"", b"second")

    with pytest.raises(ValueError):
        service.compare_images(b"first", b"")