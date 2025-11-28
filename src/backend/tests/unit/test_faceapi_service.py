from __future__ import annotations

from contextlib import AbstractContextManager, contextmanager
from types import SimpleNamespace
from typing import Any, Callable, ContextManager, Iterator, cast

import pytest  # type: ignore[import]

from azure.ai.vision.face.models import QualityForRecognition

from app.services.faceapi_service import ClientFactory, FaceAPIService


class _FakePoller:
    def wait(self) -> None:  # pragma: no cover - trivial
        return None


class _FakeLargePersonGroup:
    def __init__(self) -> None:
        self.deleted = False

    def create(self, **_: object) -> None:
        return None

    def create_person(self, **_: object) -> SimpleNamespace:
        return SimpleNamespace(person_id="person-1")

    def add_face_from_url(self, **_: object) -> None:
        return None

    def begin_train(self, **_: object) -> _FakePoller:
        return _FakePoller()

    def delete(self, group_id: str) -> None:
        self.deleted = True
        self.group_id = group_id


class _FakeAdminClient(AbstractContextManager):
    def __init__(self) -> None:
        self.large_person_group = _FakeLargePersonGroup()

    def __exit__(self, *args: object) -> bool:  # pragma: no cover - trivial
        return False


class _FakeFaceClient(AbstractContextManager):
    def __init__(self, detection_sequences: list[list[SimpleNamespace]], verify_results: list[SimpleNamespace]) -> None:
        self._detection_sequences = detection_sequences
        self._verify_results = verify_results

    def detect_from_url(self, **_: object) -> list[SimpleNamespace]:
        return self._detection_sequences.pop(0)

    def verify_from_large_person_group(self, **_: object) -> SimpleNamespace:
        assert self._verify_results, "No verify results left"
        return self._verify_results.pop(0)

    def __exit__(self, *args: object) -> bool:  # pragma: no cover - trivial
        return False


def _make_detection(face_id: str, quality: QualityForRecognition) -> SimpleNamespace:
    return SimpleNamespace(
        face_id=face_id,
        face_attributes=SimpleNamespace(quality_for_recognition=quality),
    )


def _client_factory_builder(
    verify_results: list[SimpleNamespace],
    training_quality: QualityForRecognition = QualityForRecognition.HIGH,
    candidate_quality: QualityForRecognition = QualityForRecognition.HIGH,
) -> Callable[[], ContextManager[tuple[Any, Any]]]:

    verify_queue = verify_results.copy()

    @contextmanager
    def _factory() -> Iterator[tuple[Any, Any]]:
        detections = [
            [_make_detection("ref", training_quality)],
            [_make_detection("candidate", candidate_quality)],
        ]
        admin = _FakeAdminClient()
        face = _FakeFaceClient(detection_sequences=detections, verify_results=verify_queue)
        yield admin, face

    return _factory


def test_face_api_service_state_transitions(monkeypatch: pytest.MonkeyPatch) -> None:
    verify_sequence = [
        SimpleNamespace(confidence=0.995, is_identical=True),
        SimpleNamespace(confidence=0.4, is_identical=False),
    ]

    factory = cast(ClientFactory, _client_factory_builder(verify_sequence))

    service = FaceAPIService(
        threshold=0.99,
        endpoint="https://example.com",
        api_key="test-key",
        client_factory=factory,
        uuid_factory=lambda: "group",
    )

    result_first = service.compare_from_urls("ref-url", "candidate-url")
    assert result_first.status == "approved"
    assert pytest.approx(result_first.similarity, abs=1e-6) == 0.995
    assert result_first.is_identical is True

    result_second = service.compare_from_urls("ref-url", "candidate-url")
    assert result_second.status == "not approved"
    assert result_second.is_identical is False


def test_face_api_service_validates_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    factory = cast(ClientFactory, _client_factory_builder([SimpleNamespace(confidence=0.5, is_identical=True)]))

    service = FaceAPIService(
        threshold=0.9,
        endpoint="https://example.com",
        api_key="test-key",
        client_factory=factory,
        uuid_factory=lambda: "group",
    )

    with pytest.raises(ValueError):
        service.compare_from_urls("", "candidate")

    with pytest.raises(ValueError):
        service.compare_from_urls("reference", "")