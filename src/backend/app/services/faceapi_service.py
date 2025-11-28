"""Face API service orchestrating verification using the state pattern."""

from __future__ import annotations

import os
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, ContextManager

from azure.ai.vision.face import FaceAdministrationClient, FaceClient
from azure.ai.vision.face.models import (
    FaceAttributeTypeRecognition04,
    FaceDetectionModel,
    FaceRecognitionModel,
    QualityForRecognition,
)
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import AzureError
from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class FaceAPIResult:
    """Represents the outcome of a Face API verification."""

    similarity: float
    status: str
    is_identical: bool
    confidence: float
    reason: str


class ApprovalState:
    """Base state for approval logic."""

    name: str

    def evaluate(
        self, context: "FaceAPIService", similarity: float
    ) -> None:  # pragma: no cover - interface
        raise NotImplementedError


class ApprovedState(ApprovalState):
    """Represents the approved state."""

    name = "approved"

    def evaluate(self, context: "FaceAPIService", similarity: float) -> None:
        if similarity < context.threshold:
            context.set_state(NotApprovedState())


class NotApprovedState(ApprovalState):
    """Represents the not approved state."""

    name = "not approved"

    def evaluate(self, context: "FaceAPIService", similarity: float) -> None:
        if similarity >= context.threshold:
            context.set_state(ApprovedState())


ClientFactory = Callable[
    [], ContextManager[tuple[FaceAdministrationClient, FaceClient]]
]


class FaceAPIService:
    """Compares two faces using Azure Face API."""

    def __init__(
        self,
        threshold: float = 0.99,
        *,
        endpoint: str | None = None,
        api_key: str | None = None,
        client_factory: ClientFactory | None = None,
        uuid_factory: Callable[[], str] | None = None,
        polling_interval: int = 5,
    ) -> None:
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be within (0.0, 1.0]")

        endpoint_value = endpoint or os.getenv("FACE_ENDPOINT")
        if not endpoint_value:
            raise RuntimeError("FACE_ENDPOINT environment variable is missing")

        api_key_value = api_key or os.getenv("FACE_APIKEY") or os.getenv("FACE_API_KEY")
        if not api_key_value:
            raise RuntimeError("FACE_APIKEY environment variable is missing")

        self._threshold = threshold
        self._state: ApprovalState = NotApprovedState()
        self._endpoint = endpoint_value
        self._api_key = api_key_value
        self._polling_interval = polling_interval
        self._client_factory = client_factory or self._default_client_factory
        self._uuid_factory = uuid_factory or (lambda: uuid.uuid4().hex)

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def status(self) -> str:
        return self._state.name

    def set_state(self, state: ApprovalState) -> None:
        self._state = state

    def compare_from_urls(
        self, reference_image_url: str, candidate_image_url: str
    ) -> FaceAPIResult:
        if not reference_image_url or not candidate_image_url:
            raise ValueError("Image URLs must not be empty")

        group_id = self._uuid_factory().lower()

        with self._client_factory() as (admin_client, face_client):
            try:
                self._create_group(admin_client, group_id)
                person = self._create_person(admin_client, group_id)
                self._register_reference_face(
                    face_client,
                    admin_client,
                    group_id,
                    person.person_id,
                    reference_image_url,
                )
                self._train_group(admin_client, group_id)
                candidate_face_id = self._detect_candidate_face(
                    face_client, candidate_image_url
                )
                verify_result = self._verify_candidate(
                    face_client, group_id, person.person_id, candidate_face_id
                )
            finally:
                self._safe_delete_group(admin_client, group_id)

        confidence_raw = getattr(verify_result, "confidence", 0.0) or 0.0
        confidence = max(0.0, min(float(confidence_raw), 1.0))
        is_identical = bool(getattr(verify_result, "is_identical", False))

        self._state.evaluate(self, confidence if is_identical else 0.0)

        reason = (
            f"Verification returned {'identical' if is_identical else 'different'} faces "
            f"with confidence {confidence:.4f}."
        )

        return FaceAPIResult(
            similarity=confidence,
            status=self.status,
            is_identical=is_identical,
            confidence=confidence,
            reason=reason,
        )

    def _create_group(
        self, admin_client: FaceAdministrationClient, group_id: str
    ) -> None:
        try:
            admin_client.large_person_group.create(
                large_person_group_id=group_id,
                name=group_id,
                recognition_model=FaceRecognitionModel.RECOGNITION04,
            )
        except AzureError as exc:  # pragma: no cover - network error path
            raise RuntimeError("Failed to create Face API person group") from exc

    def _create_person(
        self, admin_client: FaceAdministrationClient, group_id: str
    ) -> Any:
        try:
            return admin_client.large_person_group.create_person(
                large_person_group_id=group_id,
                name="reference",
            )
        except AzureError as exc:  # pragma: no cover - network error path
            raise RuntimeError("Failed to create Face API person") from exc

    def _register_reference_face(
        self,
        face_client: FaceClient,
        admin_client: FaceAdministrationClient,
        group_id: str,
        person_id: str,
        image_url: str,
    ) -> None:
        detection = self._select_single_high_quality_face(
            face_client.detect_from_url(
                url=image_url,
                detection_model=FaceDetectionModel.DETECTION03,
                recognition_model=FaceRecognitionModel.RECOGNITION04,
                return_face_id=True,
                return_face_attributes=[
                    FaceAttributeTypeRecognition04.QUALITY_FOR_RECOGNITION
                ],
            ),
            high_quality_only=True,
        )
        if detection is None:
            raise ValueError("Reference image does not contain a valid face")

        try:
            admin_client.large_person_group.add_face_from_url(
                large_person_group_id=group_id,
                person_id=person_id,
                url=image_url,
                detection_model=FaceDetectionModel.DETECTION03,
            )
        except AzureError as exc:  # pragma: no cover - network error path
            raise RuntimeError(
                "Failed to add reference face to Face API group"
            ) from exc

    def _train_group(
        self, admin_client: FaceAdministrationClient, group_id: str
    ) -> None:
        try:
            poller = admin_client.large_person_group.begin_train(
                large_person_group_id=group_id,
                polling_interval=self._polling_interval,
            )
            poller.wait()
        except AzureError as exc:  # pragma: no cover - network error path
            raise RuntimeError("Training Face API person group failed") from exc

    def _detect_candidate_face(
        self, face_client: FaceClient, candidate_url: str
    ) -> str:
        detection = self._select_single_high_quality_face(
            face_client.detect_from_url(
                url=candidate_url,
                detection_model=FaceDetectionModel.DETECTION03,
                recognition_model=FaceRecognitionModel.RECOGNITION04,
                return_face_id=True,
                return_face_attributes=[
                    FaceAttributeTypeRecognition04.QUALITY_FOR_RECOGNITION
                ],
            ),
            high_quality_only=False,
        )
        if detection is None:
            raise ValueError("Candidate image does not contain a recognizable face")
        return str(detection.face_id)

    def _verify_candidate(
        self,
        face_client: FaceClient,
        group_id: str,
        person_id: str,
        face_id: str,
    ) -> Any:
        try:
            return face_client.verify_from_large_person_group(
                face_id=face_id,
                large_person_group_id=group_id,
                person_id=person_id,
            )
        except AzureError as exc:  # pragma: no cover - network error path
            raise RuntimeError("Face verification failed") from exc

    def _safe_delete_group(
        self, admin_client: FaceAdministrationClient, group_id: str
    ) -> None:
        try:
            admin_client.large_person_group.delete(group_id)
        except AzureError:  # pragma: no cover - best effort cleanup
            pass

    def _select_single_high_quality_face(
        self,
        detections: list[Any],
        *,
        high_quality_only: bool,
    ) -> Any | None:
        valid_faces = []
        for face in detections:
            quality = getattr(face.face_attributes, "quality_for_recognition", None)
            if quality is None:
                continue
            if high_quality_only and quality != QualityForRecognition.HIGH:
                continue
            if not high_quality_only and quality == QualityForRecognition.LOW:
                continue
            valid_faces.append(face)

        if len(valid_faces) != 1:
            return None
        return valid_faces[0]

    @contextmanager
    def _default_client_factory(
        self,
    ) -> Iterator[tuple[FaceAdministrationClient, FaceClient]]:
        credential = AzureKeyCredential(self._api_key)
        with FaceAdministrationClient(
            endpoint=self._endpoint, credential=credential
        ) as admin_client, FaceClient(
            endpoint=self._endpoint,
            credential=credential,
        ) as face_client:
            yield admin_client, face_client


__all__ = ["FaceAPIService", "FaceAPIResult", "ApprovedState", "NotApprovedState"]
