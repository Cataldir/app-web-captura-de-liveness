"""Image similarity evaluation service based on cosine similarity."""

from __future__ import annotations

import base64
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass

from urllib import error as urllib_error
from urllib import request as urllib_request

import numpy as np
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity


load_dotenv()


@dataclass(frozen=True)
class SimilarityResult:
    """Stores the outcome of an image similarity evaluation."""

    similarity: float
    status: str


class ApprovalState(ABC):
    """Defines the contract for approval states."""

    name: str

    @abstractmethod
    def evaluate(self, context: EmbeddingsService, similarity: float) -> None:
        """Updates the context according to the computed similarity."""


class ApprovedState(ApprovalState):
    """Represents the approved state."""

    name = "approved"

    def evaluate(self, context: EmbeddingsService, similarity: float) -> None:
        if similarity < context.threshold:
            context.set_state(NotApprovedState())


class NotApprovedState(ApprovalState):
    """Represents the not-approved state."""

    name = "not approved"

    def evaluate(self, context: EmbeddingsService, similarity: float) -> None:
        if similarity >= context.threshold:
            context.set_state(ApprovedState())


class EmbeddingsService:
    """Evaluates image similarity while transitioning between approval states."""

    def __init__(
        self,
        threshold: float = 0.99,
        *,
        endpoint_url: str | None = None,
        api_key: str | None = None,
        request_timeout: float = 10.0,
    ) -> None:
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be within (0.0, 1.0]")

        self._threshold = threshold
        self._state: ApprovalState = NotApprovedState()

        endpoint_candidate = endpoint_url or os.getenv("EMBEDDING_ENDPOINT_URL")
        if not endpoint_candidate:
            raise RuntimeError("Embedding endpoint URL is not configured")
        self._endpoint_url = endpoint_candidate

        api_key_candidate = api_key or os.getenv("EMBEDDING_API_KEY")
        if not api_key_candidate:
            raise RuntimeError("Embedding API key is not configured")
        self._api_key = api_key_candidate

        self._timeout = request_timeout

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def status(self) -> str:
        return self._state.name

    def set_state(self, state: ApprovalState) -> None:
        self._state = state

    def evaluate_pair(
        self, first_image: bytes, second_image: bytes
    ) -> SimilarityResult:
        if not first_image or not second_image:
            raise ValueError("Image payloads must not be empty")

        first_vector = self._generate_embedding(first_image)
        second_vector = self._generate_embedding(second_image)

        first_matrix = np.expand_dims(first_vector, axis=0)
        second_matrix = np.expand_dims(second_vector, axis=0)
        similarity_matrix = cosine_similarity(first_matrix, second_matrix)
        similarity = float(similarity_matrix[0][0])
        similarity = max(0.0, min(similarity, 1.0))

        self._state.evaluate(self, similarity)
        return SimilarityResult(similarity=similarity, status=self.status)

    def _generate_embedding(self, payload: bytes) -> np.ndarray:
        encoded_payload = base64.b64encode(payload).decode("utf-8")
        request_body = json.dumps({"image": encoded_payload}).encode("utf-8")

        request_headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }

        request_object = urllib_request.Request(
            self._endpoint_url,
            data=request_body,
            headers=request_headers,
            method="POST",
        )

        try:
            with urllib_request.urlopen(
                request_object, timeout=self._timeout
            ) as response:
                response_payload = response.read().decode("utf-8")
        except urllib_error.HTTPError as exc:
            detail = exc.read().decode("utf8", "ignore")
            raise RuntimeError(
                f"Embedding endpoint failed with status {exc.code}: {detail or 'no details'}"
            ) from exc
        except urllib_error.URLError as exc:
            raise RuntimeError("Embedding endpoint is unreachable") from exc

        try:
            parsed = json.loads(response_payload)
        except json.JSONDecodeError as exc:
            raise ValueError("Embedding endpoint returned invalid JSON") from exc

        vector = parsed.get("embedding")
        if not isinstance(vector, list) or not vector:
            raise ValueError("Embedding endpoint returned an empty embedding")

        try:
            array = np.array(vector, dtype=np.float32)
        except ValueError as exc:
            raise ValueError("Embedding endpoint returned non-numeric values") from exc

        if array.ndim != 1:
            raise ValueError("Embedding endpoint returned an invalid vector shape")

        return array


__all__ = ["EmbeddingsService", "SimilarityResult", "ApprovedState", "NotApprovedState"]
