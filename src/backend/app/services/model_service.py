"""Model-backed similarity evaluation using Azure OpenAI."""

from __future__ import annotations

import base64
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, cast

from dotenv import load_dotenv
from openai import AzureOpenAI
from openai.types.chat import ChatCompletionMessageParam


load_dotenv()


@dataclass(frozen=True)
class ModelComparisonResult:
    """Structured outcome produced by the Azure OpenAI evaluator."""

    similarity: float
    status: str
    same_person: bool
    explanation: str


class ApprovalState(ABC):
    """Defines the contract for approval states."""

    name: str

    @abstractmethod
    def evaluate(self, context: ModelService, similarity: float) -> None:
        """Updates the context according to the computed similarity."""


class ApprovedState(ApprovalState):
    """Represents the approved state."""

    name = "approved"

    def evaluate(self, context: ModelService, similarity: float) -> None:
        if similarity < context.threshold:
            context.set_state(NotApprovedState())


class NotApprovedState(ApprovalState):
    """Represents the not-approved state."""

    name = "not approved"

    def evaluate(self, context: ModelService, similarity: float) -> None:
        if similarity >= context.threshold:
            context.set_state(ApprovedState())


class ModelService:
    """Evaluates similarity between two images by prompting an Azure OpenAI model."""

    def __init__(
        self,
        threshold: float = 0.99,
        *,
        deployment: str | None = None,
        endpoint: str | None = None,
        api_key: str | None = None,
        api_version: str | None = None,
        client: AzureOpenAI | None = None,
        max_tokens: int = 2048,
    ) -> None:
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be within (0.0, 1.0]")

        self._threshold = threshold
        self._state: ApprovalState = NotApprovedState()

        deployment_name = (
            deployment
            or os.getenv("AZURE_OPENAI_DEPLOYMENT")
            or os.getenv("DEPLOYMENT")
            or os.getenv("MODEL_NAME")
        )
        if not deployment_name:
            raise RuntimeError("Azure OpenAI deployment name is not configured")

        endpoint_url = (
            endpoint
            or os.getenv("AZURE_OPENAI_ENDPOINT")
            or os.getenv("FOUNDRY_ENDPOINT")
        )
        if not endpoint_url:
            raise RuntimeError("Azure OpenAI endpoint is not configured")

        api_key_value = (
            api_key
            or os.getenv("AZURE_OPENAI_API_KEY")
            or os.getenv("EMBEDDING_API_KEY")
            or os.getenv("API_KEY")
        )
        if not api_key_value:
            raise RuntimeError("Azure OpenAI API key is not configured")

        version_value = (
            api_version
            or os.getenv("AZURE_OPENAI_API_VERSION")
            or os.getenv("API_VERSION")
            or "2024-12-01-preview"
        )
        if not version_value:
            raise RuntimeError("Azure OpenAI API version is not configured")

        self._deployment = deployment_name
        self._max_tokens = max_tokens
        self._client = client or AzureOpenAI(
            api_version=version_value,
            azure_endpoint=endpoint_url,
            api_key=api_key_value,
        )

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def status(self) -> str:
        return self._state.name

    def set_state(self, state: ApprovalState) -> None:
        self._state = state

    def compare_images(
        self, first_image: bytes, second_image: bytes
    ) -> ModelComparisonResult:
        if not first_image or not second_image:
            raise ValueError("Image payloads must not be empty")

        response_payload = self._request_similarity(first_image, second_image)
        data = self._parse_model_payload(response_payload)

        similarity = float(data["similarity"])
        similarity = max(0.0, min(similarity, 1.0))
        same_person = bool(data["same_person"])
        explanation = str(data.get("explanation") or "")

        self._state.evaluate(self, similarity if same_person else 0.0)
        return ModelComparisonResult(
            similarity=similarity,
            status=self.status,
            same_person=same_person,
            explanation=explanation,
        )

    def _request_similarity(self, first_image: bytes, second_image: bytes) -> str:
        first_encoded = base64.b64encode(first_image).decode("utf-8")
        second_encoded = base64.b64encode(second_image).decode("utf-8")

        system_prompt = (
            "You are an identity verification assistant. Compare the two provided facial images and "
            "respond with strict JSON containing keys similarity (float 0-1), same_person (boolean), "
            "and explanation (string describing visual evidence)."
        )

        user_instructions = (
            "Evaluate how similar the two faces are and decide if they belong to the same person. "
            "Explain distinct facial traits, lighting, pose, and any discrepancies."
        )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": user_instructions},
                    {"type": "input_image", "image_base64": first_encoded},
                    {"type": "input_image", "image_base64": second_encoded},
                ],
            },
        ]

        response = self._client.chat.completions.create(
            messages=cast(list[ChatCompletionMessageParam], messages),
            max_completion_tokens=self._max_tokens,
            model=self._deployment,
            response_format={"type": "json_object"},
        )

        message = response.choices[0].message
        content = message.content
        if not content:
            raise ValueError("Azure OpenAI response did not include content")

        return content

    def _parse_model_payload(self, payload: str) -> dict[str, Any]:
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ValueError("Azure OpenAI returned invalid JSON") from exc

        if {"similarity", "same_person", "explanation"} - parsed.keys():
            raise ValueError("Azure OpenAI response is missing required fields")

        return parsed


__all__ = [
    "ModelService",
    "ModelComparisonResult",
    "ApprovedState",
    "NotApprovedState",
]
