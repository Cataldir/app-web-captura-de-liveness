"""Liveness detection engine with an extensible strategy interface."""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional


class SingletonMeta(type):
    """Thread-safe Singleton metaclass based on the refactoring.guru example."""

    _instances: Dict[type, Any] = {}

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:  # noqa: D401
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


@dataclass(slots=True, frozen=True)
class LivenessResult:
    """Structured result produced by a liveness detector."""

    is_live: bool
    confidence: float
    reason: str
    timestamp: datetime


class LivenessDetector(ABC):
    """Provides the contract for pluggable liveness detection strategies."""

    @abstractmethod
    def evaluate(self, frame: bytes, *, context: Optional[Dict[str, Any]] = None) -> LivenessResult:
        """Return a liveness result for a raw video frame payload."""


class SimpleHeuristicDetector(LivenessDetector):
    """Deterministic heuristic used as default while a real detector is not available."""

    def evaluate(self, frame: bytes, *, context: Optional[Dict[str, Any]] = None) -> LivenessResult:
        if not frame:
            return LivenessResult(
                is_live=False,
                confidence=0.0,
                reason="Empty frame received",
                timestamp=datetime.now(tz=timezone.utc),
            )

        digest = hashlib.sha256(frame).digest()
        # Normalize the first byte of the digest to obtain a pseudo-confidence score.
        confidence = round(digest[0] / 255, 3)
        is_live = confidence >= 0.5
        reason = "Confidence threshold satisfied" if is_live else "Confidence threshold not met"
        if context and context.get("attempt"):
            reason = f"{reason} on attempt {context['attempt']}"

        return LivenessResult(
            is_live=is_live,
            confidence=confidence,
            reason=reason,
            timestamp=datetime.now(tz=timezone.utc),
        )


class LivenessEngine(metaclass=SingletonMeta):
    """Coordinates the selected detector and session-scoped metadata."""

    def __init__(self, detector: Optional[LivenessDetector] = None) -> None:
        self._detector = detector or SimpleHeuristicDetector()
        self._attempt = 0

    def evaluate(self, frame: bytes) -> LivenessResult:
        self._attempt += 1
        return self._detector.evaluate(frame, context={"attempt": self._attempt})

    def set_detector(self, detector: LivenessDetector) -> None:
        self._detector = detector

    def reset(self) -> None:
        self._attempt = 0
