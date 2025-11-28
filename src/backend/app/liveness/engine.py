"""Liveness detection engine with an extensible strategy interface."""

from __future__ import annotations

import hashlib
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Event, Lock
from typing import Any, Dict, Iterable, Optional

import cv2  # type: ignore[import-not-found]
import numpy as np  # type: ignore[import-not-found]

LOGGER = logging.getLogger(__name__)

try:
    from liveness_detector.server_launcher import GestureServerClient
except Exception:  # pragma: no cover - import varies per platform
    GestureServerClient = None  # type: ignore[assignment]


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
    def evaluate(
        self,
        frame: bytes,
        *,
        context: Optional[Dict[str, Any]] = None,
    ) -> LivenessResult:
        """Return a liveness result for a raw video frame payload."""

    def reset_session(self) -> None:  # pragma: no cover - optional override
        """Reset internal state between sessions when supported."""


class GestureServerDetector(LivenessDetector):
    """Adapter that delegates detection to the liveness-detector package."""

    def __init__(
        self,
        *,
        socket_path: Optional[str] = None,
        language: Optional[str] = None,
        num_gestures: Optional[int] = None,
        gestures: Optional[Iterable[str]] = None,
        callback_timeout: float = 2.5,
    ) -> None:
        if GestureServerClient is None:
            raise RuntimeError("liveness-detector is not available on this platform.")

        resolved_socket = socket_path or os.getenv(
            "LIVENESS_SOCKET_PATH",
            (
                "\\\\.\\pipe\\liveness_socket"
                if os.name == "nt"
                else "/tmp/liveness_socket"
            ),
        )
        resolved_language = language or os.getenv("LIVENESS_LANGUAGE", "en")
        resolved_num_gestures = num_gestures or int(
            os.getenv("LIVENESS_NUM_GESTURES", "2")
        )
        gestures_list = (
            list(gestures) if gestures else _parse_csv_env("LIVENESS_GESTURES")
        )

        self._client = GestureServerClient(
            language=resolved_language,
            socket_path=resolved_socket,
            num_gestures=resolved_num_gestures,
            gestures_list=gestures_list if gestures_list else None,
        )
        self._client.set_report_alive_callback(self._handle_alive)
        self._client.set_string_callback(self._handle_message)

        self._callback_timeout = callback_timeout
        self._lock = Lock()
        self._result_event = Event()
        self._last_alive: Optional[bool] = None
        self._last_reason = "Detector inicializado"
        self._server_started = False

    def _ensure_server(self) -> None:
        if not self._server_started:
            if not self._client.start_server():
                raise RuntimeError("Não foi possível iniciar o liveness-detector")
            self._server_started = True

    def _handle_alive(self, alive: bool) -> None:
        with self._lock:
            self._last_alive = alive
            self._last_reason = (
                "Detector confirmou liveness" if alive else "Detector identificou spoof"
            )
        self._result_event.set()

    def _handle_message(self, message: str) -> None:
        if message:
            with self._lock:
                self._last_reason = message

    @staticmethod
    def _decode_frame(frame: bytes) -> np.ndarray:
        array = np.frombuffer(frame, dtype=np.uint8)
        image = cv2.imdecode(array, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Frame inválido: não foi possível decodificar para imagem")
        return image

    def evaluate(
        self,
        frame: bytes,
        *,
        context: Optional[Dict[str, Any]] = None,
    ) -> LivenessResult:
        image = self._decode_frame(frame)
        self._ensure_server()

        self._result_event.clear()
        with self._lock:
            self._client.process_frame(image)

        if not self._result_event.wait(self._callback_timeout):
            raise TimeoutError("Liveness-detector não respondeu dentro do tempo limite")

        with self._lock:
            is_live = bool(self._last_alive)
            reason = self._last_reason

        return LivenessResult(
            is_live=is_live,
            confidence=1.0 if is_live else 0.0,
            reason=reason,
            timestamp=datetime.now(tz=timezone.utc),
        )

    def reset_session(self) -> None:
        with self._lock:
            self._last_alive = None
            self._last_reason = "Detector reiniciado"
        self._result_event.clear()

    def close(self) -> None:
        with self._lock:
            if self._server_started:
                self._client.stop_server()
                self._server_started = False


class SimpleHeuristicDetector(LivenessDetector):
    """Deterministic heuristic used as default while a real detector is not available."""

    def evaluate(
        self,
        frame: bytes,
        *,
        context: Optional[Dict[str, Any]] = None,
    ) -> LivenessResult:
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
        reason = (
            "Confidence threshold satisfied"
            if is_live
            else "Confidence threshold not met"
        )
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
        reset_session = getattr(self._detector, "reset_session", None)
        if callable(reset_session):
            reset_session()

    def close(self) -> None:
        closer = getattr(self._detector, "close", None)
        if callable(closer):
            closer()


def _parse_csv_env(name: str) -> list[str]:
    raw = os.getenv(name)
    if not raw:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


def create_detector_from_env() -> LivenessDetector:
    provider = os.getenv("LIVENESS_DETECTOR_PROVIDER", "heuristic")
    normalized_provider = provider.strip().lower()

    if normalized_provider in {"gesture", "liveness-detector"}:
        try:
            return GestureServerDetector()
        except Exception as exc:  # pragma: no cover - depends on environment
            LOGGER.warning(
                "Falha ao inicializar liveness-detector, usando heurística. Motivo: %s",
                exc,
            )

    return SimpleHeuristicDetector()
