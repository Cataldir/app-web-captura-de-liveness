from __future__ import annotations

import pytest  # type: ignore[import-not-found]

from app.liveness import engine as engine_module
from app.liveness.engine import (
    GestureServerDetector,
    LivenessEngine,
    LivenessResult,
    SimpleHeuristicDetector,
    create_detector_from_env,
)


def test_singleton_engine_reuses_instance() -> None:
    engine_a = LivenessEngine()
    engine_b = LivenessEngine()
    assert engine_a is engine_b


def test_simple_detector_returns_deterministic_result() -> None:
    engine = LivenessEngine(detector=SimpleHeuristicDetector())
    payload = b"frame"
    first = engine.evaluate(payload)
    second = engine.evaluate(payload)
    assert isinstance(first, LivenessResult)
    assert first.is_live == second.is_live
    assert first.confidence == second.confidence


def test_create_detector_from_env_falls_back_when_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LIVENESS_DETECTOR_PROVIDER", "gesture")
    monkeypatch.setattr(engine_module, "GestureServerClient", None)
    detector = create_detector_from_env()
    assert isinstance(detector, SimpleHeuristicDetector)
    monkeypatch.delenv("LIVENESS_DETECTOR_PROVIDER", raising=False)


def test_gesture_detector_requires_library(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(engine_module, "GestureServerClient", None)
    with pytest.raises(RuntimeError):
        GestureServerDetector()
