from __future__ import annotations

from app.liveness.engine import LivenessEngine, LivenessResult, SimpleHeuristicDetector


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
