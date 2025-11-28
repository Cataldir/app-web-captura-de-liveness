from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator
from typing import Generator

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture()
def client() -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(autouse=True)
def embeddings_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMBEDDING_ENDPOINT_URL", "https://embedding.test/score")
    monkeypatch.setenv("EMBEDDING_API_KEY", "test-key")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5-nano")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://openai.test")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
    monkeypatch.setenv("FACE_ENDPOINT", "https://face.test")
    monkeypatch.setenv("FACE_APIKEY", "face-key")
