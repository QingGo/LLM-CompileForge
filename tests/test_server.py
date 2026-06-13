"""Unit tests for the server layer (routes + app).

Uses FastAPI TestClient with a mock engine (no torch import needed).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pytest
from fastapi.testclient import TestClient

from python_runtime.server.app import create_app

# ═══════════════════════════════════════════════════════════
# Mock Engine — fast, no torch dependency
# ═══════════════════════════════════════════════════════════


@dataclass
class FakeResult:
    """Minimal stand-in for engine.batch.GenerationResult."""

    request_id: str
    new_tokens: list[int]
    is_finished: bool


class FakeScheduler:
    max_batch_size: int = 32


class _FakeTokenizer:
    """Minimal tokenizer mock for server tests."""

    def decode(self, token_ids: list[int]) -> str:
        return " ".join(str(t) for t in token_ids)


class FakeEngine:
    """Mock LLMEngine that returns predictable token sequences."""

    def __init__(self, token_sequence: list[int] | None = None) -> None:
        self._rid: str | None = None
        self._step_count = 0
        self._max_tokens = 256
        self._token_sequence = token_sequence or [65, 66, 67, 68, 69]  # A,B,C,D,E
        self._calls: list[dict[str, Any]] = []
        self.scheduler = FakeScheduler()  # type: ignore[assignment]
        self._tokenizer = _FakeTokenizer()

    def add_request(
        self,
        prompt: str | list[int],
        max_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        priority: int = 0,
    ) -> str:
        self._rid = "fake-rid-1"
        self._step_count = 0
        self._max_tokens = max_tokens
        self._calls.append(
            {
                "method": "add_request",
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        )
        return self._rid

    def step(self) -> list[FakeResult]:
        self._step_count += 1

        if self._rid is None:
            return []

        total_available = min(self._max_tokens, len(self._token_sequence))

        if self._step_count > total_available:
            return []

        idx = min(self._step_count - 1, len(self._token_sequence) - 1)
        token = self._token_sequence[idx]
        finished = self._step_count >= total_available

        return [FakeResult(request_id=self._rid, new_tokens=[token], is_finished=finished)]

    @property
    def is_idle(self) -> bool:
        return self._rid is None or self._step_count >= self._max_tokens


# ═══════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════


@pytest.fixture
def engine() -> FakeEngine:
    return FakeEngine()


@pytest.fixture
def client(engine: FakeEngine) -> TestClient:
    app = create_app(engine)
    return TestClient(app)


# ═══════════════════════════════════════════════════════════
# Health
# ═══════════════════════════════════════════════════════════


@pytest.mark.unit
class TestHealthEndpoint:
    def test_health_returns_ok(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["version"] == "0.1.0"


# ═══════════════════════════════════════════════════════════
# Completion — Non-streaming
# ═══════════════════════════════════════════════════════════


@pytest.mark.unit
class TestCompletionNonStreaming:
    def test_basic_completion(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/completions",
            json={"prompt": "Hello", "max_tokens": 3, "stream": False},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["object"] == "text_completion"
        assert len(body["choices"]) == 1
        assert body["choices"][0]["text"] == "65 66 67"
        assert body["choices"][0]["finish_reason"] == "stop"
        assert body["choices"][0]["index"] == 0

    def test_completion_with_prompt_list(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/completions",
            json={"prompt": [1, 2, 3], "max_tokens": 2, "stream": False},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["choices"][0]["text"] == "65 66"

    def test_usage_stats(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/completions",
            json={"prompt": "Hello world", "max_tokens": 2, "stream": False},
        )
        assert resp.status_code == 200
        body = resp.json()
        usage = body["usage"]
        assert usage["prompt_tokens"] == 2  # "Hello world" → 2 words
        assert usage["completion_tokens"] == 2
        assert usage["total_tokens"] == 4

    def test_model_field(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/completions",
            json={"prompt": "test", "max_tokens": 1, "model": "llama3-8b", "stream": False},
        )
        assert resp.status_code == 200
        assert resp.json()["model"] == "llama3-8b"

    def test_has_id(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/completions",
            json={"prompt": "x", "max_tokens": 1, "stream": False},
        )
        assert resp.status_code == 200
        assert "id" in resp.json()
        assert len(resp.json()["id"]) > 0

    def test_completion_single_token(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/completions",
            json={"prompt": "one", "max_tokens": 1, "stream": False},
        )
        assert resp.status_code == 200
        assert resp.json()["choices"][0]["text"] == "65"


# ═══════════════════════════════════════════════════════════
# Completion — Streaming
# ═══════════════════════════════════════════════════════════


def _collect_sse_events(response: Any) -> list[dict[str, Any]]:
    """Parse SSE streaming response, extracting JSON events."""
    events: list[dict[str, Any]] = []
    for line in response.iter_lines():
        line_str = line if isinstance(line, str) else line.decode("utf-8")
        if line_str.startswith("data: ") and line_str != "data: [DONE]":
            events.append(json.loads(line_str[6:]))
    return events


@pytest.mark.unit
class TestCompletionStreaming:
    def test_streaming_returns_sse(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/completions",
            json={"prompt": "Hello", "max_tokens": 3, "stream": True},
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")

    def test_streaming_events_count(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/completions",
            json={"prompt": "Hello", "max_tokens": 3, "stream": True},
        )
        events = _collect_sse_events(resp)
        assert len(events) == 3  # One event per generated token

    def test_streaming_final_event_has_finish_reason(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/completions",
            json={"prompt": "Hello", "max_tokens": 2, "stream": True},
        )
        events = _collect_sse_events(resp)
        assert len(events) >= 1
        # Last event should have finish_reason
        assert events[-1]["choices"][0]["finish_reason"] == "stop"

    def test_streaming_done_sentinel(self, client: TestClient) -> None:
        """Verify SSE stream ends with 'data: [DONE]'."""
        resp = client.post(
            "/v1/completions",
            json={"prompt": "Hello", "max_tokens": 1, "stream": True},
        )
        raw_body = resp.text
        assert "data: [DONE]" in raw_body

    def test_streaming_event_schema(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/completions",
            json={"prompt": "test", "max_tokens": 1, "stream": True},
        )
        events = _collect_sse_events(resp)
        assert len(events) >= 1
        evt = events[0]
        assert evt["object"] == "text_completion"
        assert "id" in evt
        assert "choices" in evt
        assert "text" in evt["choices"][0]


# ═══════════════════════════════════════════════════════════
# Chat Completion — Non-streaming
# ═══════════════════════════════════════════════════════════


@pytest.mark.unit
class TestChatCompletionNonStreaming:
    def test_basic_chat(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 3,
                "stream": False,
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["object"] == "chat.completion"
        assert len(body["choices"]) == 1
        assert body["choices"][0]["message"]["role"] == "assistant"
        assert body["choices"][0]["message"]["content"] == "65 66 67"
        assert body["choices"][0]["finish_reason"] == "stop"

    def test_chat_multi_message(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/chat/completions",
            json={
                "messages": [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Hello"},
                ],
                "max_tokens": 2,
                "stream": False,
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["choices"][0]["message"]["content"] == "65 66"

    def test_chat_usage_stats(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hello world"}],
                "max_tokens": 2,
                "stream": False,
            },
        )
        assert resp.status_code == 200
        usage = resp.json()["usage"]
        assert usage["completion_tokens"] == 2
        assert usage["total_tokens"] > 2


# ═══════════════════════════════════════════════════════════
# Chat Completion — Streaming
# ═══════════════════════════════════════════════════════════


@pytest.mark.unit
class TestChatCompletionStreaming:
    def test_chat_streaming_returns_sse(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 2,
                "stream": True,
            },
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")

    def test_chat_streaming_events(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 3,
                "stream": True,
            },
        )
        events = _collect_sse_events(resp)
        assert len(events) == 3

    def test_chat_streaming_delta_format(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 1,
                "stream": True,
            },
        )
        events = _collect_sse_events(resp)
        assert len(events) >= 1
        evt = events[0]
        assert evt["object"] == "chat.completion.chunk"
        assert "delta" in evt["choices"][0]
        assert "content" in evt["choices"][0]["delta"]

    def test_chat_streaming_done(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 1,
                "stream": True,
            },
        )
        assert "data: [DONE]" in resp.text


# ═══════════════════════════════════════════════════════════
# Parameter validation
# ═══════════════════════════════════════════════════════════


@pytest.mark.unit
class TestParameterValidation:
    def test_default_max_tokens(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/completions",
            json={"prompt": "hello", "stream": False},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["usage"]["completion_tokens"] == 5  # default max_tokens=256 but tokens=5

    def test_temperature_bounds(self, client: TestClient) -> None:
        # outside [0, 2] should be rejected by Pydantic
        resp = client.post(
            "/v1/completions",
            json={"prompt": "x", "temperature": 3.0, "stream": False},
        )
        assert resp.status_code == 422

        resp = client.post(
            "/v1/completions",
            json={"prompt": "x", "temperature": -0.1, "stream": False},
        )
        assert resp.status_code == 422

    def test_top_p_bounds(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/completions",
            json={"prompt": "x", "top_p": 1.5, "stream": False},
        )
        assert resp.status_code == 422

    def test_messages_required_for_chat(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/chat/completions",
            json={"max_tokens": 1, "stream": False},
        )
        assert resp.status_code == 422

    def test_empty_messages_rejected(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/chat/completions",
            json={"messages": [], "max_tokens": 1, "stream": False},
        )
        assert resp.status_code == 422


# ═══════════════════════════════════════════════════════════
# Edge cases
# ═══════════════════════════════════════════════════════════


@pytest.mark.unit
class TestEdgeCases:
    def test_max_tokens_zero_rejected(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/completions",
            json={"prompt": "x", "max_tokens": 0, "stream": False},
        )
        assert resp.status_code == 422

    def test_large_max_tokens_ok(self, client: TestClient) -> None:
        # Should work but FakeEngine caps at 5 tokens
        resp = client.post(
            "/v1/completions",
            json={"prompt": "long", "max_tokens": 20000, "stream": False},
        )
        assert resp.status_code == 200

    def test_chat_completion_prompt_construction(self, client: TestClient, engine: FakeEngine) -> None:
        """Verify chat endpoint builds prompt correctly."""
        client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 1,
                "stream": False,
            },
        )
        assert len(engine._calls) == 1
        add_req = engine._calls[0]
        assert "User: Hello" in add_req["prompt"]
        assert "Assistant:" in add_req["prompt"]
