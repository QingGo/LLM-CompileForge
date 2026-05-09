"""End-to-end integration tests — full pipeline with real LLMEngine.

Creates a minimal MlirModule with deterministic weights, a real
PyTorchBackend, and a real LLMEngine. The engine is wrapped in a
FastAPI app via create_app() and tested with TestClient.

The test model produces token 42 on every step (temperature=0 greedy),
so the output is fully predictable.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
import torch
from fastapi.testclient import TestClient

from compiler.mlir_artifact import MlirFunction, MlirModule, MlirOp
from engine.llm_engine import LLMEngine
from hal import PyTorchBackend
from server.app import create_app

# ── Simple tokenizer for e2e tests ─────────────────────────


class _SimpleTokenizer:
    """Minimal tokenizer for testing — maps characters to their ASCII codes."""

    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, tokens: list[int]) -> str:
        return " ".join(str(t) for t in tokens)

# ── Deterministic test model ───────────────────────────────

TOKEN_42 = 42
VOCAB_SIZE = 100
HIDDEN_SIZE = 8


def _make_deterministic_module() -> MlirModule:
    """Create a minimal MlirModule that always returns token 42 as argmax."""
    embed = torch.ones(1, HIDDEN_SIZE, dtype=torch.float32)
    w = torch.zeros(HIDDEN_SIZE, VOCAB_SIZE, dtype=torch.float32)
    w[0, TOKEN_42] = 1.0

    func = MlirFunction(
        name="main",
        inputs=[("%input_ids", "tensor<?xi64>")],
        outputs=[("%logits", f"tensor<?x{VOCAB_SIZE}xf32>")],
        ops=[
            MlirOp(name="sf.constant", dialect="sf", op_name="constant",
                   operands=["embed"], results=["%1"]),
            MlirOp(name="sf.matmul", dialect="sf", op_name="matmul",
                   operands=["%1", "w"], results=["%logits"]),
        ],
        weights={"embed": embed, "w": w},
    )
    return MlirModule(functions=[func], metadata={"vocab_size": VOCAB_SIZE, "hidden_size": HIDDEN_SIZE})


def _create_test_engine(
    max_batch_size: int = 8,
    num_blocks: int = 200,
    chunk_size: int = 256,
) -> LLMEngine:
    """Create a real LLMEngine with the deterministic test module."""
    backend = PyTorchBackend("cpu")
    module = _make_deterministic_module()
    engine = LLMEngine(
        module,
        backend,
        max_batch_size=max_batch_size,
        max_tokens_per_step=512,
        chunk_size=chunk_size,
        num_blocks=num_blocks,
        block_size=16,
    )
    engine.set_tokenizer(_SimpleTokenizer(), eos_token_id=None)
    return engine


# ── Fixtures ───────────────────────────────────────────────


@pytest.fixture(scope="module")
def e2e_client() -> TestClient:
    """Module-scoped fixture: create engine + app once per test module."""
    engine = _create_test_engine()
    app = create_app(engine)
    return TestClient(app)


# ── Helpers ────────────────────────────────────────────────


def _collect_sse_events(response: Any) -> list[dict[str, Any]]:
    """Parse SSE streaming response into a list of JSON event dicts."""
    events: list[dict[str, Any]] = []
    for line in response.iter_lines():
        line_str = line if isinstance(line, str) else line.decode("utf-8")
        if line_str.startswith("data: ") and line_str != "data: [DONE]":
            events.append(json.loads(line_str[6:]))
    return events


# ═══════════════════════════════════════════════════════════
# Health (via real engine)
# ═══════════════════════════════════════════════════════════


@pytest.mark.integration
class TestE2EHealth:
    def test_health(self, e2e_client: TestClient) -> None:
        resp = e2e_client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


# ═══════════════════════════════════════════════════════════
# Non-streaming completion
# ═══════════════════════════════════════════════════════════


@pytest.mark.integration
class TestE2ECompletionNonStreaming:
    def test_single_request_produces_tokens(self, e2e_client: TestClient) -> None:
        """Full pipeline: add_request → step() loop → collect → JSON response."""
        resp = e2e_client.post(
            "/v1/completions",
            json={"prompt": [1, 2, 3], "max_tokens": 2, "temperature": 0.0, "stream": False},
        )
        assert resp.status_code == 200
        body = resp.json()

        assert body["object"] == "text_completion"
        assert len(body["choices"]) == 1
        assert body["choices"][0]["index"] == 0
        assert body["choices"][0]["finish_reason"] == "stop"

        # Token 42 → text "42", two tokens → "42 42"
        assert body["choices"][0]["text"] == "42 42"

        # Usage stats
        assert body["usage"]["prompt_tokens"] == 3
        assert body["usage"]["completion_tokens"] == 2
        assert body["usage"]["total_tokens"] == 5

    def test_completion_with_prompt_str(self, e2e_client: TestClient) -> None:
        """Prompt as string — engine treats as list of char codes."""
        resp = e2e_client.post(
            "/v1/completions",
            json={"prompt": "hello", "max_tokens": 1, "temperature": 0.0, "stream": False},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["choices"][0]["text"] == "42"
        assert body["usage"]["completion_tokens"] == 1

    def test_multiple_steps(self, e2e_client: TestClient) -> None:
        """Request with max_tokens=3 — engine loops 3 step() calls."""
        resp = e2e_client.post(
            "/v1/completions",
            json={"prompt": [1], "max_tokens": 3, "temperature": 0.0, "stream": False},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["choices"][0]["text"] == "42 42 42"
        assert body["usage"]["completion_tokens"] == 3

    def test_id_is_non_empty(self, e2e_client: TestClient) -> None:
        resp = e2e_client.post(
            "/v1/completions",
            json={"prompt": [1], "max_tokens": 1, "temperature": 0.0, "stream": False},
        )
        assert resp.status_code == 200
        assert len(resp.json()["id"]) > 0

    def test_created_is_timestamp(self, e2e_client: TestClient) -> None:
        resp = e2e_client.post(
            "/v1/completions",
            json={"prompt": [1], "max_tokens": 1, "temperature": 0.0, "stream": False},
        )
        assert resp.status_code == 200
        created = resp.json()["created"]
        assert isinstance(created, int)
        assert created > 1700000000  # some time after 2023

    def test_second_request_works(self, e2e_client: TestClient) -> None:
        """Ensure engine is reusable across requests."""
        resp1 = e2e_client.post(
            "/v1/completions",
            json={"prompt": [1], "max_tokens": 1, "temperature": 0.0, "stream": False},
        )
        assert resp1.status_code == 200

        resp2 = e2e_client.post(
            "/v1/completions",
            json={"prompt": [2, 3, 4], "max_tokens": 2, "temperature": 0.0, "stream": False},
        )
        assert resp2.status_code == 200
        assert resp2.json()["choices"][0]["text"] == "42 42"


# ═══════════════════════════════════════════════════════════
# Streaming completion
# ═══════════════════════════════════════════════════════════


@pytest.mark.integration
class TestE2ECompletionStreaming:
    def test_streaming_produces_events(self, e2e_client: TestClient) -> None:
        resp = e2e_client.post(
            "/v1/completions",
            json={"prompt": [1, 2], "max_tokens": 2, "temperature": 0.0, "stream": True},
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")

    def test_streaming_event_count(self, e2e_client: TestClient) -> None:
        resp = e2e_client.post(
            "/v1/completions",
            json={"prompt": [1], "max_tokens": 3, "temperature": 0.0, "stream": True},
        )
        events = _collect_sse_events(resp)
        assert len(events) == 3

    def test_streaming_event_content(self, e2e_client: TestClient) -> None:
        """Each SSE event contains token 42's character '＊' or '42'.'"""
        resp = e2e_client.post(
            "/v1/completions",
            json={"prompt": [1], "max_tokens": 2, "temperature": 0.0, "stream": True},
        )
        events = _collect_sse_events(resp)
        assert len(events) == 2
        # Token 42 is '*' (ASCII). The streaming handler converts it via chr().
        assert events[0]["choices"][0]["text"] == "*"
        assert events[1]["choices"][0]["text"] == "*"

    def test_streaming_finish_reason(self, e2e_client: TestClient) -> None:
        resp = e2e_client.post(
            "/v1/completions",
            json={"prompt": [1], "max_tokens": 1, "temperature": 0.0, "stream": True},
        )
        events = _collect_sse_events(resp)
        assert events[-1]["choices"][0]["finish_reason"] == "stop"

    def test_streaming_done_sentinel(self, e2e_client: TestClient) -> None:
        resp = e2e_client.post(
            "/v1/completions",
            json={"prompt": [1], "max_tokens": 1, "temperature": 0.0, "stream": True},
        )
        assert "data: [DONE]" in resp.text


# ═══════════════════════════════════════════════════════════
# Chat Completions
# ═══════════════════════════════════════════════════════════


@pytest.mark.integration
class TestE2EChatCompletions:
    def test_chat_non_streaming(self, e2e_client: TestClient) -> None:
        resp = e2e_client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 2,
                "temperature": 0.0,
                "stream": False,
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["object"] == "chat.completion"
        assert body["choices"][0]["message"]["role"] == "assistant"
        assert body["choices"][0]["message"]["content"] == "42 42"

    def test_chat_streaming(self, e2e_client: TestClient) -> None:
        resp = e2e_client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 1,
                "temperature": 0.0,
                "stream": True,
            },
        )
        assert resp.status_code == 200
        events = _collect_sse_events(resp)
        assert len(events) == 1
        assert events[0]["object"] == "chat.completion.chunk"
        assert "delta" in events[0]["choices"][0]


# ═══════════════════════════════════════════════════════════
# Engine reusability
# ═══════════════════════════════════════════════════════════


@pytest.mark.integration
class TestE2EEngineReuse:
    def test_consecutive_requests_dont_accumulate_state(self, e2e_client: TestClient) -> None:
        """Engine should be clean after each request completes."""
        for i in range(3):
            resp = e2e_client.post(
                "/v1/completions",
                json={"prompt": [i + 1], "max_tokens": 1, "temperature": 0.0, "stream": False},
            )
            assert resp.status_code == 200
            assert resp.json()["choices"][0]["text"] == "42"
