"""OpenAI-compatible API routes.

POST /v1/completions      — text completion
POST /v1/chat/completions — chat completion
GET  /health              — health check

Streaming is supported via SSE (Server-Sent Events).
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from python_runtime.engine.llm_engine import LLMEngine

# ── Pydantic helpers ────────────────────────────────────────


def _default_uuid() -> str:
    return f"cmpl-{uuid.uuid4().hex[:12]}"


# ── Request Models ──────────────────────────────────────────


class CompletionRequest(BaseModel):
    """OpenAI /v1/completions request schema."""

    model: str = Field(default="default", description="Model name (informational for MVP).")
    prompt: str | list[int] = Field(..., description="Text prompt or token ID list.")
    max_tokens: int = Field(default=256, ge=1, le=32768, description="Max tokens to generate.")
    temperature: float = Field(default=1.0, ge=0.0, le=2.0, description="Sampling temperature.")
    top_p: float = Field(default=1.0, ge=0.0, le=1.0, description="Nucleus sampling threshold.")
    top_k: int = Field(default=0, ge=0, description="Top-k sampling filter (0 = off).")
    stream: bool = Field(default=False, description="Enable SSE streaming.")


class Message(BaseModel):
    """A chat message (role + content)."""

    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    """OpenAI /v1/chat/completions request schema."""

    model: str = Field(default="default")
    messages: list[Message] = Field(..., min_length=1)
    max_tokens: int = Field(default=256, ge=1, le=32768)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    top_k: int = Field(default=0, ge=0)
    stream: bool = Field(default=False)


# ── Response Models ─────────────────────────────────────────


class UsageStats(BaseModel):
    """Token usage counters."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class CompletionChoice(BaseModel):
    text: str
    index: int = 0
    finish_reason: str | None = None


class CompletionResponse(BaseModel):
    id: str = Field(default_factory=_default_uuid)
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = "default"
    choices: list[CompletionChoice]
    usage: UsageStats = Field(default_factory=UsageStats)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str | None = None


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=_default_uuid)
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = "default"
    choices: list[ChatChoice]
    usage: UsageStats = Field(default_factory=UsageStats)


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.1.0"


# ── Router ─────────────────────────────────────────────────

router = APIRouter()

# Internal request counter
_request_counter = 0

# Async lock to serialise engine access (MVP constraint).
# The lock is held during add_request + step loop for a single request.
# For MVP this is acceptable — true concurrency is Phase 2 work.
_engine_lock: asyncio.Lock = asyncio.Lock()


def _next_req_id() -> str:
    global _request_counter
    _request_counter += 1
    return f"server-{_request_counter}"


def _get_engine(request: Request) -> LLMEngine:
    """Extract LLMEngine from app state."""
    engine: LLMEngine = request.app.state.engine
    if engine is None:
        raise RuntimeError("LLMEngine not initialised. Call create_app(engine) first.")
    return engine


def _prompt_tokens(prompt: str | list[int]) -> int:
    """Count tokens in a prompt for usage stats."""
    if isinstance(prompt, str):
        return len(prompt.split())
    return len(prompt)


# ── Completions ────────────────────────────────────────────


@router.post("/v1/completions", response_model=None)
async def completions(req: CompletionRequest, request: Request) -> Any:
    """Text completion endpoint (OpenAI-compatible).

    Supports both streaming (SSE) and non-streaming modes.
    """
    engine = _get_engine(request)

    if req.stream:
        return StreamingResponse(
            _stream_completions(engine, req),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # Non-streaming: acquire lock, run step loop, return JSON
    async with _engine_lock:
        rid = engine.add_request(
            req.prompt,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
        )

        all_tokens: list[int] = []
        finish_reason: str | None = None

        while True:
            results = engine.step()
            for r in results:
                if r.request_id == rid:
                    all_tokens.extend(r.new_tokens)
                    if r.is_finished:
                        finish_reason = "stop" if len(all_tokens) >= req.max_tokens or not r.new_tokens else "stop"
                        break
            if finish_reason is not None:
                break

        text = engine._tokenizer.decode(all_tokens)

        return CompletionResponse(
            id=rid,
            model=req.model,
            choices=[CompletionChoice(text=text, index=0, finish_reason=finish_reason)],
            usage=UsageStats(
                prompt_tokens=_prompt_tokens(req.prompt),
                completion_tokens=len(all_tokens),
                total_tokens=_prompt_tokens(req.prompt) + len(all_tokens),
            ),
        )


async def _stream_completions(engine: LLMEngine, req: CompletionRequest) -> AsyncGenerator[str, None]:
    """SSE streaming generator for text completions.

    Holds the engine lock for the entire streaming lifetime so
    add_request+step() form an atomic unit.
    """
    async with _engine_lock:
        rid = engine.add_request(
            req.prompt,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
        )

        while True:
            results = engine.step()
            for r in results:
                if r.request_id == rid:
                    token_val = r.new_tokens[0] if r.new_tokens else 0
                    token_text = engine._tokenizer.decode([token_val])

                    chunk: dict[str, Any] = {
                        "id": rid,
                        "object": "text_completion",
                        "created": int(time.time()),
                        "model": req.model,
                        "choices": [
                            {
                                "text": token_text,
                                "index": 0,
                                "finish_reason": "stop" if r.is_finished else None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

                    if r.is_finished:
                        yield "data: [DONE]\n\n"
                        return


# ── Chat Completions ───────────────────────────────────────


@router.post("/v1/chat/completions", response_model=None)
async def chat_completions(req: ChatCompletionRequest, request: Request) -> Any:
    """Chat completion endpoint (OpenAI-compatible).

    Extracts the user message content as the prompt.
    Supports both streaming and non-streaming modes.
    """
    engine = _get_engine(request)

    # Build prompt from messages (simple concatenation for MVP)
    prompt_parts: list[str] = []
    for msg in req.messages:
        role_label = msg.role.capitalize()
        prompt_parts.append(f"{role_label}: {msg.content}")
    prompt = "\n".join(prompt_parts) + "\nAssistant: "

    prompt_token_count = _prompt_tokens(prompt)

    if req.stream:
        return StreamingResponse(
            _stream_chat(engine, req, prompt),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    async with _engine_lock:
        rid = engine.add_request(
            prompt,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
        )

        all_tokens: list[int] = []
        finish_reason: str | None = None

        while True:
            results = engine.step()
            for r in results:
                if r.request_id == rid:
                    all_tokens.extend(r.new_tokens)
                    if r.is_finished:
                        finish_reason = "stop"
                        break
            if finish_reason is not None:
                break

        text = engine._tokenizer.decode(all_tokens)

        return ChatCompletionResponse(
            id=rid,
            model=req.model,
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=text),
                    finish_reason=finish_reason,
                )
            ],
            usage=UsageStats(
                prompt_tokens=prompt_token_count,
                completion_tokens=len(all_tokens),
                total_tokens=prompt_token_count + len(all_tokens),
            ),
        )


async def _stream_chat(engine: LLMEngine, req: ChatCompletionRequest, prompt: str) -> AsyncGenerator[str, None]:
    """SSE streaming generator for chat completions."""
    async with _engine_lock:
        rid = engine.add_request(
            prompt,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
        )

        while True:
            results = engine.step()
            for r in results:
                if r.request_id == rid:
                    token_val = r.new_tokens[0] if r.new_tokens else 0
                    token_text = engine._tokenizer.decode([token_val])

                    chunk: dict[str, Any] = {
                        "id": rid,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": req.model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": token_text},
                                "finish_reason": "stop" if r.is_finished else None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

                    if r.is_finished:
                        yield "data: [DONE]\n\n"
                        return


# ── Health ─────────────────────────────────────────────────


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(status="ok", version="0.1.0")
