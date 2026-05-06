"""FastAPI application factory.

Creates a FastAPI app with mounted routes and LLMEngine lifecycle.
The engine is created lazily (outside this module) to avoid importing
torch at import time.

Usage:
    from server.app import create_app
    from server import create_engine

    engine = create_engine()
    app = create_app(engine)
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import FastAPI

from server.routes import router

if TYPE_CHECKING:
    from engine.llm_engine import LLMEngine

logger = logging.getLogger("serveforge.server")


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan: logs startup/shutdown."""
    logger.info("LLM-ServeForge server starting (v0.1.0)")
    engine = app.state.engine
    if engine is not None:
        logger.info("LLMEngine attached: max_batch=%s", engine.scheduler.max_batch_size)
    yield
    logger.info("LLM-ServeForge server shutting down")


def create_app(engine: LLMEngine | None = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        engine: An LLMEngine instance. If None, the engine must be set
                later via app.state.engine before serving requests.

    Returns:
        A fully configured FastAPI application.
    """
    app = FastAPI(
        title="LLM-ServeForge",
        version="0.1.0",
        description="Hardware-agnostic LLM inference server — Phase 1 MVP",
        lifespan=_lifespan,
    )

    app.state.engine = engine
    app.include_router(router)

    return app


def create_engine(
    device: str = "cpu",
    max_batch_size: int = 32,
    max_tokens_per_step: int = 512,
    chunk_size: int = 256,
    num_blocks: int = 1000,
    block_size: int = 16,
) -> LLMEngine:
    """Create an LLMEngine with a minimal compiled module and HAL backend.

    This function imports torch (slow on first call) — it should be
    called at application startup, not at module import time.

    Args:
        device: HAL device type ("cpu" or "cuda").
        max_batch_size: Maximum concurrent requests per batch.
        max_tokens_per_step: Total tokens per forward pass.
        chunk_size: Max prefill tokens per request per step.
        num_blocks: KV cache block pool size.
        block_size: Tokens per KV cache block.

    Returns:
        A configured LLMEngine instance.
    """
    from compiler.ir import IrFunction, IrModule
    from engine.llm_engine import LLMEngine  # concrete import for mypy
    from hal import PyTorchBackend

    backend = PyTorchBackend(device)
    module = IrModule(functions=[IrFunction(name="main")])
    engine: LLMEngine = LLMEngine(
        module,
        backend,
        max_batch_size=max_batch_size,
        max_tokens_per_step=max_tokens_per_step,
        chunk_size=chunk_size,
        num_blocks=num_blocks,
        block_size=block_size,
    )
    return engine
