"""AI Shoot FastAPI application — entry point."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router
from src.core.config import settings

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logger.info("AI Shoot API starting — %s", settings.app_name)
    yield
    logger.info("AI Shoot API shutting down")


app = FastAPI(
    title="AI Shoot API",
    description=(
        "Real-time basketball shot analysis powered by computer vision, "
        "biomechanics, and agentic AI (Google ADK 2.0 + Gemini Flash)."
    ),
    version="0.4.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
