"""Health check route — GET /health."""

from __future__ import annotations

import logging

from fastapi import APIRouter

from src.api.schemas.responses import HealthComponent, HealthResponse

router = APIRouter(tags=["health"])
logger = logging.getLogger(__name__)


@router.get("/health", response_model=HealthResponse, summary="Liveness + readiness check")
async def health() -> HealthResponse:
    """
    Return the API health status and component availability.

    Optional dependencies (ultralytics, mediapipe, google-adk) are checked
    without importing their heavy model weights. A ``degraded`` component
    means the pipeline will return stub data for that step rather than
    failing hard.
    """
    components: dict[str, HealthComponent] = {}

    for component, module in [
        ("pipeline", "src.agents.orchestrator"),
        ("memory", "src.agents.memory"),
    ]:
        try:
            __import__(module)
            components[component] = HealthComponent(status="ok")
        except Exception as exc:
            logger.warning("Component %s unavailable: %s", component, exc)
            components[component] = HealthComponent(status="down", detail=str(exc))

    for component, module in [
        ("perception", "src.analysis.shot_detector"),
        ("vlm", "src.vlm.gemini_client"),
    ]:
        try:
            __import__(module)
            components[component] = HealthComponent(status="ok")
        except ImportError:
            components[component] = HealthComponent(
                status="degraded",
                detail="Optional dependency not installed — stub mode active",
            )
        except Exception as exc:
            components[component] = HealthComponent(status="down", detail=str(exc))

    if any(c.status == "down" for c in components.values()):
        overall = "down"
    elif any(c.status == "degraded" for c in components.values()):
        overall = "degraded"
    else:
        overall = "ok"
    return HealthResponse(status=overall, components=components)
