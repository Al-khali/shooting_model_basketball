"""Route aggregation — import this module to register all API routes."""

from fastapi import APIRouter

from src.api.routes.analyze import router as analyze_router
from src.api.routes.health import router as health_router
from src.api.routes.players import router as players_router

router = APIRouter()
router.include_router(health_router)
router.include_router(analyze_router)
router.include_router(players_router)

__all__ = ["router"]
