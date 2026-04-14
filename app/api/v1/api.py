from fastapi import APIRouter

from app.api.v1.endpoints import health, auth, ranking

api_router = APIRouter()

# Health (no prefix — top-level /health)
api_router.include_router(health.router)

# Versioned API
api_router.include_router(auth.router, prefix="/api/v1")
api_router.include_router(ranking.router, prefix="/api/v1")
