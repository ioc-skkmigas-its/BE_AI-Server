from fastapi import APIRouter

from app.presentation.api.routes import health, ranking, root

api_router = APIRouter()
api_router.include_router(root.router)
api_router.include_router(health.router)
api_router.include_router(ranking.router)
