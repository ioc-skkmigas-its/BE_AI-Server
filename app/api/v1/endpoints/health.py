from fastapi import APIRouter
from app.services.model_loader import model_status
from app.core.config import settings

router = APIRouter(tags=["Health"])


@router.get("/health", summary="Basic health check")
async def health():
    return {
        "status": "ok",
        "app": settings.app_name,
        "version": settings.app_version,
    }


@router.get("/health/model", summary="AutoGluon model load status")
async def health_model():
    status = model_status()
    return {
        "model_ready": status["ready"],
        "hf_repo": status["hf_repo"],
        "cache_dir": status["cache_dir"],
        "error": status["error"],
    }
