from fastapi import APIRouter, Depends

from app.application.health_service import HealthService
from app.presentation.api.dependencies import get_health_service

router = APIRouter()


@router.get("/health")
def health(service: HealthService = Depends(get_health_service)) -> dict[str, str]:
    return service.app_health()


@router.get("/supabase/health")
def supabase_health(
    service: HealthService = Depends(get_health_service),
) -> dict[str, str]:
    return service.supabase_health()
