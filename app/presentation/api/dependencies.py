from fastapi import Request

from app.application.health_service import HealthService
from app.application.ranking_service import WellRankingService


def get_health_service(request: Request) -> HealthService:
    return request.app.state.health_service


def get_ranking_service(request: Request) -> WellRankingService:
    return request.app.state.ranking_service
