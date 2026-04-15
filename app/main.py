from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI

from app.application.health_service import HealthService
from app.application.ranking_service import WellRankingService
from app.core.config import get_settings
from app.infrastructure.db_migrator import run_auto_migration
from app.infrastructure.model_loader import XGBoostModelLoader
from app.infrastructure.ranking_repository import RankingRepository
from app.infrastructure.supabase_client import SupabaseGateway
from app.infrastructure.trident_repository import TridentRepository
from app.presentation.api.router import api_router


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    migration_result = run_auto_migration(settings)
    logger.info(migration_result["detail"])

    supabase = SupabaseGateway(settings)
    trident_repo = TridentRepository(settings, supabase)
    ranking_repo = RankingRepository(settings, supabase)
    model_loader = XGBoostModelLoader(settings)

    app.state.settings = settings
    app.state.supabase = supabase
    app.state.health_service = HealthService(settings, supabase)
    app.state.ranking_service = WellRankingService(
        settings=settings,
        trident_repo=trident_repo,
        model_loader=model_loader,
        ranking_repo=ranking_repo,
    )
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        lifespan=lifespan,
    )
    app.include_router(api_router)
    return app


app = create_app()
