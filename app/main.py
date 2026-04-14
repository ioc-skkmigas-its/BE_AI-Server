"""
main.py
───────
FastAPI application entry point.

Lifespan:
  1. Init SQLite tables (user auth)
  2. Download + load AutoGluon model from Hugging Face
  3. Start APScheduler for weekly ranking job
  4. On shutdown: stop scheduler
"""

import logging
from contextlib import asynccontextmanager

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.api import api_router
from app.core.config import settings
from app.db.engine import init_db
from app.services.model_loader import load_model
from app.services.ranking_job import run_ranking_job

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

scheduler = AsyncIOScheduler(timezone="UTC")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──────────────────────────────────────────────
    logger.info("=== %s v%s starting up ===", settings.app_name, settings.app_version)

    # 1. SQLite — create user table if not exists
    init_db()
    logger.info("SQLite tables initialised.")

    # 2. Load AutoGluon model (download from HF if not cached)
    try:
        load_model()
    except Exception as exc:
        logger.error(
            "Model failed to load: %s\n"
            "The API will start but ranking jobs will not work until the model is available.",
            exc,
        )

    # 3. Schedule weekly ranking job
    scheduler.add_job(
        run_ranking_job,
        trigger="cron",
        day_of_week=settings.ranking_schedule_day,
        hour=settings.ranking_schedule_hour,
        minute=settings.ranking_schedule_minute,
        args=["scheduler"],
        id="weekly_ranking",
        replace_existing=True,
    )
    scheduler.start()
    logger.info(
        "Scheduler started — ranking job scheduled every %s at %02d:%02d UTC",
        settings.ranking_schedule_day,
        settings.ranking_schedule_hour,
        settings.ranking_schedule_minute,
    )

    yield

    # ── Shutdown ─────────────────────────────────────────────
    scheduler.shutdown(wait=False)
    logger.info("=== %s shut down ===", settings.app_name)


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "AI-powered MSF well ranking backend. "
            "Runs AutoGluon inference weekly, stores results in Supabase."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # CORS — adjust origins as needed for your frontend
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router)
    return app


app = create_app()
