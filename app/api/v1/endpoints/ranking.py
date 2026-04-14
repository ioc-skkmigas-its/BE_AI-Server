import asyncio
import logging

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks

from app.core.security import get_current_user
from app.db.models import User
from app.schemas.ranking_schema import (
    RankingTriggerResponse,
    RankingRunStatus,
    LatestRankingsResponse,
    WellRankingResult,
)
from app.services import ranking_job, supabase_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ranking", tags=["Ranking"])


@router.post(
    "/trigger",
    response_model=RankingTriggerResponse,
    summary="Manually trigger the ranking job",
)
async def trigger_ranking(
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
):
    """
    Kick off the weekly ranking pipeline immediately.
    The job runs in the background — poll /ranking/status to check progress.
    Returns 409 if a job is already running.
    """
    if ranking_job.is_running():
        raise HTTPException(
            status_code=409,
            detail=f"A ranking job is already running (run_id={ranking_job.current_run_id()}). "
                   "Check /ranking/status for progress.",
        )

    # We need a run_id before the job starts so we can return it immediately
    import uuid
    run_id_preview = str(uuid.uuid4())

    async def _run():
        try:
            await ranking_job.run_ranking_job(triggered_by="api")
        except Exception as exc:
            logger.error("Background ranking job failed: %s", exc)

    background_tasks.add_task(_run)

    return RankingTriggerResponse(
        message="Ranking job started in background. Poll /ranking/status for progress.",
        run_id="(assigned at job start — check /ranking/status)",
        status="started",
    )


@router.get(
    "/status",
    response_model=RankingRunStatus,
    summary="Get status of the latest ranking run",
)
async def ranking_status(current_user: User = Depends(get_current_user)):
    """
    Returns the most recent ranking run log entry from Supabase.
    If a job is currently running, it will show status='running'.
    """
    latest = await supabase_service.get_latest_run()
    if not latest:
        raise HTTPException(status_code=404, detail="No ranking runs found yet.")
    return RankingRunStatus(**latest)


@router.get(
    "/latest",
    response_model=LatestRankingsResponse,
    summary="Fetch the latest ranking results",
)
async def latest_rankings(
    limit: int = Query(default=50, ge=1, le=500, description="Number of results to return"),
    offset: int = Query(default=0, ge=0, description="Pagination offset"),
    current_user: User = Depends(get_current_user),
):
    """
    Returns the top-ranked wells from the most recently completed ranking run.
    Sorted by rank_overall ascending (rank 1 = highest scoring well).
    """
    latest_run = await supabase_service.get_latest_run()
    if not latest_run:
        raise HTTPException(status_code=404, detail="No completed ranking run found.")

    run_id = latest_run["id"]
    rows = await supabase_service.get_rankings_by_run(run_id, limit=limit, offset=offset)

    return LatestRankingsResponse(
        run_id=run_id,
        total_wells=latest_run.get("wells_processed", len(rows)),
        run_started_at=latest_run.get("started_at"),
        rankings=[WellRankingResult(**r) for r in rows],
    )
