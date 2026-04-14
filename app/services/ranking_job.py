"""
ranking_job.py
──────────────
The weekly ranking orchestrator. Called by APScheduler (or manually via API).

Flow:
  1. Create a run log entry in Supabase (status=running)
  2. Fetch all well data from Supabase
  3. Run AutoGluon inference via ai_service.predict_batch()
  4. Write ranked results to Supabase well_rankings table
  5. Update run log (status=success / failed)
"""

import asyncio
import logging
import uuid
from datetime import datetime, timezone

from app.services import supabase_service
from app.services.ai_service import predict_batch

logger = logging.getLogger(__name__)

# Module-level tracking for the currently running / last-completed job
_current_run_id: str | None = None
_is_running: bool = False


def is_running() -> bool:
    return _is_running


def current_run_id() -> str | None:
    return _current_run_id


async def run_ranking_job(triggered_by: str = "scheduler") -> str:
    """
    Execute the full ranking pipeline.

    Args:
        triggered_by: 'scheduler' | 'manual' | 'api'

    Returns:
        run_id (UUID string)

    Raises:
        RuntimeError if a job is already in progress.
    """
    global _current_run_id, _is_running

    if _is_running:
        raise RuntimeError(
            f"A ranking job is already running (run_id={_current_run_id}). "
            "Please wait for it to complete."
        )

    run_id = str(uuid.uuid4())
    _current_run_id = run_id
    _is_running = True
    started_at = datetime.now(timezone.utc)

    logger.info("=== Ranking Job START [run_id=%s, triggered_by=%s] ===", run_id, triggered_by)

    try:
        # Step 1: Log start
        await supabase_service.log_run_start(run_id, triggered_by)

        # Step 2: Fetch well data
        logger.info("Fetching well data from Supabase ...")
        wells_df = await supabase_service.fetch_all_wells()
        n_wells = len(wells_df)
        logger.info("Fetched %d wells.", n_wells)

        # Step 3: Run inference (CPU-bound — run in executor to not block event loop)
        logger.info("Running AutoGluon inference ...")
        loop = asyncio.get_event_loop()
        ranked_df = await loop.run_in_executor(None, predict_batch, wells_df)

        # Step 4: Write results back to Supabase
        logger.info("Writing ranking results to Supabase ...")
        await supabase_service.write_rankings(run_id, ranked_df)

        # Step 5: Log success
        duration = (datetime.now(timezone.utc) - started_at).total_seconds()
        await supabase_service.log_run_finish(
            run_id,
            status="success",
            wells_processed=n_wells,
            duration_sec=duration,
        )

        logger.info(
            "=== Ranking Job DONE [run_id=%s, %d wells, %.1fs] ===",
            run_id, n_wells, duration,
        )
        return run_id

    except Exception as exc:
        duration = (datetime.now(timezone.utc) - started_at).total_seconds()
        logger.exception("Ranking job FAILED [run_id=%s]: %s", run_id, exc)
        await supabase_service.log_run_finish(
            run_id,
            status="failed",
            duration_sec=duration,
            error_message=str(exc),
        )
        raise

    finally:
        _is_running = False
