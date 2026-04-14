"""
supabase_service.py
------------------
Handles all Supabase I/O:
  - fetch_all_wells()    -> read well data for inference
  - write_rankings()     -> batch-insert ranked results
  - log_run_start()      -> insert a 'running' log entry
  - log_run_finish()     -> update log entry with result
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from time import sleep
from typing import Optional

import numpy as np
import pandas as pd
from supabase import Client, create_client

from app.core.config import settings

logger = logging.getLogger(__name__)

_READ_PAGE_SIZE = settings.supabase_read_page_size


def _first_non_empty_value(row: pd.Series, *keys: str) -> str | None:
    """Return first non-empty string representation for given candidate keys."""
    for key in keys:
        value = row.get(key)
        if pd.isna(value):
            continue

        text = str(value).strip()
        if text and text.lower() != "nan":
            return text

    return None


def _get_service_client() -> Client:
    """Supabase client using service_role key (bypasses RLS)."""
    return create_client(settings.supabase_url, settings.supabase_service_key)


def _get_anon_client() -> Client:
    """Supabase client using anon key for read-only operations."""
    return create_client(settings.supabase_url, settings.supabase_anon_key)


def _fetch_all_rows(client: Client, table: str) -> list[dict]:
    """Fetch all rows from a table using paginated range requests."""

    def _is_retryable(exc: Exception) -> bool:
        exc_name = type(exc).__name__.lower()
        text = str(exc).lower()
        retry_markers = [
            "server disconnected",
            "timeout",
            "connection",
            "temporarily unavailable",
        ]
        return any(marker in text for marker in retry_markers) or any(
            marker in exc_name for marker in ["timeout", "connect", "protocol"]
        )

    def _fetch_page(
        offset: int,
        window: int,
        include_count: bool = False,
        use_fresh_client: bool = False,
    ) -> tuple[list[dict], int | None]:
        max_attempts = 4
        for attempt in range(1, max_attempts + 1):
            page_client = _get_anon_client() if use_fresh_client else client
            try:
                query = page_client.table(table)
                if include_count:
                    query = query.select("*", count="exact")
                else:
                    query = query.select("*")

                response = query.range(offset, offset + window - 1).execute()
                return response.data or [], getattr(response, "count", None)
            except Exception as exc:
                if attempt == max_attempts or not _is_retryable(exc):
                    raise
                delay = 0.5 * (2 ** (attempt - 1))
                logger.warning(
                    "Retry fetch %s offset=%d attempt=%d/%d due to: %s",
                    table,
                    offset,
                    attempt,
                    max_attempts,
                    exc,
                )
                sleep(delay)
        return [], None

    first_batch, total_count = _fetch_page(0, _READ_PAGE_SIZE, include_count=True)
    if not first_batch:
        return []

    all_rows: list[dict] = list(first_batch)
    effective_window = max(len(first_batch), 1)

    logger.debug(
        "Fetched %d rows from %s (offset=0, total_count=%s, requested_window=%d, effective_window=%d)",
        len(first_batch),
        table,
        total_count,
        _READ_PAGE_SIZE,
        effective_window,
    )

    # If count not provided, fallback to sequential pagination.
    if total_count is None:
        offset = len(first_batch)
        while True:
            batch, _ = _fetch_page(offset, effective_window)
            if not batch:
                break
            all_rows.extend(batch)
            logger.debug("Fetched %d rows from %s (offset=%d)", len(batch), table, offset)
            offset += len(batch)
        return all_rows

    if total_count <= len(first_batch):
        return all_rows

    remaining_offsets = list(range(len(first_batch), total_count, effective_window))
    max_workers = max(1, settings.supabase_fetch_workers)

    if max_workers == 1:
        for offset in remaining_offsets:
            batch, _ = _fetch_page(offset, effective_window)
            if batch:
                all_rows.extend(batch)
        return all_rows

    batches_by_offset: dict[int, list[dict]] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_fetch_page, offset, effective_window, False, True): offset
            for offset in remaining_offsets
        }
        for future in as_completed(futures):
            offset = futures[future]
            batch, _ = future.result()
            batches_by_offset[offset] = batch

    for offset in remaining_offsets:
        batch = batches_by_offset.get(offset) or []
        if batch:
            all_rows.extend(batch)

    return all_rows


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Element-wise division with protection against zero/invalid denominator."""
    num = pd.to_numeric(numerator, errors="coerce")
    den = pd.to_numeric(denominator, errors="coerce")
    valid = den.notna() & (den != 0)

    out = pd.Series(np.nan, index=num.index, dtype="float64")
    out.loc[valid] = num.loc[valid] / den.loc[valid]
    return out


def _enrich_monthlies_with_static(
    monthlies_df: pd.DataFrame,
    static_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join monthly rows with static well metadata and derive additional model features.
    """
    if monthlies_df.empty or static_df.empty:
        return monthlies_df
    if "well_id" not in monthlies_df.columns or "id" not in static_df.columns:
        logger.warning(
            "Skipping monthlies enrichment: required join keys missing "
            "(monthlies.well_id / static.id)."
        )
        return monthlies_df

    merged_df = monthlies_df.merge(
        static_df,
        how="left",
        left_on="well_id",
        right_on="id",
        suffixes=("", "_well"),
    )

    spud_dt = pd.to_datetime(merged_df.get("spud_date"), errors="coerce")
    merged_df["spud_year"] = spud_dt.dt.year
    merged_df["spud_month"] = spud_dt.dt.month
    merged_df["oil_to_water_ratio"] = _safe_divide(
        merged_df.get("oil_rate_bopd"),
        merged_df.get("water_rate_bwpd"),
    )
    merged_df["gas_to_oil_ratio"] = _safe_divide(
        merged_df.get("gas_rate_mmscfd"),
        merged_df.get("oil_rate_bopd"),
    )
    merged_df["pressure_drawdown_proxy"] = (
        pd.to_numeric(merged_df.get("reservoir_pressure_init_psi"), errors="coerce")
        - pd.to_numeric(merged_df.get("reservoir_pressure_psi"), errors="coerce")
    )
    merged_df["cost_per_boe_proxy"] = _safe_divide(
        merged_df.get("operating_cost_usd"),
        merged_df.get("boe_total"),
    )
    merged_df["revenue_per_boe_proxy"] = _safe_divide(
        merged_df.get("gross_revenue_usd"),
        merged_df.get("boe_total"),
    )
    merged_df["margin_per_boe_proxy"] = (
        merged_df["revenue_per_boe_proxy"] - merged_df["cost_per_boe_proxy"]
    )

    match_rate = merged_df["uwi"].notna().mean() if "uwi" in merged_df.columns else 0.0
    logger.info(
        "Monthlies enrichment complete: %d rows, static join coverage %.1f%%",
        len(merged_df),
        match_rate * 100.0,
    )

    return merged_df


# -- Read ---------------------------------------------------------------------

async def fetch_all_wells() -> pd.DataFrame:
    """
    Fetch all rows from the source table in Supabase.
    If source is trident_well_monthlies, enrich rows from static trident_wells.

    Returns:
        DataFrame with all inference columns.
    """
    client = _get_anon_client()
    table = settings.supabase_wells_table

    logger.info("Fetching wells from Supabase table '%s' ...", table)
    all_rows = _fetch_all_rows(client, table)
    logger.info("Total wells fetched: %d", len(all_rows))

    if not all_rows:
        raise ValueError(
            f"No data found in Supabase table '{table}'. "
            "Check SUPABASE_WELLS_TABLE env var and RLS policies."
        )

    wells_df = pd.DataFrame(all_rows)

    if settings.supabase_enrich_monthlies and table.lower() == "trident_well_monthlies":
        static_table = settings.supabase_static_wells_table
        logger.info("Fetching static well metadata from '%s' ...", static_table)
        static_rows = _fetch_all_rows(client, static_table)
        logger.info("Total static wells fetched: %d", len(static_rows))
        wells_df = _enrich_monthlies_with_static(wells_df, pd.DataFrame(static_rows))

    return wells_df


# -- Write --------------------------------------------------------------------

async def write_rankings(run_id: str, ranked_df: pd.DataFrame) -> None:
    """
    Batch-insert ranking results to Supabase.
    Uses service_role key to bypass RLS.

    Args:
        run_id: UUID of the current ranking run.
        ranked_df: DataFrame with predictions + rank columns.
    """
    table = settings.supabase_rankings_table

    records = []
    for idx, row in ranked_df.iterrows():
        uwi = _first_non_empty_value(row, "UWI", "uwi", "well_id", "id", "fid")
        if not uwi:
            uwi = f"unknown-{run_id}-{idx}"

        record = {
            "uwi": uwi,
            "well_name": _first_non_empty_value(row, "WELL_NAME", "well_name"),
            "field_name": _first_non_empty_value(row, "FIELD_NAME", "field_name"),
            "area_id": _first_non_empty_value(row, "AREA_ID", "area_id"),
            "basin_cluster": _first_non_empty_value(row, "basin_cluster"),
            "predicted_score": float(row["predicted_score"]),
            "rank_overall": int(row["rank_overall"]),
            "rank_in_basin": int(row["rank_in_basin"]),
            "rank_label": str(row["rank_label"]),
            "run_id": run_id,
        }
        records.append(record)

    total = len(records)
    chunk_size = max(1, settings.supabase_write_chunk_size)
    workers = max(1, settings.supabase_write_workers)

    if total == 0:
        logger.info("No ranking records to write.")
        return

    chunks: list[tuple[int, list[dict]]] = [
        (start, records[start : start + chunk_size])
        for start in range(0, total, chunk_size)
    ]

    def _is_retryable(exc: Exception) -> bool:
        exc_name = type(exc).__name__.lower()
        text = str(exc).lower()
        retry_markers = [
            "server disconnected",
            "timeout",
            "connection",
            "temporarily unavailable",
        ]
        return any(marker in text for marker in retry_markers) or any(
            marker in exc_name for marker in ["timeout", "connect", "protocol"]
        )

    def _insert_chunk(start: int, chunk: list[dict]) -> None:
        max_attempts = 4
        for attempt in range(1, max_attempts + 1):
            try:
                _get_service_client().table(table).insert(chunk).execute()
                logger.debug("Inserted chunk %d-%d", start, start + len(chunk))
                return
            except Exception as exc:
                if attempt == max_attempts or not _is_retryable(exc):
                    raise
                delay = 0.5 * (2 ** (attempt - 1))
                logger.warning(
                    "Retry write chunk %d-%d attempt=%d/%d due to: %s",
                    start,
                    start + len(chunk),
                    attempt,
                    max_attempts,
                    exc,
                )
                sleep(delay)

    logger.info(
        "Writing %d ranking records to '%s' in %d chunks (chunk=%d, workers=%d) ...",
        total,
        table,
        len(chunks),
        chunk_size,
        workers,
    )

    if workers == 1 or len(chunks) == 1:
        for start, chunk in chunks:
            _insert_chunk(start, chunk)
    else:
        max_workers = min(workers, len(chunks))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [
                pool.submit(_insert_chunk, start, chunk)
                for start, chunk in chunks
            ]
            for future in as_completed(futures):
                future.result()

    logger.info("Rankings write complete.")


# -- Logging ------------------------------------------------------------------

async def log_run_start(run_id: str, triggered_by: str) -> None:
    """Insert a running entry in ranking_run_log."""
    client = _get_service_client()
    client.table(settings.supabase_run_log_table).insert(
        {
            "id": run_id,
            "triggered_by": triggered_by,
            "status": "running",
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
    ).execute()
    logger.info("Run log created: %s", run_id)


async def log_run_finish(
    run_id: str,
    status: str,
    wells_processed: Optional[int] = None,
    duration_sec: Optional[float] = None,
    error_message: Optional[str] = None,
) -> None:
    """Update run log entry with final status and metrics."""
    client = _get_service_client()
    update_data: dict = {
        "status": status,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    if wells_processed is not None:
        update_data["wells_processed"] = wells_processed
    if duration_sec is not None:
        update_data["duration_sec"] = duration_sec
    if error_message is not None:
        update_data["error_message"] = error_message

    client.table(settings.supabase_run_log_table).update(update_data).eq("id", run_id).execute()
    logger.info("Run log updated: %s -> %s", run_id, status)


# -- Queries ------------------------------------------------------------------

async def get_latest_run() -> Optional[dict]:
    """Return the most recent successful run log entry."""
    client = _get_anon_client()
    response = (
        client.table(settings.supabase_run_log_table)
        .select("*")
        .eq("status", "success")
        .order("finished_at", desc=True)
        .limit(1)
        .execute()
    )
    return response.data[0] if response.data else None


async def get_rankings_by_run(run_id: str, limit: int = 100, offset: int = 0) -> list[dict]:
    """Fetch ranking rows for a specific run_id."""
    client = _get_anon_client()
    response = (
        client.table(settings.supabase_rankings_table)
        .select("*")
        .eq("run_id", run_id)
        .order("rank_overall")
        .range(offset, offset + limit - 1)
        .execute()
    )
    return response.data or []
