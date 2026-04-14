from datetime import datetime
from typing import Optional
from pydantic import BaseModel


class RankingRunStatus(BaseModel):
    run_id: str
    triggered_by: str
    status: str
    wells_processed: Optional[int] = None
    duration_sec: Optional[float] = None
    error_message: Optional[str] = None
    started_at: datetime
    finished_at: Optional[datetime] = None


class WellRankingResult(BaseModel):
    uwi: str
    well_name: Optional[str] = None
    field_name: Optional[str] = None
    area_id: Optional[str] = None
    basin_cluster: Optional[str] = None
    predicted_score: float
    rank_overall: int
    rank_in_basin: int
    rank_label: str
    run_id: str
    created_at: datetime


class RankingTriggerResponse(BaseModel):
    message: str
    run_id: str
    status: str


class LatestRankingsResponse(BaseModel):
    run_id: str
    total_wells: int
    run_started_at: Optional[datetime] = None
    rankings: list[WellRankingResult]
