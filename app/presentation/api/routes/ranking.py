from fastapi import APIRouter, Depends, HTTPException, Query

from app.application.ranking_service import WellRankingService
from app.presentation.api.dependencies import get_ranking_service

router = APIRouter(prefix="/ranking", tags=["ranking"])


@router.post("/run")
def run_ranking(
    service: WellRankingService = Depends(get_ranking_service),
) -> dict[str, object]:
    try:
        return service.run_ranking()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/latest")
def latest_rankings(
    limit: int = Query(default=100, ge=1, le=5000),
    offset: int = Query(default=0, ge=0),
    run_id: str | None = Query(default=None),
    service: WellRankingService = Depends(get_ranking_service),
) -> dict[str, object]:
    try:
        return service.get_latest_rankings(limit=limit, offset=offset, run_id=run_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
