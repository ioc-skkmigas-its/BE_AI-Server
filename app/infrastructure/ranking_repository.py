from __future__ import annotations

from app.core.config import Settings
from app.domain.models.well_ranking import WellRankingPrediction
from app.infrastructure.supabase_client import SupabaseGateway


class RankingRepository:
    def __init__(self, settings: Settings, supabase: SupabaseGateway) -> None:
        self._settings = settings
        self._supabase = supabase

    def save_rankings(self, rankings: list[WellRankingPrediction]) -> int:
        rows = [ranking.to_row_dict() for ranking in rankings]
        return self._supabase.upsert_rows(
            table_name=self._settings.supabase_rankings_table,
            rows=rows,
            on_conflict=self._settings.ranking_upsert_conflict_keys,
            chunk_size=self._settings.supabase_write_chunk_size,
        )

    def get_latest_run_id(self) -> str | None:
        rows = self._supabase.fetch_rows(
            table_name=self._settings.supabase_rankings_table,
            limit=1,
            order_by="ranked_at",
            descending=True,
        )
        if not rows:
            return None

        run_id = rows[0].get("run_id")
        return str(run_id) if run_id else None

    def list_by_run(
        self,
        run_id: str,
        limit: int,
        offset: int = 0,
    ) -> list[dict[str, object]]:
        return self._supabase.fetch_rows(
            table_name=self._settings.supabase_rankings_table,
            limit=limit,
            offset=offset,
            order_by="rank_overall",
            descending=False,
            filters={"run_id": run_id},
        )
