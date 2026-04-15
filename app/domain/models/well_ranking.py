from datetime import datetime, timezone
from hashlib import sha1

from sqlmodel import Field, SQLModel


class WellRankingPrediction(SQLModel, table=True):
    """ORM for persisted ranking output generated from non-ID feature inputs."""

    __tablename__ = "well_ranking_predictions"

    record_key: str = Field(primary_key=True, max_length=64)
    run_id: str = Field(index=True, max_length=36)
    ranked_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        index=True,
    )

    uwi: str | None = Field(default=None, max_length=100, index=True)
    well_name: str | None = Field(default=None, max_length=120, index=True)
    field_name: str | None = Field(default=None, max_length=120, index=True)
    area_id: str | None = Field(default=None, max_length=100, index=True)
    basin_cluster: str | None = Field(default=None, max_length=80, index=True)
    month_start: str | None = Field(default=None, max_length=20)

    predicted_score: float
    rank_overall: int = Field(index=True)
    rank_on_field: int = Field(index=True)
    rank_on_area: int = Field(index=True)

    @staticmethod
    def build_record_key(
        run_id: str,
        uwi: str | None,
        well_name: str | None,
        field_name: str | None,
        area_id: str | None,
        month_start: str | None,
    ) -> str:
        seed = "|".join(
            [
                run_id,
                uwi or "",
                well_name or "",
                field_name or "",
                area_id or "",
                month_start or "",
            ]
        )
        return sha1(seed.encode("utf-8")).hexdigest()

    def to_row_dict(self) -> dict[str, object]:
        row = self.model_dump()
        row["ranked_at"] = self.ranked_at.isoformat()
        return row
