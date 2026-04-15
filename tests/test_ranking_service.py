from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.application.ranking_service import WellRankingService
from app.domain.models.well_ranking import WellRankingPrediction


@dataclass
class DummySettings:
    supabase_rankings_table: str = "well_ranking_predictions"


class FakeTridentRepo:
    def fetch_latest_panel(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "well_id": ["A", "B", "C"],
                "fid": [101, 102, 103],
                "catalog_id": ["CAT-1", "CAT-2", "CAT-3"],
                "uwi": ["U-1", "U-2", "U-3"],
                "well_name": ["Well 1", "Well 2", "Well 3"],
                "field_name": ["Field-X", "Field-X", "Field-Y"],
                "area_id": ["Area-1", "Area-1", "Area-2"],
                "basin_cluster": ["Basin-A", "Basin-A", "Basin-B"],
                "spud_date": ["2020-01-01", "2019-06-01", "2018-03-01"],
                "month_start": ["2025-01-01", "2025-01-01", "2025-01-01"],
                "year": [2025, 2025, 2025],
                "oil_rate_bopd": [100.0, 80.0, 120.0],
                "gas_rate_mmscfd": [3.0, 2.5, 4.0],
                "water_rate_bwpd": [25.0, 30.0, 20.0],
                "water_cut_pct": [20.0, 25.0, 15.0],
                "uptime_pct": [95.0, 92.0, 97.0],
                "active_flag": [True, True, True],
                "boe_total": [3000.0, 2700.0, 3200.0],
                "operating_cost_usd": [120000.0, 110000.0, 130000.0],
                "gross_revenue_usd": [250000.0, 230000.0, 280000.0],
                "reservoir_pressure_init_psi": [3200.0, 3000.0, 3400.0],
                "reservoir_pressure_psi": [2500.0, 2300.0, 2700.0],
            }
        )


class FakeModelLoader:
    def resolve_feature_columns(self, _input_columns: list[str]) -> list[str]:
        return ["oil_rate_bopd", "water_cut_pct"]

    def predict(self, _features_df: pd.DataFrame) -> np.ndarray:
        # Deterministic scores to verify rank logic.
        return np.array([0.8, 0.6, 0.9])


class CapturingModelLoader:
    def __init__(self) -> None:
        self.required = [
            "FID",
            "CATALOGID",
            "SPUD_DATE",
            "oil_rate_bopd",
            "gas_rate_mmscfd",
            "water_rate_bwpd",
            "spud_year",
            "well_age_years",
            "pressure_window_ok",
            "oil_to_water_ratio",
            "readiness_score",
            "eligibility_status",
            "analog_count",
        ]
        self.seen_features: pd.DataFrame | None = None

    def resolve_feature_columns(self, _input_columns: list[str]) -> list[str]:
        return self.required

    def predict(self, features_df: pd.DataFrame) -> np.ndarray:
        self.seen_features = features_df.copy()
        return np.array([0.5, 0.4, 0.6])


class FakeRankingRepo:
    def __init__(self) -> None:
        self.saved: list[WellRankingPrediction] = []
        self.latest_run_id: str | None = None

    def save_rankings(self, rankings: list[WellRankingPrediction]) -> int:
        self.saved = rankings
        if rankings:
            self.latest_run_id = rankings[0].run_id
        return len(rankings)

    def get_latest_run_id(self) -> str | None:
        return self.latest_run_id

    def list_by_run(self, run_id: str, limit: int, offset: int = 0) -> list[dict[str, object]]:
        rows = [item.to_row_dict() for item in self.saved if item.run_id == run_id]
        rows = sorted(rows, key=lambda row: row["rank_overall"])
        return rows[offset : offset + limit]


def test_run_ranking_builds_overall_field_and_area_ranks() -> None:
    ranking_repo = FakeRankingRepo()
    service = WellRankingService(
        settings=DummySettings(),
        trident_repo=FakeTridentRepo(),
        model_loader=FakeModelLoader(),
        ranking_repo=ranking_repo,
    )

    result = service.run_ranking()

    assert result["status"] == "ok"
    assert result["rows_scored"] == 3
    assert result["rows_saved"] == 3

    saved = sorted(ranking_repo.saved, key=lambda row: row.rank_overall)
    assert [row.uwi for row in saved] == ["U-3", "U-1", "U-2"]

    # Field-X contains U-1 and U-2, so their field ranks should be 1 and 2.
    field_x = [row for row in ranking_repo.saved if row.field_name == "Field-X"]
    field_x = sorted(field_x, key=lambda row: row.rank_on_field)
    assert [row.uwi for row in field_x] == ["U-1", "U-2"]

    # Area ranks should match ranking within each area.
    area_1 = [row for row in ranking_repo.saved if row.area_id == "Area-1"]
    area_1 = sorted(area_1, key=lambda row: row.rank_on_area)
    assert [row.uwi for row in area_1] == ["U-1", "U-2"]


def test_get_latest_rankings_uses_latest_run_when_run_id_not_supplied() -> None:
    ranking_repo = FakeRankingRepo()
    service = WellRankingService(
        settings=DummySettings(),
        trident_repo=FakeTridentRepo(),
        model_loader=FakeModelLoader(),
        ranking_repo=ranking_repo,
    )

    service.run_ranking()
    latest = service.get_latest_rankings(limit=2)

    assert latest["status"] == "ok"
    assert latest["run_id"] is not None
    assert latest["count"] == 2
    assert latest["items"][0]["rank_overall"] == 1
    assert latest["items"][1]["rank_overall"] == 2


def test_run_ranking_builds_required_model_feature_frame() -> None:
    ranking_repo = FakeRankingRepo()
    model_loader = CapturingModelLoader()
    service = WellRankingService(
        settings=DummySettings(),
        trident_repo=FakeTridentRepo(),
        model_loader=model_loader,
        ranking_repo=ranking_repo,
    )

    service.run_ranking()

    assert model_loader.seen_features is not None
    feature_df = model_loader.seen_features

    assert set(model_loader.required).issubset(set(feature_df.columns))
    assert feature_df["CATALOGID"].tolist() == ["CAT-1", "CAT-2", "CAT-3"]
    assert feature_df["FID"].tolist() == [101, 102, 103]
    assert feature_df["spud_year"].tolist() == [2020.0, 2019.0, 2018.0]
    assert feature_df["analog_count"].tolist() == [0, 0, 0]
