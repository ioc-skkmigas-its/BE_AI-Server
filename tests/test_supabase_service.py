import asyncio
import os
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

os.environ.setdefault("DEBUG", "false")

from app.services import supabase_service


class _FakeReadQuery:
    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows
        self.calls: list[tuple[int, int]] = []
        self._current_range = (0, -1)

    def select(self, _fields: str, **_kwargs):
        return self

    def range(self, start: int, end: int):
        self._current_range = (start, end)
        self.calls.append((start, end))
        return self

    def execute(self):
        start, end = self._current_range
        return SimpleNamespace(data=self._rows[start : end + 1])


class _FakeReadClient:
    def __init__(self, rows: list[dict]) -> None:
        self.query = _FakeReadQuery(rows)

    def table(self, _name: str):
        return self.query


class _FakeWriteQuery:
    def __init__(self) -> None:
        self.inserted_chunks: list[list[dict]] = []

    def insert(self, payload: list[dict]):
        self.inserted_chunks.append(payload)
        return self

    def execute(self):
        return SimpleNamespace(data=[])


class _FakeWriteClient:
    def __init__(self) -> None:
        self.query = _FakeWriteQuery()

    def table(self, _name: str):
        return self.query


def test_fetch_all_wells_paginates_until_last_page():
    rows = [{"well_id": f"W-{i}"} for i in range(2500)]
    fake_client = _FakeReadClient(rows)

    with patch("app.services.supabase_service._get_anon_client", return_value=fake_client):
        with patch("app.services.supabase_service.settings.supabase_wells_table", "mock_table"):
            with patch("app.services.supabase_service.settings.supabase_enrich_monthlies", False):
                with patch("app.services.supabase_service._READ_PAGE_SIZE", 1000):
                    df = asyncio.run(supabase_service.fetch_all_wells())

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2500
    assert fake_client.query.calls == [
        (0, 999),
        (1000, 1999),
        (2000, 2999),
        (2500, 3499),
    ]


def test_enrich_monthlies_with_static_derives_proxy_features():
    monthlies_df = pd.DataFrame(
        [
            {
                "well_id": "WELL-ID-1",
                "oil_rate_bopd": 100.0,
                "water_rate_bwpd": 20.0,
                "gas_rate_mmscfd": 2.0,
                "reservoir_pressure_psi": 1800.0,
                "operating_cost_usd": 1000.0,
                "gross_revenue_usd": 3000.0,
                "boe_total": 200.0,
            }
        ]
    )
    static_df = pd.DataFrame(
        [
            {
                "id": "WELL-ID-1",
                "uwi": "UWI-1",
                "spud_date": "2020-05-17",
                "well_class": "DEV",
                "reservoir_pressure_init_psi": 2000.0,
            }
        ]
    )

    out = supabase_service._enrich_monthlies_with_static(monthlies_df, static_df)

    assert len(out) == 1
    assert out.loc[0, "spud_year"] == 2020
    assert out.loc[0, "spud_month"] == 5
    assert out.loc[0, "oil_to_water_ratio"] == 5.0
    assert out.loc[0, "gas_to_oil_ratio"] == 0.02
    assert out.loc[0, "pressure_drawdown_proxy"] == 200.0
    assert out.loc[0, "cost_per_boe_proxy"] == 5.0
    assert out.loc[0, "revenue_per_boe_proxy"] == 15.0
    assert out.loc[0, "margin_per_boe_proxy"] == 10.0
    assert out.loc[0, "uwi"] == "UWI-1"


def test_write_rankings_maps_identifier_fallback_fields():
    ranked_df = pd.DataFrame(
        [
            {
                "well_id": "WELL-001",
                "well_name": "Alpha-1",
                "predicted_score": 12.3,
                "rank_overall": 1,
                "rank_in_basin": 1,
                "rank_label": "TOP_10%",
            }
        ]
    )
    fake_client = _FakeWriteClient()

    with patch("app.services.supabase_service._get_service_client", return_value=fake_client):
        asyncio.run(supabase_service.write_rankings("run-1", ranked_df))

    assert len(fake_client.query.inserted_chunks) == 1
    payload = fake_client.query.inserted_chunks[0][0]
    assert payload["uwi"] == "WELL-001"
    assert payload["well_name"] == "Alpha-1"
    assert payload["run_id"] == "run-1"


def test_write_rankings_uses_chunk_size_setting():
    ranked_df = pd.DataFrame(
        [
            {
                "well_id": f"WELL-{i}",
                "predicted_score": float(i),
                "rank_overall": i,
                "rank_in_basin": i,
                "rank_label": "GOOD",
            }
            for i in range(1, 6)
        ]
    )
    fake_client = _FakeWriteClient()

    with patch("app.services.supabase_service._get_service_client", return_value=fake_client):
        with patch("app.services.supabase_service.settings.supabase_write_chunk_size", 2):
            with patch("app.services.supabase_service.settings.supabase_write_workers", 1):
                asyncio.run(supabase_service.write_rankings("run-2", ranked_df))

    chunk_sizes = [len(chunk) for chunk in fake_client.query.inserted_chunks]
    assert chunk_sizes == [2, 2, 1]
