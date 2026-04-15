from __future__ import annotations

import pandas as pd

from app.core.config import Settings
from app.infrastructure.supabase_client import SupabaseGateway


class TridentRepository:
    def __init__(self, settings: Settings, supabase: SupabaseGateway) -> None:
        self._settings = settings
        self._supabase = supabase

    def fetch_latest_panel(self) -> pd.DataFrame:
        monthly_rows = self._supabase.fetch_all_rows(
            table_name=self._settings.source_monthly_table,
            page_size=self._settings.supabase_read_page_size,
        )
        if not monthly_rows:
            raise ValueError(
                f"No rows found in source monthly table '{self._settings.source_monthly_table}'."
            )

        monthly_df = pd.DataFrame(monthly_rows)
        monthly_df = self._enrich_monthly_features(monthly_df)
        latest_monthly_df = self._latest_monthly_per_well(monthly_df)

        static_rows = self._supabase.fetch_all_rows(
            table_name=self._settings.supabase_static_wells_table,
            page_size=self._settings.supabase_read_page_size,
        )
        static_df = pd.DataFrame(static_rows)

        if latest_monthly_df.empty or static_df.empty:
            return latest_monthly_df

        if "well_id" not in latest_monthly_df.columns or "id" not in static_df.columns:
            return latest_monthly_df

        merged_df = latest_monthly_df.merge(
            static_df,
            how="left",
            left_on="well_id",
            right_on="id",
            suffixes=("", "_well"),
        )
        return merged_df

    @staticmethod
    def _enrich_monthly_features(monthly_df: pd.DataFrame) -> pd.DataFrame:
        if monthly_df.empty:
            return monthly_df

        df = monthly_df.copy()
        if "month_start" in df.columns:
            df["month_start"] = pd.to_datetime(df["month_start"], errors="coerce")

        if "well_id" not in df.columns:
            if "month_start" in df.columns:
                return df.sort_values("month_start").reset_index(drop=True)
            return df

        sort_cols = ["well_id"]
        if "month_start" in df.columns:
            sort_cols.append("month_start")
        elif {"year", "month"}.issubset(df.columns):
            sort_cols.extend(["year", "month"])

        df = df.sort_values(sort_cols).reset_index(drop=True)
        grouped = df.groupby("well_id", sort=False)

        def rolling_mean(source_col: str, target_col: str, window: int) -> None:
            if source_col not in df.columns:
                return
            df[target_col] = grouped[source_col].transform(
                lambda s: pd.to_numeric(s, errors="coerce").rolling(window, min_periods=1).mean()
            )

        def rolling_sum(source_col: str, target_col: str, window: int) -> None:
            if source_col not in df.columns:
                return
            df[target_col] = grouped[source_col].transform(
                lambda s: pd.to_numeric(s, errors="coerce").rolling(window, min_periods=1).sum()
            )

        def cumulative_sum(source_col: str, target_col: str) -> None:
            if source_col not in df.columns:
                return
            df[target_col] = grouped[source_col].transform(
                lambda s: pd.to_numeric(s, errors="coerce").fillna(0).cumsum()
            )

        rolling_mean("oil_rate_bopd", "oil_rate_3m_avg", 3)
        rolling_mean("gas_rate_mmscfd", "gas_rate_3m_avg", 3)
        rolling_mean("water_cut_pct", "water_cut_3m_avg", 3)
        rolling_mean("uptime_pct", "uptime_3m_avg", 3)
        rolling_sum("shut_in_days", "shutin_3m_sum", 3)

        rolling_mean("oil_rate_bopd", "oil_rate_6m_avg", 6)
        rolling_mean("gas_rate_mmscfd", "gas_rate_6m_avg", 6)
        rolling_mean("water_cut_pct", "water_cut_6m_avg", 6)
        rolling_mean("uptime_pct", "uptime_6m_avg", 6)
        rolling_sum("shut_in_days", "shutin_6m_sum", 6)

        cumulative_sum("oil_volume_bbl", "cumulative_oil_bbl")
        cumulative_sum("gas_volume_mscf", "cumulative_gas_mscf")
        cumulative_sum("water_volume_bbl", "cumulative_water_bbl")

        if "snapshot_year" not in df.columns:
            if "year" in df.columns:
                df["snapshot_year"] = pd.to_numeric(df["year"], errors="coerce")
            elif "month_start" in df.columns:
                df["snapshot_year"] = df["month_start"].dt.year

        return df

    @staticmethod
    def _latest_monthly_per_well(monthly_df: pd.DataFrame) -> pd.DataFrame:
        if monthly_df.empty:
            return monthly_df

        df = monthly_df.copy()

        if "month_start" in df.columns:
            df["_sort_key"] = pd.to_datetime(df["month_start"], errors="coerce")
        elif {"year", "month"}.issubset(set(df.columns)):
            year = pd.to_numeric(df["year"], errors="coerce").fillna(0).astype(int)
            month = pd.to_numeric(df["month"], errors="coerce").fillna(0).astype(int)
            df["_sort_key"] = year * 100 + month
        else:
            raise ValueError(
                "Source monthly table must contain month_start or (year, month) columns."
            )

        if "well_id" not in df.columns:
            return df.sort_values("_sort_key").drop(columns=["_sort_key"])

        latest_df = (
            df.sort_values("_sort_key")
            .drop_duplicates(subset=["well_id"], keep="last")
            .drop(columns=["_sort_key"])
            .reset_index(drop=True)
        )
        return latest_df
