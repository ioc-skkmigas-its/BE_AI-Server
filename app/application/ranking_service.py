from __future__ import annotations

from datetime import datetime, timezone
import uuid

import numpy as np
import pandas as pd

from app.core.config import Settings
from app.domain.models.well_ranking import WellRankingPrediction
from app.infrastructure.model_loader import XGBoostModelLoader
from app.infrastructure.ranking_repository import RankingRepository
from app.infrastructure.trident_repository import TridentRepository


class WellRankingService:
    _COLUMN_ALIASES: dict[str, list[str]] = {
        "catalogid": ["catalog_id"],
        "spud_date": ["spud_date"],
        "stts_type": ["stts_type"],
        "stts_state": ["stts_state"],
        "stts_fluid": ["stts_fluid"],
        "envrmt_typ": ["envrmt_typ"],
        "lat": ["lat"],
        "long": ["long"],
        "srfce_lat": ["srfce_lat"],
        "srfce_long": ["srfce_long"],
        "utm_zone": ["utm_zone"],
        "easting": ["easting"],
        "northing": ["northing"],
        "data_quality_flag": ["data_quality_flag"],
    }

    def __init__(
        self,
        settings: Settings,
        trident_repo: TridentRepository,
        model_loader: XGBoostModelLoader,
        ranking_repo: RankingRepository,
    ) -> None:
        self._settings = settings
        self._trident_repo = trident_repo
        self._model_loader = model_loader
        self._ranking_repo = ranking_repo

    def run_ranking(self) -> dict[str, object]:
        raw_df = self._trident_repo.fetch_latest_panel()
        if raw_df.empty:
            raise ValueError("No source data is available for ranking.")

        identifiers_df = self._build_identifiers(raw_df)

        feature_cols = self._model_loader.resolve_feature_columns(list(raw_df.columns))
        if not feature_cols:
            raise ValueError("No non-ID feature columns were found for prediction.")

        feature_df = self._build_feature_frame(raw_df, feature_cols)
        predicted_scores = self._model_loader.predict(feature_df)

        ranked_df = self._build_ranked_frame(identifiers_df, predicted_scores)

        run_id = str(uuid.uuid4())
        ranked_at = datetime.now(timezone.utc)

        ranking_rows = [
            self._to_orm(run_id=run_id, ranked_at=ranked_at, row=row)
            for row in ranked_df.to_dict(orient="records")
        ]

        saved_rows = self._ranking_repo.save_rankings(ranking_rows)

        return {
            "status": "ok",
            "run_id": run_id,
            "rows_scored": int(len(ranked_df)),
            "rows_saved": int(saved_rows),
            "output_table": self._settings.supabase_rankings_table,
        }

    def get_latest_rankings(
        self,
        limit: int,
        offset: int = 0,
        run_id: str | None = None,
    ) -> dict[str, object]:
        target_run_id = run_id or self._ranking_repo.get_latest_run_id()
        if target_run_id is None:
            return {"status": "ok", "run_id": None, "count": 0, "items": []}

        rows = self._ranking_repo.list_by_run(
            run_id=target_run_id,
            limit=limit,
            offset=offset,
        )
        return {
            "status": "ok",
            "run_id": target_run_id,
            "count": len(rows),
            "items": rows,
        }

    def _build_identifiers(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        identifiers = pd.DataFrame(index=raw_df.index)

        identifiers["uwi"] = self._pick_column(raw_df, ["uwi"])
        identifiers["well_name"] = self._pick_column(raw_df, ["well_name"])
        identifiers["field_name"] = self._pick_column(raw_df, ["field_name"])
        identifiers["area_id"] = self._pick_column(raw_df, ["area_id"])
        identifiers["basin_cluster"] = self._pick_column(raw_df, ["basin_cluster"])

        month_start = self._pick_column(raw_df, ["month_start"])
        month_start_dt = pd.to_datetime(month_start, errors="coerce")
        identifiers["month_start"] = month_start_dt.dt.strftime("%Y-%m-%d")

        return identifiers

    @staticmethod
    def _pick_column(raw_df: pd.DataFrame, candidates: list[str]) -> pd.Series:
        lookup = {col.lower(): col for col in raw_df.columns}
        for candidate in candidates:
            source_col = lookup.get(candidate.lower())
            if source_col is not None:
                return raw_df[source_col]
        return pd.Series([None] * len(raw_df), index=raw_df.index, dtype="object")

    def _build_feature_frame(self, raw_df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
        source_lookup = {col.lower(): col for col in raw_df.columns}
        features = pd.DataFrame(index=raw_df.index)

        for feature in feature_cols:
            source_col = source_lookup.get(feature.lower())
            if source_col is None:
                for alias in self._COLUMN_ALIASES.get(feature.lower(), []):
                    source_col = source_lookup.get(alias)
                    if source_col is not None:
                        break

            if source_col is None:
                features[feature] = pd.Series(np.nan, index=raw_df.index)
            else:
                features[feature] = raw_df[source_col]

        self._populate_derived_features(features)

        for col in features.columns:
            if pd.api.types.is_bool_dtype(features[col]):
                features[col] = features[col].astype(int)
        return features

    @staticmethod
    def _series_or_nan(features: pd.DataFrame, col_name: str) -> pd.Series:
        if col_name not in features.columns:
            return pd.Series(np.nan, index=features.index, dtype="float64")
        return pd.to_numeric(features[col_name], errors="coerce")

    @staticmethod
    def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
        den = denominator.copy()
        den = den.where(den != 0)
        return numerator / den

    @staticmethod
    def _fill_feature(features: pd.DataFrame, col_name: str, values: pd.Series | float | int | str) -> None:
        if col_name not in features.columns:
            return
        if not isinstance(values, pd.Series):
            values = pd.Series(values, index=features.index)
        features[col_name] = features[col_name].where(features[col_name].notna(), values)

    @staticmethod
    def _populate_derived_features(features: pd.DataFrame) -> None:
        month_start_raw = (
            features["month_start"]
            if "month_start" in features.columns
            else pd.Series(pd.NaT, index=features.index)
        )
        month_start = pd.to_datetime(month_start_raw, errors="coerce")

        if "SPUD_DATE" in features.columns:
            spud_date_raw = features["SPUD_DATE"]
        elif "spud_date" in features.columns:
            spud_date_raw = features["spud_date"]
        else:
            spud_date_raw = pd.Series(pd.NaT, index=features.index)
        spud_date = pd.to_datetime(spud_date_raw, errors="coerce")

        snapshot_year = WellRankingService._series_or_nan(features, "year")
        snapshot_year = snapshot_year.where(snapshot_year.notna(), month_start.dt.year)
        WellRankingService._fill_feature(features, "snapshot_year", snapshot_year)

        WellRankingService._fill_feature(features, "spud_year", spud_date.dt.year)
        WellRankingService._fill_feature(features, "spud_month", spud_date.dt.month)

        spud_year = WellRankingService._series_or_nan(features, "spud_year")
        well_age = snapshot_year - spud_year
        WellRankingService._fill_feature(features, "well_age_years", well_age.clip(lower=0))

        oil_3m = WellRankingService._series_or_nan(features, "oil_rate_3m_avg")
        oil_6m = WellRankingService._series_or_nan(features, "oil_rate_6m_avg")
        oil_decline_short = WellRankingService._safe_divide(oil_6m - oil_3m, oil_6m.abs())
        WellRankingService._fill_feature(features, "oil_decline_short_proxy", oil_decline_short)

        water_3m = WellRankingService._series_or_nan(features, "water_cut_3m_avg")
        water_6m = WellRankingService._series_or_nan(features, "water_cut_6m_avg")
        WellRankingService._fill_feature(features, "watercut_worsening_proxy", water_3m - water_6m)

        shutin_6m_sum = WellRankingService._series_or_nan(features, "shutin_6m_sum")
        downtime_ratio_6m = WellRankingService._safe_divide(
            shutin_6m_sum,
            pd.Series(180.0, index=features.index),
        )
        WellRankingService._fill_feature(features, "downtime_ratio_6m", downtime_ratio_6m)

        oil_rate = WellRankingService._series_or_nan(features, "oil_rate_bopd")
        water_rate = WellRankingService._series_or_nan(features, "water_rate_bwpd")
        gas_rate = WellRankingService._series_or_nan(features, "gas_rate_mmscfd")
        WellRankingService._fill_feature(
            features,
            "oil_to_water_ratio",
            WellRankingService._safe_divide(oil_rate, water_rate),
        )
        WellRankingService._fill_feature(
            features,
            "gas_to_oil_ratio",
            WellRankingService._safe_divide(gas_rate, oil_rate),
        )

        pressure_init = WellRankingService._series_or_nan(features, "reservoir_pressure_init_psi")
        pressure_current = WellRankingService._series_or_nan(features, "reservoir_pressure_psi")
        pressure_drawdown = pressure_init - pressure_current
        pressure_ratio = WellRankingService._safe_divide(pressure_current, pressure_init)
        WellRankingService._fill_feature(features, "pressure_drawdown_proxy", pressure_drawdown)
        WellRankingService._fill_feature(features, "pressure_ratio_current_to_init", pressure_ratio)

        boe_total = WellRankingService._series_or_nan(features, "boe_total")
        opex = WellRankingService._series_or_nan(features, "operating_cost_usd")
        revenue = WellRankingService._series_or_nan(features, "gross_revenue_usd")
        cost_per_boe = WellRankingService._safe_divide(opex, boe_total)
        revenue_per_boe = WellRankingService._safe_divide(revenue, boe_total)
        margin_per_boe = revenue_per_boe - cost_per_boe
        WellRankingService._fill_feature(features, "cost_per_boe_proxy", cost_per_boe)
        WellRankingService._fill_feature(features, "revenue_per_boe_proxy", revenue_per_boe)
        WellRankingService._fill_feature(features, "margin_per_boe_proxy", margin_per_boe)

        intervention_cost = WellRankingService._series_or_nan(features, "intervention_cost_usd")
        oil_volume = WellRankingService._series_or_nan(features, "oil_volume_bbl")
        WellRankingService._fill_feature(
            features,
            "opex_incremental_usd_bbl_proxy",
            WellRankingService._safe_divide(intervention_cost, oil_volume),
        )

        WellRankingService._fill_feature(
            features,
            "planned_stage_count",
            WellRankingService._series_or_nan(features, "msf_stage_count"),
        )
        WellRankingService._fill_feature(
            features,
            "planned_proppant_ton",
            WellRankingService._series_or_nan(features, "proppant_ton"),
        )
        WellRankingService._fill_feature(
            features,
            "planned_fluid_bbl",
            WellRankingService._series_or_nan(features, "frac_fluid_bbl"),
        )

        pressure_window_ok = pressure_ratio.between(0.35, 1.15, inclusive="both").astype(int)
        WellRankingService._fill_feature(features, "pressure_window_ok", pressure_window_ok)

        completion_type = features.get("completion_type")
        if completion_type is not None:
            completion_text = completion_type.astype(str).str.strip().str.lower()
            completion_integrity_ok = (~completion_text.isin(["", "nan", "none", "unknown"]))
            WellRankingService._fill_feature(
                features,
                "completion_integrity_ok",
                completion_integrity_ok.astype(int),
            )

        planned_prop = WellRankingService._series_or_nan(features, "planned_proppant_ton")
        planned_fluid = WellRankingService._series_or_nan(features, "planned_fluid_bbl")
        chemical_envelope_ok = ((planned_prop > 0) & (planned_fluid > 0)).astype(int)
        WellRankingService._fill_feature(features, "chemical_proppant_envelope_ok", chemical_envelope_ok)

        event_code = features.get("event_code")
        if event_code is not None:
            event_text = event_code.astype(str).str.lower()
            facility_limitation = event_text.str.contains("facility|constraint|defer|limit", regex=True)
            WellRankingService._fill_feature(
                features,
                "facility_limitation_flag",
                facility_limitation.astype(int),
            )

        anomaly_flag = WellRankingService._series_or_nan(features, "anomaly_flag")
        WellRankingService._fill_feature(features, "hse_flag", (anomaly_flag > 0).astype(int))

        uptime_score = WellRankingService._series_or_nan(features, "uptime_3m_avg")
        uptime_score = uptime_score.where(uptime_score.notna(), WellRankingService._series_or_nan(features, "uptime_pct"))
        service_score = uptime_score.clip(lower=0, upper=100) / 100.0
        WellRankingService._fill_feature(features, "service_availability_score", service_score)

        completeness_cols = [
            col
            for col in [
                "oil_rate_bopd",
                "gas_rate_mmscfd",
                "water_rate_bwpd",
                "water_cut_pct",
                "uptime_pct",
                "reservoir_pressure_psi",
                "operating_cost_usd",
                "boe_total",
            ]
            if col in features.columns
        ]
        if completeness_cols:
            completeness_score = features[completeness_cols].notna().mean(axis=1)
        else:
            completeness_score = pd.Series(0.0, index=features.index)
        WellRankingService._fill_feature(features, "data_completeness_score", completeness_score)

        downtime_ratio = WellRankingService._series_or_nan(features, "downtime_ratio_6m").fillna(0).clip(0, 1)
        readiness_score = (
            service_score.fillna(0)
            + completeness_score.fillna(0)
            + (1 - downtime_ratio)
        ) / 3.0
        WellRankingService._fill_feature(features, "readiness_score", readiness_score.clip(lower=0, upper=1))

        maturity = features.get("maturity_class")
        if maturity is not None:
            maturity_text = maturity.astype(str).str.strip().str.lower()
            maturity_score = maturity_text.map(
                {
                    "immature": 0,
                    "early": 1,
                    "emerging": 1,
                    "mature": 2,
                    "late": 3,
                    "declining": 4,
                }
            )
            WellRankingService._fill_feature(
                features,
                "field_maturity_override",
                maturity_score,
            )

        active_flag = WellRankingService._series_or_nan(features, "active_flag").fillna(0)
        eligibility = ((active_flag > 0) & (readiness_score.fillna(0) >= 0.40)).astype(int)
        WellRankingService._fill_feature(features, "eligibility_status", eligibility)

        WellRankingService._fill_feature(features, "analog_success_rate", pd.Series(0.0, index=features.index))
        WellRankingService._fill_feature(
            features,
            "analog_median_uplift_bopd",
            WellRankingService._series_or_nan(features, "estimated_msf_uplift_bopd").fillna(0.0),
        )
        WellRankingService._fill_feature(features, "analog_count", pd.Series(0, index=features.index))

    @staticmethod
    def _build_ranked_frame(
        identifiers_df: pd.DataFrame,
        predicted_scores: np.ndarray,
    ) -> pd.DataFrame:
        ranked_df = identifiers_df.copy()
        ranked_df["predicted_score"] = predicted_scores

        ranked_df["rank_overall"] = (
            ranked_df["predicted_score"].rank(ascending=False, method="min").astype(int)
        )

        ranked_df["_field_group"] = ranked_df["field_name"].fillna("UNKNOWN_FIELD")
        ranked_df["_area_group"] = ranked_df["area_id"].fillna("UNKNOWN_AREA")

        ranked_df["rank_on_field"] = (
            ranked_df.groupby("_field_group")["predicted_score"]
            .rank(ascending=False, method="min")
            .astype(int)
        )
        ranked_df["rank_on_area"] = (
            ranked_df.groupby("_area_group")["predicted_score"]
            .rank(ascending=False, method="min")
            .astype(int)
        )

        ranked_df = ranked_df.drop(columns=["_field_group", "_area_group"])
        ranked_df = ranked_df.sort_values("rank_overall", kind="stable").reset_index(drop=True)
        return ranked_df

    @staticmethod
    def _to_orm(
        run_id: str,
        ranked_at: datetime,
        row: dict[str, object],
    ) -> WellRankingPrediction:
        uwi = WellRankingService._as_text(row.get("uwi"))
        well_name = WellRankingService._as_text(row.get("well_name"))
        field_name = WellRankingService._as_text(row.get("field_name"))
        area_id = WellRankingService._as_text(row.get("area_id"))
        basin_cluster = WellRankingService._as_text(row.get("basin_cluster"))
        month_start = WellRankingService._as_text(row.get("month_start"))

        record_key = WellRankingPrediction.build_record_key(
            run_id=run_id,
            uwi=uwi,
            well_name=well_name,
            field_name=field_name,
            area_id=area_id,
            month_start=month_start,
        )

        return WellRankingPrediction(
            record_key=record_key,
            run_id=run_id,
            ranked_at=ranked_at,
            uwi=uwi,
            well_name=well_name,
            field_name=field_name,
            area_id=area_id,
            basin_cluster=basin_cluster,
            month_start=month_start,
            predicted_score=float(row["predicted_score"]),
            rank_overall=int(row["rank_overall"]),
            rank_on_field=int(row["rank_on_field"]),
            rank_on_area=int(row["rank_on_area"]),
        )

    @staticmethod
    def _as_text(value: object) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        if text == "" or text.lower() == "nan":
            return None
        return text
