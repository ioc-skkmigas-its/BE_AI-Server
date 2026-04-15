from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "fastapi-autogluon-starter"
    app_version: str = "0.1.0"

    enable_auto_migration: bool = True
    # PostgreSQL URL used to run DDL (recommended for Supabase).
    # Example: postgresql://postgres:<password>@db.<project-ref>.supabase.co:5432/postgres?sslmode=require
    supabase_db_url: str | None = None
    # Fallback DB URL if SUPABASE_DB_URL is not set.
    database_url: str | None = None

    supabase_url: str
    supabase_anon_key: str
    supabase_service_key: str
    # Backward-compatible source table key.
    supabase_wells_table: str = "trident_well_monthlies"
    supabase_monthly_table: str | None = None
    supabase_static_wells_table: str = "trident_wells"
    supabase_rankings_table: str = "well_ranking_predictions"
    supabase_read_page_size: int = 1000
    supabase_write_chunk_size: int = 500

    ranking_upsert_conflict_keys: str = "record_key"

    artifact_bundle_zip_path: str = "./mature-field-candidate-ranking-v4-final-20260414-014711-main.zip"
    artifact_extract_dir: str = "./data/model_artifacts"

    xgboost_model_path: str = "./data/model_artifacts/uplift_base_model.joblib"
    xgboost_preprocessor_path: str | None = "./data/model_artifacts/preprocessor.joblib"
    xgboost_feature_manifest_path: str | None = "./data/model_artifacts/artifact_manifest.json"

    @property
    def source_monthly_table(self) -> str:
        return self.supabase_monthly_table or self.supabase_wells_table


@lru_cache
def get_settings() -> Settings:
    return Settings()
