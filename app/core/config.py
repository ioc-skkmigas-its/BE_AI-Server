from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ── App ──────────────────────────────────────────────────
    app_name: str = "sipantau-api"
    app_version: str = "1.0.0"
    debug: bool = False

    # ── Security ─────────────────────────────────────────────
    secret_key: str
    access_token_expire_minutes: int = 60

    # ── SQLite (user auth) ───────────────────────────────────
    database_url: str = "sqlite:///./ranking.db"

    # ── Supabase ─────────────────────────────────────────────
    supabase_url: str
    supabase_anon_key: str
    supabase_service_key: str

    supabase_wells_table: str = "wells"
    supabase_static_wells_table: str = "trident_wells"
    supabase_enrich_monthlies: bool = True
    supabase_read_page_size: int = 100000000
    supabase_fetch_workers: int = 8
    supabase_write_chunk_size: int = 1000
    supabase_write_workers: int = 8
    supabase_rankings_table: str = "well_rankings"
    supabase_run_log_table: str = "ranking_run_log"

    # ── Hugging Face ─────────────────────────────────────────
    hf_token: str
    hf_model_repo: str = "anekazek/msf-ranking-model-final-20260413-075146"
    model_bundle_filename: str = "msf_ranking_inference_bundle.zip"
    model_cache_dir: str = "./data/model"

    # ── Scheduler ────────────────────────────────────────────
    ranking_schedule_day: str = "mon"
    ranking_schedule_hour: int = 2
    ranking_schedule_minute: int = 0


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
