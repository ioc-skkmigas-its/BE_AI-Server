from __future__ import annotations

from hashlib import sha1
import re

from sqlalchemy import create_engine, text

from app.core.config import Settings


_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_POSTGRES_PREFIXES = (
    "postgresql+psycopg://",
    "postgresql://",
    "postgres://",
)


def run_auto_migration(settings: Settings) -> dict[str, str]:
    if not settings.enable_auto_migration:
        return {
            "status": "skipped",
            "detail": "Auto migration is disabled by configuration.",
        }

    raw_db_url = settings.supabase_db_url or settings.database_url
    if not raw_db_url:
        return {
            "status": "skipped",
            "detail": "SUPABASE_DB_URL or DATABASE_URL is required for auto migration.",
        }

    db_url = _resolve_db_url(settings)
    if db_url is None:
        return {
            "status": "skipped",
            "detail": "Auto migration only supports PostgreSQL URLs. Set SUPABASE_DB_URL to a Supabase Postgres connection string.",
        }

    table_name = _validate_identifier(settings.supabase_rankings_table, "table")
    normalized_url = _normalize_postgres_url(db_url)

    statements = _build_statements(table_name)
    engine = create_engine(normalized_url, pool_pre_ping=True)
    try:
        with engine.begin() as connection:
            for statement in statements:
                connection.execute(text(statement))
    finally:
        engine.dispose()

    return {
        "status": "applied",
        "detail": f"Auto migration applied for table '{table_name}'.",
    }


def _resolve_db_url(settings: Settings) -> str | None:
    db_url = settings.supabase_db_url or settings.database_url
    if not db_url:
        return None

    lowered = db_url.lower()
    if not lowered.startswith(_POSTGRES_PREFIXES):
        return None

    return db_url


def _normalize_postgres_url(db_url: str) -> str:
    if db_url.startswith("postgresql://"):
        return db_url.replace("postgresql://", "postgresql+psycopg://", 1)
    if db_url.startswith("postgres://"):
        return db_url.replace("postgres://", "postgresql+psycopg://", 1)
    return db_url


def _build_statements(table_name: str) -> list[str]:
    create_table_statement = f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        record_key VARCHAR(64) PRIMARY KEY,
        run_id VARCHAR(36) NOT NULL,
        ranked_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        uwi VARCHAR(100),
        well_name VARCHAR(120),
        field_name VARCHAR(120),
        area_id VARCHAR(100),
        basin_cluster VARCHAR(80),
        month_start VARCHAR(20),
        predicted_score DOUBLE PRECISION NOT NULL,
        rank_overall INTEGER NOT NULL,
        rank_on_field INTEGER NOT NULL,
        rank_on_area INTEGER NOT NULL
    );
    """.strip()

    indexed_columns = [
        "run_id",
        "ranked_at",
        "uwi",
        "well_name",
        "field_name",
        "area_id",
        "basin_cluster",
        "rank_overall",
        "rank_on_field",
        "rank_on_area",
    ]

    index_statements = []
    for column in indexed_columns:
        _validate_identifier(column, "column")
        index_name = _build_index_name(table_name=table_name, column_name=column)
        index_statements.append(
            f"CREATE INDEX IF NOT EXISTS {index_name} ON {table_name} ({column});"
        )

    return [create_table_statement, *index_statements]


def _build_index_name(table_name: str, column_name: str) -> str:
    raw_name = f"idx_{table_name}_{column_name}"
    if len(raw_name) <= 63:
        return raw_name

    digest = sha1(raw_name.encode("utf-8")).hexdigest()[:8]
    trimmed = raw_name[:54]
    return f"{trimmed}_{digest}"


def _validate_identifier(value: str, identifier_type: str) -> str:
    if not _IDENTIFIER_PATTERN.fullmatch(value):
        raise ValueError(
            f"Invalid {identifier_type} identifier '{value}'. "
            "Use letters, numbers, and underscores only."
        )
    return value
