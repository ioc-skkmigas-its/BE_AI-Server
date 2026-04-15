from app.core.config import Settings
from app.infrastructure.supabase_client import SupabaseGateway


class HealthService:
    def __init__(self, settings: Settings, supabase: SupabaseGateway) -> None:
        self._settings = settings
        self._supabase = supabase

    def app_health(self) -> dict[str, str]:
        return {"status": "ok"}

    def supabase_health(self) -> dict[str, str]:
        table_name = self._settings.source_monthly_table
        try:
            count = self._supabase.count_rows(table_name)
            return {
                "status": "ok",
                "table": table_name,
                "count": str(count),
            }
        except Exception as exc:
            return {
                "status": "error",
                "table": table_name,
                "detail": str(exc),
            }
