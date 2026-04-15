from typing import Any

from supabase import Client, create_client

from app.core.config import Settings


class SupabaseGateway:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self.anon_client: Client = create_client(
            settings.supabase_url,
            settings.supabase_anon_key,
        )
        self.service_client: Client = create_client(
            settings.supabase_url,
            settings.supabase_service_key,
        )

    def count_rows(self, table_name: str) -> int:
        response = (
            self.service_client.table(table_name)
            .select("*", count="exact")
            .limit(1)
            .execute()
        )
        return response.count if response.count is not None else 0

    def fetch_all_rows(
        self,
        table_name: str,
        page_size: int,
        columns: str = "*",
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        start = 0
        safe_page_size = max(1, page_size)

        while True:
            response = (
                self.anon_client.table(table_name)
                .select(columns)
                .range(start, start + safe_page_size - 1)
                .execute()
            )
            batch = response.data or []
            rows.extend(batch)

            if len(batch) < safe_page_size:
                break

            start += safe_page_size

        return rows

    def fetch_rows(
        self,
        table_name: str,
        limit: int,
        offset: int = 0,
        order_by: str | None = None,
        descending: bool = False,
        filters: dict[str, Any] | None = None,
        columns: str = "*",
    ) -> list[dict[str, Any]]:
        query = self.anon_client.table(table_name).select(columns)

        if filters:
            for key, value in filters.items():
                query = query.eq(key, value)

        if order_by:
            query = query.order(order_by, desc=descending)

        safe_limit = max(1, limit)
        safe_offset = max(0, offset)
        query = query.range(safe_offset, safe_offset + safe_limit - 1)

        response = query.execute()
        return response.data or []

    def upsert_rows(
        self,
        table_name: str,
        rows: list[dict[str, Any]],
        on_conflict: str | None = None,
        chunk_size: int = 500,
    ) -> int:
        if not rows:
            return 0

        safe_chunk_size = max(1, chunk_size)
        for start in range(0, len(rows), safe_chunk_size):
            chunk = rows[start : start + safe_chunk_size]
            query = self.service_client.table(table_name)
            if on_conflict:
                query.upsert(chunk, on_conflict=on_conflict).execute()
            else:
                query.upsert(chunk).execute()

        return len(rows)
