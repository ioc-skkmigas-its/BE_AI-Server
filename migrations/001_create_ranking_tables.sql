-- Migration 001: Create ranking output tables
-- Run this SQL in your Supabase SQL Editor
-- Supabase Dashboard → SQL Editor → New Query → paste & run

-- ─────────────────────────────────────────────────────────────
-- Table: ranking_run_log
-- Audit trail for every weekly (or manual) ranking job run
-- ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS ranking_run_log (
    id              TEXT PRIMARY KEY,          -- UUID v4
    triggered_by    TEXT NOT NULL,             -- 'scheduler' | 'manual' | 'api'
    status          TEXT NOT NULL DEFAULT 'running', -- 'running' | 'success' | 'failed'
    wells_processed INTEGER,
    duration_sec    FLOAT,
    error_message   TEXT,
    started_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at     TIMESTAMPTZ
);

-- ─────────────────────────────────────────────────────────────
-- Table: well_rankings
-- Weekly AI-generated ranking results, read by frontend
-- ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS well_rankings (
    id              BIGSERIAL PRIMARY KEY,
    uwi             TEXT NOT NULL,             -- Unique Well Identifier
    well_name       TEXT,
    field_name      TEXT,
    area_id         TEXT,
    basin_cluster   TEXT,
    predicted_score FLOAT NOT NULL,            -- AutoGluon output score
    rank_overall    INTEGER,                   -- Global rank across all wells
    rank_in_basin   INTEGER,                   -- Rank within basin_cluster group
    rank_label      TEXT NOT NULL,             -- 'TOP_10%' | 'TOP_25%' | 'GOOD' | 'AVERAGE' | 'BELOW_AVERAGE'
    run_id          TEXT NOT NULL REFERENCES ranking_run_log(id),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_well_rankings_run_id       ON well_rankings (run_id);
CREATE INDEX IF NOT EXISTS idx_well_rankings_basin        ON well_rankings (basin_cluster);
CREATE INDEX IF NOT EXISTS idx_well_rankings_uwi          ON well_rankings (uwi);
CREATE INDEX IF NOT EXISTS idx_well_rankings_score        ON well_rankings (predicted_score DESC);
CREATE INDEX IF NOT EXISTS idx_well_rankings_rank_overall ON well_rankings (rank_overall);

-- ─────────────────────────────────────────────────────────────
-- View: latest_rankings
-- Always shows results from the most recent successful run
-- Use this view in your frontend queries
-- ─────────────────────────────────────────────────────────────
CREATE OR REPLACE VIEW latest_rankings AS
SELECT wr.*
FROM well_rankings wr
INNER JOIN (
    SELECT id
    FROM ranking_run_log
    WHERE status = 'success'
    ORDER BY finished_at DESC
    LIMIT 1
) latest_run ON wr.run_id = latest_run.id;

-- ─────────────────────────────────────────────────────────────
-- RLS: Enable Row Level Security (recommended for production)
-- ─────────────────────────────────────────────────────────────
-- Note: The backend uses service_role key which bypasses RLS.
-- These policies are for direct frontend/anon access.

ALTER TABLE ranking_run_log ENABLE ROW LEVEL SECURITY;
ALTER TABLE well_rankings   ENABLE ROW LEVEL SECURITY;

-- Allow public read access to ranking results
CREATE POLICY "Public read ranking_run_log"
    ON ranking_run_log FOR SELECT USING (true);

CREATE POLICY "Public read well_rankings"
    ON well_rankings FOR SELECT USING (true);

-- Deny all writes from anon (only service_role can write)
-- (No INSERT/UPDATE/DELETE policies for anon = denied by default)
