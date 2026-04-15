---
title: IOC MIGAS Ranking API
emoji: 🛢️
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# IOC MIGAS Ranking API

FastAPI backend for candidate well ranking with XGBoost artifacts.

## Runtime

- Entry: `main:app`
- Port: `7860`
- Health: `GET /health`
- Run ranking: `POST /ranking/run`
- Latest ranking: `GET /ranking/latest`

## Required Space Secrets

Set these in Hugging Face Space Settings -> Variables and secrets:

- `SUPABASE_URL`
- `SUPABASE_ANON_KEY`
- `SUPABASE_SERVICE_KEY`
- `SUPABASE_DB_URL` (for auto migration)

Optional overrides:

- `SUPABASE_MONTHLY_TABLE`
- `SUPABASE_STATIC_WELLS_TABLE`
- `SUPABASE_RANKINGS_TABLE`
- `RANKING_UPSERT_CONFLICT_KEYS`

Model artifact defaults are configured to use:

- `./mature-field-candidate-ranking-v4-final-20260414-014711-main.zip`
- extraction dir `./data/model_artifacts`

## Notes

- The app performs auto migration on startup when `ENABLE_AUTO_MIGRATION=true`.
- If Space startup fails, verify all required secrets are present and valid.