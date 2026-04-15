---
title: IOC MIGAS Ranking API
emoji: 🛢️
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

## IOC MIGAS Ranking API

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
- `HF_TOKEN` (only needed if model bundle is downloaded from HF Hub)

Required when ZIP model bundle is not shipped in Space repo:

- `ARTIFACT_BUNDLE_HF_REPO`
- `ARTIFACT_BUNDLE_HF_FILENAME`

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

## GitHub Actions Deployment

This repository includes a complete workflow at `.github/workflows/deploy-hf-space.yml`.

### What it does

- Runs unit tests: `tests/test_ranking_service.py`
- Deploys to Hugging Face Space after tests pass
- Supports both automatic deploy on `main` and manual deploy with `workflow_dispatch`
- Builds a deploy mirror that excludes binary files rejected by HF git hook (`*.zip`, `*.joblib`, etc.)
- Can include tracked `data/` files when requested (non-binary only)

### Required GitHub Secrets

Add this secret in GitHub Repository Settings -> Secrets and variables -> Actions:

- `HF_TOKEN`: Hugging Face write token with access to `XRyZ/ioc-migas`

Optional GitHub repository variable:

- `HF_SPACE_REPO`: override target space (`owner/name`).
  If not set, workflow uses `XRyZ/ioc-migas`.
- `HF_INCLUDE_DATA`: set to `true` if `data/` should be included on automatic deploys.

Optional manual dispatch input:

- `include_data`: set to `true` to include tracked `data/` in deploy mirror.
  Binary model artifacts (`*.joblib`, `*.pkl`, `*.zip`, `*.ubj`, `*.bst`) remain excluded.
  `data/model_artifacts` is also gitignored in this repository.

### Trigger behavior

- `pull_request`: run tests only
- `push` to `main`: run tests and deploy
- `workflow_dispatch`: run tests and deploy with optional custom target space
