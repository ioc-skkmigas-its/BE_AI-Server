# sipantau-api

> Backend service for AI-powered MSF well ranking using AutoGluon.  
> Runs a weekly ranking job, stores results in Supabase, and exposes a JWT-protected REST API.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   sipantau-api                      │
│                                                     │
│  APScheduler (weekly) ─→ AutoGluon inference        │
│       │                        │                   │
│  Supabase (read wells)   Supabase (write rankings)  │
└─────────────────────────────────────────────────────┘
                    ▲
              Frontend reads well_rankings
```

## Quick Start

### 1. Setup Environment

```bash
cp .env.example .env
# Edit .env — fill in all required values (see below)
```

### 2. Required Environment Variables

| Variable | Where to find |
|---|---|
| `SECRET_KEY` | Generate: `python -c "import secrets; print(secrets.token_hex(32))"` |
| `SUPABASE_URL` | `https://jwxvmwcqqieqdmxkwslf.supabase.co` |
| `SUPABASE_ANON_KEY` | Supabase Dashboard → Project Settings → API → `anon` key |
| `SUPABASE_SERVICE_KEY` | Supabase Dashboard → Project Settings → API → `service_role` key |
| `HF_TOKEN` | https://huggingface.co/settings/tokens → New token → **Read** type |

### 3. Run Supabase Migration

Open Supabase Dashboard → SQL Editor → paste contents of:
```
migrations/001_create_ranking_tables.sql
```

### 4. Install & Run (Local Dev)

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/macOS

pip install -r requirements.txt

uvicorn app.main:app --reload
```

> **Note**: On first startup, the app will download the 1.41 GB model bundle from Hugging Face.  
> This only happens once — subsequent starts load from `data/model/`.

Open Swagger UI: http://localhost:8000/docs

### 5. Run with Docker

```bash
docker compose up --build
```

---

## API Reference

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| GET | `/health` | ❌ | App health |
| GET | `/health/model` | ❌ | Model load status |
| POST | `/api/v1/auth/register` | ❌ | Register user |
| POST | `/api/v1/auth/login` | ❌ | Login → JWT |
| GET | `/api/v1/auth/me` | ✅ JWT | Current user |
| POST | `/api/v1/ranking/trigger` | ✅ JWT | Manual trigger ranking job |
| GET | `/api/v1/ranking/status` | ✅ JWT | Latest run status |
| GET | `/api/v1/ranking/latest` | ✅ JWT | Latest ranking results |

### Authentication

All protected endpoints require:
```
Authorization: Bearer <jwt_token>
```

Get a token via `POST /api/v1/auth/login` (OAuth2PasswordRequestForm).

---

## Ranking Labels

| Label | Meaning |
|---|---|
| `TOP_10%` | Top 10% by predicted score globally |
| `TOP_25%` | Top 11–25% |
| `GOOD` | Top 26–50% |
| `AVERAGE` | Top 51–75% |
| `BELOW_AVERAGE` | Bottom 25% |

---

## Supabase Tables

| Table | Purpose |
|---|---|
| `wells` | Source well data (read by backend) |
| `well_rankings` | AI ranking output (written weekly) |
| `ranking_run_log` | Audit trail for each run |
| `latest_rankings` | View — always shows most recent run |

---

## Scheduler

The ranking job runs automatically every **Monday at 02:00 UTC**.  
Override via `.env`:
```
RANKING_SCHEDULE_DAY=mon
RANKING_SCHEDULE_HOUR=2
RANKING_SCHEDULE_MINUTE=0
```

Use `POST /api/v1/ranking/trigger` to run immediately without waiting.

---

## Development

```bash
# Run tests (no model download needed — mocked)
pytest tests/ -v

# Check code structure
python -c "from app.main import app; print('Import OK')"
```

---

## Model Info

| Property | Value |
|---|---|
| Repo | `anekazek/msf-ranking-model-final-20260413-075146` |
| Type | AutoGluon TabularPredictor |
| Target | `synthetic_demo_score` |
| Test NDCG@10 | 1.000 |
| Test NDCG@25 | 0.980 |
| Features | 69 columns |
| Grouping | `basin_cluster` |
