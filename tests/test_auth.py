import pytest
from fastapi.testclient import TestClient
from sqlmodel import SQLModel, create_engine, Session
from unittest.mock import patch

# Use an in-memory SQLite for tests
TEST_DB_URL = "sqlite://"

with patch("app.services.model_loader.load_model", return_value=None):
    with patch("app.core.config.settings") as mock_settings:
        mock_settings.app_name = "sipantau-api"
        mock_settings.app_version = "test"
        mock_settings.secret_key = "test-secret-key-for-testing-only"
        mock_settings.access_token_expire_minutes = 60
        mock_settings.database_url = TEST_DB_URL
        mock_settings.ranking_schedule_day = "mon"
        mock_settings.ranking_schedule_hour = 2
        mock_settings.ranking_schedule_minute = 0
        mock_settings.model_cache_dir = "./data/model"
        mock_settings.hf_model_repo = "test/repo"
        mock_settings.hf_token = "test"
        from app.main import app

client = TestClient(app)

TEST_EMAIL = "test@sipantau.id"
TEST_PASSWORD = "TestPass123!"


def test_register():
    response = client.post(
        "/api/v1/auth/register",
        json={"email": TEST_EMAIL, "password": TEST_PASSWORD},
    )
    assert response.status_code in (201, 409)  # 409 if already registered


def test_register_duplicate():
    # Register once
    client.post(
        "/api/v1/auth/register",
        json={"email": "dup@sipantau.id", "password": TEST_PASSWORD},
    )
    # Register again — should 409
    response = client.post(
        "/api/v1/auth/register",
        json={"email": "dup@sipantau.id", "password": TEST_PASSWORD},
    )
    assert response.status_code == 409


def test_login_success():
    client.post(
        "/api/v1/auth/register",
        json={"email": "login@sipantau.id", "password": TEST_PASSWORD},
    )
    response = client.post(
        "/api/v1/auth/login",
        data={"username": "login@sipantau.id", "password": TEST_PASSWORD},
    )
    assert response.status_code == 200
    assert "access_token" in response.json()


def test_login_wrong_password():
    response = client.post(
        "/api/v1/auth/login",
        data={"username": TEST_EMAIL, "password": "wrong"},
    )
    assert response.status_code == 401


def test_me_requires_auth():
    response = client.get("/api/v1/auth/me")
    assert response.status_code == 401
