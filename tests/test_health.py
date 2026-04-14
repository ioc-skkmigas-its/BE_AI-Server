import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

# Patch model load so tests don't need the actual 1.4GB model
with patch("app.services.model_loader.load_model", return_value=None):
    with patch("app.services.model_loader._model_ready", True):
        from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "app" in data
    assert "version" in data


def test_health_model():
    response = client.get("/health/model")
    assert response.status_code == 200
    data = response.json()
    assert "model_ready" in data
    assert "hf_repo" in data
