"""
Тесты для FastAPI-сервиса классификации тональности отзывов.

Запуск:
    pytest tests/test_api.py -v
"""

import sys
from pathlib import Path

import pytest
from httpx import AsyncClient, ASGITransport

sys.path.insert(0, str(Path(__file__).parent.parent))

from IMDb_review.src.api import app


SAMPLE_PAYLOAD = {"text": "This movie was absolutely wonderful, great acting!"}


@pytest.fixture
def client():
    """Создаёт тестовый клиент FastAPI."""
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


@pytest.mark.asyncio
async def test_health_endpoint(client):
    """GET /health должен возвращать статус сервиса."""
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data


@pytest.mark.asyncio
async def test_health_response_schema(client):
    """Проверка схемы ответа /health."""
    response = await client.get("/health")
    data = response.json()
    assert isinstance(data.get("status"), str)
    assert isinstance(data.get("model_loaded"), bool)


@pytest.mark.asyncio
async def test_predict_endpoint_invalid_data(client):
    """POST /predict с пустым текстом должен вернуть 422 (min_length=1)."""
    response = await client.post("/predict", json={"text": ""})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_predict_endpoint_missing_fields(client):
    """POST /predict без поля text должен вернуть 422."""
    response = await client.post("/predict", json={})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_batch_predict_endpoint_empty(client):
    """POST /predict/batch с пустым списком должен вернуть 422 (min_length=1)."""
    response = await client.post("/predict/batch", json={"texts": []})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_predict_response_schema_or_503(client):
    """
    Если модель загружена — проверяем структуру ответа.
    Если не загружена — сервис корректно возвращает 503.
    """
    response = await client.post("/predict", json=SAMPLE_PAYLOAD)
    assert response.headers["content-type"] == "application/json"

    if response.status_code == 200:
        data = response.json()
        assert data["sentiment"] in ("positive", "negative")
        assert "confidence" in data
    else:
        assert response.status_code == 503
        assert "detail" in response.json()


@pytest.mark.asyncio
async def test_batch_predict_response_or_503(client):
    """Аналогично для батч-предсказания."""
    response = await client.post(
        "/predict/batch",
        json={"texts": [SAMPLE_PAYLOAD["text"], "This was a terrible, boring film."]},
    )
    assert response.headers["content-type"] == "application/json"

    if response.status_code == 200:
        data = response.json()
        assert len(data["results"]) == 2
        for r in data["results"]:
            assert r["sentiment"] in ("positive", "negative")
    else:
        assert response.status_code == 503