"""
FastAPI-сервис для инференса модели классификации тональности отзывов.

Эндпоинты:
- GET  /health — проверка состояния сервиса
- POST /predict — предсказание для одного текста
- POST /predict/batch — пакетное предсказание
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from IMDb_review.src.predict import predictor
from IMDb_review.src.utils import setup_logging, load_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic-схемы
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    """Запрос на предсказание для одного текста."""

    text: str = Field(..., min_length=1, description="Текст отзыва")

    class Config:
        json_schema_extra = {
            "example": {"text": "This movie was absolutely wonderful, great acting!"}
        }


class PredictResponse(BaseModel):
    """Ответ с предсказанием."""

    sentiment: str = Field(..., description="Предсказанная тональность: positive/negative")
    confidence: float | None = Field(None, description="Уверенность модели (decision_function)")


class BatchPredictRequest(BaseModel):
    """Запрос на пакетное предсказание."""

    texts: list[str] = Field(
        ..., min_length=1, max_length=1000,
        description="Список текстов отзывов (до 1000)"
    )


class BatchPredictResponse(BaseModel):
    """Ответ на пакетное предсказание."""

    results: list[PredictResponse] = Field(..., description="Список предсказаний")


class HealthResponse(BaseModel):
    """Ответ на health-check."""

    status: str = Field("ok", description="Статус сервиса")
    model_loaded: bool = Field(..., description="Загружена ли модель")


# ---------------------------------------------------------------------------
# Приложение FastAPI
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Загружает модель при старте приложения."""
    logger.info("Загрузка модели...")
    try:
        predictor.load()
        logger.info("Модель успешно загружена")
    except FileNotFoundError as e:
        logger.warning("Модель не найдена: %s. Сервис работает без модели.", e)
    yield


config = load_config()
api_config = config.get("api", {})
setup_logging(config.get("logging", {}))

app = FastAPI(
    title=api_config.get("title", "Sentiment Classification API"),
    description="REST API для инференса модели классификации тональности отзывов",
    version=api_config.get("version", "1.0.0"),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Эндпоинты
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health_check():
    """Проверка состояния сервиса."""
    return HealthResponse(
        status="ok" if predictor.is_loaded else "degraded",
        model_loaded=predictor.is_loaded,
    )


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
def predict(request: PredictRequest):
    """Предсказать тональность для одного текста отзыва."""
    if not predictor.is_loaded:
        raise HTTPException(status_code=503, detail="Модель не загружена")

    try:
        result = predictor.predict(request.text)
        logger.info("Prediction: %.50s... -> %s", request.text, result["sentiment"])
        return PredictResponse(**result)
    except Exception as e:
        logger.error("Ошибка предсказания: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["Inference"])
def predict_batch(request: BatchPredictRequest):
    """Пакетное предсказание для нескольких текстов (до 1000)."""
    if not predictor.is_loaded:
        raise HTTPException(status_code=503, detail="Модель не загружена")

    try:
        results = predictor.predict_batch(request.texts)
        logger.info("Batch prediction: %d texts", len(request.texts))
        return BatchPredictResponse(results=[PredictResponse(**r) for r in results])
    except Exception as e:
        logger.error("Ошибка пакетного предсказания: %s", e)
        raise HTTPException(status_code=500, detail=str(e))