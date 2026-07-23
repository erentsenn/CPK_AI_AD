"""REST API для инференса модели классификации тональности отзывов.

Эндпоинты:
    GET  /health         — проверка работоспособности сервиса и статуса модели
    POST /predict        — предсказание тональности одного текста
    POST /predict/batch  — предсказание тональности для набора текстов
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status

from .model_loader import sentiment_model
from .schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    PredictRequest,
    PredictResponse,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Загружаем модель один раз при старте сервиса.
    sentiment_model.load()
    yield


app = FastAPI(
    title="Sentiment Classification API",
    description=(
        "REST-обёртка над ML-моделью бинарной классификации тональности "
        "текстовых отзывов (positive / negative)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, tags=["service"])
def health() -> HealthResponse:
    """Возвращает статус сервиса и информацию о загруженной модели."""
    if sentiment_model.is_loaded:
        return HealthResponse(
            status="ok",
            model_loaded=True,
            model_name=sentiment_model.model_name,
            model_path=sentiment_model.model_path,
        )
    return HealthResponse(
        status="error",
        model_loaded=False,
        detail=sentiment_model.load_error,
    )


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
def predict(request: PredictRequest) -> PredictResponse:
    """Предсказывает тональность одного текстового отзыва."""
    if not sentiment_model.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=sentiment_model.load_error or "Модель не загружена",
        )
    result = sentiment_model.predict([request.text])[0]
    return PredictResponse(prediction=result)


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["inference"])
def predict_batch(request: BatchPredictRequest) -> BatchPredictResponse:
    """Предсказывает тональность для набора текстовых отзывов."""
    if not sentiment_model.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=sentiment_model.load_error or "Модель не загружена",
        )
    results = sentiment_model.predict(request.texts)
    return BatchPredictResponse(predictions=results, count=len(results))


@app.get("/", tags=["service"])
def root() -> dict:
    """Краткая информация о сервисе и доступных эндпоинтах."""
    return {
        "service": "Sentiment Classification API",
        "endpoints": ["/health", "/predict", "/predict/batch", "/docs"],
    }
