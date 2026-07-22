"""Pydantic-схемы запросов и ответов REST API."""

from typing import List, Optional

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Запрос на предсказание для одного отзыва."""

    text: str = Field(..., min_length=1, description="Текст отзыва для классификации")

    model_config = {
        "json_schema_extra": {
            "example": {"text": "This movie was absolutely fantastic and touching!"}
        }
    }


class BatchPredictRequest(BaseModel):
    """Запрос на предсказание для набора отзывов."""

    texts: List[str] = Field(
        ..., min_length=1, description="Список текстов отзывов для классификации"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "texts": [
                    "One of the best films I have ever seen.",
                    "Terrible plot and boring acting, a waste of time.",
                ]
            }
        }
    }


class PredictionItem(BaseModel):
    """Результат предсказания для одного текста."""

    text: str = Field(..., description="Исходный текст отзыва")
    sentiment: str = Field(..., description="Предсказанная тональность: positive / negative")
    confidence: Optional[float] = Field(
        None, description="Уверенность модели в предсказании (если доступна)"
    )


class PredictResponse(BaseModel):
    """Ответ эндпоинта /predict."""

    prediction: PredictionItem


class BatchPredictResponse(BaseModel):
    """Ответ эндпоинта /predict/batch."""

    predictions: List[PredictionItem]
    count: int = Field(..., description="Количество обработанных текстов")


class HealthResponse(BaseModel):
    """Ответ эндпоинта /health."""

    status: str = Field(..., description="Статус сервиса: ok / error")
    model_loaded: bool = Field(..., description="Загружена ли модель в память")
    model_name: Optional[str] = Field(None, description="Название загруженной модели")
    model_path: Optional[str] = Field(None, description="Путь к загруженному файлу модели")
    detail: Optional[str] = Field(None, description="Дополнительная информация об ошибке")
