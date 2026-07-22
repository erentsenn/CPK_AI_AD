"""Загрузка сериализованной модели классификации тональности.

Модель сохраняется в ноутбуке в виде словаря:
    {"model_name": str, "vectorizer": TfidfVectorizer, "model": <sklearn estimator>}

Путь к файлу берётся из переменной окружения MODEL_PATH, иначе
перебираются типовые расположения (в т.ч. требуемое заданием /model/model.pkl).
"""

import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .preprocessing import clean

# Кандидаты путей: сначала env, затем варианты из ноутбука и требования задания.
_CANDIDATE_PATHS = [
    os.getenv("MODEL_PATH"),
    "models/baseline_model.pkl",
    "model/model.pkl",
    "model/baseline_model.pkl",
    "models/model.pkl",
]


def _resolve_model_path() -> Optional[Path]:
    for candidate in _CANDIDATE_PATHS:
        if not candidate:
            continue
        path = Path(candidate)
        if path.is_file():
            return path
    return None


class SentimentModel:
    """Обёртка над векторизатором и классификатором с единым интерфейсом инференса."""

    def __init__(self) -> None:
        self.model_name: Optional[str] = None
        self.vectorizer = None
        self.model = None
        self.model_path: Optional[str] = None
        self.is_loaded: bool = False
        self.load_error: Optional[str] = None

    def load(self) -> None:
        path = _resolve_model_path()
        if path is None:
            self.load_error = (
                "Файл модели не найден. Задайте переменную окружения MODEL_PATH "
                "или положите pkl в models/baseline_model.pkl (или model/model.pkl)."
            )
            self.is_loaded = False
            return

        try:
            with open(path, "rb") as f:
                data: Dict[str, Any] = pickle.load(f)
            self.vectorizer = data["vectorizer"]
            self.model = data["model"]
            self.model_name = data.get("model_name", type(self.model).__name__)
            self.model_path = str(path)
            self.is_loaded = True
            self.load_error = None
        except Exception as exc:  # noqa: BLE001 - хотим сообщить любую ошибку загрузки
            self.is_loaded = False
            self.load_error = f"Ошибка загрузки модели: {exc}"

    def _confidence(self, features) -> Optional[np.ndarray]:
        """Возвращает уверенность модели в предсказании, если она доступна."""
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(features)
            return proba.max(axis=1)
        if hasattr(self.model, "decision_function"):
            # LinearSVC: превращаем расстояние до гиперплоскости в псевдо-вероятность.
            scores = self.model.decision_function(features)
            scores = np.atleast_1d(scores)
            return 1.0 / (1.0 + np.exp(-np.abs(scores)))
        return None

    def predict(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Предсказывает тональность для списка текстов."""
        if not self.is_loaded:
            raise RuntimeError(self.load_error or "Модель не загружена")

        cleaned = [clean(t) for t in texts]
        features = self.vectorizer.transform(cleaned)
        labels = self.model.predict(features)
        confidences = self._confidence(features)

        results: List[Dict[str, Any]] = []
        for i, (text, label) in enumerate(zip(texts, labels)):
            item: Dict[str, Any] = {"text": text, "sentiment": str(label)}
            if confidences is not None:
                item["confidence"] = round(float(confidences[i]), 4)
            results.append(item)
        return results


# Единственный экземпляр модели на процесс.
sentiment_model = SentimentModel()
