"""
Загрузка модели и выполнение предсказания.

Модель: LinearSVC + TfidfVectorizer (Sentiment Classification)
Файлы: model.pkl, vectorizer.pkl
"""

import joblib
import logging
from pathlib import Path

from IMDb_review.src.utils import load_config, clean_text

logger = logging.getLogger(__name__)


class Predictor:
    """
    Загрузчик и обёртка для модели классификации тональности.

    Пример использования:
        predictor = Predictor()
        predictor.load()
        result = predictor.predict("This movie was great!")
    """

    def __init__(self, config_path: str = "config.yaml"):
        config = load_config(config_path)
        self.model_path = Path(config["paths"]["model"])
        self.vectorizer_path = Path(config["paths"]["vectorizer"])
        self._model = None
        self._vectorizer = None

    def load(self) -> None:
        """Загружает модель и векторизатор из pickle-файлов."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Модель не найдена: {self.model_path}")
        if not self.vectorizer_path.exists():
            raise FileNotFoundError(f"Векторизатор не найден: {self.vectorizer_path}")

        self._model = joblib.load(self.model_path)
        self._vectorizer = joblib.load(self.vectorizer_path)

        logger.info("Модель и векторизатор загружены: %s, %s",
                     self.model_path.name, self.vectorizer_path.name)

    @property
    def is_loaded(self) -> bool:
        """Проверяет, загружена ли модель."""
        return self._model is not None and self._vectorizer is not None

    def _predict_with_confidence(self, vec):
        """Общая логика получения label + confidence из decision_function."""
        labels = self._model.predict(vec)
        confidences = None
        if hasattr(self._model, "decision_function"):
            confidences = self._model.decision_function(vec)
        return labels, confidences

    def predict(self, text: str) -> dict:
        """
        Выполняет предсказание тональности для одного текста.

        Параметры
        ----------
        text : str
            Текст отзыва.

        Возвращает
        ----------
        dict
            {"sentiment": "positive"/"negative", "confidence": float}
        """
        if not self.is_loaded:
            raise RuntimeError("Модель не загружена. Вызовите load() перед predict().")

        cleaned = clean_text(text)
        vec = self._vectorizer.transform([cleaned])
        labels, confidences = self._predict_with_confidence(vec)

        result = {"sentiment": str(labels[0])}
        if confidences is not None:
            result["confidence"] = float(confidences[0])
        return result

    def predict_batch(self, texts: list[str]) -> list[dict]:
        """
        Выполняет предсказание для нескольких текстов.

        Параметры
        ----------
        texts : list[str]
            Список текстов отзывов.

        Возвращает
        ----------
        list[dict]
            Список результатов вида {"sentiment": ..., "confidence": ...}
        """
        if not self.is_loaded:
            raise RuntimeError("Модель не загружена. Вызовите load() перед predict().")

        cleaned = [clean_text(t) for t in texts]
        vecs = self._vectorizer.transform(cleaned)
        labels, confidences = self._predict_with_confidence(vecs)

        results = [{"sentiment": str(label)} for label in labels]
        if confidences is not None:
            for r, c in zip(results, confidences):
                r["confidence"] = float(c)
        return results


# Глобальный экземпляр для использования в API
predictor = Predictor()