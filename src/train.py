"""
Обучение модели классификации тональности отзывов.

Используется датасет отзывов о фильмах (positive/negative).
Пайплайн: TF-IDF векторизация + LinearSVC (scikit-learn).

Результат:
  - models/model.pkl        — сериализованная модель
  - models/vectorizer.pkl    — сериализованный TF-IDF векторизатор
"""

import argparse
import logging
import re
import joblib
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, precision_score, recall_score

from IMDb_review.src.utils import setup_logging, load_config, load_dataset

logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """Предобработка текста — должна совпадать с той, что используется в API."""
    text = re.sub(r'<br\s*/?>', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text


def train(config: dict) -> None:
    """
    Полный цикл обучения модели.

    Параметры
    ----------
    config : dict
        Конфигурация из config.yaml.
    """
    np.random.seed(42)

    data_cfg = config["data"]
    vec_cfg = config["vectorizer"]
    model_params = config["model"]["params"]
    model_path = Path(config["paths"]["model"])
    vectorizer_path = Path(config["paths"]["vectorizer"])

    # 1. Загрузка данных
    logger.info("Загрузка данных...")
    df = load_dataset(config)
    logger.info("Загружено %d записей, %d колонок", df.shape[0], df.shape[1])

    text_col = data_cfg["text_column"]
    target_col = data_cfg["target_column"]

    # 2. Предобработка
    logger.info("Предобработка текста...")
    df["text_clean"] = df[text_col].apply(clean_text)

    X = df["text_clean"]
    y = df[target_col]

    # 3. Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=data_cfg["test_size"],
        random_state=data_cfg["random_state"],
        stratify=y,
    )
    logger.info("Train: %d, Test: %d", len(X_train), len(X_test))

    # 4. Векторизация (TF-IDF)
    logger.info("Векторизация (TF-IDF)...")
    vectorizer = TfidfVectorizer(
        max_features=vec_cfg["max_features"],
        ngram_range=tuple(vec_cfg["ngram_range"]),
        min_df=vec_cfg["min_df"],
        stop_words=vec_cfg["stop_words"],
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    logger.info("Размерность признаков: %s", X_train_vec.shape)

    # 5. Обучение (с подбором C через GridSearchCV)
    logger.info("Подбор гиперпараметров (GridSearchCV)...")
    param_grid = {"C": [0.01, 0.1, 0.5, 1.0, 2.0, 3.0]}
    grid_search = GridSearchCV(
        LinearSVC(random_state=model_params["random_state"], max_iter=model_params["max_iter"]),
        param_grid,
        scoring="f1_macro",
        cv=3,
        n_jobs=-1,
    )
    grid_search.fit(X_train_vec, y_train)
    model = grid_search.best_estimator_
    logger.info("Лучший C: %s", grid_search.best_params_["C"])
    logger.info("Модель обучена")

    # 6. Оценка
    y_pred = model.predict(X_test_vec)
    f1 = f1_score(y_test, y_pred, pos_label="positive")
    precision = precision_score(y_test, y_pred, pos_label="positive")
    recall = recall_score(y_test, y_pred, pos_label="positive")

    logger.info("=" * 50)
    logger.info("МЕТРИКИ")
    logger.info("=" * 50)
    logger.info("F1-score:  %.4f  (порог > 0.75)", f1)
    logger.info("Precision: %.4f  (порог > 0.70)", precision)
    logger.info("Recall:    %.4f  (порог > 0.70)", recall)
    logger.info("=" * 50)

    # 7. Сохранение модели и векторизатора
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)

    logger.info("Модель сохранена: %s", model_path)
    logger.info("Векторизатор сохранён: %s", vectorizer_path)


def main() -> None:
    """Точка входа для CLI."""
    parser = argparse.ArgumentParser(description="Обучение модели классификации тональности")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Путь к файлу конфигурации",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config.get("logging", {}))
    train(config)


if __name__ == "__main__":
    main()