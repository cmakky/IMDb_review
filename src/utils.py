"""
Вспомогательные функции для проекта классификации тональности отзывов.
"""

import logging
import re
from pathlib import Path

import pandas as pd
import yaml


def load_config(config_path: str = "config.yaml") -> dict:
    """Загружает конфигурацию из YAML-файла."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Файл конфигурации не найден: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logging(logging_config: dict) -> None:
    """Настраивает базовое логирование."""
    level = logging_config.get("level", "INFO")
    fmt = logging_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logging.basicConfig(level=level, format=fmt)


def clean_text(text: str) -> str:
    """
    Предобработка текста отзыва, должна точно совпадать с тем,
    что использовалось при обучении модели (см. ноутбук).
    """
    text = re.sub(r'<br\s*/?>', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text


def load_dataset(config: dict) -> pd.DataFrame:
    """Загружает датасет отзывов из CSV, указанного в конфигурации."""
    data_path = Path(config["data"]["path"])
    if not data_path.exists():
        raise FileNotFoundError(f"Датасет не найден: {data_path}")
    return pd.read_csv(data_path)