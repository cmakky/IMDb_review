# Классификация тональности отзывов о фильмах 

**Проект для практики "Архитектор ИИ"**

Полный цикл ML-проекта: от исследовательского анализа данных до развёртывания модели в виде REST API.

## Датасет

- Источник: [data/texts/review.csv](data/texts/review.csv)
- Целевая переменная: `sentiment` — тональность отзыва (positive / negative)
- Признак: review — текст отзыва о фильме
- Размер: 60000 отзывов, классы сбалансорванны (30000/30000)

## Структура репозитория

```text
├── README.md                 # Описание проекта
├── requirements.txt          # Зависимости Python
├── Dockerfile                # Docker-образ для API
├── .gitignore                # Игнорируемые файлы
├── config.yaml               # Конфигурация (пути, гиперпараметры)
│
├── src/                      # Исходный код
│   ├── __init__.py
│   ├── train.py              # Обучение модели
│   ├── predict.py            # Инференс (загрузка + предсказание)
│   ├── api.py                # FastAPI-сервис
│   └── utils.py              # Вспомогательные функции
│
├── notebooks/                # Jupyter ноутбуки
│   └── Project_LastName.ipynb  # EDA и пайплайн обучения и оценки модели
│
├── models/                    # Сериализованные модели
│   └── model.pkl             # Обученная модель (LinearSVC)
│   └── vectorizer.pkl        # Обученный TF-IDF векторизатор
│
├── tests/                    # Тесты
│   ├── __init__.py
│   └── test_api.py           # Тесты для API
│
└── scripts/                  # Скрипты для развёртывания
    ├── deploy.sh             # Сборка и запуск Docker
    └── test_request.sh       # Тестовые запросы к API
```

## Быстрый старт

### 1. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 2. Запуск ноутбука с EDA и обучением

```bash
jupyter notebook notebooks/Project_Tuaev.ipynb
```

После выполнения ноутбука модель и векторизатор сохранятся в `models/model.pkl` и `models/vectorizer.pkl`.

### 3. Запуск API

```bash
# Через uvicorn
uvicorn src.api:app --reload --port 8000

# Или через Docker
docker build -t sentiment-api .
docker run -p 8000:8000 sentiment-api
```

### 4. Тестирование API

```bash
# Health check
curl http://localhost:8000/health

# Предсказание
bash scripts/test_request.sh
```

## Эндпоинты API

| Метод | Путь       | Описание                                            |
| ---------- | -------------- | ----------------------------------------------------------- |
| GET        | /health        | Проверка состояния сервиса          |
| POST       | /predict       | Предсказание для одного объекта |
| POST       | /predict/batch | Пакетное предсказание (до 1000)       |

Документация Swagger: http://localhost:8000/docs

## Модель

- **Пайплайн:** TF-IDF векторизация (unigram + bigram, до 20 000 признаков) + LinearSVC (scikit-learn)
- **Подбор гиперпараметров:** GridSearchCV по параметру C
- **Метрики на тесте:**
  - F1-score:  ~0.95
  - Precision: ~0.95
  - Recall:    ~0.95
- **Baseline для сравнения:** LogisticRegression (F1 ~0.92)
