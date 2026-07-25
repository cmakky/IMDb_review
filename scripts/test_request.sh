#!/usr/bin/env bash
# ============================================================================
# test_request.sh — скрипт для тестирования API
# ============================================================================
# Использование:
#   bash scripts/test_request.sh              # health + predict
#   bash scripts/test_request.sh --health     # только health
#   bash scripts/test_request.sh --predict    # только predict
#   bash scripts/test_request.sh --batch      # пакетное предсказание
# ============================================================================

set -euo pipefail

BASE_URL="${API_URL:-http://localhost:8000}"

# Тестовые данные — примеры отзывов с явной тональностью
POSITIVE_TEXT="This movie was absolutely wonderful, great acting and a fantastic story!"
NEGATIVE_TEXT="Terrible movie, waste of time, awful acting and a boring plot."

check_health() {
    echo "🔍 Health check: ${BASE_URL}/health"
    curl -s "${BASE_URL}/health" | python3 -m json.tool
    echo
}

predict() {
    echo "📊 Предсказание (positive пример): ${BASE_URL}/predict"
    curl -s -X POST "${BASE_URL}/predict" \
        -H "Content-Type: application/json" \
        -d "{\"text\": \"${POSITIVE_TEXT}\"}" | python3 -m json.tool
    echo

    echo "📊 Предсказание (negative пример): ${BASE_URL}/predict"
    curl -s -X POST "${BASE_URL}/predict" \
        -H "Content-Type: application/json" \
        -d "{\"text\": \"${NEGATIVE_TEXT}\"}" | python3 -m json.tool
    echo
}

predict_batch() {
    echo "📦 Пакетное предсказание: ${BASE_URL}/predict/batch"
    curl -s -X POST "${BASE_URL}/predict/batch" \
        -H "Content-Type: application/json" \
        -d "{\"texts\": [\"${POSITIVE_TEXT}\", \"${NEGATIVE_TEXT}\"]}" | python3 -m json.tool
    echo
}

# --- main ---

case "${1:-}" in
    --health)
        check_health
        ;;
    --predict)
        predict
        ;;
    --batch)
        predict_batch
        ;;
    *)
        echo "=========================================="
        echo "  Тестирование API классификации тональности"
        echo "  Базовый URL: ${BASE_URL}"
        echo "=========================================="
        echo
        check_health
        predict
        predict_batch
        ;;
esac