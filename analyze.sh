#!/bin/bash
# Скрипт для быстрого анализа изображений/видео с обученной моделью

MODEL="runs/detect/ball_detector_quick/weights/best.pt"

if [ ! -f "$MODEL" ]; then
    echo "❌ Модель не найдена: $MODEL"
    echo "Запустите обучение: ./start_training.sh"
    exit 1
fi

if [ $# -eq 0 ]; then
    echo "Использование:"
    echo "  ./analyze.sh image.jpg              # анализ изображения"
    echo "  ./analyze.sh video.mp4 --video      # анализ видео"
    echo ""
    echo "Дополнительные параметры:"
    echo "  --confidence 0.6                    # порог уверенности"
    echo "  --net-method hough|contour|color    # метод детекции сетки"
    echo "  --output result.jpg                 # путь для сохранения"
    echo ""
    echo "Примеры:"
    echo "  ./analyze.sh test_image.jpg"
    echo "  ./analyze.sh match.mp4 --video --confidence 0.7"
    echo "  ./analyze.sh photo.jpg --net-method color --output result.jpg"
    exit 0
fi

INPUT="$1"
shift

if [ ! -f "$INPUT" ]; then
    echo "❌ Файл не найден: $INPUT"
    exit 1
fi

echo "Анализ: $INPUT"
echo "Модель: $MODEL"
echo ""

.venv/bin/python cli.py "$INPUT" --model "$MODEL" "$@"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✅ Анализ завершен успешно!"
else
    echo ""
    echo "❌ Ошибка при анализе"
    exit $EXIT_CODE
fi
