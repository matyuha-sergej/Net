#!/bin/bash
# Скрипт для быстрого запуска обучения модели

echo "=========================================="
echo "Обучение модели детекции мяча"
echo "=========================================="
echo ""
echo "Датасет: 1467 изображений (1026 train, 293 val, 148 test)"
echo ""
echo "Выберите вариант обучения:"
echo "  1) Быстрое обучение (50 эпох, ~15-30 мин)"
echo "  2) Полное обучение (100 эпох, ~30-60 мин)"
echo "  3) Улучшенное качество (100 эпох, большая модель, ~60-90 мин)"
echo "  4) Свои параметры"
echo ""
read -p "Ваш выбор (1-4): " choice

case $choice in
    1)
        echo ""
        echo "Запуск быстрого обучения..."
        .venv/bin/python train_yolo.py \
            --data datasets/merged_ball_dataset/data.yaml \
            --model yolov8n.pt \
            --epochs 50 \
            --batch 16 \
            --name ball_detector_quick
        ;;
    2)
        echo ""
        echo "Запуск полного обучения..."
        .venv/bin/python train_yolo.py \
            --data datasets/merged_ball_dataset/data.yaml \
            --model yolov8n.pt \
            --epochs 100 \
            --batch 16 \
            --name ball_detector_full
        ;;
    3)
        echo ""
        echo "Запуск улучшенного обучения..."
        .venv/bin/python train_yolo.py \
            --data datasets/merged_ball_dataset/data.yaml \
            --model yolov8s.pt \
            --epochs 100 \
            --batch 8 \
            --name ball_detector_enhanced
        ;;
    4)
        echo ""
        read -p "Модель (yolov8n.pt/yolov8s.pt/yolov8m.pt): " model
        read -p "Количество эпох: " epochs
        read -p "Размер батча: " batch
        read -p "Имя эксперимента: " name
        
        echo ""
        echo "Запуск обучения с параметрами:"
        echo "  Модель: $model"
        echo "  Эпохи: $epochs"
        echo "  Батч: $batch"
        echo "  Имя: $name"
        echo ""
        
        .venv/bin/python train_yolo.py \
            --data datasets/merged_ball_dataset/data.yaml \
            --model "$model" \
            --epochs "$epochs" \
            --batch "$batch" \
            --name "$name"
        ;;
    *)
        echo "Неверный выбор!"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "Обучение завершено!"
echo "=========================================="
echo ""
echo "Модель сохранена в: runs/detect/[имя]/weights/best.pt"
echo ""
echo "Для тестирования модели:"
echo "  .venv/bin/python cli.py test_image.jpg --model runs/detect/[имя]/weights/best.pt"
echo ""
echo "Для запуска GUI:"
echo "  .venv/bin/python main.py"
