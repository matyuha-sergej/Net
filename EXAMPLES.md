# Примеры использования

## Пример 1: Быстрый тест

### Создание тестового изображения и проверка

```bash
# Запустите тестовый скрипт
python test_app.py

# Это создаст:
# - test_image.jpg (синтетическое изображение с мячом и сеткой)
# - test_result.jpg (результат анализа)
```

### Анализ тестового изображения

```bash
# Простой анализ
python cli.py test_image.jpg

# С настройками
python cli.py test_image.jpg --confidence 0.3 --net-method hough
```

## Пример 2: Работа с изображением волейбольного матча

```bash
# Скачайте или используйте свое изображение
# Анализ с предобученной моделью
python cli.py volleyball_game.jpg \
    --confidence 0.5 \
    --net-method hough \
    --output volleyball_result.jpg

# Результат будет сохранен в volleyball_result.jpg
```

## Пример 3: Обучение модели на своих данных

### Подготовка датасета

```
1. Подготовьте изображения с разметкой в формате YOLO
2. Организуйте структуру:

my_ball_dataset/
├── images/
│   ├── train/
│   │   ├── frame_001.jpg
│   │   ├── frame_002.jpg
│   │   └── ...
│   └── val/
│       ├── frame_100.jpg
│       └── ...
└── labels/
    ├── train/
    │   ├── frame_001.txt
    │   ├── frame_002.txt
    │   └── ...
    └── val/
        ├── frame_100.txt
        └── ...
```

### Создание конфигурации

```bash
# Создайте пример data.yaml
python train_yolo.py --create-example

# Отредактируйте data.yaml:
```

```yaml
path: /home/sergii/PycharmProjects/Net/datasets/my_ball_dataset
train: images/train
val: images/val

nc: 1
names: ['ball']
```

### Обучение

```bash
# Базовое обучение (легкая модель, быстро)
python train_yolo.py \
    --data data.yaml \
    --model yolov8n.pt \
    --epochs 50 \
    --batch 16 \
    --name my_ball_detector

# Улучшенное обучение (средняя модель, лучше качество)
python train_yolo.py \
    --data data.yaml \
    --model yolov8m.pt \
    --epochs 100 \
    --batch 8 \
    --name my_ball_detector_v2

# Обучение на GPU
python train_yolo.py \
    --data data.yaml \
    --model yolov8s.pt \
    --epochs 100 \
    --batch 32 \
    --device 0

# Обучение на CPU (медленнее)
python train_yolo.py \
    --data data.yaml \
    --model yolov8n.pt \
    --epochs 50 \
    --batch 8 \
    --device cpu
```

### Использование обученной модели

```bash
# Путь к лучшей модели после обучения
MODEL_PATH="runs/detect/my_ball_detector/weights/best.pt"

# CLI
python cli.py image.jpg --model $MODEL_PATH

# GUI
python main.py
# Выберите "Своя модель..." и укажите путь к best.pt
```

## Пример 4: Обработка видео матча

```bash
# Базовая обработка
python cli.py volleyball_match.mp4 --video

# С настройками
python cli.py volleyball_match.mp4 \
    --video \
    --model runs/detect/my_ball_detector/weights/best.pt \
    --confidence 0.6 \
    --net-method hough \
    --output analyzed_match.mp4

# Результат:
# - Каждый кадр будет обработан
# - Мяч будет отмечен (зеленый = перед сеткой, красный = за сеткой)
# - Сетка будет отмечена голубой линией
# - Видео сохранится в analyzed_match.mp4
```

## Пример 5: Пакетная обработка изображений

```bash
# Создайте скрипт для обработки всех изображений в папке
cat > batch_process.sh << 'EOF'
#!/bin/bash

INPUT_DIR="$1"
OUTPUT_DIR="$2"
MODEL="${3:-yolov8n.pt}"

mkdir -p "$OUTPUT_DIR"

for img in "$INPUT_DIR"/*.{jpg,jpeg,png}; do
    if [ -f "$img" ]; then
        filename=$(basename "$img")
        echo "Processing: $filename"
        python cli.py "$img" \
            --model "$MODEL" \
            --confidence 0.5 \
            --net-method hough \
            --output "$OUTPUT_DIR/$filename"
    fi
done

echo "Done! Processed images saved to $OUTPUT_DIR"
EOF

chmod +x batch_process.sh

# Использование
./batch_process.sh input_images/ output_results/ my_model.pt
```

## Пример 6: Интерактивная работа в GUI

```bash
# Запустите GUI
python main.py
```

**Пошаговая инструкция:**

1. **Загрузка изображения**
   - Нажмите "Загрузить изображение"
   - Выберите файл с мячом и сеткой

2. **Настройка параметров**
   - Модель YOLO: выберите yolov8n.pt или свою модель
   - Метод детекции сетки: попробуйте hough, затем другие
   - Порог уверенности: настройте слайдером (0.5 - хороший старт)

3. **Анализ**
   - Нажмите "Анализировать"
   - Результат отобразится справа
   - Внизу появится текст с результатом

4. **Работа с видео**
   - Нажмите "Загрузить видео"
   - Выберите видеофайл
   - Обработка займет время
   - Результат сохранится рядом с исходным файлом

## Пример 7: Сравнение методов детекции сетки

```bash
# Создайте скрипт для сравнения
cat > compare_methods.py << 'EOF'
import cv2
from net_detector import NetDetector

img = cv2.imread('test_image.jpg')
detector = NetDetector()

methods = ['hough', 'contour', 'color']
for method in methods:
    net_line = detector.detect_net_line(img, method)
    if net_line:
        result = img.copy()
        x1, y1, x2, y2 = net_line
        cv2.line(result, (x1, y1), (x2, y2), (0, 255, 255), 3)
        cv2.imwrite(f'net_detection_{method}.jpg', result)
        print(f"{method}: {net_line}")
    else:
        print(f"{method}: не обнаружена")
EOF

python compare_methods.py
```

## Пример 8: Настройка параметров детекции

### Изменение чувствительности детекции сетки

Отредактируйте `net_detector.py`:

```python
# Для метода Hough - измените порог
lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50,  # было 100
                        minLineLength=50, maxLineGap=20)  # было 100, 10

# Для метода Color - измените диапазон цветов
lower_color = (180, 180, 180)  # было (200, 200, 200)
upper_color = (255, 255, 255)
```

## Пример 9: Статистика по видео

```bash
# Модифицируйте cli.py для вывода подробной статистики
python cli.py match.mp4 --video --model my_model.pt

# Вывод будет включать:
# - Количество кадров с мячом перед сеткой
# - Количество кадров с мячом за сеткой
# - Процент уверенности для каждого положения
```

## Пример 10: Интеграция в свой проект

```python
from ball_detector import BallDetector
from net_detector import NetDetector
from position_analyzer import PositionAnalyzer
import cv2

# Инициализация
ball_detector = BallDetector('my_model.pt')
net_detector = NetDetector()
analyzer = PositionAnalyzer()

# Загрузка изображения
image = cv2.imread('frame.jpg')

# Детекция
balls = ball_detector.detect(image, confidence=0.5)
if balls:
    ball_bbox = max(balls, key=lambda x: x[4])
    ball_center = ball_detector.get_ball_center(ball_bbox)
    
    net_line = net_detector.detect_net_line(image, 'hough')
    if net_line:
        net_x = net_detector.get_net_x_position(net_line)
        position, confidence = analyzer.analyze_with_depth(
            ball_center, ball_bbox, net_x, image
        )
        
        print(f"Мяч {position.value} с уверенностью {confidence:.2%}")
        
        # Визуализация
        result = analyzer.visualize_result(
            image, ball_bbox, net_line, position, confidence
        )
        cv2.imwrite('result.jpg', result)
```

## Полезные советы

### 1. Оптимизация скорости

```bash
# Используйте легкую модель для быстрой обработки
python cli.py video.mp4 --video --model yolov8n.pt --batch 32

# Для лучшего качества (медленнее)
python cli.py video.mp4 --video --model yolov8l.pt --batch 8
```

### 2. Улучшение точности

```bash
# Увеличьте порог уверенности для уменьшения ложных срабатываний
python cli.py image.jpg --confidence 0.7

# Обучите модель на большем датасете
python train_yolo.py --data data.yaml --epochs 200
```

### 3. Работа с разными видами спорта

```python
# Волейбол - сетка обычно в центре, высокая
analyzer = PositionAnalyzer(camera_position='side')

# Теннис - сетка ниже, может быть под углом
# Настройте параметры в net_detector.py
```

### 4. Отладка

```bash
# Запустите тесты для проверки модулей
python test_app.py

# Проверьте только детекцию сетки
python -c "
import cv2
from net_detector import NetDetector
img = cv2.imread('your_image.jpg')
detector = NetDetector()
for method in ['hough', 'contour', 'color']:
    result = detector.detect_net_line(img, method)
    print(f'{method}: {result}')
"
```

## Заключение

Эти примеры покрывают основные сценарии использования. Экспериментируйте с параметрами и методами для достижения лучших результатов на ваших данных!
