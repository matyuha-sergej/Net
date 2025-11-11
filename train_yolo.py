"""
Скрипт для обучения YOLO модели на собственных датасетах
"""
import argparse
from ultralytics import YOLO
from pathlib import Path


def train_model(data_yaml: str, 
                base_model: str = 'yolov8n.pt',
                epochs: int = 100,
                imgsz: int = 640,
                batch: int = 16,
                name: str = 'ball_detector',
                device: str = '0'):
    """
    Обучить YOLO модель на собственных данных
    
    Args:
        data_yaml: путь к файлу конфигурации датасета
        base_model: базовая модель для transfer learning
        epochs: количество эпох обучения
        imgsz: размер изображений
        batch: размер батча
        name: имя эксперимента
        device: устройство для обучения ('0' для GPU, 'cpu' для CPU)
    """
    print(f"Начало обучения модели на датасете: {data_yaml}")
    print(f"Базовая модель: {base_model}")
    print(f"Параметры: epochs={epochs}, imgsz={imgsz}, batch={batch}")
    print(f"Устройство: {device}")
    
    # Загрузка базовой модели
    model = YOLO(base_model)
    
    # Обучение
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        name=name,
        device=device,
        patience=50,  # Early stopping
        save=True,
        save_period=10,  # Сохранять каждые 10 эпох
        cache=False,
        augment=True,
        verbose=True,
        workers=8,
        project='runs/detect',
    )
    
    print("\n" + "="*60)
    print("Обучение завершено!")
    print(f"Лучшая модель сохранена в: runs/detect/{name}/weights/best.pt")
    print(f"Последняя модель: runs/detect/{name}/weights/last.pt")
    print("="*60)
    
    # Валидация
    print("\nЗапуск валидации...")
    metrics = model.val()
    
    print(f"\nМетрики:")
    print(f"  mAP50: {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")
    print(f"  Precision: {metrics.box.mp:.4f}")
    print(f"  Recall: {metrics.box.mr:.4f}")
    
    return f"runs/detect/{name}/weights/best.pt"


def create_example_data_yaml(output_path: str = 'data.yaml'):
    """
    Создать пример файла конфигурации датасета
    
    Args:
        output_path: путь для сохранения файла
    """
    example_yaml = """# Конфигурация датасета для обучения YOLO
# Путь к корневой директории датасета
path: /path/to/your/dataset  # ИЗМЕНИТЕ НА ВАШ ПУТЬ

# Пути к поддиректориям (относительно path)
train: images/train
val: images/val
test: images/test  # опционально

# Количество классов
nc: 1

# Названия классов
names: ['ball']

# Примечания:
# - Структура датасета должна быть:
#   dataset/
#   ├── images/
#   │   ├── train/
#   │   ├── val/
#   │   └── test/
#   └── labels/
#       ├── train/
#       ├── val/
#       └── test/
#
# - Файлы разметки должны быть в формате YOLO (.txt)
# - Каждая строка в .txt файле: class x_center y_center width height
#   (все значения нормализованы от 0 до 1)
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(example_yaml)
    
    print(f"Пример конфигурации сохранен в: {output_path}")
    print("Отредактируйте этот файл, указав правильные пути к вашему датасету")


def main():
    """Главная функция"""
    parser = argparse.ArgumentParser(
        description='Обучение YOLO модели для детекции мяча'
    )
    
    parser.add_argument('--data', type=str, 
                       help='Путь к файлу конфигурации датасета (data.yaml)')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                       help='Базовая модель (yolov8n/s/m/l/x.pt)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Количество эпох обучения')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Размер изображений')
    parser.add_argument('--batch', type=int, default=16,
                       help='Размер батча')
    parser.add_argument('--name', type=str, default='ball_detector',
                       help='Имя эксперимента')
    parser.add_argument('--device', type=str, default='0',
                       help='Устройство (0 для GPU, cpu для CPU)')
    parser.add_argument('--create-example', action='store_true',
                       help='Создать пример файла data.yaml')
    
    args = parser.parse_args()
    
    if args.create_example:
        create_example_data_yaml()
        return
    
    if not args.data:
        print("Ошибка: укажите путь к файлу конфигурации датасета (--data)")
        print("Используйте --create-example для создания примера data.yaml")
        return
    
    try:
        model_path = train_model(
            args.data,
            args.model,
            args.epochs,
            args.imgsz,
            args.batch,
            args.name,
            args.device
        )
        
        print(f"\nДля использования модели запустите:")
        print(f"python cli.py image.jpg --model {model_path}")
        
    except Exception as e:
        print(f"Ошибка при обучении: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
