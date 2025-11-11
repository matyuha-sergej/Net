"""
Скрипт для объединения нескольких датасетов в один и разделения на train/val/test
"""
import os
import sys
import shutil
from pathlib import Path
import random
import argparse
from typing import List, Tuple


def collect_datasets(datasets_dir: str) -> List[Tuple[Path, int]]:
    """
    Собрать все датасеты из папки
    
    Args:
        datasets_dir: путь к папке с датасетами
        
    Returns:
        список (путь_к_датасету, количество_изображений)
    """
    datasets_path = Path(datasets_dir)
    datasets = []
    
    for subdir in datasets_path.iterdir():
        if subdir.is_dir():
            images_dir = subdir / "images"
            labels_dir = subdir / "labels"
            
            if images_dir.exists() and labels_dir.exists():
                image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
                label_files = list(labels_dir.glob("*.txt"))
                
                # Проверяем, что количество изображений и меток совпадает
                if len(image_files) == len(label_files) > 0:
                    datasets.append((subdir, len(image_files)))
                    print(f"✓ Найден датасет: {subdir.name} ({len(image_files)} изображений)")
                else:
                    print(f"⚠ Пропущен датасет: {subdir.name} (несоответствие файлов)")
    
    return datasets


def merge_datasets(datasets: List[Tuple[Path, int]], 
                  output_dir: str,
                  train_ratio: float = 0.7,
                  val_ratio: float = 0.2,
                  test_ratio: float = 0.1,
                  copy_files: bool = True) -> dict:
    """
    Объединить датасеты и разделить на train/val/test
    
    Args:
        datasets: список датасетов
        output_dir: выходная папка
        train_ratio: доля для обучения
        val_ratio: доля для валидации
        test_ratio: доля для тестирования
        copy_files: копировать файлы (True) или создавать симлинки (False)
        
    Returns:
        статистика
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Создаем структуру папок
    for split in ['train', 'val', 'test']:
        (output_path / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_path / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    # Собираем все пары (изображение, метка)
    all_pairs = []
    
    for dataset_path, _ in datasets:
        images_dir = dataset_path / "images"
        labels_dir = dataset_path / "labels"
        
        for img_file in images_dir.glob("*.jpg"):
            label_file = labels_dir / f"{img_file.stem}.txt"
            if label_file.exists():
                all_pairs.append((img_file, label_file))
    
    print(f"\nВсего пар изображение-метка: {len(all_pairs)}")
    
    # Перемешиваем
    random.shuffle(all_pairs)
    
    # Разделяем на train/val/test
    total = len(all_pairs)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    splits = {
        'train': all_pairs[:train_end],
        'val': all_pairs[train_end:val_end],
        'test': all_pairs[val_end:]
    }
    
    # Копируем/линкуем файлы
    stats = {}
    
    for split_name, pairs in splits.items():
        print(f"\nОбработка {split_name}: {len(pairs)} изображений...")
        
        for i, (img_file, label_file) in enumerate(pairs):
            # Создаем уникальное имя файла
            new_name = f"{split_name}_{i:06d}{img_file.suffix}"
            
            img_dest = output_path / 'images' / split_name / new_name
            label_dest = output_path / 'labels' / split_name / f"{Path(new_name).stem}.txt"
            
            if copy_files:
                shutil.copy2(img_file, img_dest)
                shutil.copy2(label_file, label_dest)
            else:
                os.symlink(img_file.absolute(), img_dest)
                os.symlink(label_file.absolute(), label_dest)
        
        stats[split_name] = len(pairs)
        print(f"✓ {split_name}: {len(pairs)} изображений скопировано")
    
    return stats


def create_data_yaml(output_dir: str, class_names: List[str], stats: dict):
    """
    Создать файл data.yaml для обучения YOLO
    
    Args:
        output_dir: папка с объединенным датасетом
        class_names: названия классов
        stats: статистика по разделению
    """
    output_path = Path(output_dir)
    
    yaml_content = f"""# Конфигурация датасета для обучения YOLO
# Автоматически создан prepare_dataset.py

path: {output_path.absolute()}  # путь к корневой директории
train: images/train  # относительный путь к обучающим изображениям
val: images/val      # относительный путь к валидационным изображениям
test: images/test    # относительный путь к тестовым изображениям

# Классы
nc: {len(class_names)}  # количество классов
names: {class_names}  # названия классов

# Статистика
# Train: {stats.get('train', 0)} изображений
# Val: {stats.get('val', 0)} изображений
# Test: {stats.get('test', 0)} изображений
# Всего: {sum(stats.values())} изображений
"""
    
    yaml_path = output_path / 'data.yaml'
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    
    print(f"\n✓ Создан файл конфигурации: {yaml_path}")
    return yaml_path


def analyze_labels(datasets: List[Tuple[Path, int]]) -> List[str]:
    """
    Проанализировать метки и определить классы
    
    Args:
        datasets: список датасетов
        
    Returns:
        список названий классов
    """
    print("\nАнализ меток...")
    
    classes_found = set()
    sample_labels = []
    
    # Берем первый датасет для анализа
    if datasets:
        first_dataset = datasets[0][0]
        labels_dir = first_dataset / "labels"
        
        # Читаем несколько меток для анализа
        for label_file in list(labels_dir.glob("*.txt"))[:5]:
            with open(label_file, 'r') as f:
                content = f.read().strip()
                sample_labels.append(content)
                
                # Извлекаем номер класса
                for line in content.split('\n'):
                    if line.strip():
                        class_id = int(line.split()[0])
                        classes_found.add(class_id)
    
    print(f"Найдены классы: {sorted(classes_found)}")
    print(f"Примеры меток:")
    for i, label in enumerate(sample_labels[:3], 1):
        print(f"  Пример {i}: {label[:50]}...")
    
    # По умолчанию предполагаем, что это мяч
    # Пользователь может изменить в data.yaml
    class_names = ['ball'] * (max(classes_found) + 1 if classes_found else 1)
    
    return class_names


def main():
    """Главная функция"""
    parser = argparse.ArgumentParser(description='Подготовка датасета для обучения YOLO')
    parser.add_argument('--input', default='datasets', help='Папка с исходными датасетами')
    parser.add_argument('--output', default='datasets/merged_ball_dataset', help='Папка для объединенного датасета')
    parser.add_argument('--train-ratio', type=float, default=0.7, help='Доля для обучения')
    parser.add_argument('--val-ratio', type=float, default=0.2, help='Доля для валидации')
    parser.add_argument('--test-ratio', type=float, default=0.1, help='Доля для тестирования')
    parser.add_argument('--yes', '-y', action='store_true', help='Автоматически подтвердить')
    args = parser.parse_args()
    
    print("="*60)
    print("ПОДГОТОВКА ДАТАСЕТА ДЛЯ ОБУЧЕНИЯ")
    print("="*60)
    
    # Настройки
    datasets_dir = args.input
    output_dir = args.output
    
    # 1. Собираем датасеты
    print(f"\n1. Поиск датасетов в папке: {datasets_dir}")
    datasets = collect_datasets(datasets_dir)
    
    if not datasets:
        print("❌ Датасеты не найдены!")
        return
    
    total_images = sum(count for _, count in datasets)
    print(f"\n✓ Найдено датасетов: {len(datasets)}")
    print(f"✓ Всего изображений: {total_images}")
    
    # 2. Анализируем метки
    class_names = analyze_labels(datasets)
    
    # 3. Объединяем датасеты
    print(f"\n2. Объединение датасетов в: {output_dir}")
    print(f"   Разделение: {args.train_ratio*100:.0f}% train, {args.val_ratio*100:.0f}% val, {args.test_ratio*100:.0f}% test")
    
    if not args.yes:
        try:
            response = input("\nПродолжить? (y/n): ")
            if response.lower() != 'y':
                print("Отменено.")
                return
        except EOFError:
            print("\nАвтоматическое подтверждение (используйте --yes для пропуска)")
    
    stats = merge_datasets(
        datasets, 
        output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        copy_files=True  # Используйте False для симлинков (быстрее, но требует сохранения исходных файлов)
    )
    
    # 4. Создаем data.yaml
    print("\n3. Создание конфигурационного файла data.yaml")
    yaml_path = create_data_yaml(output_dir, class_names, stats)
    
    # 5. Итоги
    print("\n" + "="*60)
    print("ГОТОВО!")
    print("="*60)
    print(f"\nОбъединенный датасет: {output_dir}")
    print(f"Конфигурация: {yaml_path}")
    print(f"\nСтатистика:")
    print(f"  Train: {stats['train']} изображений")
    print(f"  Val:   {stats['val']} изображений")
    print(f"  Test:  {stats['test']} изображений")
    print(f"  Всего: {sum(stats.values())} изображений")
    
    print(f"\nДля обучения модели выполните:")
    print(f"  .venv/bin/python train_yolo.py \\")
    print(f"      --data {yaml_path} \\")
    print(f"      --epochs 100 \\")
    print(f"      --batch 16 \\")
    print(f"      --name ball_detector")
    
    print(f"\nИли быстрое обучение (меньше эпох):")
    print(f"  .venv/bin/python train_yolo.py \\")
    print(f"      --data {yaml_path} \\")
    print(f"      --epochs 50 \\")
    print(f"      --batch 16")


if __name__ == '__main__':
    # Устанавливаем seed для воспроизводимости
    random.seed(42)
    main()
