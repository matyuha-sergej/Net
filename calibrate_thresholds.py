"""
Скрипт для калибровки порогов размера мяча
Анализирует изображения и показывает статистику размеров мячей
"""
import cv2
import numpy as np
from pathlib import Path
import argparse
from ball_detector import BallDetector


def analyze_images(images_dir: str, model_path: str, confidence: float = 0.5):
    """
    Анализировать изображения и собрать статистику размеров мячей
    
    Args:
        images_dir: папка с изображениями
        model_path: путь к модели YOLO
        confidence: порог уверенности
    """
    print("="*60)
    print("КАЛИБРОВКА ПОРОГОВ РАЗМЕРА МЯЧА")
    print("="*60)
    
    # Инициализация детектора
    ball_detector = BallDetector(model_path)
    
    # Находим все изображения
    images_path = Path(images_dir)
    image_files = list(images_path.glob("*.jpg")) + list(images_path.glob("*.png"))
    
    if not image_files:
        print(f"❌ Изображения не найдены в {images_dir}")
        return
    
    print(f"\n✓ Найдено изображений: {len(image_files)}")
    print(f"Анализ первых 50 изображений...\n")
    
    # Статистика
    ball_sizes = []
    ball_diameters = []
    ball_y_positions = []
    
    for i, img_file in enumerate(image_files[:50]):
        image = cv2.imread(str(img_file))
        if image is None:
            continue
        
        img_height, img_width = image.shape[:2]
        
        # Детекция мяча
        balls = ball_detector.detect(image, confidence)
        
        if balls:
            # Берем мяч с максимальной уверенностью
            ball_bbox = max(balls, key=lambda x: x[4])
            x1, y1, x2, y2, conf = ball_bbox
            
            ball_width = x2 - x1
            ball_height = y2 - y1
            ball_diameter = (ball_width + ball_height) / 2
            ball_y = (y1 + y2) / 2
            normalized_y = ball_y / img_height
            
            ball_sizes.append((ball_width, ball_height))
            ball_diameters.append(ball_diameter)
            ball_y_positions.append(normalized_y)
            
            print(f"{i+1:3d}. {img_file.name:40s} | "
                  f"Размер: {ball_width:3.0f}x{ball_height:3.0f} px | "
                  f"Диаметр: {ball_diameter:5.1f} px | "
                  f"Y: {normalized_y:.2f}")
    
    if not ball_diameters:
        print("\n❌ Мячи не обнаружены ни на одном изображении")
        return
    
    # Вычисляем статистику
    diameters_array = np.array(ball_diameters)
    
    print("\n" + "="*60)
    print("СТАТИСТИКА РАЗМЕРОВ МЯЧА")
    print("="*60)
    
    print(f"\nВсего проанализировано: {len(ball_diameters)} мячей")
    print(f"\nДиаметр мяча (пиксели):")
    print(f"  Минимальный:  {np.min(diameters_array):.1f} px")
    print(f"  Максимальный: {np.max(diameters_array):.1f} px")
    print(f"  Средний:      {np.mean(diameters_array):.1f} px")
    print(f"  Медиана:      {np.median(diameters_array):.1f} px")
    print(f"  Std Dev:      {np.std(diameters_array):.1f} px")
    
    # Квартили
    q25 = np.percentile(diameters_array, 25)
    q75 = np.percentile(diameters_array, 75)
    
    print(f"\nКвартили:")
    print(f"  25% (Q1):     {q25:.1f} px")
    print(f"  75% (Q3):     {q75:.1f} px")
    
    # Анализ Y-координат
    y_array = np.array(ball_y_positions)
    print(f"\nY-координата (нормализованная, 0=верх, 1=низ):")
    print(f"  Минимальная:  {np.min(y_array):.2f}")
    print(f"  Максимальная: {np.max(y_array):.2f}")
    print(f"  Средняя:      {np.mean(y_array):.2f}")
    
    # Рекомендации по порогам
    print("\n" + "="*60)
    print("РЕКОМЕНДУЕМЫЕ ПОРОГИ")
    print("="*60)
    
    # Определяем пороги на основе квартилей
    threshold_far = q25 - 5  # Мячи меньше этого - за сеткой
    threshold_near = q75 + 5  # Мячи больше этого - перед сеткой
    
    print(f"\nДля камеры ЗА воротами:")
    print(f"  size_threshold_far:  {threshold_far:.0f} px  (мяч в воротах/за сеткой)")
    print(f"  size_threshold_near: {threshold_near:.0f} px  (мяч перед сеткой)")
    
    print(f"\nТекущие значения в position_analyzer.py:")
    print(f"  size_threshold_far:  35 px")
    print(f"  size_threshold_near: 50 px")
    
    if abs(threshold_far - 35) > 10 or abs(threshold_near - 50) > 10:
        print(f"\n⚠️  РЕКОМЕНДУЕТСЯ обновить пороги в position_analyzer.py!")
        print(f"\n   Откройте position_analyzer.py и измените:")
        print(f"   size_threshold_far = {threshold_far:.0f}  # строка ~168")
        print(f"   size_threshold_near = {threshold_near:.0f}  # строка ~169")
    else:
        print(f"\n✓ Текущие пороги в допустимых пределах")
    
    # Гистограмма (текстовая)
    print(f"\nРаспределение размеров (гистограмма):")
    hist, bins = np.histogram(diameters_array, bins=10)
    max_count = max(hist)
    
    for i in range(len(hist)):
        bar_length = int(40 * hist[i] / max_count) if max_count > 0 else 0
        print(f"  {bins[i]:5.1f}-{bins[i+1]:5.1f} px: {'█' * bar_length} ({hist[i]})")
    
    print("\n" + "="*60)
    print("СОВЕТ: Посмотрите на изображения с минимальными и максимальными")
    print("       размерами мяча, чтобы убедиться в корректности порогов.")
    print("="*60)


def main():
    """Главная функция"""
    parser = argparse.ArgumentParser(description='Калибровка порогов размера мяча')
    parser.add_argument('--images', default='datasets/merged_ball_dataset/images/test',
                       help='Папка с изображениями для анализа')
    parser.add_argument('--model', default='runs/detect/ball_detector_quick/weights/best.pt',
                       help='Путь к модели YOLO')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Порог уверенности детекции')
    
    args = parser.parse_args()
    
    try:
        analyze_images(args.images, args.model, args.confidence)
    except Exception as e:
        print(f"\n❌ Ошибка: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
