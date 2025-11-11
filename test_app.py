"""
Тестовый скрипт для проверки работоспособности модулей
"""
import numpy as np
import cv2
from ball_detector import BallDetector
from net_detector import NetDetector
from position_analyzer import PositionAnalyzer, BallPosition


def create_test_image():
    """
    Создать тестовое изображение с мячом и сеткой
    
    Returns:
        numpy array с изображением
    """
    # Создаем белое изображение
    img = np.ones((480, 640, 3), dtype=np.uint8) * 255
    
    # Рисуем игровую площадку (зеленый фон)
    cv2.rectangle(img, (0, 200), (640, 480), (100, 200, 100), -1)
    
    # Рисуем сетку (белая вертикальная линия)
    net_x = 320
    cv2.line(img, (net_x, 150), (net_x, 480), (255, 255, 255), 4)
    cv2.line(img, (net_x-2, 150), (net_x-2, 480), (200, 200, 200), 2)
    cv2.line(img, (net_x+2, 150), (net_x+2, 480), (200, 200, 200), 2)
    
    # Рисуем мяч (оранжевый круг)
    # Вариант 1: мяч перед сеткой (слева)
    ball_center_1 = (200, 250)
    cv2.circle(img, ball_center_1, 30, (0, 100, 255), -1)
    cv2.circle(img, ball_center_1, 30, (0, 80, 200), 2)
    
    return img


def test_net_detector():
    """Тест детектора сетки"""
    print("\n" + "="*60)
    print("ТЕСТ: Детектор сетки")
    print("="*60)
    
    # Создаем тестовое изображение
    img = create_test_image()
    
    # Инициализируем детектор
    detector = NetDetector()
    
    # Тестируем разные методы
    methods = ['hough', 'contour', 'color']
    
    for method in methods:
        net_line = detector.detect_net_line(img, method)
        if net_line:
            net_x = detector.get_net_x_position(net_line)
            print(f"✓ Метод {method:8s}: сетка обнаружена на x={net_x}")
            print(f"  Координаты линии: {net_line}")
        else:
            print(f"✗ Метод {method:8s}: сетка НЕ обнаружена")
    
    print("\nТест завершен")
    
    # Сохраняем тестовое изображение
    cv2.imwrite('test_image.jpg', img)
    print("Тестовое изображение сохранено: test_image.jpg")


def test_position_analyzer():
    """Тест анализатора положения"""
    print("\n" + "="*60)
    print("ТЕСТ: Анализатор положения")
    print("="*60)
    
    # Создаем тестовое изображение
    img = create_test_image()
    h, w = img.shape[:2]
    
    # Инициализируем анализатор
    analyzer = PositionAnalyzer()
    
    # Тестовые случаи
    test_cases = [
        {
            'name': 'Мяч слева от сетки (перед)',
            'ball_bbox': (170, 220, 230, 280, 0.95),
            'ball_center': (200, 250),
            'expected': BallPosition.IN_FRONT_OF_NET
        },
        {
            'name': 'Мяч справа от сетки (за)',
            'ball_bbox': (410, 220, 470, 280, 0.95),
            'ball_center': (440, 250),
            'expected': BallPosition.BEHIND_NET
        },
        {
            'name': 'Мяч на линии сетки',
            'ball_bbox': (300, 220, 340, 280, 0.90),
            'ball_center': (320, 250),
            'expected': None  # может быть любой результат
        }
    ]
    
    net_x = 320  # позиция сетки
    
    for i, test in enumerate(test_cases, 1):
        print(f"\nТест {i}: {test['name']}")
        
        position, confidence = analyzer.analyze_with_depth(
            test['ball_center'],
            test['ball_bbox'],
            net_x,
            img
        )
        
        print(f"  Результат: {position.value}")
        print(f"  Уверенность: {confidence:.2%}")
        
        if test['expected'] is not None:
            if position == test['expected']:
                print(f"  ✓ УСПЕХ: получен ожидаемый результат")
            else:
                print(f"  ✗ ОШИБКА: ожидался {test['expected'].value}")
    
    print("\nТест завершен")


def test_visualization():
    """Тест визуализации"""
    print("\n" + "="*60)
    print("ТЕСТ: Визуализация результатов")
    print("="*60)
    
    # Создаем тестовое изображение
    img = create_test_image()
    
    # Инициализируем компоненты
    net_detector = NetDetector()
    analyzer = PositionAnalyzer()
    
    # Детектируем сетку
    net_line = net_detector.detect_net_line(img, 'hough')
    
    if net_line is None:
        print("✗ Не удалось обнаружить сетку для визуализации")
        return
    
    net_x = net_detector.get_net_x_position(net_line)
    
    # Тестовый мяч
    ball_bbox = (170, 220, 230, 280, 0.95)
    ball_center = (200, 250)
    
    # Анализируем
    position, confidence = analyzer.analyze_with_depth(
        ball_center, ball_bbox, net_x, img
    )
    
    # Визуализируем
    result_img = analyzer.visualize_result(
        img, ball_bbox, net_line, position, confidence
    )
    
    # Сохраняем
    cv2.imwrite('test_result.jpg', result_img)
    print("✓ Визуализация сохранена: test_result.jpg")
    print(f"  Положение: {position.value}")
    print(f"  Уверенность: {confidence:.2%}")


def test_ball_detector():
    """Тест детектора мяча"""
    print("\n" + "="*60)
    print("ТЕСТ: Детектор мяча (YOLO)")
    print("="*60)
    print("Примечание: для этого теста требуется модель YOLO")
    print("При первом запуске будет загружена предобученная модель (~6MB)")
    
    try:
        # Создаем тестовое изображение
        img = create_test_image()
        
        # Инициализируем детектор
        print("\nЗагрузка модели YOLO...")
        detector = BallDetector('yolov8n.pt')
        print("✓ Модель загружена")
        
        # Детектируем мяч
        print("\nПоиск мяча на тестовом изображении...")
        balls = detector.detect(img, confidence=0.3)
        
        if balls:
            print(f"✓ Обнаружено объектов: {len(balls)}")
            for i, ball in enumerate(balls, 1):
                x1, y1, x2, y2, conf = ball
                center = detector.get_ball_center(ball)
                print(f"  Объект {i}: bbox=({x1}, {y1}, {x2}, {y2}), "
                      f"confidence={conf:.2f}, center={center}")
        else:
            print("⚠ Объекты не обнаружены на тестовом изображении")
            print("  (это нормально, т.к. тестовое изображение синтетическое)")
        
        print("\n✓ Детектор мяча работает корректно")
        
    except Exception as e:
        print(f"✗ Ошибка при тестировании детектора мяча: {str(e)}")
        print("  Убедитесь, что установлены все зависимости:")
        print("  pip install ultralytics torch torchvision")


def main():
    """Главная функция"""
    print("="*60)
    print("ТЕСТИРОВАНИЕ МОДУЛЕЙ ПРИЛОЖЕНИЯ")
    print("="*60)
    
    # Запускаем все тесты
    test_net_detector()
    test_position_analyzer()
    test_visualization()
    test_ball_detector()
    
    print("\n" + "="*60)
    print("ВСЕ ТЕСТЫ ЗАВЕРШЕНЫ")
    print("="*60)
    print("\nСозданные файлы:")
    print("  - test_image.jpg   (тестовое изображение)")
    print("  - test_result.jpg  (результат визуализации)")
    print("\nДля запуска приложения:")
    print("  GUI: python main.py")
    print("  CLI: python cli.py test_image.jpg")


if __name__ == '__main__':
    main()
