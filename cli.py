"""
Скрипт для анализа изображений/видео из командной строки
"""
import argparse
import cv2
import sys
from pathlib import Path

from ball_detector import BallDetector
from net_detector import NetDetector
from position_analyzer import PositionAnalyzer


def process_image(image_path: str, model_path: str, confidence: float, 
                 net_method: str, output_path: str = None):
    """
    Обработать изображение
    
    Args:
        image_path: путь к изображению
        model_path: путь к модели YOLO
        confidence: порог уверенности
        net_method: метод детекции сетки
        output_path: путь для сохранения результата
    """
    # Инициализация
    ball_detector = BallDetector(model_path)
    net_detector = NetDetector()
    position_analyzer = PositionAnalyzer()
    
    # Загрузка изображения
    image = cv2.imread(image_path)
    if image is None:
        print(f"Ошибка: не удалось загрузить изображение {image_path}")
        return
    
    print(f"Обработка изображения: {image_path}")
    
    # Детекция мяча
    balls = ball_detector.detect(image, confidence)
    
    if not balls:
        print("Мяч не обнаружен на изображении")
        return
    
    print(f"Обнаружено мячей: {len(balls)}")
    
    # Берем мяч с максимальной уверенностью
    ball_bbox = max(balls, key=lambda x: x[4])
    ball_center = ball_detector.get_ball_center(ball_bbox)
    
    # Вычисляем размер мяча для отладки
    x1, y1, x2, y2, conf = ball_bbox
    ball_width = x2 - x1
    ball_height = y2 - y1
    ball_diameter = (ball_width + ball_height) / 2
    
    print(f"Мяч: bbox={ball_bbox[:4]}, confidence={conf:.2f}")
    print(f"Размер мяча: {ball_width}x{ball_height} px (диаметр: {ball_diameter:.1f} px)")
    
    # Детекция сетки
    net_line = net_detector.detect_net_line(image, net_method)
    
    if net_line is None:
        print(f"Сетка не обнаружена методом '{net_method}'. Попробуйте другой метод.")
        return
    
    print(f"Сетка обнаружена: {net_line}")
    
    net_x = net_detector.get_net_x_position(net_line)
    
    # Анализ положения
    position, conf = position_analyzer.analyze_with_depth(
        ball_center, ball_bbox, net_x, image
    )
    
    # Дополнительная информация об окклюзии
    occlusion_score = position_analyzer._check_net_occlusion(ball_bbox, image)
    
    print(f"\n{'='*50}")
    print(f"РЕЗУЛЬТАТ: {position.value}")
    print(f"Уверенность: {conf:.2%}")
    print(f"Окклюзия сеткой: {occlusion_score:.2%}")
    print(f"{'='*50}\n")
    
    # Визуализация
    result_img = position_analyzer.visualize_result(
        image, ball_bbox, net_line, position, conf
    )
    
    # Сохранение результата
    if output_path is None:
        path = Path(image_path)
        output_path = str(path.parent / f"{path.stem}_result{path.suffix}")
    
    cv2.imwrite(output_path, result_img)
    print(f"Результат сохранен: {output_path}")


def process_video(video_path: str, model_path: str, confidence: float,
                 net_method: str, output_path: str = None):
    """
    Обработать видео
    
    Args:
        video_path: путь к видео
        model_path: путь к модели YOLO
        confidence: порог уверенности
        net_method: метод детекции сетки
        output_path: путь для сохранения результата
    """
    # Инициализация
    ball_detector = BallDetector(model_path)
    net_detector = NetDetector()
    position_analyzer = PositionAnalyzer()
    
    # Открытие видео
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Ошибка: не удалось открыть видео {video_path}")
        return
    
    # Получаем параметры видео
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Обработка видео: {video_path}")
    print(f"Параметры: {width}x{height}, {fps} FPS, {total_frames} кадров")
    
    # Создаем writer для сохранения результата
    if output_path is None:
        path = Path(video_path)
        output_path = str(path.parent / f"{path.stem}_result{path.suffix}")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    detections = {'behind': 0, 'front': 0, 'unknown': 0}
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Детекция мяча
        balls = ball_detector.detect(frame, confidence)
        
        result_frame = frame.copy()
        
        if balls:
            ball_bbox = max(balls, key=lambda x: x[4])
            ball_center = ball_detector.get_ball_center(ball_bbox)
            
            # Детекция сетки
            net_line = net_detector.detect_net_line(frame, net_method)
            
            if net_line is not None:
                net_x = net_detector.get_net_x_position(net_line)
                position, conf = position_analyzer.analyze_with_depth(
                    ball_center, ball_bbox, net_x, frame
                )
                
                result_frame = position_analyzer.visualize_result(
                    frame, ball_bbox, net_line, position, conf
                )
                
                # Статистика
                if position.name == 'BEHIND_NET':
                    detections['behind'] += 1
                elif position.name == 'IN_FRONT_OF_NET':
                    detections['front'] += 1
                else:
                    detections['unknown'] += 1
        
        out.write(result_frame)
        
        # Прогресс
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"Прогресс: {progress:.1f}% ({frame_count}/{total_frames})", end='\r')
    
    cap.release()
    out.release()
    
    print(f"\n\nВидео обработано и сохранено: {output_path}")
    print(f"\nСтатистика:")
    print(f"  Мяч перед сеткой: {detections['front']} кадров")
    print(f"  Мяч за сеткой: {detections['behind']} кадров")
    print(f"  Неопределенно: {detections['unknown']} кадров")


def main():
    """Главная функция"""
    parser = argparse.ArgumentParser(
        description='Определение положения мяча относительно сетки'
    )
    
    parser.add_argument('input', help='Путь к изображению или видео')
    parser.add_argument('-m', '--model', default='yolov8n.pt',
                       help='Путь к модели YOLO (по умолчанию: yolov8n.pt)')
    parser.add_argument('-c', '--confidence', type=float, default=0.5,
                       help='Порог уверенности (по умолчанию: 0.5)')
    parser.add_argument('-n', '--net-method', choices=['hough', 'contour', 'color'],
                       default='hough', help='Метод детекции сетки (по умолчанию: hough)')
    parser.add_argument('-o', '--output', help='Путь для сохранения результата')
    parser.add_argument('-v', '--video', action='store_true',
                       help='Обработать как видео')
    
    args = parser.parse_args()
    
    try:
        if args.video:
            process_video(args.input, args.model, args.confidence, 
                         args.net_method, args.output)
        else:
            process_image(args.input, args.model, args.confidence,
                         args.net_method, args.output)
    except Exception as e:
        print(f"Ошибка: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
