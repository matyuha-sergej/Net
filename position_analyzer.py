"""
Модуль для анализа положения мяча относительно сетки
"""
import numpy as np
from typing import Tuple, Optional
from enum import Enum


class BallPosition(Enum):
    """Положение мяча относительно сетки"""
    BEHIND_NET = "behind the net"
    IN_FRONT_OF_NET = "front of the net"
    UNKNOWN = "unknown"


class PositionAnalyzer:
    """Класс для определения положения мяча относительно сетки"""
    
    def __init__(self, camera_position: str = 'behind_goal'):
        """
        Инициализация анализатора
        
        Args:
            camera_position: положение камеры 
                - 'behind_goal' - за воротами (по умолчанию)
                - 'side' - сбоку
        """
        self.camera_position = camera_position
        # Эталонные размеры мяча (пиксели) - калибруются автоматически
        self.reference_ball_sizes = []
    
    def analyze_position(self, 
                        ball_center: Tuple[int, int],
                        ball_bbox: Tuple[int, int, int, int, float],
                        net_x: int,
                        image_shape: Tuple[int, int]) -> BallPosition:
        """
        Определить положение мяча относительно сетки
        
        Args:
            ball_center: центр мяча (x, y)
            ball_bbox: bbox мяча (x1, y1, x2, y2, confidence)
            net_x: X-координата сетки
            image_shape: размер изображения (height, width)
            
        Returns:
            положение мяча
        """
        ball_x, ball_y = ball_center
        x1, y1, x2, y2, _ = ball_bbox
        
        # Простой метод: сравнение X-координат
        # Если камера сбоку, используем размер мяча и перспективу
        ball_width = x2 - x1
        ball_height = y2 - y1
        
        if self.camera_position == 'side':
            # Анализируем размер мяча - мяч дальше от камеры кажется меньше
            # Также учитываем X-координату
            
            if ball_x < net_x:
                # Мяч слева от сетки
                return BallPosition.IN_FRONT_OF_NET
            else:
                # Мяч справа от сетки
                return BallPosition.BEHIND_NET
        
        return BallPosition.UNKNOWN
    
    def analyze_with_depth(self,
                          ball_center: Tuple[int, int],
                          ball_bbox: Tuple[int, int, int, int, float],
                          net_x: int,
                          image: np.ndarray) -> Tuple[BallPosition, float]:
        """
        Определить положение мяча с оценкой уверенности
        
        Для камеры за воротами используется анализ размера мяча:
        - Мяч дальше (в воротах/за сеткой) выглядит МЕНЬШЕ
        - Мяч ближе (перед сеткой) выглядит БОЛЬШЕ
        
        Args:
            ball_center: центр мяча (x, y)
            ball_bbox: bbox мяча (x1, y1, x2, y2, confidence)
            net_x: X-координата сетки (не используется для behind_goal)
            image: исходное изображение
            
        Returns:
            (положение мяча, уверенность в определении)
        """
        ball_x, ball_y = ball_center
        x1, y1, x2, y2, conf = ball_bbox
        
        # Вычисляем размер мяча
        ball_width = x2 - x1
        ball_height = y2 - y1
        ball_size = ball_width * ball_height
        ball_diameter = (ball_width + ball_height) / 2
        
        if self.camera_position == 'behind_goal':
            # Камера за воротами - анализируем глубину по размеру мяча
            position, confidence = self._analyze_depth_by_size(
                ball_bbox, ball_center, ball_size, ball_diameter, image, conf
            )
        else:
            # Старая логика для камеры сбоку
            occlusion_score = self._check_occlusion(ball_bbox, net_x, image)
            
            if ball_x < net_x - 20:
                position = BallPosition.IN_FRONT_OF_NET
                confidence = min(0.95, conf + occlusion_score * 0.3)
            elif ball_x > net_x + 20:
                position = BallPosition.BEHIND_NET
                confidence = min(0.95, conf + occlusion_score * 0.3)
            else:
                if occlusion_score > 0.3:
                    position = BallPosition.BEHIND_NET
                    confidence = 0.7 + occlusion_score * 0.2
                else:
                    position = BallPosition.IN_FRONT_OF_NET
                    confidence = 0.7 - occlusion_score * 0.2
        
        return position, confidence
    
    def _analyze_depth_by_size(self,
                               ball_bbox: Tuple[int, int, int, int, float],
                               ball_center: Tuple[int, int],
                               ball_size: float,
                               ball_diameter: float,
                               image: np.ndarray,
                               detection_conf: float) -> Tuple[BallPosition, float]:
        """
        Анализ положения мяча по его размеру (для камеры за воротами)
        
        Логика:
        1. Мяч дальше (в воротах) → меньше по размеру
        2. Мяч ближе (перед сеткой) → больше по размеру
        3. Также учитываем Y-координату и окклюзию
        
        Args:
            ball_bbox: bbox мяча
            ball_center: центр мяча
            ball_size: площадь мяча в пикселях
            ball_diameter: средний диаметр мяча
            image: изображение
            detection_conf: уверенность детекции YOLO
            
        Returns:
            (положение, уверенность)
        """
        x1, y1, x2, y2, _ = ball_bbox
        ball_x, ball_y = ball_center
        img_height, img_width = image.shape[:2]
        
        # Анализируем окклюзию сеткой
        occlusion_score = self._check_net_occlusion(ball_bbox, image)
        
        # Нормализованная Y-координата (0 = верх, 1 = низ)
        normalized_y = ball_y / img_height
        
        # Критерии для определения положения:
        
        # 1. РАЗМЕР МЯЧА
        # Калибровано на реальных данных:
        # - Мячи 22-27 px (Q1): за сеткой/в воротах (дальше от камеры)
        # - Мячи 56-72 px (Q3+): перед сеткой (ближе к камере)
        # - Медиана 31.8 px: промежуточная зона
        
        size_threshold_far = 28  # Если меньше - скорее всего за сеткой
        size_threshold_near = 48  # Если больше - скорее всего перед сеткой
        
        # 2. Y-КООРДИНАТА
        # Мяч за сеткой обычно в верхней/средней части кадра
        # Мяч перед сеткой может быть ниже
        y_threshold_high = 0.4  # Верхняя часть кадра
        
        # 3. ОККЛЮЗИЯ
        # Если сетка сильно перекрывает мяч - он за ней
        
        # АНАЛИЗ
        scores = {
            'behind': 0.0,
            'front': 0.0
        }
        
        # Критерий 1: Размер мяча (основной фактор)
        if ball_diameter < size_threshold_far:
            # Маленький мяч - скорее всего за сеткой
            scores['behind'] += 0.5
        elif ball_diameter > size_threshold_near:
            # Большой мяч - скорее всего перед сеткой
            scores['front'] += 0.5
        else:
            # Промежуточный размер - анализируем по другим признакам
            # Линейная интерполяция
            ratio = (ball_diameter - size_threshold_far) / (size_threshold_near - size_threshold_far)
            scores['front'] += ratio * 0.3
            scores['behind'] += (1 - ratio) * 0.3
        
        # Критерий 2: Y-координата
        if normalized_y < y_threshold_high:
            # Мяч в верхней части - скорее за сеткой
            scores['behind'] += 0.2
        else:
            # Мяч в нижней части - скорее перед сеткой
            scores['front'] += 0.15
        
        # Критерий 3: Окклюзия сеткой (усиленный вес)
        if occlusion_score > 0.4:
            # Сильная окклюзия - мяч определенно за сеткой
            scores['behind'] += 0.5 * occlusion_score
        elif occlusion_score > 0.2:
            # Средняя окклюзия - скорее за сеткой
            scores['behind'] += 0.3 * occlusion_score
        else:
            # Слабая или нет окклюзии - мяч перед сеткой
            scores['front'] += 0.15
        
        # Определяем финальное положение
        if scores['behind'] > scores['front']:
            position = BallPosition.BEHIND_NET
            confidence = min(0.95, scores['behind'] * detection_conf)
        else:
            position = BallPosition.IN_FRONT_OF_NET
            confidence = min(0.95, scores['front'] * detection_conf)
        
        # Повышаем уверенность для явных случаев
        if ball_diameter < size_threshold_far - 5:
            confidence = max(confidence, 0.85)
        elif ball_diameter > size_threshold_near + 5:
            confidence = max(confidence, 0.85)
        
        return position, confidence
    
    def _check_net_occlusion(self, 
                            ball_bbox: Tuple[int, int, int, int, float],
                            image: np.ndarray) -> float:
        """
        Проверить, перекрыт ли мяч сеткой (усиленный анализ текстуры)
        
        Метод:
        1. Анализ линий сетки (Canny edges)
        2. Детекция паттерна сетки (решетчатая структура)
        3. Анализ текстурных особенностей
        
        Args:
            ball_bbox: bbox мяча
            image: изображение
            
        Returns:
            степень окклюзии (0-1)
        """
        import cv2
        
        x1, y1, x2, y2, _ = ball_bbox
        
        # Расширяем область вокруг мяча для анализа сетки
        margin = 15  # Увеличен для лучшего анализа контекста
        x1_ext = max(0, x1 - margin)
        y1_ext = max(0, y1 - margin)
        x2_ext = min(image.shape[1], x2 + margin)
        y2_ext = min(image.shape[0], y2 + margin)
        
        # Извлекаем область
        ball_region = image[y1_ext:y2_ext, x1_ext:x2_ext]
        
        if ball_region.size == 0:
            return 0.0
        
        # Конвертируем в grayscale
        if len(ball_region.shape) == 3:
            gray = cv2.cvtColor(ball_region, cv2.COLOR_BGR2GRAY)
        else:
            gray = ball_region
        
        occlusion_scores = []
        
        # Критерий 1: Плотность ребер (линии сетки)
        edges = cv2.Canny(gray, 30, 100)  # Более чувствительные пороги
        edge_density = np.sum(edges > 0) / edges.size
        occlusion_scores.append(min(1.0, edge_density * 6))
        
        # Критерий 2: Детекция линий Хафа (вертикальные и горизонтальные линии сетки)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=20, 
                                minLineLength=10, maxLineGap=5)
        if lines is not None:
            # Много линий = сильная окклюзия
            line_score = min(1.0, len(lines) / 20.0)
            occlusion_scores.append(line_score)
        else:
            occlusion_scores.append(0.0)
        
        # Критерий 3: Текстурный анализ (дисперсия)
        # Сетка создает высокую дисперсию из-за чередования белых/черных участков
        texture_variance = np.var(gray)
        # Нормализуем: высокая дисперсия (>1000) = высокая окклюзия
        texture_score = min(1.0, texture_variance / 2000.0)
        occlusion_scores.append(texture_score)
        
        # Критерий 4: Анализ градиентов (сетка создает много резких переходов)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        gradient_score = min(1.0, np.mean(gradient_magnitude) / 50.0)
        occlusion_scores.append(gradient_score)
        
        # Финальная оценка окклюзии (взвешенное среднее)
        weights = [0.35, 0.30, 0.20, 0.15]  # Веса для каждого критерия
        final_occlusion = sum(s * w for s, w in zip(occlusion_scores, weights))
        
        return min(1.0, final_occlusion)
    
    def _check_occlusion(self, 
                        ball_bbox: Tuple[int, int, int, int, float],
                        net_x: int,
                        image: np.ndarray) -> float:
        """
        Проверить перекрытие мяча сеткой
        
        Args:
            ball_bbox: bbox мяча
            net_x: X-координата сетки
            image: изображение
            
        Returns:
            степень перекрытия (0-1)
        """
        x1, y1, x2, y2, _ = ball_bbox
        
        # Проверяем, пересекает ли сетка bbox мяча
        if x1 < net_x < x2:
            # Сетка пересекает мяч
            overlap_width = min(x2 - net_x, net_x - x1)
            ball_width = x2 - x1
            return overlap_width / ball_width
        
        return 0.0
    
    def visualize_result(self,
                        image: np.ndarray,
                        ball_bbox: Tuple[int, int, int, int, float],
                        net_line: Optional[Tuple[int, int, int, int]],
                        position: BallPosition,
                        confidence: float) -> np.ndarray:
        """
        Визуализировать результат на изображении
        
        Args:
            image: исходное изображение
            ball_bbox: bbox мяча
            net_line: линия сетки
            position: положение мяча
            confidence: уверенность
            
        Returns:
            изображение с визуализацией
        """
        import cv2
        
        result = image.copy()
        x1, y1, x2, y2, _ = ball_bbox
        
        # Рисуем bbox мяча
        if position == BallPosition.BEHIND_NET:
            color = (0, 0, 255)  # Красный (BGR)
            label = "BEHIND THE NET"
        elif position == BallPosition.IN_FRONT_OF_NET:
            color = (0, 255, 0)  # Зеленый (BGR)
            label = "FRONT OF THE NET"
        else:
            color = (128, 128, 128)  # Серый
            label = "UNKNOWN"
        
        # Рисуем утолщенный bbox мяча
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 3)
        
        # Рисуем сетку (желтая линия)
        if net_line is not None:
            nx1, ny1, nx2, ny2 = net_line
            cv2.line(result, (nx1, ny1), (nx2, ny2), (0, 255, 255), 3)
        
        # Добавляем текст с результатом
        # Основной текст (верхний)
        text = f"{label}"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        
        # Фон для текста (для лучшей читаемости)
        text_x = max(10, x1)
        text_y = max(30, y1 - 15)
        
        cv2.rectangle(result, 
                     (text_x - 5, text_y - text_size[1] - 5),
                     (text_x + text_size[0] + 5, text_y + 5),
                     (0, 0, 0), -1)  # Черный фон
        
        cv2.putText(result, text, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Дополнительный текст с уверенностью (нижний)
        conf_text = f"Confidence: {confidence:.1%}"
        cv2.putText(result, conf_text, (text_x, text_y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return result
