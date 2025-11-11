"""
Модуль для анализа положения мяча относительно сетки
"""
import numpy as np
from typing import Tuple, Optional
from enum import Enum


class BallPosition(Enum):
    """Положение мяча относительно сетки"""
    BEHIND_NET = "За сеткой"
    IN_FRONT_OF_NET = "Перед сеткой"
    UNKNOWN = "Неизвестно"


class PositionAnalyzer:
    """Класс для определения положения мяча относительно сетки"""
    
    def __init__(self, camera_position: str = 'side'):
        """
        Инициализация анализатора
        
        Args:
            camera_position: положение камеры ('side' - сбоку, 'front' - спереди)
        """
        self.camera_position = camera_position
    
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
        
        Args:
            ball_center: центр мяча (x, y)
            ball_bbox: bbox мяча (x1, y1, x2, y2, confidence)
            net_x: X-координата сетки
            image: исходное изображение
            
        Returns:
            (положение мяча, уверенность в определении)
        """
        ball_x, ball_y = ball_center
        x1, y1, x2, y2, conf = ball_bbox
        
        # Вычисляем размер мяча
        ball_size = (x2 - x1) * (y2 - y1)
        
        # Анализируем перекрытие с сеткой
        occlusion_score = self._check_occlusion(ball_bbox, net_x, image)
        
        # Определяем положение
        if ball_x < net_x - 20:
            position = BallPosition.IN_FRONT_OF_NET
            confidence = min(0.95, conf + occlusion_score * 0.3)
        elif ball_x > net_x + 20:
            position = BallPosition.BEHIND_NET
            confidence = min(0.95, conf + occlusion_score * 0.3)
        else:
            # Мяч близко к сетке - анализируем окклюзию
            if occlusion_score > 0.3:
                position = BallPosition.BEHIND_NET
                confidence = 0.7 + occlusion_score * 0.2
            else:
                position = BallPosition.IN_FRONT_OF_NET
                confidence = 0.7 - occlusion_score * 0.2
        
        return position, confidence
    
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
            color = (0, 0, 255)  # Красный
        elif position == BallPosition.IN_FRONT_OF_NET:
            color = (0, 255, 0)  # Зеленый
        else:
            color = (128, 128, 128)  # Серый
        
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
        
        # Рисуем сетку
        if net_line is not None:
            nx1, ny1, nx2, ny2 = net_line
            cv2.line(result, (nx1, ny1), (nx2, ny2), (255, 255, 0), 2)
        
        # Добавляем текст с результатом
        text = f"{position.value} ({confidence:.2f})"
        cv2.putText(result, text, (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return result
