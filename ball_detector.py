"""
Модуль для детекции мяча с использованием YOLO
"""
import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Optional


class BallDetector:
    """Класс для обнаружения мяча на изображении"""
    
    def __init__(self, model_path: str = 'yolov8n.pt'):
        """
        Инициализация детектора мяча
        
        Args:
            model_path: путь к модели YOLO (можно использовать предобученную или свою)
        """
        self.model = YOLO(model_path)
        
    def detect(self, image: np.ndarray, confidence: float = 0.5) -> List[Tuple[int, int, int, int, float]]:
        """
        Обнаружение мяча на изображении
        
        Args:
            image: изображение в формате numpy array (BGR)
            confidence: минимальная уверенность детекции
            
        Returns:
            список bbox в формате (x1, y1, x2, y2, confidence)
        """
        results = self.model(image, conf=confidence, verbose=False)
        
        balls = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Фильтруем только мячи (sports ball class = 32 в COCO)
                # Если у вас своя модель только для мяча, убираем эту проверку
                cls = int(box.cls[0])
                if cls == 32 or True:  # Принимаем все объекты если модель обучена только на мячах
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    balls.append((int(x1), int(y1), int(x2), int(y2), conf))
        
        return balls
    
    def get_ball_center(self, bbox: Tuple[int, int, int, int, float]) -> Tuple[int, int]:
        """
        Получить центр мяча
        
        Args:
            bbox: bbox в формате (x1, y1, x2, y2, confidence)
            
        Returns:
            координаты центра (x, y)
        """
        x1, y1, x2, y2, _ = bbox
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        return center_x, center_y
