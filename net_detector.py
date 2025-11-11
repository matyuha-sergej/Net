"""
Модуль для детекции сетки на изображении
"""
import cv2
import numpy as np
from typing import Optional, Tuple, List


class NetDetector:
    """Класс для обнаружения сетки на изображении"""
    
    def __init__(self):
        """Инициализация детектора сетки"""
        pass
    
    def detect_net_line(self, image: np.ndarray, method: str = 'hough') -> Optional[Tuple[int, int, int, int]]:
        """
        Обнаружение линии сетки на изображении
        
        Args:
            image: изображение в формате numpy array (BGR)
            method: метод детекции ('hough', 'contour', 'color')
            
        Returns:
            координаты линии сетки (x1, y1, x2, y2) или None
        """
        if method == 'hough':
            return self._detect_by_hough(image)
        elif method == 'contour':
            return self._detect_by_contour(image)
        elif method == 'color':
            return self._detect_by_color(image)
        else:
            return self._detect_by_hough(image)
    
    def _detect_by_hough(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Детекция сетки методом Hough преобразования
        
        Args:
            image: изображение в формате numpy array (BGR)
            
        Returns:
            координаты вертикальной линии сетки (x1, y1, x2, y2)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Обнаружение линий
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, 
                                minLineLength=100, maxLineGap=10)
        
        if lines is None:
            return None
        
        # Находим наиболее вертикальную линию в центральной части изображения
        h, w = image.shape[:2]
        best_line = None
        best_score = 0
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Проверяем вертикальность
            if abs(x2 - x1) < 50:  # Линия должна быть почти вертикальной
                # Проверяем, находится ли в центральной части
                line_x = (x1 + x2) / 2
                if 0.3 * w < line_x < 0.7 * w:
                    # Вычисляем длину линии
                    length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                    verticality = abs(x2 - x1) / (abs(y2 - y1) + 1)  # Чем меньше, тем вертикальнее
                    score = length / (verticality + 1)
                    
                    if score > best_score:
                        best_score = score
                        best_line = (x1, y1, x2, y2)
        
        return best_line
    
    def _detect_by_contour(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Детекция сетки методом поиска контуров
        
        Args:
            image: изображение в формате numpy array (BGR)
            
        Returns:
            координаты вертикальной линии сетки
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        h, w = image.shape[:2]
        best_line = None
        
        for contour in contours:
            x, y, w_c, h_c = cv2.boundingRect(contour)
            
            # Ищем узкий вертикальный контур
            if h_c > h * 0.5 and w_c < 20:
                x_center = x + w_c // 2
                best_line = (x_center, y, x_center, y + h_c)
                break
        
        return best_line
    
    def _detect_by_color(self, image: np.ndarray, 
                        lower_color: Tuple[int, int, int] = (200, 200, 200),
                        upper_color: Tuple[int, int, int] = (255, 255, 255)) -> Optional[Tuple[int, int, int, int]]:
        """
        Детекция сетки по цвету (обычно сетка белая)
        
        Args:
            image: изображение в формате numpy array (BGR)
            lower_color: нижняя граница цвета
            upper_color: верхняя граница цвета
            
        Returns:
            координаты вертикальной линии сетки
        """
        # Создаем маску по цвету
        mask = cv2.inRange(image, lower_color, upper_color)
        
        # Морфологические операции для улучшения
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Находим контуры
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        h, w = image.shape[:2]
        best_line = None
        max_height = 0
        
        for contour in contours:
            x, y, w_c, h_c = cv2.boundingRect(contour)
            
            # Ищем самый высокий узкий контур в центре
            if h_c > max_height and w_c < 30:
                line_x = x + w_c // 2
                if 0.3 * w < line_x < 0.7 * w:
                    max_height = h_c
                    best_line = (line_x, y, line_x, y + h_c)
        
        return best_line
    
    def get_net_x_position(self, net_line: Optional[Tuple[int, int, int, int]]) -> Optional[int]:
        """
        Получить X-координату сетки
        
        Args:
            net_line: линия сетки (x1, y1, x2, y2)
            
        Returns:
            X-координата сетки или None
        """
        if net_line is None:
            return None
        
        x1, _, x2, _ = net_line
        return (x1 + x2) // 2
