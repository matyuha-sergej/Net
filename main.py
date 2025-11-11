"""
Основное приложение для определения положения мяча относительно сетки
"""
import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFileDialog,
                             QComboBox, QSlider, QGroupBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap

from ball_detector import BallDetector
from net_detector import NetDetector
from position_analyzer import PositionAnalyzer, BallPosition


class BallPositionApp(QMainWindow):
    """Главное окно приложения"""
    
    def __init__(self):
        super().__init__()
        self.image = None
        self.result_image = None
        
        # Инициализация детекторов
        self.ball_detector = None
        self.net_detector = NetDetector()
        self.position_analyzer = PositionAnalyzer()
        
        self.init_ui()
    
    def init_ui(self):
        """Инициализация интерфейса"""
        self.setWindowTitle('Определение положения мяча относительно сетки')
        self.setGeometry(100, 100, 1200, 800)
        
        # Центральный виджет
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Главный layout
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)
        
        # Панель управления
        control_panel = self._create_control_panel()
        main_layout.addWidget(control_panel)
        
        # Панель изображений
        image_panel = self._create_image_panel()
        main_layout.addLayout(image_panel)
        
        # Панель результатов
        result_panel = self._create_result_panel()
        main_layout.addWidget(result_panel)
    
    def _create_control_panel(self) -> QGroupBox:
        """Создать панель управления"""
        group_box = QGroupBox("Управление")
        layout = QHBoxLayout()
        
        # Кнопка загрузки изображения
        self.load_btn = QPushButton('Загрузить изображение')
        self.load_btn.clicked.connect(self.load_image)
        layout.addWidget(self.load_btn)
        
        # Кнопка загрузки видео
        self.load_video_btn = QPushButton('Загрузить видео')
        self.load_video_btn.clicked.connect(self.load_video)
        layout.addWidget(self.load_video_btn)
        
        # Выбор модели YOLO
        layout.addWidget(QLabel('Модель YOLO:'))
        self.model_combo = QComboBox()
        self.model_combo.addItems(['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'Своя модель...'])
        self.model_combo.currentIndexChanged.connect(self.load_model)
        layout.addWidget(self.model_combo)
        
        # Метод детекции сетки
        layout.addWidget(QLabel('Метод детекции сетки:'))
        self.net_method_combo = QComboBox()
        self.net_method_combo.addItems(['hough', 'contour', 'color'])
        layout.addWidget(self.net_method_combo)
        
        # Кнопка анализа
        self.analyze_btn = QPushButton('Анализировать')
        self.analyze_btn.clicked.connect(self.analyze_position)
        self.analyze_btn.setEnabled(False)
        layout.addWidget(self.analyze_btn)
        
        # Слайдер уверенности
        layout.addWidget(QLabel('Порог уверенности:'))
        self.confidence_slider = QSlider(Qt.Horizontal)
        self.confidence_slider.setMinimum(10)
        self.confidence_slider.setMaximum(95)
        self.confidence_slider.setValue(50)
        self.confidence_slider.setTickPosition(QSlider.TicksBelow)
        self.confidence_slider.setTickInterval(10)
        layout.addWidget(self.confidence_slider)
        
        self.confidence_label = QLabel('0.50')
        layout.addWidget(self.confidence_label)
        self.confidence_slider.valueChanged.connect(
            lambda v: self.confidence_label.setText(f'{v/100:.2f}')
        )
        
        group_box.setLayout(layout)
        return group_box
    
    def _create_image_panel(self) -> QHBoxLayout:
        """Создать панель изображений"""
        layout = QHBoxLayout()
        
        # Исходное изображение
        left_layout = QVBoxLayout()
        left_layout.addWidget(QLabel('Исходное изображение:'))
        self.original_label = QLabel()
        self.original_label.setMinimumSize(500, 400)
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_label.setStyleSheet("border: 1px solid black; background-color: #f0f0f0;")
        left_layout.addWidget(self.original_label)
        layout.addLayout(left_layout)
        
        # Результат
        right_layout = QVBoxLayout()
        right_layout.addWidget(QLabel('Результат:'))
        self.result_label = QLabel()
        self.result_label.setMinimumSize(500, 400)
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("border: 1px solid black; background-color: #f0f0f0;")
        right_layout.addWidget(self.result_label)
        layout.addLayout(right_layout)
        
        return layout
    
    def _create_result_panel(self) -> QGroupBox:
        """Создать панель результатов"""
        group_box = QGroupBox("Результаты анализа")
        layout = QHBoxLayout()
        
        self.result_text = QLabel('Загрузите изображение и нажмите "Анализировать"')
        self.result_text.setStyleSheet("font-size: 14pt; padding: 10px;")
        self.result_text.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.result_text)
        
        group_box.setLayout(layout)
        return group_box
    
    def load_model(self):
        """Загрузить модель YOLO"""
        model_name = self.model_combo.currentText()
        
        if model_name == 'Своя модель...':
            file_name, _ = QFileDialog.getOpenFileName(
                self, 'Выберите модель YOLO', '', 'Model Files (*.pt *.pth)'
            )
            if file_name:
                model_name = file_name
            else:
                return
        
        try:
            self.ball_detector = BallDetector(model_name)
            self.result_text.setText(f'Модель {model_name} загружена успешно')
        except Exception as e:
            self.result_text.setText(f'Ошибка загрузки модели: {str(e)}')
    
    def load_image(self):
        """Загрузить изображение"""
        file_name, _ = QFileDialog.getOpenFileName(
            self, 'Выберите изображение', '', 
            'Images (*.png *.jpg *.jpeg *.bmp)'
        )
        
        if file_name:
            self.image = cv2.imread(file_name)
            if self.image is not None:
                self.display_image(self.image, self.original_label)
                self.analyze_btn.setEnabled(True)
                self.result_text.setText('Изображение загружено. Нажмите "Анализировать"')
                
                # Автоматически загружаем модель, если еще не загружена
                if self.ball_detector is None:
                    self.load_model()
    
    def load_video(self):
        """Загрузить и обработать видео"""
        file_name, _ = QFileDialog.getOpenFileName(
            self, 'Выберите видео', '', 
            'Videos (*.mp4 *.avi *.mov *.mkv)'
        )
        
        if file_name:
            self.process_video(file_name)
    
    def display_image(self, image: np.ndarray, label: QLabel):
        """Отобразить изображение в label"""
        # Конвертируем BGR в RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        
        # Создаем QImage
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Масштабируем под размер label
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(
            label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        label.setPixmap(scaled_pixmap)
    
    def analyze_position(self):
        """Анализировать положение мяча"""
        if self.image is None:
            self.result_text.setText('Сначала загрузите изображение')
            return
        
        if self.ball_detector is None:
            self.load_model()
            if self.ball_detector is None:
                self.result_text.setText('Не удалось загрузить модель YOLO')
                return
        
        try:
            # Детекция мяча
            confidence = self.confidence_slider.value() / 100
            balls = self.ball_detector.detect(self.image, confidence)
            
            if not balls:
                self.result_text.setText('Мяч не обнаружен на изображении')
                return
            
            # Берем мяч с максимальной уверенностью
            ball_bbox = max(balls, key=lambda x: x[4])
            ball_center = self.ball_detector.get_ball_center(ball_bbox)
            
            # Детекция сетки
            net_method = self.net_method_combo.currentText()
            net_line = self.net_detector.detect_net_line(self.image, net_method)
            
            if net_line is None:
                self.result_text.setText('Сетка не обнаружена. Попробуйте другой метод детекции.')
                # Все равно показываем мяч
                result_img = self.image.copy()
                x1, y1, x2, y2, conf = ball_bbox
                cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(result_img, f'Ball: {conf:.2f}', (x1, y1 - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                self.display_image(result_img, self.result_label)
                return
            
            net_x = self.net_detector.get_net_x_position(net_line)
            
            # Анализ положения
            position, conf = self.position_analyzer.analyze_with_depth(
                ball_center, ball_bbox, net_x, self.image
            )
            
            # Визуализация
            result_img = self.position_analyzer.visualize_result(
                self.image, ball_bbox, net_line, position, conf
            )
            
            self.display_image(result_img, self.result_label)
            
            # Обновляем текст результата
            result_str = f'Мяч находится: {position.value} (уверенность: {conf:.2%})'
            if len(balls) > 1:
                result_str += f'\n(Обнаружено мячей: {len(balls)})'
            self.result_text.setText(result_str)
            
        except Exception as e:
            self.result_text.setText(f'Ошибка при анализе: {str(e)}')
            import traceback
            traceback.print_exc()
    
    def process_video(self, video_path: str):
        """Обработать видео"""
        if self.ball_detector is None:
            self.load_model()
        
        cap = cv2.VideoCapture(video_path)
        
        # Получаем параметры видео
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Создаем writer для сохранения результата
        output_path = video_path.rsplit('.', 1)[0] + '_analyzed.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        confidence = self.confidence_slider.value() / 100
        net_method = self.net_method_combo.currentText()
        
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Детекция мяча
            balls = self.ball_detector.detect(frame, confidence)
            
            result_frame = frame.copy()
            
            if balls:
                ball_bbox = max(balls, key=lambda x: x[4])
                ball_center = self.ball_detector.get_ball_center(ball_bbox)
                
                # Детекция сетки
                net_line = self.net_detector.detect_net_line(frame, net_method)
                
                if net_line is not None:
                    net_x = self.net_detector.get_net_x_position(net_line)
                    position, conf = self.position_analyzer.analyze_with_depth(
                        ball_center, ball_bbox, net_x, frame
                    )
                    
                    result_frame = self.position_analyzer.visualize_result(
                        frame, ball_bbox, net_line, position, conf
                    )
            
            out.write(result_frame)
            
            # Обновляем GUI каждые 30 кадров
            if frame_count % 30 == 0:
                self.display_image(result_frame, self.result_label)
                self.result_text.setText(f'Обработка видео: кадр {frame_count}')
                QApplication.processEvents()
        
        cap.release()
        out.release()
        
        self.result_text.setText(f'Видео обработано и сохранено: {output_path}')


def main():
    """Главная функция"""
    app = QApplication(sys.argv)
    window = BallPositionApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
