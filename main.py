import sys
import numpy as np
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QFileDialog
from view.visualizer_widget import VisualizerWidget
from iot.pointcloud_io import load_from_file
from model.point_cloud import PointCloud

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Point Cloud Filter")
        self.setGeometry(100, 100, 800, 600)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Кнопка загрузки файла
        self.btn_load = QPushButton("Загрузить файл")
        self.btn_load.clicked.connect(self.load_file)
        layout.addWidget(self.btn_load)

        # Кнопка для теста (пока оставим)
        self.btn_test = QPushButton("Тестовое облако")
        self.btn_test.clicked.connect(self.update_cloud)
        layout.addWidget(self.btn_test)

        # Виджет визуализатора
        self.vis_widget = VisualizerWidget()
        layout.addWidget(self.vis_widget)

        self.statusBar().showMessage("Готово")
        self.current_cloud: PointCloud = None

    def load_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Выберите файл облака точек", "",
            "Point Cloud files (*.ply *.pcd *.xyz);;All files (*)"
        )
        if not file_path:
            return

        try:
            self.current_cloud = load_from_file(file_path)
            points = self.current_cloud.get_xyz()
            self.vis_widget.update_cloud(points)
            self.statusBar().showMessage(f"Загружено {len(self.current_cloud)} точек из {file_path}")
        except Exception as e:
            self.statusBar().showMessage(f"Ошибка: {e}")

    def update_cloud(self):
        # Тестовая генерация случайного облака
        points = np.random.rand(1000, 3) * 10
        self.vis_widget.update_cloud(points)
        self.statusBar().showMessage(f"Обновлено тестовое облако: {len(points)} точек")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())