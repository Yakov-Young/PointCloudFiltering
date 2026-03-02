import sys
import numpy as np
from PyQt6.QtWidgets import QApplication, QDialog, QMainWindow, QVBoxLayout, QWidget, QPushButton, QFileDialog, QHBoxLayout
from view.visualizer_widget import VisualizerWidget
from view.filter_dialog import FilterDialog
from controller.main_controller import MainController

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Point Cloud Filter")
        self.setGeometry(100, 100, 900, 700)

        # Создаём контроллер и связываем с view
        self.controller = MainController()
        self.controller.set_view(self)

        # Центральный виджет
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Панель инструментов (горизонтальная)
        toolbar = QHBoxLayout()
        self.btn_load = QPushButton("Загрузить")
        self.btn_load.clicked.connect(self.on_load)
        toolbar.addWidget(self.btn_load)

        #self.btn_reset = QPushButton("Сброс")
        #self.btn_reset.clicked.connect(self.on_reset)
        #toolbar.addWidget(self.btn_reset)

        # Также добавим кнопку "Сброс" (вернуться к исходному облаку)
        self.btn_reset = QPushButton("Сбросить к исходному")
        self.btn_reset.clicked.connect(self.reset_to_original)
        toolbar.addWidget(self.btn_reset)

        #self.btn_filter = QPushButton("Фильтр")
        #self.btn_filter.clicked.connect(self.on_filter)
        #toolbar.addWidget(self.btn_filter)

        # В __init__ после других кнопок:
        self.btn_filter = QPushButton("Применить фильтр")
        self.btn_filter.clicked.connect(self.apply_filter_dialog)
        toolbar.addWidget(self.btn_filter)

        self.btn_evaluate = QPushButton("Оценка")
        self.btn_evaluate.clicked.connect(self.on_evaluate)
        toolbar.addWidget(self.btn_evaluate)

        self.btn_save = QPushButton("Сохранить")
        self.btn_save.clicked.connect(self.on_save)
        toolbar.addWidget(self.btn_save)

        main_layout.addLayout(toolbar)

        # Виджет визуализатора
        self.vis_widget = VisualizerWidget()
        main_layout.addWidget(self.vis_widget)

        self.statusBar().showMessage("Готово")

    # --- Методы обратного вызова (вызываются контроллером) ---
    def update_visualizer(self, points):
        """Обновляет отображение облака в визуализаторе."""
        self.vis_widget.update_cloud(points)

    def update_cloud(self, points):
        """Обновляет отображение облака в визуализаторе."""
        self.vis_widget.update_cloud(points)

    def show_status(self, message):
        """Выводит сообщение в строку состояния."""
        self.statusBar().showMessage(message)

    # --- Обработчики событий от кнопок ---
    def on_load(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Выберите файл облака точек", "",
            "Point Cloud files (*.ply *.pcd *.xyz);;All files (*)"
        )
        if file_path:
            self.controller.load_file(file_path)

    def on_reset(self):
        self.controller.reset_to_original()

    def apply_filter_dialog(self):
        dlg = FilterDialog(self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            filter_instance = dlg.get_filter()
            if filter_instance:
                success, msg = self.controller.apply_filter(filter_instance)
                self.statusBar().showMessage(msg)

    def reset_to_original(self):
        success, msg = self.controller.reset_to_original()
        self.statusBar().showMessage(msg)
        def on_filter(self):
            # Пока просто заглушка для демонстрации
            # Здесь будет диалог выбора фильтра
            self.show_status("Выбор фильтра ещё не реализован")

    def on_evaluate(self):
        # Заглушка для оценки
        self.show_status("Оценка ещё не реализована")

    def on_save(self):
        if self.controller.get_current_cloud() is None:
            self.show_status("Нет данных для сохранения")
            return
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить облако", "",
            "PLY files (*.ply);;PCD files (*.pcd);;XYZ files (*.xyz)"
        )
        if file_path:
            self.controller.save_current(file_path)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())