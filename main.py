import sys
import numpy as np
from PyQt6.QtWidgets import QApplication, QComboBox, QDialog, QLabel, QMainWindow, QTextEdit, QVBoxLayout, QWidget, QPushButton, QFileDialog, QHBoxLayout
from view.visualizer_widget import VisualizerWidget
from view.filter_dialog import FilterDialog
from controller.main_controller import MainController
from view.pipeline_dialog import PipelineDialog

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

        # Пайплайн
        self.btn_pipeline = QPushButton("Пайплайн")
        self.btn_pipeline.clicked.connect(self.manage_pipeline)
        toolbar.addWidget(self.btn_pipeline)

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

        # Текстовое поле для отчёта
        self.report_text = QTextEdit()
        self.report_text.setReadOnly(True)
        self.report_text.setMaximumHeight(150)
        main_layout.addWidget(self.report_text)
        
        # Переключатель
        toolbar.addWidget(QLabel("  Режим:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Однотонный", "Цвета", "Выбросы"])
        self.mode_combo.currentTextChanged.connect(self.on_mode_changed)
        #self.mode_combo.setCurrentText("Цвета")
        toolbar.addWidget(self.mode_combo)

        self.statusBar().showMessage("Готово")

    # --- Методы обратного вызова (вызываются контроллером) ---
    #def update_visualizer(self, points):
    #    """Обновляет отображение облака в визуализаторе."""
    #    self.vis_widget.update_cloud(points)

    def update_cloud(self, points=None):
        """
        Обновляет визуализатор, используя текущее облако из контроллера.
        Параметр points игнорируется (сохранён для совместимости с вызовами из контроллера).
        """
        if self.controller.current_cloud is not None:
            xyz = self.controller.current_cloud.get_xyz()
            rgb = self.controller.current_cloud.get_rgb()
            self.vis_widget.update_cloud(xyz, rgb)

    def refresh_visualization(self):
        """Обновляет отображение облака с учётом текущего режима окраски."""
        if self.controller.current_cloud is None:
            return

        xyz = self.controller.current_cloud.get_xyz()
        mode = self.mode_combo.currentText()
        cloud = self.controller.current_cloud

        if mode == "Цвета":
            if all(f in cloud.points.dtype.names for f in ('red', 'green', 'blue')):
                colors = np.zeros((len(xyz), 3), dtype=np.float32)
                colors[:, 0] = cloud.points['red'] / 255.0
                colors[:, 1] = cloud.points['green'] / 255.0
                colors[:, 2] = cloud.points['blue'] / 255.0
                if np.all(colors == 0):
                    colors = None
            else:
                colors = None

        elif mode == "Выбросы":
            if 'scalar_isGarbage' in cloud.points.dtype.names:
                garbage = cloud.points['scalar_isGarbage']
                colors = np.zeros((len(xyz), 3), dtype=np.float32)
                # Красный для выбросов (значение 1)
                colors[garbage == 1] = [1.0, 0.0, 0.0]
                # Синий для нормальных точек (значение 0)
                colors[garbage == 0] = [0.0, 0.0, 1.0]
                # Для всех остальных значений (например, NaN или другие числа) — серый
                mask_other = (garbage != 0) & (garbage != 1)
                if np.any(mask_other):
                    colors[mask_other] = [0.5, 0.5, 0.5]
                # Если все цвета нулевые (нет точек с 0 или 1), передаём None
                if np.all(colors == 0):
                    colors = None
            else:
                colors = None

        else:  # Однотонный
            colors = None

        self.vis_widget.update_cloud(xyz, colors)

    def on_mode_changed(self, mode):
        self.refresh_visualization()

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
            self.report_text.clear()  # очищаем отчёт

    def on_reset(self):
        self.controller.reset_to_original()

    def apply_filter_dialog(self):
        dlg = FilterDialog(self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            filter_instance = dlg.get_filter()
            if filter_instance:
                success, msg = self.controller.apply_filter(filter_instance)
                #self.statusBar().showMessage(msg)

    def reset_to_original(self):
        success, msg = self.controller.reset_to_original()
        if success:
            self.report_text.clear()
        self.statusBar().showMessage(msg)

    def manage_pipeline(self):
        dlg = PipelineDialog(self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            filters = dlg.get_pipeline()
            if filters:
                success, msg = self.controller.apply_pipeline(filters)
                self.show_status(msg)
                self.refresh_visualization()

    def on_evaluate(self):
        report, error = self.controller.evaluate()
        if error:
            self.report_text.setText(f"Ошибка: {error}")
            self.show_status(error)
        else:
            self.report_text.setText(report.get_report_string())
            self.show_status("Оценка завершена")

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