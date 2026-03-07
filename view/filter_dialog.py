# view/filter_dialog.py
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QSpinBox, QDoubleSpinBox, QPushButton, QFormLayout, QDialogButtonBox, QWidget
from filters.statistical_outlier import StatisticalOutlierFilter
from filters.radius_outlier import RadiusOutlierFilter

class FilterDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Выбор фильтра")
        self.setModal(True)
        
        # Список доступных фильтров (добавляем ROR)
        self.filter_classes = {
            "Statistical Outlier Removal": StatisticalOutlierFilter,
            "Radius Outlier Removal": RadiusOutlierFilter
        }
        
        self.selected_filter = None
        self.current_filter_instance = None
        
        self._init_ui()
        
    def _init_ui(self):
        layout = QVBoxLayout(self)
        
        # Выпадающий список для выбора типа фильтра
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Тип фильтра:"))
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(self.filter_classes.keys())
        self.filter_combo.currentTextChanged.connect(self.on_filter_changed)
        filter_layout.addWidget(self.filter_combo)
        layout.addLayout(filter_layout)
        
        # Область для параметров (будет заполняться динамически)
        self.params_widget = QWidget()
        self.params_layout = QFormLayout(self.params_widget)
        layout.addWidget(self.params_widget)
        
        # Кнопки ОК/Отмена
        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)
        
        # Инициализируем первый фильтр
        self.on_filter_changed(self.filter_combo.currentText())
    
    def on_filter_changed(self, filter_name):
        # Очищаем предыдущие параметры
        self._clear_layout(self.params_layout)
        
        # Создаём экземпляр выбранного фильтра
        filter_class = self.filter_classes[filter_name]
        self.current_filter_instance = filter_class()
        
        # Добавляем поля для параметров
        params = self.current_filter_instance.get_parameters()
        self.param_inputs = {}
        for param_name, value in params.items():
            if isinstance(value, int):
                widget = QSpinBox()
                widget.setRange(-1000000, 1000000)  # разумные пределы
                widget.setValue(value)
            elif isinstance(value, float):
                widget = QDoubleSpinBox()
                widget.setRange(-1000000.0, 1000000.0)
                widget.setValue(value)
                widget.setDecimals(3)
            else:
                continue  # неподдерживаемый тип
            self.params_layout.addRow(param_name, widget)
            self.param_inputs[param_name] = widget
    
    def _clear_layout(self, layout):
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
    
    def get_filter(self):
        """Возвращает настроенный экземпляр фильтра."""
        if self.current_filter_instance is None:
            return None
        
        # Собираем значения из полей ввода
        params = {}
        for param_name, widget in self.param_inputs.items():
            params[param_name] = widget.value()
        
        self.current_filter_instance.set_parameters(**params)
        return self.current_filter_instance