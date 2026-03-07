from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QListWidget,
                             QPushButton, QListWidgetItem, QMessageBox)
from PyQt6.QtCore import Qt
from view.filter_dialog import FilterDialog
from filters.base_filter import Filter

class PipelineDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Управление пайплайном фильтров")
        self.setModal(True)
        self.resize(500, 400)

        self.filters = []  # список экземпляров фильтров

        layout = QVBoxLayout(self)

        # Список фильтров
        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        layout.addWidget(self.list_widget)

        # Кнопки управления
        btn_layout = QHBoxLayout()
        self.btn_add = QPushButton("Добавить фильтр")
        self.btn_add.clicked.connect(self.add_filter)
        btn_layout.addWidget(self.btn_add)

        self.btn_remove = QPushButton("Удалить")
        self.btn_remove.clicked.connect(self.remove_filter)
        btn_layout.addWidget(self.btn_remove)

        self.btn_clear = QPushButton("Очистить")
        self.btn_clear.clicked.connect(self.clear_filters)
        btn_layout.addWidget(self.btn_clear)

        layout.addLayout(btn_layout)

        # Кнопки диалога
        self.btn_apply = QPushButton("Применить все")
        self.btn_apply.clicked.connect(self.accept)  # accept закроет диалог, а мы потом обработаем
        self.btn_cancel = QPushButton("Отмена")
        self.btn_cancel.clicked.connect(self.reject)

        dialog_btn_layout = QHBoxLayout()
        dialog_btn_layout.addWidget(self.btn_apply)
        dialog_btn_layout.addWidget(self.btn_cancel)
        layout.addLayout(dialog_btn_layout)

        self.update_list()

    def add_filter(self):
        dlg = FilterDialog(self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            filter_instance = dlg.get_filter()
            if filter_instance:
                self.filters.append(filter_instance)
                self.update_list()

    def remove_filter(self):
        current_row = self.list_widget.currentRow()
        if current_row >= 0:
            del self.filters[current_row]
            self.update_list()

    def clear_filters(self):
        self.filters.clear()
        self.update_list()

    def update_list(self):
        self.list_widget.clear()
        for i, f in enumerate(self.filters):
            params = f.get_parameters()
            param_str = ", ".join(f"{k}={v}" for k, v in params.items())
            item_text = f"{i+1}. {f.name} [{param_str}]"
            self.list_widget.addItem(item_text)

    def get_pipeline(self):
        """Возвращает список фильтров для применения."""
        return self.filters