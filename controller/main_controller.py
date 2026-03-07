import numpy as np
from model.point_cloud import PointCloud
from iot.pointcloud_io import load_from_file, save_to_file
from evaluation.report import EvaluationReport
from typing import Optional

class MainController:
    """Контроллер приложения. Управляет данными и взаимодействием с моделью."""

    def __init__(self):
        self.original_cloud: Optional[PointCloud] = None  # исходное облако (неизменяемое)
        self.current_cloud: Optional[PointCloud] = None   # текущее облако (после фильтров)
        self.view = None  # будет установлен позже для обновления интерфейса
        self.last_removal_mask = None  # маска удаления после последнего фильтра

    def set_view(self, view):
        """Устанавливает ссылку на главное окно для обратной связи."""
        self.view = view

    def load_file(self, path):
        try:
            cloud = load_from_file(path)
            self.original_cloud = cloud
            self.current_cloud = cloud.copy()
            if self.view:
                self.view.update_cloud(self.current_cloud.get_xyz())
                self.view.show_status(f"Загружено {len(cloud)} точек из {path}")
            return True, f"Загружено {len(cloud)} точек"
        except Exception as e:
            if self.view:
                self.view.show_status(f"Ошибка загрузки: {e}")
            return False, str(e)

    def reset_to_original(self):
        if self.original_cloud is None:
            msg = "Нет исходного облака"
            if self.view:
                self.view.show_status(msg)
            return False, msg
        self.current_cloud = self.original_cloud.copy()
        self.last_removal_mask = None  # сбрасываем маску
        if self.view:
            self.view.update_cloud(self.current_cloud.get_xyz())
            self.view.show_status(f"Сброшено к исходному ({len(self.current_cloud)} точек)")
        return True, f"Сброшено к исходному ({len(self.current_cloud)} точек)"


    def apply_filter(self, filter_instance):
        if self.current_cloud is None:
            return False, "Нет загруженного облака"
        try:
            filtered = filter_instance.apply(self.current_cloud)
            self.current_cloud = filtered
            # Маска удаления для исходного облака (для оценки) будет вычислена в evaluate
            if self.view:
                self.view.update_cloud(self.current_cloud.get_xyz())
                self.view.show_status(f"Фильтр '{filter_instance.name}' применён. Осталось точек: {len(filtered)}")
            return True, f"Фильтр применён. Осталось точек: {len(filtered)}"
        except Exception as e:
            return False, f"Ошибка при применении фильтра: {e}"
        
    def apply_pipeline(self, filters):
        if self.current_cloud is None:
            return False, "Нет загруженного облака"
        
        current = self.current_cloud
        for filter_instance in filters:
            try:
                current = filter_instance.apply(current)
            except Exception as e:
                return False, f"Ошибка при применении фильтра {filter_instance.name}: {e}"
        
        self.current_cloud = current
        if self.view:
            self.view.update_cloud(self.current_cloud.get_xyz())
            self.view.show_status(f"Применено {len(filters)} фильтров. Осталось точек: {len(current)}")
        return True, f"Применено {len(filters)} фильтров. Осталось точек: {len(current)}"
    
    def save_current(self, path):
        if self.current_cloud is None:
            msg = "Нет данных для сохранения"
            if self.view:
                self.view.show_status(msg)
            return False, msg
        try:
            save_to_file(self.current_cloud, path)
            msg = f"Сохранено в {path}"
            if self.view:
                self.view.show_status(msg)
            return True, msg
        except Exception as e:
            msg = f"Ошибка сохранения: {e}"
            if self.view:
                self.view.show_status(msg)
            return False, msg
        
    def evaluate(self, k=10):
        if self.original_cloud is None or self.current_cloud is None:
            return None, "Нет данных для оценки"
        try:
            report = EvaluationReport(self.original_cloud, self.current_cloud)
            report.compute_basic_metrics()
            report.compute_knn_metrics(k)
            
            # Вычисляем маску удаления для исходного облака
            if self.original_cloud is not None and self.current_cloud is not None:
                removal_mask = np.ones(len(self.original_cloud), dtype=bool)
                # Индексы точек, которые остались в current_cloud (в исходном облаке)
                survived_indices = self.current_cloud.original_indices
                removal_mask[survived_indices] = False  # эти точки не удалены
                report.compute_classification_metrics(removal_mask)
            
            return report, None
        except Exception as e:
            return None, f"Ошибка при вычислении метрик: {e}"

    def get_current_cloud(self) -> Optional[PointCloud]:
        return self.current_cloud

    def get_original_cloud(self) -> Optional[PointCloud]:
        return self.original_cloud