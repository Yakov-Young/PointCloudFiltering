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
        if self.view:
            self.view.update_cloud(self.current_cloud.get_xyz())
            self.view.show_status(f"Сброшено к исходному ({len(self.current_cloud)} точек)")
        return True, f"Сброшено к исходному ({len(self.current_cloud)} точек)"


    def apply_filter(self, filter_instance):
        if self.current_cloud is None:
            msg = "Нет загруженного облака"
            if self.view:
                self.view.show_status(msg)
            return False, msg
        try:
            filtered = filter_instance.apply(self.current_cloud)
            self.current_cloud = filtered
            if self.view:
                self.view.update_cloud(self.current_cloud.get_xyz())
                self.view.show_status(f"Фильтр '{filter_instance.name}' применён. Осталось точек: {len(filtered)}")
            return True, f"Фильтр применён. Осталось точек: {len(filtered)}"
        except Exception as e:
            msg = f"Ошибка при применении фильтра: {e}"
            if self.view:
                self.view.show_status(msg)
            return False, msg

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
        """Выполняет оценку текущего облака относительно исходного."""
        if self.original_cloud is None or self.current_cloud is None:
            msg = "Нет данных для оценки"
            if self.view:
                self.view.show_status(msg)
            return None, msg
        
        try:
            report = EvaluationReport(self.original_cloud, self.current_cloud)
            report.compute_all_metrics(k)
            if self.view:
                self.view.show_status("Оценка завершена")
            return report, None
        except Exception as e:
            msg = f"Ошибка при вычислении метрик: {e}"
            if self.view:
                self.view.show_status(msg)
            return None, msg

    def get_current_cloud(self) -> Optional[PointCloud]:
        return self.current_cloud

    def get_original_cloud(self) -> Optional[PointCloud]:
        return self.original_cloud