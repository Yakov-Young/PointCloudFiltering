from model.point_cloud import PointCloud
from iot.pointcloud_io import load_from_file, save_to_file
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

    def load_file(self, file_path: str) -> bool:
        """Загружает облако из файла, сохраняет как исходное и текущее."""
        try:
            cloud = load_from_file(file_path)
            self.original_cloud = cloud.copy()  # сохраняем копию как исходное
            self.current_cloud = cloud
            if self.view:
                self.view.update_visualizer(self.current_cloud.get_xyz())
                self.view.show_status(f"Загружено {len(cloud)} точек из {file_path}")
            return True
        except Exception as e:
            if self.view:
                self.view.show_status(f"Ошибка загрузки: {e}")
            return False

    def reset_to_original(self):
        """Сбрасывает текущее облако к исходному."""
        if self.original_cloud is not None:
            self.current_cloud = self.original_cloud.copy()
            if self.view:
                self.view.update_visualizer(self.current_cloud.get_xyz())
                self.view.show_status("Сброшено к исходному облаку")


    def apply_filter(self, filter_instance):
        """Применяет фильтр к текущему облаку."""
        if self.current_cloud is None:
            return False, "Нет загруженного облака"
        
        try:
            filtered = filter_instance.apply(self.current_cloud)
            self.current_cloud = filtered
            # Обновляем визуализацию
            self.view.update_cloud(self.current_cloud.get_xyz())
            return True, f"Фильтр '{filter_instance.name}' применён. Осталось точек: {len(filtered)}"
        except Exception as e:
            return False, f"Ошибка при применении фильтра: {e}"

    def save_current(self, file_path: str):
        """Сохраняет текущее облако в файл."""
        if self.current_cloud is None:
            if self.view:
                self.view.show_status("Нет данных для сохранения")
            return
        try:
            save_to_file(self.current_cloud, file_path)
            if self.view:
                self.view.show_status(f"Сохранено в {file_path}")
        except Exception as e:
            if self.view:
                self.view.show_status(f"Ошибка сохранения: {e}")

    def get_current_cloud(self) -> Optional[PointCloud]:
        return self.current_cloud

    def get_original_cloud(self) -> Optional[PointCloud]:
        return self.original_cloud