import open3d as o3d
import numpy as np
from .base_filter import Filter
from model.point_cloud import PointCloud

class RadiusOutlierFilter(Filter):
    """Удаление выбросов на основе количества соседей в заданном радиусе."""
    
    def __init__(self, radius: float = 0.050, min_neighbors: int = 10):
        super().__init__("Radius Outlier Removal")
        self.radius = radius
        self.min_neighbors = min_neighbors
        self.last_mask = None  # для хранения маски последнего применения

    def apply(self, cloud: PointCloud) -> PointCloud:
        # Конвертируем в формат Open3D
        o3d_cloud = o3d.geometry.PointCloud()
        o3d_cloud.points = o3d.utility.Vector3dVector(cloud.get_xyz())
        
        # Применяем фильтр
        _, inlier_indices = o3d_cloud.remove_radius_outlier(
            nb_points=self.min_neighbors,
            radius=self.radius
        )
        
        # Создаём маску: True для inliers (оставлены), False для outliers (удалены)
        mask = np.zeros(len(cloud), dtype=bool)
        mask[inlier_indices] = True
        self.last_mask = mask  # сохраняем
        
        # Создаём новое облако из inliers
        filtered_data = cloud.points[inlier_indices].copy()
        filtered_indices = cloud.original_indices[inlier_indices].copy()  # сохраняем исходные индексы
        return PointCloud(filtered_data, filtered_indices)

    def get_parameters(self) -> dict:
        return {
            "radius": self.radius,
            "min_neighbors": self.min_neighbors
        }

    def set_parameters(self, **kwargs):
        if "radius" in kwargs:
            self.radius = float(kwargs["radius"])
        if "min_neighbors" in kwargs:
            self.min_neighbors = int(kwargs["min_neighbors"])