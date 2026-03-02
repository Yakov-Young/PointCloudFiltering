# filters/statistical_outlier.py
import open3d as o3d
import numpy as np
from .base_filter import Filter
from model.point_cloud import PointCloud

class StatisticalOutlierFilter(Filter):
    """Удаление выбросов на основе статистического анализа расстояний до соседей."""
    
    def __init__(self, nb_neighbors: int = 20, std_ratio: float = 1.0):
        super().__init__("Statistical Outlier Removal")
        self.nb_neighbors = nb_neighbors
        self.std_ratio = std_ratio

    def apply(self, cloud: PointCloud) -> PointCloud:
        # Конвертируем в формат Open3D
        o3d_cloud = o3d.geometry.PointCloud()
        o3d_cloud.points = o3d.utility.Vector3dVector(cloud.get_xyz())
        
        # Применяем фильтр
        _, inlier_indices = o3d_cloud.remove_statistical_outlier(
            nb_neighbors=self.nb_neighbors,
            std_ratio=self.std_ratio
        )
        
        # Создаём новое облако из индексов inliers
        filtered_data = cloud.points[inlier_indices].copy()
        return PointCloud(filtered_data)

    def get_parameters(self) -> dict:
        return {
            "nb_neighbors": self.nb_neighbors,
            "std_ratio": self.std_ratio
        }

    def set_parameters(self, **kwargs):
        if "nb_neighbors" in kwargs:
            self.nb_neighbors = int(kwargs["nb_neighbors"])
        if "std_ratio" in kwargs:
            self.std_ratio = float(kwargs["std_ratio"])