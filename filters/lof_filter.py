import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from .base_filter import Filter
from model.point_cloud import PointCloud

class LOFilter(Filter):
    """Фильтр на основе Local Outlier Factor (LOF) из scikit-learn."""
    
    def __init__(self, n_neighbors: int = 20, contamination: float = 0.1):
        super().__init__("Local Outlier Factor")
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.last_mask = None  # для хранения маски последнего применения

    def apply(self, cloud: PointCloud) -> PointCloud:
        # Получаем координаты
        xyz = cloud.get_xyz()
        if len(xyz) == 0:
            return cloud
        
        # Применяем LOF
        clf = LocalOutlierFactor(n_neighbors=self.n_neighbors, contamination=self.contamination)
        # Предсказание: -1 для выбросов, 1 для нормальных точек
        y_pred = clf.fit_predict(xyz)
        
        # Создаём маску inliers (точки, которые НЕ выбросы)
        inlier_mask = y_pred == 1
        self.last_mask = inlier_mask  # сохраняем маску оставшихся (inliers)
        
        # Индексы inliers
        inlier_indices = np.where(inlier_mask)[0]
        filtered_data = cloud.points[inlier_indices].copy()
        filtered_indices = cloud.original_indices[inlier_indices].copy()
        
        return PointCloud(filtered_data, filtered_indices)

    def get_parameters(self) -> dict:
        return {
            "n_neighbors": self.n_neighbors,
            "contamination": self.contamination
        }

    def set_parameters(self, **kwargs):
        if "n_neighbors" in kwargs:
            self.n_neighbors = int(kwargs["n_neighbors"])
        if "contamination" in kwargs:
            self.contamination = float(kwargs["contamination"])