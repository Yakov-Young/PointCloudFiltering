import numpy as np
import open3d as o3d
from .base_filter import Filter
from model.point_cloud import PointCloud

class PCACurvatureFilter(Filter):
    """
    Фильтр на основе PCA-кривизны (surface variation).
    Для каждой точки оценивается кривизна σ = λ_min / (λ1+λ2+λ3),
    где λ — собственные числа ковариационной матрицы k соседей.
    Точки с кривизной выше заданного процентиля считаются выбросами и удаляются.
    """
    
    def __init__(self, k: int = 30, percentile: float = 97.0):
        """
        k: число соседей для оценки ковариации.
        percentile: процентиль кривизны для порога (точки выше этого процентиля удаляются).
        """
        super().__init__("PCA Curvature Filter")
        self.k = k
        self.percentile = percentile
        self.last_mask = None  # маска inliers (оставшиеся точки)

    def apply(self, cloud: PointCloud) -> PointCloud:
        if len(cloud) == 0:
            return cloud

        # Создаем Open3D PointCloud для использования estimate_covariances
        o3d_cloud = o3d.geometry.PointCloud()
        o3d_cloud.points = o3d.utility.Vector3dVector(cloud.get_xyz())
        
        # Поиск k соседей и вычисление ковариационных матриц
        search_param = o3d.geometry.KDTreeSearchParamKNN(knn=self.k)
        o3d_cloud.estimate_covariances(search_param)  # заполняет .covariances
        
        # Получаем ковариационные матрицы (N, 3, 3)
        covariances = np.asarray(o3d_cloud.covariances)
        
        # Вычисляем собственные числа (векторизованно, сортировка по возрастанию)
        eigenvalues = np.linalg.eigvalsh(covariances)  # (N, 3)
        
        # Кривизна = λ_min / (сумма λ) (добавляем эпсилон для избегания деления на 0)
        eps = 1e-12
        curvature = eigenvalues[:, 0] / (eigenvalues.sum(axis=1) + eps)
        
        # Определяем порог как заданный процентиль
        thr = np.percentile(curvature, self.percentile)
        
        # Маска inliers: точки с кривизной <= порога
        inlier_mask = curvature <= thr
        self.last_mask = inlier_mask
        
        # Индексы inliers
        inlier_indices = np.where(inlier_mask)[0]
        filtered_data = cloud.points[inlier_indices].copy()
        filtered_indices = cloud.original_indices[inlier_indices].copy()
        return PointCloud(filtered_data, filtered_indices)

    def get_parameters(self) -> dict:
        return {
            "k": self.k,
            "percentile": self.percentile
        }

    def set_parameters(self, **kwargs):
        if "k" in kwargs:
            self.k = int(kwargs["k"])
        if "percentile" in kwargs:
            self.percentile = float(kwargs["percentile"])