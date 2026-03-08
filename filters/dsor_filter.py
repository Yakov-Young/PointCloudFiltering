import open3d as o3d
import numpy as np
from .base_filter import Filter
from model.point_cloud import PointCloud


class DSORFilter(Filter):
    """
    Dynamic Statistical Outlier Removal (DSOR).

    Для каждой точки:
    - считаем среднюю дистанцию до k ближайших соседей;
    - считаем глобальный порог как в SOR;
    - увеличиваем этот порог с расстоянием до сенсора.

    Точка остается, если ее средняя дистанция до соседей
    не превышает динамический порог.
    """

    def __init__(
        self,
        k: int = 20,
        std_ratio: float = 1.0,
        range_multiplier: float = 0.02,
    ):
        super().__init__("Dynamic SOR")
        self.k = k
        self.std_ratio = std_ratio
        self.range_multiplier = range_multiplier
        self.last_mask = None

    def apply(self, cloud: PointCloud) -> PointCloud:
        if len(cloud) == 0:
            self.last_mask = np.array([], dtype=bool)
            return cloud

        xyz = cloud.get_xyz()
        n_points = len(xyz)

        if n_points <= 1:
            self.last_mask = np.ones(n_points, dtype=bool)
            return PointCloud(cloud.points.copy(), cloud.original_indices.copy())

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        tree = o3d.geometry.KDTreeFlann(pcd)

        k_search = min(self.k + 1, n_points)
        mean_distances = np.empty(n_points, dtype=np.float64)

        for i in range(n_points):
            _, idx, dists2 = tree.search_knn_vector_3d(xyz[i], k_search)

            if len(idx) <= 1:
                mean_distances[i] = np.inf
                continue

            idx = np.asarray(idx)
            dists = np.sqrt(np.asarray(dists2, dtype=np.float64))

            if idx[0] == i:
                dists = dists[1:]
            else:
                dists = dists[:self.k]

            if len(dists) == 0:
                mean_distances[i] = np.inf
            else:
                mean_distances[i] = dists[:self.k].mean()

        finite_mask = np.isfinite(mean_distances)
        if not np.any(finite_mask):
            self.last_mask = np.zeros(n_points, dtype=bool)
            return PointCloud(cloud.points[:0].copy(), cloud.original_indices[:0].copy())

        mu = mean_distances[finite_mask].mean()
        sigma = mean_distances[finite_mask].std()
        global_threshold = mu + self.std_ratio * sigma

        ranges = np.linalg.norm(xyz, axis=1)

        dynamic_thresholds = global_threshold * (1.0 + self.range_multiplier * ranges)

        inlier_mask = mean_distances <= dynamic_thresholds
        self.last_mask = inlier_mask

        inlier_indices = np.where(inlier_mask)[0]
        filtered_data = cloud.points[inlier_indices].copy()
        filtered_indices = cloud.original_indices[inlier_indices].copy()

        return PointCloud(filtered_data, filtered_indices)

    def get_parameters(self) -> dict:
        return {
            "k": self.k,
            "std_ratio": self.std_ratio,
            "range_multiplier": self.range_multiplier,
        }

    def set_parameters(self, **kwargs):
        if "k" in kwargs:
            self.k = int(kwargs["k"])
        if "std_ratio" in kwargs:
            self.std_ratio = float(kwargs["std_ratio"])
        if "range_multiplier" in kwargs:
            self.range_multiplier = float(kwargs["range_multiplier"])
