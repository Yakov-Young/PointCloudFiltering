import open3d as o3d
import numpy as np
from .base_filter import Filter
from model.point_cloud import PointCloud


class DSORFilter(Filter):
    """
    Classical Dynamic Statistical Outlier Removal (DSOR).

    Для каждой точки:
    - вычисляется средняя дистанция до k ближайших соседей;
    - по всему облаку считается глобальный порог SOR:
          T_g = mu + std_ratio * sigma
    - для каждой точки строится динамический порог:
          T_i = T_g * range_multiplier * d_i
      где d_i — расстояние точки до сенсора (начала координат).

    Точка сохраняется, если ее средняя дистанция до соседей
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
        xyz = cloud.get_xyz()
        n_points = len(xyz)

        if n_points == 0:
            self.last_mask = np.array([], dtype=bool)
            return cloud

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        tree = o3d.geometry.KDTreeFlann(pcd)

        k_search = min(self.k + 1, n_points)
        mean_distances = np.full(n_points, np.inf, dtype=np.float64)

        for i in range(n_points):
            _, idx, dists2 = tree.search_knn_vector_3d(xyz[i], k_search)

            if len(dists2) == 0:
                continue

            idx = np.asarray(idx)
            dists = np.sqrt(np.asarray(dists2, dtype=np.float64))

            if len(idx) > 0 and idx[0] == i:
                dists = dists[1:]

            if dists.size > 0:
                mean_distances[i] = dists[:self.k].mean()

        finite_mask = np.isfinite(mean_distances)

        if not np.any(finite_mask):
            self.last_mask = np.zeros(n_points, dtype=bool)
            empty_indices = np.array([], dtype=int)
            filtered_data = cloud.points[empty_indices].copy()
            filtered_indices = cloud.original_indices[empty_indices].copy()
            return PointCloud(filtered_data, filtered_indices)

        mu = mean_distances[finite_mask].mean()
        sigma = mean_distances[finite_mask].std()
        global_threshold = mu + self.std_ratio * sigma

        ranges = np.linalg.norm(xyz, axis=1)
        dynamic_thresholds = global_threshold * self.range_multiplier * ranges

        inlier_mask = finite_mask & (mean_distances <= dynamic_thresholds)
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