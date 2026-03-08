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
        xyz = cloud.get_xyz()
        n_points = len(xyz)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        tree = o3d.geometry.KDTreeFlann(pcd)

        k_search = min(self.k + 1, n_points)
        mean_distances = np.empty(n_points, dtype=np.float64)

        for i in range(n_points):
            _, idx, dists2 = tree.search_knn_vector_3d(xyz[i], k_search)
            dists = np.sqrt(np.asarray(dists2, dtype=np.float64))
            if idx[0] == i:
                dists = dists[1:]
            mean_distances[i] = dists[:self.k].mean() if len(dists) > 0 else np.inf

        # Дальность от сканера (начало координат)
        ranges = np.linalg.norm(xyz, axis=1)

        # Разбиение на бины по дальности
        num_bins = int(self.range_multiplier)  # используем range_multiplier как число бинов
        bin_edges = np.linspace(ranges.min(), ranges.max(), num_bins + 1)

        inlier_mask = np.ones(n_points, dtype=bool)

        for b in range(num_bins):
            in_bin = (ranges >= bin_edges[b]) & (ranges < bin_edges[b + 1])
            if b == num_bins - 1:  # последний бин включает правую границу
                in_bin = (ranges >= bin_edges[b]) & (ranges <= bin_edges[b + 1])

            if in_bin.sum() == 0:
                continue

            local_dists = mean_distances[in_bin]
            finite = np.isfinite(local_dists)
            if not np.any(finite):
                continue

            mu_local = local_dists[finite].mean()
            sigma_local = local_dists[finite].std()
            threshold = mu_local + self.std_ratio * sigma_local

            outliers_in_bin = in_bin & (mean_distances > threshold)
            inlier_mask[outliers_in_bin] = False

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
