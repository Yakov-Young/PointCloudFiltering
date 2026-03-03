# evaluation/report.py
import numpy as np
import open3d as o3d
from model.point_cloud import PointCloud

class EvaluationReport:
    """Отчёт о сравнении исходного и отфильтрованного облаков."""
    
    def __init__(self, original: PointCloud, filtered: PointCloud):
        self.original = original
        self.filtered = filtered
        self.metrics = {}

    def compute_basic_metrics(self):
        """Вычисляет базовые метрики: количество точек, процент удаления."""
        n_orig = len(self.original)
        n_filt = len(self.filtered)
        
        self.metrics['original_count'] = n_orig
        self.metrics['filtered_count'] = n_filt
        self.metrics['removed_count'] = n_orig - n_filt
        self.metrics['removed_percent'] = 100 * (1 - n_filt / n_orig) if n_orig > 0 else 0

    def compute_knn_metrics(self, k=10):
        """
        Вычисляет среднее расстояние до k ближайших соседей для исходного
        и отфильтрованного облаков.
        """
        if len(self.original) == 0 or len(self.filtered) == 0:
            self.metrics['original_mean_knn'] = 0
            self.metrics['filtered_mean_knn'] = 0
            self.metrics['knn_change_percent'] = 0
            return

        def mean_knn_distance(points, k):
            # points: np.ndarray (N,3)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            tree = o3d.geometry.KDTreeFlann(pcd)
            distances = []
            for i in range(len(points)):
                [_, idx, dist_sq] = tree.search_knn_vector_3d(pcd.points[i], k + 1)
                if len(dist_sq) > 1:
                    distances.extend(np.sqrt(dist_sq[1:]))
            if distances:
                return np.mean(distances)
            else:
                return 0.0

        orig_xyz = self.original.get_xyz()
        filt_xyz = self.filtered.get_xyz()
        
        orig_mean = mean_knn_distance(orig_xyz, k)
        filt_mean = mean_knn_distance(filt_xyz, k)
        
        self.metrics['original_mean_knn'] = orig_mean
        self.metrics['filtered_mean_knn'] = filt_mean
        if orig_mean > 0:
            self.metrics['knn_change_percent'] = 100 * (filt_mean - orig_mean) / orig_mean
        else:
            self.metrics['knn_change_percent'] = 0

    def compute_all_metrics(self, k=10):
        """Вычисляет все доступные метрики."""
        self.compute_basic_metrics()
        self.compute_knn_metrics(k)

    def get_report_string(self) -> str:
        """Возвращает форматированную строку с отчётом."""
        lines = ["=== Оценка результатов ==="]
        if 'original_count' in self.metrics:
            lines.append(f"Исходных точек: {self.metrics['original_count']}")
            lines.append(f"После фильтрации: {self.metrics['filtered_count']}")
            lines.append(f"Удалено точек: {self.metrics['removed_count']} ({self.metrics['removed_percent']:.2f}%)")
        if 'original_mean_knn' in self.metrics:
            lines.append(f"Среднее расстояние до {self.metrics.get('k', 10)} соседей (исх.): {self.metrics['original_mean_knn']:.4f}")
            lines.append(f"Среднее расстояние до соседей (фильтр): {self.metrics['filtered_mean_knn']:.4f}")
            lines.append(f"Изменение среднего расстояния: {self.metrics['knn_change_percent']:.2f}%")
        return "\n".join(lines)