# evaluation/report.py
import numpy as np
import open3d as o3d
from model.point_cloud import PointCloud
from scipy.spatial import cKDTree

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

    def compute_classification_metrics(self, removal_mask):
        """
        Вычисляет precision, recall, F1 на основе ground truth (scalar_isGarbage).
        removal_mask: булев массив длины len(original), где True означает, что точка была удалена (предсказан как выброс).
        """
        if 'scalar_isGarbage' not in self.original.points.dtype.names:
            return  # поле отсутствует – ничего не делаем

        gt = self.original.points['scalar_isGarbage']
        pred = removal_mask.astype(int)  # 1 – удалена (выброс), 0 – оставлена

        tp = np.sum((gt == 1) & (pred == 1))
        fp = np.sum((gt == 0) & (pred == 1))
        fn = np.sum((gt == 1) & (pred == 0))
        tn = np.sum((gt == 0) & (pred == 0))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        self.metrics['precision'] = precision
        self.metrics['recall'] = recall
        self.metrics['f1'] = f1
        self.metrics['tp'] = int(tp)
        self.metrics['fp'] = int(fp)
        self.metrics['fn'] = int(fn)
        self.metrics['tn'] = int(tn)

    from scipy.spatial import cKDTree

    def compute_knn_metrics(self, k=10, n_jobs=8):
        if len(self.original) == 0 or len(self.filtered) == 0:
            self.metrics['original_mean_knn'] = 0
            self.metrics['filtered_mean_knn'] = 0
            self.metrics['knn_change_percent'] = 0
            return

        def mean_knn_distance_parallel(points, k, n_jobs):
            tree = cKDTree(points)
            # query возвращает расстояния и индексы для k ближайших соседей, включая саму точку
            distances, _ = tree.query(points, k=k+1, workers=6)
            # исключаем расстояние до самой точки (первый столбец)
            mean_dist = np.mean(distances[:, 1:])
            return mean_dist

        orig_xyz = self.original.get_xyz()
        filt_xyz = self.filtered.get_xyz()
        
        orig_mean = mean_knn_distance_parallel(orig_xyz, k, n_jobs)
        filt_mean = mean_knn_distance_parallel(filt_xyz, k, n_jobs)
        
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
        lines = ["=== Оценка результатов ==="]
        if 'original_count' in self.metrics:
            lines.append(f"Исходных точек: {self.metrics['original_count']}")
            lines.append(f"После фильтрации: {self.metrics['filtered_count']}")
            lines.append(f"Удалено точек: {self.metrics['removed_count']} ({self.metrics['removed_percent']:.2f}%)")
        if 'original_mean_knn' in self.metrics:
            lines.append(f"Среднее расстояние до 10 соседей (исх.): {self.metrics['original_mean_knn']:.4f}")
            lines.append(f"Среднее расстояние до соседей (фильтр): {self.metrics['filtered_mean_knn']:.4f}")
            lines.append(f"Изменение среднего расстояния: {self.metrics['knn_change_percent']:.2f}%")
        if 'precision' in self.metrics:
            lines.append(f"Точность (Precision): {self.metrics['precision']:.4f}")
            lines.append(f"Полнота (Recall): {self.metrics['recall']:.4f}")
            lines.append(f"F1-мера: {self.metrics['f1']:.4f}")
            lines.append(f"TP: {self.metrics['tp']}, FP: {self.metrics['fp']}, FN: {self.metrics['fn']}, TN: {self.metrics['tn']}")
        return "\n".join(lines)