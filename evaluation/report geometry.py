import numpy as np
from scipy.spatial import cKDTree
from model.point_cloud import PointCloud

class EvaluationReport:
    def __init__(self, original: PointCloud, filtered: PointCloud):
        self.original = original
        self.filtered = filtered
        self.metrics = {}

    def compute_basic_metrics(self):
        n_orig = len(self.original)
        n_filt = len(self.filtered)
        self.metrics['original_count'] = n_orig
        self.metrics['filtered_count'] = n_filt
        self.metrics['removed_count'] = n_orig - n_filt
        self.metrics['removed_percent'] = 100 * (1 - n_filt / n_orig) if n_orig > 0 else 0

    def compute_knn_metrics(self, k=10, n_jobs=8):
        if len(self.original) == 0 or len(self.filtered) == 0:
            self._set_knn_defaults(k)
            return
        def stats(points, k, n_jobs):
            tree = cKDTree(points)
            dist, _ = tree.query(points, k=k+1, workers=n_jobs)
            neighbor_dists = dist[:, 1:]
            mean_per_point = np.mean(neighbor_dists, axis=1)
            return np.mean(mean_per_point), np.std(mean_per_point)
        orig_xyz = self.original.get_xyz()
        filt_xyz = self.filtered.get_xyz()
        orig_mean, orig_std = stats(orig_xyz, k, n_jobs)
        filt_mean, filt_std = stats(filt_xyz, k, n_jobs)
        self.metrics['original_mean_knn'] = orig_mean
        self.metrics['original_std_knn'] = orig_std
        self.metrics['filtered_mean_knn'] = filt_mean
        self.metrics['filtered_std_knn'] = filt_std
        self.metrics['knn_k'] = k
        self.metrics['knn_change_percent'] = 100 * (filt_mean - orig_mean) / orig_mean if orig_mean > 0 else 0

    def compute_roughness_and_planarity(self, k=30, n_jobs=8):
        """
        Вычисляет roughness и planarity для исходного и отфильтрованного облаков.
        roughness: среднее и 95-й перцентиль расстояний от точки до аппроксимирующей плоскости.
        planarity: среднее значение (λ2 - λ3) / λ1.
        """
        if len(self.original) == 0 or len(self.filtered) == 0:
            self._set_roughness_planarity_defaults()
            return

        def compute_stats(points, k, n_jobs):
            tree = cKDTree(points)
            n = len(points)
            roughness_per_point = np.zeros(n)
            planarity_per_point = np.zeros(n)

            # Для каждой точки ищем k ближайших соседей (включая саму себя)
            dist, idx = tree.query(points, k=k+1, workers=n_jobs)
            # idx имеет размер (n, k+1)

            for i in range(n):
                neighbor_indices = idx[i, 1:]  # исключаем саму точку
                neighbors = points[neighbor_indices]  # (k, 3)
                # Центроид
                centroid = np.mean(neighbors, axis=0)
                # Центрированные координаты
                centered = neighbors - centroid
                # Ковариационная матрица (3x3)
                cov = np.dot(centered.T, centered) / (k - 1) if k > 1 else np.eye(3)
                # Собственные значения (сортировка по возрастанию)
                evals = np.linalg.eigvalsh(cov)
                evals = np.sort(evals)[::-1]  # λ1 ≥ λ2 ≥ λ3
                λ1, λ2, λ3 = evals[0], evals[1], evals[2]
                # Planarity
                if λ1 > 0:
                    planarity = (λ2 - λ3) / λ1
                else:
                    planarity = 0.0
                # Roughness = расстояние от точки до аппроксимирующей плоскости
                # Плоскость задана нормалью = собственный вектор, соответствующий λ3
                # Но для расстояния можно использовать проекцию на третий собственный вектор
                # Однако проще: расстояние до плоскости = |(point - centroid)·n|, где n — единичная нормаль
                # Получим третий собственный вектор (соответствует λ3)
                _, eigvecs = np.linalg.eigh(cov)
                n3 = eigvecs[:, 0]  # наименьшее собственное значение -> первый столбец (т.к. eigh возвращает по возрастанию)
                # Нормализация уже выполнена
                point_vec = points[i] - centroid
                roughness = abs(np.dot(point_vec, n3))
                roughness_per_point[i] = roughness
                planarity_per_point[i] = planarity

            mean_roughness = np.mean(roughness_per_point)
            percentile_95_roughness = np.percentile(roughness_per_point, 95)
            mean_planarity = np.mean(planarity_per_point)
            return mean_roughness, percentile_95_roughness, mean_planarity

        orig_xyz = self.original.get_xyz()
        filt_xyz = self.filtered.get_xyz()

        orig_mean_r, orig_p95_r, orig_mean_p = compute_stats(orig_xyz, k, n_jobs)
        filt_mean_r, filt_p95_r, filt_mean_p = compute_stats(filt_xyz, k, n_jobs)

        self.metrics['roughness_k'] = k
        self.metrics['original_mean_roughness'] = orig_mean_r
        self.metrics['original_95p_roughness'] = orig_p95_r
        self.metrics['filtered_mean_roughness'] = filt_mean_r
        self.metrics['filtered_95p_roughness'] = filt_p95_r
        self.metrics['original_mean_planarity'] = orig_mean_p
        self.metrics['filtered_mean_planarity'] = filt_mean_p
        # Относительное изменение
        if orig_mean_r > 0:
            self.metrics['roughness_change_percent'] = 100 * (filt_mean_r - orig_mean_r) / orig_mean_r
        else:
            self.metrics['roughness_change_percent'] = 0.0
        self.metrics['planarity_change_percent'] = 100 * (filt_mean_p - orig_mean_p) / orig_mean_p if orig_mean_p > 0 else 0.0

    def compute_classification_metrics(self, removal_mask):
        if 'scalar_isGarbage' not in self.original.points.dtype.names:
            return
        gt = self.original.points['scalar_isGarbage']
        pred = removal_mask.astype(int)
        tp = np.sum((gt == 1) & (pred == 1))
        fp = np.sum((gt == 0) & (pred == 1))
        fn = np.sum((gt == 1) & (pred == 0))
        tn = np.sum((gt == 0) & (pred == 0))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        self.metrics.update({
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn),
            'tn': int(tn)
        })

    def compute_all_metrics(self, k_knn=10, k_roughness=30, n_jobs=8):
        self.compute_basic_metrics()
        self.compute_knn_metrics(k_knn, n_jobs)
        self.compute_roughness_and_planarity(k_roughness, n_jobs)

    def get_report_string(self) -> str:
        lines = ["=== Оценка результатов ==="]
        if 'original_count' in self.metrics:
            lines.append(f"Исходных точек: {self.metrics['original_count']}")
            lines.append(f"После фильтрации: {self.metrics['filtered_count']}")
            lines.append(f"Удалено точек: {self.metrics['removed_count']} ({self.metrics['removed_percent']:.2f}%)")

        if 'original_mean_knn' in self.metrics:
            k = self.metrics.get('knn_k', 10)
            lines.append(f"Среднее расстояние до {k} соседей (исх.): {self.metrics['original_mean_knn']:.6f} ± {self.metrics['original_std_knn']:.6f}")
            lines.append(f"Среднее расстояние до {k} соседей (фильтр): {self.metrics['filtered_mean_knn']:.6f} ± {self.metrics['filtered_std_knn']:.6f}")
            lines.append(f"Изменение среднего расстояния: {self.metrics['knn_change_percent']:.2f}%")

        if 'original_mean_roughness' in self.metrics:
            k = self.metrics.get('roughness_k', 30)
            lines.append(f"Шероховатость (k={k}) средняя (исх.): {self.metrics['original_mean_roughness']:.6f}")
            lines.append(f"Шероховатость (k={k}) средняя (фильтр): {self.metrics['filtered_mean_roughness']:.6f}")
            lines.append(f"Шероховатость (k={k}) 95-й перцентиль (исх.): {self.metrics['original_95p_roughness']:.6f}")
            lines.append(f"Шероховатость (k={k}) 95-й перцентиль (фильтр): {self.metrics['filtered_95p_roughness']:.6f}")
            lines.append(f"Изменение средней шероховатости: {self.metrics['roughness_change_percent']:.2f}%")
            lines.append(f"Планарность средняя (исх.): {self.metrics['original_mean_planarity']:.6f}")
            lines.append(f"Планарность средняя (фильтр): {self.metrics['filtered_mean_planarity']:.6f}")
            lines.append(f"Изменение планарности: {self.metrics['planarity_change_percent']:.2f}%")

        if 'precision' in self.metrics:
            lines.append(f"Точность (Precision): {self.metrics['precision']:.4f}")
            lines.append(f"Полнота (Recall): {self.metrics['recall']:.4f}")
            lines.append(f"F1-мера: {self.metrics['f1']:.4f}")
            lines.append(f"TP: {self.metrics['tp']}, FP: {self.metrics['fp']}, FN: {self.metrics['fn']}, TN: {self.metrics['tn']}")

        return "\n".join(lines)

    def _set_knn_defaults(self, k):
        self.metrics['original_mean_knn'] = 0
        self.metrics['original_std_knn'] = 0
        self.metrics['filtered_mean_knn'] = 0
        self.metrics['filtered_std_knn'] = 0
        self.metrics['knn_k'] = k
        self.metrics['knn_change_percent'] = 0

    def _set_roughness_planarity_defaults(self):
        self.metrics['roughness_k'] = 30
        self.metrics['original_mean_roughness'] = 0
        self.metrics['original_95p_roughness'] = 0
        self.metrics['filtered_mean_roughness'] = 0
        self.metrics['filtered_95p_roughness'] = 0
        self.metrics['roughness_change_percent'] = 0
        self.metrics['original_mean_planarity'] = 0
        self.metrics['filtered_mean_planarity'] = 0
        self.metrics['planarity_change_percent'] = 0