"""
Скрипт для пакетного тестирования различных фильтров с тремя уровнями агрессивности.
Запуск: python batch_analysis.py <путь_к_облаку>
Результат сохраняется в CSV файл в корне проекта.
"""

import sys
import os
import csv
import datetime
import numpy as np

# Импортируем необходимые модули из нашего приложения
from model.point_cloud import PointCloud
from iot.pointcloud_io import load_from_file
from filters.statistical_outlier import StatisticalOutlierFilter
from filters.radius_outlier import RadiusOutlierFilter
from filters.lof_filter import LOFilter
from filters.dsor_filter import DSORFilter
from filters.pca_curvature_filter import PCACurvatureFilter
from evaluation.report import EvaluationReport

def main():
    if len(sys.argv) < 2:
        print("Использование: python batch_analysis.py <путь_к_облаку>")
        sys.exit(1)

    input_path = sys.argv[1]
    if not os.path.isfile(input_path):
        print(f"Файл не найден: {input_path}")
        sys.exit(1)

    print(f"Загружаем облако из {input_path} ...")
    try:
        original_cloud = load_from_file(input_path)
    except Exception as e:
        print(f"Ошибка загрузки: {e}")
        sys.exit(1)

    print(f"Загружено {len(original_cloud)} точек.")
    has_gt = 'scalar_isGarbage' in original_cloud.points.dtype.names
    if has_gt:
        print("Поле ground truth (scalar_isGarbage) присутствует.")
    else:
        print("Поле ground truth отсутствует — метрики классификации не будут вычислены.")

    # Конфигурации фильтров: (имя_фильтра, класс, список кортежей (уровень, параметры))
    configs = [
        ("SOR", StatisticalOutlierFilter, [
            ("soft", {"nb_neighbors": 20, "std_ratio": 2.5}),
            ("moderate", {"nb_neighbors": 20, "std_ratio": 2.0}),
            ("aggressive", {"nb_neighbors": 20, "std_ratio": 1.5})
        ]),
        ("ROR", RadiusOutlierFilter, [
            ("soft", {"radius": 0.025, "min_neighbors": 3}),
            ("moderate", {"radius": 0.02, "min_neighbors": 5}),
            ("aggressive", {"radius": 0.015, "min_neighbors": 8})
        ]),
        ("LOF", LOFilter, [
            ("soft", {"n_neighbors": 20, "contamination": 0.02}),
            ("moderate", {"n_neighbors": 20, "contamination": 0.04}),
            ("aggressive", {"n_neighbors": 20, "contamination": 0.08})
        ]),
        ("DSOR", DSORFilter, [
            ("soft", {"k": 20, "range_multiplier": 8.0, "std_ratio": 2.5}),
            ("moderate", {"k": 20, "range_multiplier": 8.0, "std_ratio": 2.0}),
            ("aggressive", {"k": 20, "range_multiplier": 8.0, "std_ratio": 1.5})
        ]),
        ("PCA", PCACurvatureFilter, [
            ("soft", {"k": 30, "percentile": 99}),
            ("moderate", {"k": 30, "percentile": 97}),
            ("aggressive", {"k": 30, "percentile": 95})
        ])
    ]

    results = []

    for filter_name, filter_class, levels in configs:
        for level_name, params in levels:
            print(f"\nПрименяем {filter_name} ({level_name}) с параметрами {params}...")

            # Создаём экземпляр фильтра
            filter_instance = filter_class(**params)

            # Применяем к копии исходного облака (чтобы не изменять оригинал)
            try:
                filtered_cloud = filter_instance.apply(original_cloud.copy())
            except Exception as e:
                print(f"  Ошибка применения: {e}")
                continue

            # Создаём отчёт
            report = EvaluationReport(original_cloud, filtered_cloud)
            report.compute_basic_metrics()
            report.compute_knn_metrics(k=10, n_jobs=8)  # используем 8 потоков

            # Если есть ground truth, добавляем классификационные метрики
            if has_gt and filter_instance.last_mask is not None:
                # last_mask: True – оставлена (inlier), False – удалена
                removal_mask = ~filter_instance.last_mask
                report.compute_classification_metrics(removal_mask)

            # Собираем данные для CSV
            row = {
                'filter': filter_name,
                'level': level_name,
                'params': str(params),
                'original_count': report.metrics.get('original_count', 0),
                'filtered_count': report.metrics.get('filtered_count', 0),
                'removed_percent': report.metrics.get('removed_percent', 0.0),
                'precision': report.metrics.get('precision', ''),
                'recall': report.metrics.get('recall', ''),
                'f1': report.metrics.get('f1', ''),
                'tp': report.metrics.get('tp', ''),
                'fp': report.metrics.get('fp', ''),
                'fn': report.metrics.get('fn', ''),
                'tn': report.metrics.get('tn', ''),
                'mean_knn_orig': report.metrics.get('original_mean_knn', 0.0),
                'std_knn_orig': report.metrics.get('original_std_knn', 0.0),
                'mean_knn_filt': report.metrics.get('filtered_mean_knn', 0.0),
                'std_knn_filt': report.metrics.get('filtered_std_knn', 0.0),
                'knn_change_percent': report.metrics.get('knn_change_percent', 0.0)
            }
            results.append(row)
            print(f"  Осталось точек: {row['filtered_count']} ({row['removed_percent']:.2f}%)")

    # Формируем имя выходного файла
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = f"{base_name}_analysis_{timestamp}.csv"
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), out_file)

    # Записываем CSV
    fieldnames = [
        'filter', 'level', 'params',
        'original_count', 'filtered_count', 'removed_percent',
        'precision', 'recall', 'f1', 'tp', 'fp', 'fn', 'tn',
        'mean_knn_orig', 'std_knn_orig', 'mean_knn_filt', 'std_knn_filt', 'knn_change_percent'
    ]
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nРезультаты сохранены в {out_path}")

if __name__ == "__main__":
    main()