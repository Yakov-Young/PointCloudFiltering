# test_model.py
from model.point_cloud import PointCloud
from iot.pointcloud_io import load_from_file, save_to_file
import numpy as np

# Создаём тестовое облако вручную
data = np.zeros(100, dtype=PointCloud.DTYPE)
data['x'] = np.random.rand(100) * 10
data['y'] = np.random.rand(100) * 10
data['z'] = np.random.rand(100) * 10
data['r'] = 255
data['g'] = 0
data['b'] = 0

cloud = PointCloud(data)
print(f"Создано облако с {len(cloud)} точками")
xyz = cloud.get_xyz()
print(f"Координаты первых 5 точек:\n{xyz[:5]}")

# Сохраняем
save_to_file(cloud, "test.ply")
print("Сохранено в test.ply")

# Загружаем обратно
cloud2 = load_from_file("test.ply")
print(f"Загружено облако с {len(cloud2)} точками")

# Проверяем цвета
print(f"Цвет первой точки: ({cloud2.points['r'][0]}, {cloud2.points['g'][0]}, {cloud2.points['b'][0]})")