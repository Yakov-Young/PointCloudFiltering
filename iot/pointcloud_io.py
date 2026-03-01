import open3d as o3d
import numpy as np
from model.point_cloud import PointCloud

def load_from_file(path: str) -> PointCloud:
    """Загружает облако из файла, поддерживаемые форматы: pcd, ply, xyz, etc."""
    o3d_cloud = o3d.io.read_point_cloud(path)
    if not o3d_cloud.has_points():
        raise ValueError(f"Не удалось загрузить точки из {path}")

    # Получаем точки (N,3)
    points = np.asarray(o3d_cloud.points, dtype=np.float32)
    n = points.shape[0]

    # Создаём структурированный массив
    data = np.zeros(n, dtype=PointCloud.DTYPE)
    data['x'] = points[:, 0]
    data['y'] = points[:, 1]
    data['z'] = points[:, 2]

    # Если есть цвета (RGB от 0 до 255)
    if o3d_cloud.has_colors():
        colors = np.asarray(o3d_cloud.colors, dtype=np.float32)  # в диапазоне [0,1]
        data['r'] = (colors[:, 0] * 255).clip(0, 255).astype(np.uint8)
        data['g'] = (colors[:, 1] * 255).clip(0, 255).astype(np.uint8)
        data['b'] = (colors[:, 2] * 255).clip(0, 255).astype(np.uint8)

    # Open3D не хранит интенсивность, классификацию и номер возврата в общем виде,
    # поэтому эти поля останутся нулевыми. Для LAS/LAZ потребуется отдельный загрузчик.

    return PointCloud(data)

def save_to_file(cloud: PointCloud, path: str):
    """Сохраняет облако в файл. Формат определяется расширением."""
    # Создаём Open3D PointCloud
    o3d_cloud = o3d.geometry.PointCloud()
    o3d_cloud.points = o3d.utility.Vector3dVector(cloud.get_xyz())

    # Если есть цвета (хотя бы один ненулевой), добавляем
    if np.any(cloud.points['r'] | cloud.points['g'] | cloud.points['b']):
        colors = np.zeros((len(cloud), 3), dtype=np.float32)
        colors[:, 0] = cloud.points['r'] / 255.0
        colors[:, 1] = cloud.points['g'] / 255.0
        colors[:, 2] = cloud.points['b'] / 255.0
        o3d_cloud.colors = o3d.utility.Vector3dVector(colors)

    o3d.io.write_point_cloud(path, o3d_cloud)