import open3d as o3d
import numpy as np
import plyfile
from model.point_cloud import PointCloud

def load_ply_with_all_fields(path: str) -> PointCloud:
    """Загружает PLY с сохранением всех полей через plyfile."""
    plydata = plyfile.PlyData.read(path)
    vertex = plydata['vertex'].data
    n = len(vertex)
    data = np.zeros(n, dtype=PointCloud.DTYPE)
    for name in PointCloud.DTYPE.names:
        if name in vertex.dtype.names:
            data[name] = vertex[name].astype(PointCloud.DTYPE[name])
    return PointCloud(data, original_indices=np.arange(n))

def load_from_file(path: str) -> PointCloud:
    """Загружает облако из файла. Для PLY использует plyfile, для остальных — Open3D."""
    if path.lower().endswith('.ply'):
        return load_ply_with_all_fields(path)
    o3d_cloud = o3d.io.read_point_cloud(path)
    if not o3d_cloud.has_points():
        raise ValueError(f"Не удалось загрузить точки из {path}")

    points = np.asarray(o3d_cloud.points, dtype=np.float32)
    n = points.shape[0]
    data = np.zeros(n, dtype=PointCloud.DTYPE)
    data['x'] = points[:, 0]
    data['y'] = points[:, 1]
    data['z'] = points[:, 2]

    if o3d_cloud.has_colors():
        colors = np.asarray(o3d_cloud.colors, dtype=np.float32)
        data['red'] = (colors[:, 0] * 255).clip(0, 255).astype(np.uint8)
        data['green'] = (colors[:, 1] * 255).clip(0, 255).astype(np.uint8)
        data['blue'] = (colors[:, 2] * 255).clip(0, 255).astype(np.uint8)

    if o3d_cloud.has_normals():
        normals = np.asarray(o3d_cloud.normals, dtype=np.float32)
        data['nx'] = normals[:, 0]
        data['ny'] = normals[:, 1]
        data['nz'] = normals[:, 2]

    return PointCloud(data, original_indices=np.arange(n))

def save_to_ply_with_all_fields(cloud: PointCloud, path: str):
    """Сохраняет облако в PLY, используя исходные типы полей."""
    # cloud.points уже имеет нужный dtype (f4, u1, f4...)
    vertex_element = plyfile.PlyElement.describe(cloud.points, 'vertex')
    plyfile.PlyData([vertex_element], text=False).write(path)

def save_to_file(cloud: PointCloud, path: str):
    """Сохраняет облако в файл. Для PLY сохраняет все поля, для других форматов использует Open3D (с потерей дополнительных полей)."""
    if path.lower().endswith('.ply'):
        save_to_ply_with_all_fields(cloud, path)
    else:
        o3d_cloud = o3d.geometry.PointCloud()
        o3d_cloud.points = o3d.utility.Vector3dVector(cloud.get_xyz())

        if np.any(cloud.points['red'] | cloud.points['green'] | cloud.points['blue']):
            colors = np.zeros((len(cloud), 3), dtype=np.float32)
            colors[:, 0] = cloud.points['red'] / 255.0
            colors[:, 1] = cloud.points['green'] / 255.0
            colors[:, 2] = cloud.points['blue'] / 255.0
            o3d_cloud.colors = o3d.utility.Vector3dVector(colors)

        if np.any(cloud.points['nx'] | cloud.points['ny'] | cloud.points['nz']):
            normals = np.zeros((len(cloud), 3), dtype=np.float32)
            normals[:, 0] = cloud.points['nx']
            normals[:, 1] = cloud.points['ny']
            normals[:, 2] = cloud.points['nz']
            o3d_cloud.normals = o3d.utility.Vector3dVector(normals)

        o3d.io.write_point_cloud(path, o3d_cloud)
        print(f"Предупреждение: при сохранении в {path.split('.')[-1]} формате дополнительные поля (интенсивность, мусор) были потеряны. Используйте PLY для сохранения всех атрибутов.")