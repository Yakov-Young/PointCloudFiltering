import numpy as np
from typing import Optional, Tuple
#from .point import Point  # для совместимости

class PointCloud:
    """Облако точек с атрибутами."""
    
    # Описание полей для структурированного массива
    DTYPE = np.dtype([
        ('x', 'f4'),
        ('y', 'f4'),
        ('z', 'f4'),
        ('red', 'u1'),
        ('green', 'u1'),
        ('blue', 'u1'),
        ('nx', 'f4'),
        ('ny', 'f4'),
        ('nz', 'f4'),
        ('scalar_intensity', 'f4'),
        ('scalar_isGarbage', 'f4')
    ])
    
    def __init__(self, points: Optional[np.ndarray] = None, original_indices: Optional[np.ndarray] = None):
        if points is None:
            self.points = np.empty(0, dtype=self.DTYPE)
        else:
            self.points = points
        if original_indices is not None:
            self.original_indices = original_indices
        else:
            self.original_indices = np.arange(len(self.points))  # по умолчанию соответствие самому себе
        self.coord_sys: str = ""

    def __len__(self) -> int:
        return len(self.points)

    def get_xyz(self) -> np.ndarray:
        """Возвращает массив координат (N, 3) как float32."""
        xyz = np.empty((len(self), 3), dtype=np.float32)
        xyz[:, 0] = self.points['x']
        xyz[:, 1] = self.points['y']
        xyz[:, 2] = self.points['z']
        return xyz

    def set_xyz(self, xyz: np.ndarray):
        """Устанавливает координаты из массива (N,3)."""
        if xyz.shape[0] != len(self):
            raise ValueError("Количество точек не совпадает")
        self.points['x'] = xyz[:, 0]
        self.points['y'] = xyz[:, 1]
        self.points['z'] = xyz[:, 2]
    
    def get_rgb(self) -> np.ndarray:
        """Возвращает массив цветов (N, 3) в диапазоне [0, 1] для Open3D."""
        # Проверяем, есть ли поля цвета в структурированном массиве
        if 'r' in self.points.dtype.names and 'g' in self.points.dtype.names and 'b' in self.points.dtype.names:
            rgb = np.empty((len(self), 3), dtype=np.float32)
            rgb[:, 0] = self.points['r'] / 255.0
            rgb[:, 1] = self.points['g'] / 255.0
            rgb[:, 2] = self.points['b'] / 255.0
            # Если все цвета нулевые (или не заданы), вернём None, чтобы использовать серый
            if np.all(rgb == 0):
                return None
            return rgb
        else:
            return None

    def copy(self) -> 'PointCloud':
        new_cloud = PointCloud(self.points.copy(), self.original_indices.copy())
        new_cloud.coord_sys = self.coord_sys
        return new_cloud
