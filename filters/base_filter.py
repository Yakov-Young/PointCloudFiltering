# filters/base_filter.py
from abc import ABC, abstractmethod
from model.point_cloud import PointCloud

class Filter(ABC):
    """Абстрактный базовый класс для всех фильтров."""
    
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def apply(self, cloud: PointCloud) -> PointCloud:
        """
        Применяет фильтр к облаку точек.
        Должен возвращать новое облако (копию с отфильтрованными точками).
        """
        pass

    @abstractmethod
    def get_parameters(self) -> dict:
        """
        Возвращает словарь параметров фильтра с их текущими значениями.
        Используется для отображения в GUI.
        """
        pass

    @abstractmethod
    def set_parameters(self, **kwargs):
        """Устанавливает параметры фильтра."""
        pass