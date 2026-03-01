import open3d as o3d
import numpy as np
from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import QTimer

class VisualizerWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name='Open3D', width=640, height=480, visible=True)
        
        self.pcd = o3d.geometry.PointCloud()
        self.vis.add_geometry(self.pcd)
        
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_render)
        self.timer.start(50)

        opt = self.vis.get_render_option()
        opt.point_size = 3.0
        opt.background_color = np.array([0, 0, 0])

    def _update_render(self):
        self.vis.poll_events()
        self.vis.update_renderer()

    def update_cloud(self, points):
        if points is None or len(points) == 0:
            return
        self.pcd.points = o3d.utility.Vector3dVector(points)
        self.pcd.paint_uniform_color([0.5, 0.5, 0.5])
        
        self.vis.remove_geometry(self.pcd)
        self.vis.add_geometry(self.pcd)

        view_control = self.vis.get_view_control()
        view_control.set_front([0, 0, -1])
        view_control.set_up([0, -1, 0])
        view_control.set_zoom(0.8)

    def closeEvent(self, event):
        self.timer.stop()
        self.vis.destroy_window()
        super().closeEvent(event)