import os
import numpy as np
import umap
import vtk
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton, QWidget
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from pipeline_conf.conf import PATHS
from pipeline_gui.viewers.point_cloud_viewer import PointCloudViewer
from pipeline_gui.utils.vtk_utils import o3d_pcd_to_vtk

import sys

class UMAPPlot(FigureCanvas):
    def __init__(self, parent=None, on_point_click=None):
        fig, self.ax = plt.subplots()
        super(UMAPPlot, self).__init__(fig)
        self.setParent(parent)
        self.ax.set_title('UMAP of Feature Vectors')
        self.embedding = None
        self.on_point_click = on_point_click
        self.cid = self.mpl_connect('button_press_event', self.onclick)

    def calculate_umap(self, features):
        reducer = umap.UMAP()
        self.embedding = reducer.fit_transform(features)

    def plot(self, labels, highlight_index):
        self.ax.clear()
        unique_labels = np.unique(labels)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
        for label, color in zip(unique_labels, colors):
            self.ax.scatter(self.embedding[labels == label, 0], self.embedding[labels == label, 1], c=[color], label=f'Cluster {label}')
        
        highlight_color = colors[labels[highlight_index]]
        self.ax.scatter(self.embedding[highlight_index, 0], self.embedding[highlight_index, 1], c=[highlight_color], edgecolors='black', s=100, label='Current instance')
        
        self.ax.legend()
        self.draw()

    def onclick(self, event):
        if event.inaxes is not None and self.embedding is not None:
            distances = np.linalg.norm(self.embedding - np.array([event.xdata, event.ydata]), axis=1)
            closest_index = np.argmin(distances)
            if self.on_point_click:
                self.on_point_click(closest_index)

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle('Point Cloud Viewer with UMAP')
        self.setGeometry(100, 100, 1200, 600)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)

        self.viewer1 = PointCloudViewer()
        self.umap_plot = UMAPPlot(main_widget, self.on_umap_point_click)
        self.next_button = QPushButton('Next')
        self.next_button.clicked.connect(self.load_next_instance)

        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()

        left_layout.addWidget(self.viewer1.widget)
        right_layout.addWidget(self.umap_plot)
        right_layout.addWidget(self.next_button)

        layout.addLayout(left_layout)
        layout.addLayout(right_layout)

        self.current_index = 0
        self.feature_data, self.label_data, self.point_data = self.load_data()
        self.umap_plot.calculate_umap(self.feature_data)
        self.update_view()

    def load_data(self):
        folder_path = PATHS.scannet_instance_output_dir
        instances = [entry.name for entry in os.scandir(folder_path) if entry.name.endswith('.npz')]
        features, labels, points = [], [], []

        for scannet_instance in instances:
            data = np.load(os.path.join(folder_path, scannet_instance), allow_pickle=True)
            features.append(data['features'])
            labels.append(data['label'])
            points.append(data['points'])

        return np.array(features), np.array(labels), points

    def load_next_instance(self):
        self.current_index = (self.current_index + 1) % len(self.point_data)
        self.update_view()

    def on_umap_point_click(self, index):
        self.current_index = index
        self.update_view()

    def update_view(self):
        # Update VTK viewer
        current_points = np.array(self.point_data[self.current_index])
        self.viewer1.clear_point_cloud()
        self.viewer1.set_point_cloud(current_points, has_color=False)
        self.iren1 = self.viewer1.widget.GetRenderWindow().GetInteractor()
        self.viewer1.widget.GetRenderWindow().Render()
        style1 = vtk.vtkInteractorStyleTrackballCamera()
        self.iren1.SetInteractorStyle(style1)
        self.iren1.Initialize()
        self.viewer1.renderer.ResetCamera()
        self.viewer1.widget.GetRenderWindow().Render()

        # Update UMAP plot
        self.umap_plot.plot(self.label_data, self.current_index)

def main():
    app = QApplication([])
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()