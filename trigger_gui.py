import os
import numpy as np
import umap
import vtk
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton, QWidget, QLabel
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from scipy.spatial.distance import cosine

from pipeline_conf.conf import PATHS
from pipeline_gui.viewers.point_cloud_viewer import PointCloudViewer
from pipeline_gui.utils.vtk_utils import o3d_pcd_to_vtk
from pipeline_tasks.scan_loading import load_scannet_scene
from pipeline_tasks.scan_preprocessing import preprocess_scannet_scene

import open3d as o3d

import numpy as np
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
        
        label_to_color = {label: color for label, color in zip(unique_labels, colors)}
        
        for label, color in zip(unique_labels, colors):
            self.ax.scatter(self.embedding[labels == label, 0], self.embedding[labels == label, 1], c=[color], label=f'Cluster {label}')
        
        current_label = labels[highlight_index]
        highlight_color = label_to_color[current_label]
        
        self.ax.scatter(self.embedding[highlight_index, 0], self.embedding[highlight_index, 1], 
                        c=[highlight_color], edgecolors='black', s=100, label='Current instance')
        
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
        self.viewer2 = PointCloudViewer()
        self.umap_plot = UMAPPlot(main_widget, self.on_umap_point_click)
        
        self.next_button = QPushButton('Next (Furthest in Cluster)')
        self.next_button.clicked.connect(self.load_next_instance)
        
        self.next_cluster_button = QPushButton('Next Cluster')
        self.next_cluster_button.clicked.connect(self.load_next_cluster)
        
        self.gt_label_widget = QLabel("Ground Truth Label: None")
        self.gt_label_widget.setFont(QFont('Arial', 12, QFont.Bold))
        self.gt_label_widget.setAlignment(Qt.AlignCenter)
        self.gt_label_widget.setStyleSheet("background-color: #f0f0f0; padding: 10px; border-radius: 5px;")
        self.gt_label_widget.setMinimumHeight(40)

        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()
        buttons_layout = QHBoxLayout()

        buttons_layout.addWidget(self.next_button)
        buttons_layout.addWidget(self.next_cluster_button)

        left_layout.addWidget(self.viewer1.widget)
        left_layout.addWidget(self.viewer2.widget)
        left_layout.addWidget(self.gt_label_widget) 
        
        right_layout.addWidget(self.umap_plot)
        right_layout.addLayout(buttons_layout)

        layout.addLayout(left_layout)
        layout.addLayout(right_layout)

        self.current_index = 0
        self.current_cluster = None
        self.visited_indices = []
        
        self.feature_data, self.label_data, self.point_data, self.gt_labels, self.point_colors = self.load_data()
        self.umap_plot.calculate_umap(self.feature_data)
        
        self.select_initial_cluster_point()
        self.update_view()

    def select_initial_cluster_point(self):
        if self.current_cluster is None:
            unique_clusters = np.unique(self.label_data)
            self.current_cluster = unique_clusters[0]
        
        cluster_indices = np.where(self.label_data == self.current_cluster)[0]
        cluster_features = self.feature_data[cluster_indices]
        centroid = np.mean(cluster_features, axis=0)
        
        closest_idx = None
        min_distance = float('inf')
        for idx in cluster_indices:
            # Cosine distance (1 - cosine similarity)
            distance = cosine(self.feature_data[idx], centroid)
            if distance < min_distance:
                min_distance = distance
                closest_idx = idx
        
        if closest_idx is not None:
            self.current_index = closest_idx
            self.visited_indices = [closest_idx]

    def find_furthest_point(self):
        cluster_indices = np.where(self.label_data == self.current_cluster)[0]
        unvisited = [idx for idx in cluster_indices if idx not in self.visited_indices]
        
        if not unvisited:
            self.visited_indices = [self.visited_indices[0]]
            unvisited = [idx for idx in cluster_indices if idx not in self.visited_indices]
            furthest_idx = None
        max_min_distance = -float('inf')
        
        for idx in unvisited:
            min_distance = float('inf')
            for visited_idx in self.visited_indices:
                # Cosine distance (1 - cosine similarity)
                distance = cosine(self.feature_data[idx], self.feature_data[visited_idx])
                min_distance = min(min_distance, distance)
            
            if min_distance > max_min_distance:
                max_min_distance = min_distance
                furthest_idx = idx
        
        return furthest_idx

    def load_next_cluster(self):
        unique_clusters = np.unique(self.label_data)
        current_idx = np.where(unique_clusters == self.current_cluster)[0][0]
        next_idx = (current_idx + 1) % len(unique_clusters)
        self.current_cluster = unique_clusters[next_idx]
        self.visited_indices = []
        self.select_initial_cluster_point()
        self.update_view()

    def random_downsample(self, coords, colors, target_count=1000):
        """
        Randomly downsample coordinates and colors arrays to a target number of points
        while maintaining correspondence between them.
        
        Args:
            coords: numpy array of shape (N, 3) containing 3D coordinates
            colors: numpy array of shape (N, 3) containing RGB colors (normalized)
            target_count: desired number of points after downsampling
            
        Returns:
            downsampled_coords: numpy array of shape (target_count, 3)
            downsampled_colors: numpy array of shape (target_count, 3)
        """
        num_points = coords.shape[0]
        
        if num_points <= target_count:
            return coords, colors
        
        indices = np.random.choice(num_points, target_count, replace=False)
        downsampled_coords = coords[indices]
        downsampled_colors = colors[indices]
        
        return downsampled_coords, downsampled_colors

    def load_data(self):
        #folder_path = PATHS.scannet_instance_output_dir
        folder_path = PATHS.scannet_gt_instance_output_dir
        #instances = [entry.name for entry in os.scandir(folder_path) if entry.name.endswith('.npz')]
        instances = [entry.name for entry in os.scandir(folder_path) if entry.name.endswith('.npz') and entry.name.startswith('scene0007_00')]
        features, labels, points, gt_labels, colors = [], [], [], [], []

        scannet_scene_name = ""
        for scannet_instance in instances:
            data = np.load(os.path.join(folder_path, scannet_instance), allow_pickle=True)
            features.append(data['features'])
            labels.append(data['cluster_label'])
            points.append(data['points'])
            colors.append(data['colors'])
            
            # Load the ground truth label
            if 'gt_label' in data:
                gt_labels.append(data['gt_label'])
            else:
                gt_labels.append(data['cluster_label'])
        
            scannet_scene_name = scannet_instance[:12]

        scannet_scene_mesh, scannet_scene_segs_json, scannet_scene_aggr_json = load_scannet_scene(PATHS.scannet_scenes, scannet_scene_name)
        _, _, inverse_map, coords, colors_normalized, ground_truth_labels = preprocess_scannet_scene(scannet_scene_mesh, scannet_scene_segs_json, scannet_scene_aggr_json)

        coords, colors_normalized = self.random_downsample(coords, colors_normalized, target_count=10000)

        original_point_cloud = o3d.geometry.PointCloud()
        original_point_cloud.points = o3d.utility.Vector3dVector(coords)
        original_point_cloud.colors = o3d.utility.Vector3dVector(colors_normalized)
        self.background_pointcloud = original_point_cloud
        
        if labels:
            self.current_cluster = labels[0]

        return np.array(features), np.array(labels), points, gt_labels, colors

    def load_next_instance(self):
        next_idx = self.find_furthest_point()
        
        if next_idx is not None:
            self.current_index = next_idx
            self.visited_indices.append(next_idx)
            self.update_view()

    def on_umap_point_click(self, index):
        self.current_index = index
        self.current_cluster = self.label_data[index]
        self.visited_indices = [index]
        self.update_view()

    def update_view(self):
        # Update VTK viewer
        current_points = np.array(self.point_data[self.current_index])
        current_point_colors = np.array(self.point_colors[self.current_index])
        self.viewer1.clear_point_cloud()
        self.viewer1.set_point_cloud((current_points, current_point_colors), input_type="numpy", colored=True, point_size=2)
        self.iren1 = self.viewer1.widget.GetRenderWindow().GetInteractor()
        self.viewer1.widget.GetRenderWindow().Render()
        style1 = vtk.vtkInteractorStyleTrackballCamera()
        self.iren1.SetInteractorStyle(style1)
        self.iren1.Initialize()
        self.viewer1.renderer.ResetCamera()
        self.viewer1.widget.GetRenderWindow().Render()

        self.viewer2.clear_point_cloud()
        self.viewer2.set_point_cloud(self.background_pointcloud, input_type="o3d", colored=False, point_size=1)
        self.viewer2.set_point_cloud((current_points, current_point_colors), input_type="numpy", colored=True, point_size=2)
        self.viewer2.add_coordinate_axes(scale=0.3)
        self.iren2 = self.viewer2.widget.GetRenderWindow().GetInteractor()
        style2 = vtk.vtkInteractorStyleTrackballCamera()
        self.iren2.SetInteractorStyle(style2)
        self.iren2.Initialize()
        self.viewer2.renderer.SetBackground(0.1, 0.1, 0.1)
        self.viewer2.renderer.ResetCamera()
        self.viewer2.widget.GetRenderWindow().Render()
    
        # Update UMAP plot
        self.umap_plot.plot(self.label_data, self.current_index)
        
        # Update the ground truth label widget
        if self.current_index < len(self.gt_labels):
            gt_label_text = self.gt_labels[self.current_index]
            cluster_info = f"Cluster: {self.current_cluster}"
            self.gt_label_widget.setText(f"Ground Truth Label: {gt_label_text} | {cluster_info}")
            self.gt_label_widget.setStyleSheet(f"padding: 10px; border-radius: 5px;")

def main():
    app = QApplication([])
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()