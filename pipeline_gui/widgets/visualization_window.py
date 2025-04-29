from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton
from ..viewers.point_cloud_viewer import PointCloudViewer
import vtk
from pipeline_tasks.instances_filtering import filter_scannet_instances
from ..utils.o3d_utils import create_scene_point_clouds

class VisualizationWindow(QMainWindow):
    def __init__(self, original_cloud, foreground_cloud, background_cloud, scene_name):
        super().__init__()
        self.setWindowTitle(f"Scene Visualization - {scene_name}")
        self.threshold = 0.9

        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # Create visualization layout
        vis_layout = QHBoxLayout()
        
        # Create viewers
        self.viewer1 = PointCloudViewer()
        self.viewer2 = PointCloudViewer()
        
        vis_layout.addWidget(self.viewer1.widget)
        vis_layout.addWidget(self.viewer2.widget)
        layout.addLayout(vis_layout)

        # Create threshold controls
        self.setup_threshold_controls(layout)

        # Initialize visualization
        self.setup_visualization(original_cloud, foreground_cloud, background_cloud)
        
        # Set window size
        self.setMinimumSize(1300, 600)

    def setup_threshold_controls(self, layout):
        threshold_layout = QHBoxLayout()
        threshold_label = QLabel("Threshold:")
        self.threshold_value = QLabel(f"{self.threshold:.1f}")
        
        minus_button = QPushButton("-")
        plus_button = QPushButton("+")
        minus_button.setFixedWidth(40)
        plus_button.setFixedWidth(40)
        
        minus_button.clicked.connect(self.decrease_threshold)
        plus_button.clicked.connect(self.increase_threshold)
        
        threshold_layout.addStretch()
        threshold_layout.addWidget(threshold_label)
        threshold_layout.addWidget(minus_button)
        threshold_layout.addWidget(self.threshold_value)
        threshold_layout.addWidget(plus_button)
        threshold_layout.addStretch()
        layout.addLayout(threshold_layout)

    def setup_visualization(self, original_cloud, foreground_cloud, background_cloud):
        # Set point clouds
        self.viewer1.set_point_cloud(original_cloud)
        self.viewer2.set_point_cloud(foreground_cloud)
        self.viewer2.set_semitransparent_point_cloud(background_cloud)

        # Add coordinate axes
        self.viewer1.add_coordinate_axes(scale=0.3)
        self.viewer2.add_coordinate_axes(scale=0.3)

        # Set background color
        self.viewer1.renderer.SetBackground(0.1, 0.1, 0.1)
        self.viewer2.renderer.SetBackground(0.1, 0.1, 0.1)

        # Setup interactors
        self.setup_interactors()

    def setup_interactors(self):
        # Get interactors
        self.iren1 = self.viewer1.widget.GetRenderWindow().GetInteractor()
        self.iren2 = self.viewer2.widget.GetRenderWindow().GetInteractor()

        # Set interactor styles
        style1 = vtk.vtkInteractorStyleTrackballCamera()
        style2 = vtk.vtkInteractorStyleTrackballCamera()
        self.iren1.SetInteractorStyle(style1)
        self.iren2.SetInteractorStyle(style2)

        # Add orientation widgets
        self.orientation_widget1 = self.viewer1.add_orientation_widget()
        self.orientation_widget2 = self.viewer2.add_orientation_widget()

        # Setup synchronization
        def sync_renders(obj, event):
            self.viewer1.widget.GetRenderWindow().Render()
            self.viewer2.widget.GetRenderWindow().Render()

        for iren in [self.iren1, self.iren2]:
            iren.AddObserver('InteractionEvent', sync_renders)
            iren.AddObserver('EndInteractionEvent', sync_renders)

        # Initialize
        self.iren1.Initialize()
        self.iren2.Initialize()

        # Reset cameras
        self.viewer1.renderer.ResetCamera()
        self.viewer2.renderer.ResetCamera()

        # Link cameras
        self.viewer2.renderer.SetActiveCamera(self.viewer1.renderer.GetActiveCamera())

        # Initial render
        self.viewer1.widget.GetRenderWindow().Render()
        self.viewer2.widget.GetRenderWindow().Render()

    def decrease_threshold(self):
        if self.threshold > 0.0:
            self.threshold = round(max(0.0, self.threshold - 0.1), 1)
            self.threshold_value.setText(f"{self.threshold:.1f}")

    def increase_threshold(self):
        if self.threshold < 1.0:
            self.threshold = round(min(1.0, self.threshold + 0.1), 1)
            self.threshold_value.setText(f"{self.threshold:.1f}")

    def closeEvent(self, event):
        self.viewer1.widget.GetRenderWindow().Finalize()
        self.viewer2.widget.GetRenderWindow().Finalize()
        self.iren1.TerminateApp()
        self.iren2.TerminateApp()
        event.accept()