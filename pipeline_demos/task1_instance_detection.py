# Set cuda environment variables
import os
# Project paths
os.environ["PROJECT_ROOT"] = "/home/shared/"
os.environ["CONDA_ROOT"] = "/home/shared/miniconda3"
os.environ["CONDA_DEFAULT_ENV"] = "classify-unseen-objects"
os.environ["CONDA_PYTHON_EXE"] = f"{os.environ['CONDA_ROOT']}/bin/python"
os.environ["CONDA_PREFIX"] = f"{os.environ['CONDA_ROOT']}/envs/classify-unseen-objects"
os.environ["CONDA_PREFIX_1"] = f"{os.environ['CONDA_ROOT']}/miniconda3"
# CUDA paths and settings
os.environ["CUDA_HOME"] = f"{os.environ['PROJECT_ROOT']}/cuda-11.6/toolkit"
os.environ["LD_LIBRARY_PATH"] = f"{os.environ['PROJECT_ROOT']}/cuda-11.6/toolkit/lib64"
os.environ["CUDNN_LIB_DIR"] = f"{os.environ['PROJECT_ROOT']}/cuda-11.6/toolkit/lib64"
os.environ["TORCH_CUDA_ARCH_LIST"] = "6.0;7.0;7.5"
# Compiler settings
os.environ["CXX"] = "g++-9"
os.environ["CC"] = "gcc-9"
# Update PATH to include CUDA binaries
os.environ["PATH"] = f"{os.environ['CUDA_HOME']}/bin:{os.environ.get('PATH', '')}"

import sys
import numpy as np
import vtk
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

from pipeline_models.sai3d_wrapper import SAI3D_Wrapper
from pipeline_models.unscene3d_wrapper import UnScene3D_Wrapper
from pipeline_conf.conf import CONFIG
from pipeline_utils.data_utils import load_scannet_gt_instances, load_scannet_scene_data, load_scannet_scene_data_open3d

# Demo for showcasing instance detection with UnScene3D and SAI3D on ScanNet scenes.

class PointCloudViewer:
    def __init__(self):
        self.widget = QVTKRenderWindowInteractor()
        self.renderer = vtk.vtkRenderer()
        self.widget.GetRenderWindow().AddRenderer(self.renderer)
        
    def add_coordinate_axes(self, scale=1.0, origin=(0, 0, 0)):
        axes = vtk.vtkAxesActor()
        axes.SetTotalLength(scale, scale, scale)
        axes.SetPosition(origin)
        axes.SetShaftType(0)
        axes.SetAxisLabels(0)
        self.renderer.AddActor(axes)

    def add_colored_points(self, points, colors, point_size=1, opacity=1.0):
        if len(points) != len(colors):
            raise ValueError("Number of points and colors must match")
        
        vtk_points = vtk.vtkPoints()
        vtk_cells = vtk.vtkCellArray()
        vtk_colors = vtk.vtkUnsignedCharArray()
        vtk_colors.SetNumberOfComponents(3)
        vtk_colors.SetName("Colors")
        
        for i, (point, color) in enumerate(zip(points, colors)):
            point_id = vtk_points.InsertNextPoint(point)
            vtk_cells.InsertNextCell(1, [point_id])
            # Convert normalized colors [0-1] to [0-255]
            r, g, b = color[0], color[1], color[2]
            vtk_colors.InsertNextTuple3(r, g, b)
        
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(vtk_points)
        polydata.SetVerts(vtk_cells)
        polydata.GetPointData().SetScalars(vtk_colors)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetPointSize(point_size)
        actor.GetProperty().SetOpacity(opacity)
        
        self.renderer.AddActor(actor)
        return actor

class VisualizationWindow(QMainWindow):
    def __init__(self, points, colors, instances):
        super().__init__()

        self.setWindowTitle("Point Cloud Visualization")
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        vis_layout = QHBoxLayout()
        self.viewer1 = PointCloudViewer()  # Original point cloud
        self.viewer2 = PointCloudViewer()  # Segmented point cloud
        
        vis_layout.addWidget(self.viewer1.widget)
        vis_layout.addWidget(self.viewer2.widget)
        layout.addLayout(vis_layout)
        self.setup_visualization(points, colors, instances)
        self.setMinimumSize(1300, 600)

    def setup_visualization(self, points, colors, instances):
        self.viewer1.add_colored_points(points, colors, point_size=2)
        instance_points, instance_colors = self.create_instance_colored_points(points, colors, instances)
        self.viewer2.add_colored_points(instance_points, instance_colors, point_size=3)

        self.viewer1.add_coordinate_axes(scale=0.3)
        self.viewer2.add_coordinate_axes(scale=0.3)

        self.viewer1.renderer.SetBackground(0.1, 0.1, 0.1)
        self.viewer2.renderer.SetBackground(0.1, 0.1, 0.1)

        self.setup_interactors()

    def create_instance_colored_points(self, points, colors, instances):

        num_instances = len(instances)
        instance_color_map = generate_distinct_colors(num_instances)

        combined_mask = np.zeros(len(points), dtype=bool)

        print(len(instance["binary_mask"]) for instance in instances)
        for instance in instances:
            combined_mask = combined_mask | instance["binary_mask"]

        foreground_points = points[combined_mask]
        foreground_colors = np.zeros((len(foreground_points), 3))
        fg_indices = np.where(combined_mask)[0]
        fg_map = {orig: i for i, orig in enumerate(fg_indices)}
        for i, instance in enumerate(instances):
            if np.any(instance["binary_mask"]):
                instance_indices = np.where(instance["binary_mask"])[0]
                mapped_indices = [fg_map[idx] for idx in instance_indices if idx in fg_map]
                foreground_colors[mapped_indices] = instance_color_map[i]

        background_points = points[~combined_mask]
        background_color = np.ones((len(background_points), 3)) * 0.5

        all_points = np.vstack([foreground_points, background_points])
        all_colors = np.vstack([foreground_colors, background_color])

        return all_points, all_colors

    def setup_interactors(self):
        self.iren1 = self.viewer1.widget.GetRenderWindow().GetInteractor()
        self.iren2 = self.viewer2.widget.GetRenderWindow().GetInteractor()
        
        style1 = vtk.vtkInteractorStyleTrackballCamera()
        style2 = vtk.vtkInteractorStyleTrackballCamera()
        self.iren1.SetInteractorStyle(style1)
        self.iren2.SetInteractorStyle(style2)
        
        # Sync camera movements
        def sync_renders(obj, event):
            self.viewer1.widget.GetRenderWindow().Render()
            self.viewer2.widget.GetRenderWindow().Render()

        for iren in [self.iren1, self.iren2]:
            iren.AddObserver('InteractionEvent', sync_renders)
            iren.AddObserver('EndInteractionEvent', sync_renders)
            
        self.iren1.Initialize()
        self.iren2.Initialize()
        
        self.viewer1.renderer.ResetCamera()
        self.viewer2.renderer.ResetCamera()
        
        self.viewer2.renderer.SetActiveCamera(self.viewer1.renderer.GetActiveCamera())
        
        self.viewer1.widget.GetRenderWindow().Render()
        self.viewer2.widget.GetRenderWindow().Render()

    def closeEvent(self, event):
        self.viewer1.widget.GetRenderWindow().Finalize()
        self.viewer2.widget.GetRenderWindow().Finalize()
        self.iren1.TerminateApp()
        self.iren2.TerminateApp()
        event.accept()

def generate_distinct_colors(n):
    colors = []
    for i in range(n):
        hue = i / n
        saturation = 0.9
        value = 0.9
        
        h = hue * 6
        c = value * saturation
        x = c * (1 - abs(h % 2 - 1))
        m = value - c
        
        if h < 1:
            r, g, b = c, x, 0
        elif h < 2:
            r, g, b = x, c, 0
        elif h < 3:
            r, g, b = 0, c, x
        elif h < 4:
            r, g, b = 0, x, c
        elif h < 5:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x
            
        colors.append(np.array([r + m, g + m, b + m]))
    
    colors = np.array(colors) * 255
    return colors

def main():
    # Check command line arguments
    if len(sys.argv) < 2 or sys.argv[1] not in CONFIG.instance_detection_mode_dict:
        print("Usage: python3 show_unscene3d_instances.py <instance_detection_mode>")
        print("Available instance detection modes:", list(CONFIG.instance_detection_mode_dict.keys()))
        sys.exit(1)

    scene_name = 'scene0000_00'
    points, colors, gt_labels = load_scannet_scene_data(scene_name)

    instance_detection_mode = CONFIG.instance_detection_mode_dict[sys.argv[1]]["name"]
    if instance_detection_mode == CONFIG.instance_detection_mode_dict["unscene3d"]["name"]:
        unscene3d_wrapper = UnScene3D_Wrapper()
        instances = unscene3d_wrapper.detect_instances(points, colors)
    elif instance_detection_mode == CONFIG.instance_detection_mode_dict["sai3d"]["name"]:
        sai3d_wrapper = SAI3D_Wrapper()
        instances = sai3d_wrapper.detect_instances(points, colors, scene_name)
    elif instance_detection_mode == CONFIG.instance_detection_mode_dict["gt"]["name"]:
        instances = load_scannet_gt_instances(scene_name)

    app = QApplication(sys.argv)
    window = VisualizationWindow(points, colors, instances)
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()