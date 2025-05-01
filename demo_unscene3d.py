import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout
import vtk
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

from pipeline_tasks.scan_loading import load_scannet_scene
from pipeline_tasks.scan_preprocessing import preprocess_scannet_scene
from pipeline_tasks.scan_segmentation import segment_scannet_scene
from pipeline_tasks.segments_postprocessing import postprocess_scannet_segments
from pipeline_tasks.segments_merging import merge_scannet_segments
from pipeline_tasks.instances_filtering import filter_scannet_instances
from pipeline_models.segmentation_model import Unscene3D_Wrapper
from pipeline_conf.conf import PATHS

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
            r, g, b = int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)
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
    def __init__(self, coords, colors_original, instance_masks):
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
        self.setup_visualization(coords, colors_original, instance_masks)
        self.setMinimumSize(1300, 600)

    def setup_visualization(self, coords, colors_original, instance_masks):
        self.viewer1.add_colored_points(coords, colors_original, point_size=2)
        instance_points, instance_colors = self.create_instance_colored_points(coords, colors_original, instance_masks)
        self.viewer2.add_colored_points(instance_points, instance_colors, point_size=3)

        self.viewer1.add_coordinate_axes(scale=0.3)
        self.viewer2.add_coordinate_axes(scale=0.3)

        self.viewer1.renderer.SetBackground(0.1, 0.1, 0.1)
        self.viewer2.renderer.SetBackground(0.1, 0.1, 0.1)

        self.setup_interactors()

    def create_instance_colored_points(self, coords, colors_original, instance_masks):

        num_instances = len(instance_masks)
        instance_color_map = generate_distinct_colors(num_instances)

        combined_mask = np.zeros(len(coords), dtype=bool)
        for mask in instance_masks:
            combined_mask = combined_mask | mask
        foreground_points = coords[combined_mask]
        foreground_colors = np.zeros((len(foreground_points), 3))
        fg_indices = np.where(combined_mask)[0]
        fg_map = {orig: i for i, orig in enumerate(fg_indices)}
        for i, mask in enumerate(instance_masks):
            if np.any(mask):
                instance_indices = np.where(mask)[0]
                mapped_indices = [fg_map[idx] for idx in instance_indices if idx in fg_map]
                foreground_colors[mapped_indices] = instance_color_map[i]

        background_points = coords[~combined_mask]
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
    
    return np.array(colors)

def main():
    # Instantiate model and process scene
    model = Unscene3D_Wrapper()
    scannet_scene_name = 'scene0000_00'
    
    # Load and process scene
    scannet_scene_mesh, scannet_scene_segs_json, scannet_scene_aggr_json = load_scannet_scene(PATHS.scannet_scenes, scannet_scene_name)
    data, features, inverse_map, coords, colors_normalized, ground_truth_labels = preprocess_scannet_scene(
        scannet_scene_mesh, scannet_scene_segs_json, scannet_scene_aggr_json, voxelization=True)
    outputs = segment_scannet_scene(model, data, features)
    masks_binary, mask_confidences, label_confidences = postprocess_scannet_segments(
        outputs, inverse_map, coords, colors_normalized, ground_truth_labels)
    masks_binary, mask_confidences, label_confidences = merge_scannet_segments(
        masks_binary, mask_confidences, label_confidences)
    masks_binary, mask_confidences, label_confidences = filter_scannet_instances(
        masks_binary, mask_confidences, label_confidences, 0.9)
    
    # Launch visualization
    app = QApplication(sys.argv)
    window = VisualizationWindow(coords, colors_normalized, masks_binary)
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()