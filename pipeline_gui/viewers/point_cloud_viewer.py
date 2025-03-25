import vtk
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from ..utils.vtk_utils import o3d_pcd_to_vtk
import os

class PointCloudViewer:
    def __init__(self):
        self.widget = QVTKRenderWindowInteractor()
        self.renderer = vtk.vtkRenderer()
        self.widget.GetRenderWindow().AddRenderer(self.renderer)
        
    def add_coordinate_axes(self, scale=1.0, origin=(0, 0, 0)):
        """Add a coordinate axes actor to the renderer"""
        axes = vtk.vtkAxesActor()
        axes.SetTotalLength(scale, scale, scale)
        axes.SetPosition(origin)
        axes.SetShaftType(0)
        axes.SetAxisLabels(0)
        self.renderer.AddActor(axes)

    def add_orientation_widget(self):
        """Add an orientation marker widget to the renderer"""
        axes = vtk.vtkAxesActor()
        widget = vtk.vtkOrientationMarkerWidget()
        widget.SetOrientationMarker(axes)
        widget.SetInteractor(self.widget.GetRenderWindow().GetInteractor())
        widget.SetViewport(0.0, 0.0, 0.2, 0.2)
        widget.SetEnabled(1)
        widget.InteractiveOff()
        return widget

    def set_point_cloud(self, point_cloud):
        """Set the point cloud to display"""
        vtk_cloud = o3d_pcd_to_vtk(point_cloud)
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(vtk_cloud)
        mapper.SetColorModeToDirectScalars()
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetPointSize(1)
        
        self.renderer.AddActor(actor)

    def set_semitransparent_point_cloud(self, point_cloud, opacity=0.1):
        """Set a semi-transparent point cloud to display"""
        vtk_cloud = o3d_pcd_to_vtk(point_cloud)
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(vtk_cloud)
        mapper.SetColorModeToDirectScalars()
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetPointSize(1)
        
        # Enable transparency
        actor.GetProperty().SetOpacity(opacity)
        actor.GetProperty().SetBackfaceCulling(0)
        actor.GetProperty().SetFrontfaceCulling(0)
        
        self.renderer.AddActor(actor)