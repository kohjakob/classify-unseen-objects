import vtk
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from ..utils.vtk_utils import o3d_pcd_to_vtk, pcd_list_to_vtk
import os
import numpy as np

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

    def clear_point_cloud(self):
        """Clear the displayed pointclouds"""
        self.renderer.RemoveAllViewProps()


    def add_colored_points(self, points, colors, point_size=1, opacity=1):
        """Add points with corresponding colors to the renderer"""
        if len(points) != len(colors):
            raise ValueError("Number of points and colors must match")
        
        # Create a vtkPoints object and a vtkCellArray to store points
        vtk_points = vtk.vtkPoints()
        vtk_cells = vtk.vtkCellArray()
        
        # Create color array
        vtk_colors = vtk.vtkUnsignedCharArray()
        vtk_colors.SetNumberOfComponents(3)
        vtk_colors.SetName("Colors")
        
        # Add points and colors
        for i, (point, color) in enumerate(zip(points, colors)):
            point_id = vtk_points.InsertNextPoint(point)
            vtk_cells.InsertNextCell(1, [point_id])
            vtk_colors.InsertNextTuple3(int(color[0]*255), int(color[1]*255), int(color[2]*255))
        
        # Create a polydata object
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(vtk_points)
        polydata.SetVerts(vtk_cells)
        polydata.GetPointData().SetScalars(vtk_colors)
        
        # Create mapper and actor
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetPointSize(point_size)
        actor.GetProperty().SetOpacity(opacity)
        
        self.renderer.AddActor(actor)
        return actor
    
    def add_points(self, points, color, point_size=1, opacity=1):
        """Add points with set color to the renderer"""
        if len(points) == 0:
            raise ValueError("No points to display")
        
        # Create a vtkPoints object and a vtkCellArray to store points
        vtk_points = vtk.vtkPoints()
        vtk_cells = vtk.vtkCellArray()
        
        # Add points
        for point in points:
            point_id = vtk_points.InsertNextPoint(point)
            vtk_cells.InsertNextCell(1, [point_id])
        
        # Create a polydata object
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(vtk_points)
        polydata.SetVerts(vtk_cells)
        
        # Create mapper and actor
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(color)
        actor.GetProperty().SetOpacity(opacity)
        actor.GetProperty().SetPointSize(point_size)
        
        self.renderer.AddActor(actor)
        return actor

    def set_point_cloud(self, point_cloud, point_size=1, colored=False, color=(1,1,1), input_type="o3d"):
        """Set the point cloud to display
        
        Args:
            point_cloud: Either an Open3D point cloud or a tuple of (points, colors) or just points
            point_size: Size of points to display
            colored: Whether to use colors from the point cloud (if available)
            color: Default color to use if not colored or colors not available
            input_type: Type of input - "o3d" for Open3D point cloud, "numpy" for numpy arrays
        """
        
        if input_type == "o3d":
            # Extract points and colors from Open3D point cloud
            points = np.asarray(point_cloud.points)
            if colored and hasattr(point_cloud, 'colors') and len(point_cloud.colors) > 0:
                colors = np.asarray(point_cloud.colors)
                return self.add_colored_points(points, colors, point_size)
            else:
                return self.add_points(points, color, point_size)
        elif input_type == "numpy":
            if colored and isinstance(point_cloud, tuple) and len(point_cloud) == 2:
                points, colors = point_cloud
                return self.add_colored_points(points, colors, point_size)
            else:
                points = point_cloud
                return self.add_points(points, color, point_size)
        else:
            raise ValueError(f"Unsupported input_type: {input_type}")
