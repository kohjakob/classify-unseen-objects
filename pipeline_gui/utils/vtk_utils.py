import vtk
import numpy as np
import open3d as o3d

def o3d_pcd_to_vtk(o3d_pcd):
    """Convert Open3D point cloud to VTK PolyData with correct color handling"""
    points = np.asarray(o3d_pcd.points)
    colors = np.asarray(o3d_pcd.colors)

    # Create vtk points
    vtk_points = vtk.vtkPoints()
    vtk_vertices = vtk.vtkCellArray()
    vtk_colors = vtk.vtkUnsignedCharArray()
    vtk_colors.SetNumberOfComponents(3)
    vtk_colors.SetName('Colors')

    # Add points and colors
    for i in range(len(points)):
        point_id = vtk_points.InsertNextPoint(points[i])
        vtk_vertices.InsertNextCell(1)
        vtk_vertices.InsertCellPoint(point_id)
        
        # Convert colors from [0,1] float to [0,255] int
        color = colors[i]
        r = int(np.clip(color[0] * 255, 0, 255))
        g = int(np.clip(color[1] * 255, 0, 255))
        b = int(np.clip(color[2] * 255, 0, 255))
        vtk_colors.InsertNextTuple3(r, g, b)

    # Create polydata
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(vtk_points)
    polydata.SetVerts(vtk_vertices)
    polydata.GetPointData().SetScalars(vtk_colors)

    return polydata

def pcd_list_to_vtk(pcd_list):
    """Convert point cloud list to VTK PolyData with correct color handling"""
    points = np.asarray(pcd_list)

    # Create vtk points
    vtk_points = vtk.vtkPoints()
    vtk_vertices = vtk.vtkCellArray()
    vtk_colors = vtk.vtkUnsignedCharArray()
    vtk_colors.SetNumberOfComponents(3)
    vtk_colors.SetName('Colors')

    # Add points
    for i in range(len(points)):
        point_id = vtk_points.InsertNextPoint(points[i])
        vtk_vertices.InsertNextCell(1)
        vtk_vertices.InsertCellPoint(point_id)
        vtk_colors.InsertNextTuple3(150,150,150)

    # Create polydata
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(vtk_points)
    polydata.SetVerts(vtk_vertices)
    polydata.GetPointData().SetScalars(vtk_colors)

    return polydata