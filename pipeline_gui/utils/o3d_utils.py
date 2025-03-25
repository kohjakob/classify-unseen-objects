import numpy as np
import open3d as o3d
from .color_utils import generate_distinct_colors

def create_scene_point_clouds(coords, colors_normalized, masks_binary):
    """
    Create original, foreground (segmented), and background point clouds from scene data.
    
    Args:
        coords (np.ndarray): Point cloud coordinates
        colors_normalized (np.ndarray): Normalized colors for original point cloud
        masks_binary (list): List of binary masks for segmentation
        
    Returns:
        tuple: (original_point_cloud, foreground_point_cloud, background_point_cloud)
    """
    # Create original point cloud
    original_point_cloud = o3d.geometry.PointCloud()
    original_point_cloud.points = o3d.utility.Vector3dVector(coords)
    original_point_cloud.colors = o3d.utility.Vector3dVector(colors_normalized)

    # Get combined mask for all instances
    combined_mask = np.zeros(len(coords), dtype=bool)
    for mask in masks_binary:
        combined_mask = combined_mask | mask

    # Create foreground point cloud (segmented instances)
    foreground_points = coords[combined_mask]
    foreground_point_cloud = o3d.geometry.PointCloud()
    foreground_point_cloud.points = o3d.utility.Vector3dVector(foreground_points)
    
    # Color the foreground points
    num_instances = sum(1 for mask in masks_binary if np.any(mask == 1))
    instance_colors = generate_distinct_colors(num_instances)
    
    # Initialize foreground colors array
    foreground_colors = np.zeros_like(foreground_points)
    
    # Create a mapping from original indices to foreground indices
    foreground_indices = np.where(combined_mask)[0]
    index_mapping = {orig: new for new, orig in enumerate(foreground_indices)}
    
    # Color each instance
    current_idx = 0
    for mask in masks_binary:
        instance_point_indices = np.where(mask == 1)[0]
        if len(instance_point_indices) > 0:
            # Map original indices to foreground indices
            mapped_indices = [index_mapping[idx] for idx in instance_point_indices]
            foreground_colors[mapped_indices] = instance_colors[current_idx]
            current_idx += 1
    
    foreground_point_cloud.colors = o3d.utility.Vector3dVector(foreground_colors)

    # Create background point cloud (gray points)
    background_points = coords[~combined_mask]
    background_point_cloud = o3d.geometry.PointCloud()
    background_point_cloud.points = o3d.utility.Vector3dVector(background_points)
    
    # Color background points gray
    grey_color = np.array([0.7, 0.7, 0.7])
    background_colors = np.ones((len(background_points), 3)) * grey_color
    background_point_cloud.colors = o3d.utility.Vector3dVector(background_colors)
    
    return original_point_cloud, foreground_point_cloud, background_point_cloud