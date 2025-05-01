import sys
import numpy as np
import open3d as o3d
import torch
import json
import os

from pipeline_tasks.scan_loading import load_scannet_scene
from pipeline_conf.conf import PATHS

from pipeline_models.feature_extraction_model import PointMAE_Wrapper

def random_downsample(points, target_count=8912):
    if len(points) <= target_count:
        return points
    indices = np.random.choice(len(points), target_count, replace=False)
    return points[indices]

def load_scannet_gt_instances(scannet_scene_name, scannet_scenes_path):
    """
    Load ground truth instances from ScanNet scene
    
    Args:
        scannet_scene_name: Name of the ScanNet scene (e.g., 'scene0000_00')
        scannet_scenes_path: Path to the ScanNet scenes directory
        
    Returns:
        List of tuples containing (instance_points, instance_colors, label)
    """
    scene_path = os.path.join(scannet_scenes_path, scannet_scene_name)
    
    # Load the .ply file
    mesh_file = os.path.join(scene_path, f"{scannet_scene_name}_vh_clean.ply")
    mesh = o3d.io.read_triangle_mesh(mesh_file)
    
    # Extract vertex positions and colors from the mesh
    vertices = np.asarray(mesh.vertices)
    colors = np.asarray(mesh.vertex_colors)
    
    # Load the .aggregation.json file
    agg_file = os.path.join(scene_path, f"{scannet_scene_name}_vh_clean.aggregation.json")
    with open(agg_file, 'r') as f:
        agg_data = json.load(f)
    
    # Load the .segs.json file
    segs_file = os.path.join(scene_path, f"{scannet_scene_name}_vh_clean.segs.json")
    with open(segs_file, 'r') as f:
        segs_data = json.load(f)
    
    # Get the segment indices for each vertex
    seg_indices = np.array(segs_data['segIndices'])
    
    instances = []
    
    # Process each segment group (object instance)
    for seg_group in agg_data['segGroups']:
        instance_label = seg_group['label']
        instance_segments = seg_group['segments']
        
        # Find vertices that belong to this instance
        instance_mask = np.zeros(len(seg_indices), dtype=bool)
        for segment_id in instance_segments:
            instance_mask |= (seg_indices == segment_id)
        
        # Extract points and colors for this instance
        instance_points = vertices[instance_mask]
        instance_colors = colors[instance_mask]
        
        # Only add instances with enough points
        if len(instance_points) > 10:  # Minimum point threshold
            instances.append((instance_points, instance_colors, instance_label))
    
    return instances

def main():
    # Initialize the feature extraction model
    model_pointmae = PointMAE_Wrapper() 

    # Find which scannet scenes are downloaded
    folder_path = os.path.join(PATHS.base_dir, "data/scannet/scannet_scenes")
    downloaded_scannet_scenes = []
    for entry in os.scandir(folder_path):
        if entry.is_dir():
            downloaded_scannet_scenes.append(entry.name)

    # Iterate over all available scenes
    for scannet_scene in downloaded_scannet_scenes:
        scannet_scene_name = scannet_scene
        
        print(f"Processing scene: {scannet_scene_name}")
        
        # Load ground truth instances from ScanNet scene
        gt_instances = load_scannet_gt_instances(scannet_scene_name, PATHS.scannet_scenes)
        
        # Create output directory if it doesn't exist
        os.makedirs(PATHS.scannet_instance_output_dir, exist_ok=True)
        
        # Iterate over all found instances
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        for i, (instance_points, instance_colors, instance_label) in enumerate(gt_instances):
            # Downsample if needed
            downsample_indices = np.random.choice(len(instance_points), min(8912, len(instance_points)), replace=False)
            points_downsampled_a = instance_points[downsample_indices]
            colors_downsampled_a = instance_colors[downsample_indices]

            # Add batch dimension => shape (1, N, 3)
            points_downsampled = np.expand_dims(points_downsampled_a, axis=0).astype(np.float32)
            points_tensor = torch.from_numpy(points_downsampled).to(device)

            # Ensure the tensor is of float type
            points_tensor = points_tensor.float()

            # Forward pass
            with torch.no_grad():
                feats = model_pointmae.forward(points_tensor)

            # Convert to NumPy for saving
            feats_np = feats.cpu().numpy().squeeze(0)

            # Save feature vector, instance points, colors, and label to file
            instance_filename = f"{scannet_scene_name}_instance_{i}_{instance_label}.npz"
            np.savez(
                os.path.join(PATHS.scannet_gt_instance_output_dir, instance_filename),
                points=points_downsampled_a,
                colors=colors_downsampled_a,
                features=feats_np,
                gt_label=instance_label
            )
            
            print(f"  Saved instance {i}: {instance_label} with {len(points_downsampled_a)} points")

if __name__ == '__main__':
    main()

