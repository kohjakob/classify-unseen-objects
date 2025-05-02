import os
import numpy as np
import open3d as o3d
import json
from pipeline_conf.conf import PATHS

def list_available_instances_from_output_dir(output_dir = PATHS.scannet_gt_instance_output_dir):
    gt_instances = []
    for entry in os.scandir(output_dir):
        entry = entry.name
        if entry.endswith(".npz"):
            gt_instances.append(entry)
    return gt_instances

def list_available_scannet_scenes_from_output_dir(output_dir = PATHS.scannet_gt_instance_output_dir):
    scene_names = set()
    for filename in os.listdir(output_dir):
        if filename.endswith('.npz'):
            # Extract scene name (e.g., "scene0007_00" from "scene0007_00_instance_1.npz")
            parts = filename.split('_')
            if len(parts) >= 2:
                scene_name = f"{parts[0]}_{parts[1]}"
                scene_names.add(scene_name)
    return sorted(list(scene_names))

def list_available_scannet_scenes_from_scannet_scenes():
    available_scannet_scenes = []
    for entry in os.scandir(os.path.join(PATHS.scannet_scenes)):
        if entry.is_dir():
            available_scannet_scenes.append(entry.name)
    return available_scannet_scenes

def load_scannet_gt_instances(scannet_scene_name):
    """
    Load points, colors and ground truth class label of all instances of specified ScanNet scene
    
    Args:
        scannet_scene_name: Name of the ScanNet scene (e.g., 'scene0000_00')
        
    Returns:
        List of tuples containing (instance_points, instance_colors, label)
    """
    scene_path = os.path.join(PATHS.scannet_scenes, scannet_scene_name)

    mesh_file = os.path.join(scene_path, f"{scannet_scene_name}_vh_clean.ply")
    mesh = o3d.io.read_triangle_mesh(mesh_file)
    vertices = np.asarray(mesh.vertices)
    colors = np.asarray(mesh.vertex_colors)
    
    agg_file = os.path.join(scene_path, f"{scannet_scene_name}_vh_clean.aggregation.json")
    with open(agg_file, 'r') as f:
        agg_data = json.load(f)
    
    segs_file = os.path.join(scene_path, f"{scannet_scene_name}_vh_clean.segs.json")
    with open(segs_file, 'r') as f:
        segs_data = json.load(f)
    
    seg_indices = np.array(segs_data['segIndices'])
    instances = []
    
    # segGroups correspond to individual instances
    for seg_group in agg_data['segGroups']:
        instance_label = seg_group['label']
        instance_segments = seg_group['segments']
        
        # Find vertices and colors belonging to current segGroup
        instance_mask = np.zeros(len(seg_indices), dtype=bool)
        for segment_id in instance_segments:
            instance_mask |= (seg_indices == segment_id)

        instance_points = vertices[instance_mask]
        instance_colors = colors[instance_mask]
        
        # Threshold to filter very small segGroups
        if len(instance_points) > 10: 
            instances.append((instance_points, instance_colors, instance_label))
    
    return instances
