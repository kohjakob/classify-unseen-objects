import os
import numpy as np
import open3d as o3d
import json
import plyfile
from pipeline_conf.conf import CONFIG
from external.UnScene3D.datasets.scannet200.scannet200_constants import CLASS_LABELS_200
from pipeline_utils.pointcloud_utils import random_downsample

def list_available_scannet_scenes():
    available_scannet_scenes = []
    for entry in os.scandir(os.path.join(CONFIG.scannet_dir)):
        if entry.is_dir():
            available_scannet_scenes.append(entry.name)
    return available_scannet_scenes

def load_scannet_gt_instances(scannet_scene_name):
    scene_path = os.path.join(CONFIG.scannet_dir, scannet_scene_name)

    # Use plyfile instead of open3d to read mesh
    mesh_file = os.path.join(scene_path, f"{scannet_scene_name}_vh_clean_2.ply")
    ply = plyfile.PlyData.read(mesh_file)
    
    # Extract points (vertices)
    xs = np.array(ply["vertex"].data['x'])[:, None]
    ys = np.array(ply["vertex"].data['y'])[:, None]
    zs = np.array(ply["vertex"].data['z'])[:, None]
    vertices = np.concatenate((xs, ys, zs), axis=-1)
    
    # Extract colors
    rs = np.array(ply["vertex"].data['red'])[:, None]
    gs = np.array(ply["vertex"].data['green'])[:, None]
    bs = np.array(ply["vertex"].data['blue'])[:, None]
    colors = np.concatenate((rs, gs, bs), axis=-1)
    
    agg_file = os.path.join(scene_path, f"{scannet_scene_name}_vh_clean.aggregation.json")
    with open(agg_file, 'r') as f:
        agg_data = json.load(f)
    
    segs_file = os.path.join(scene_path, f"{scannet_scene_name}_vh_clean_2.0.010000.segs.json")
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
            instances.append({
                "points": instance_points,
                "colors": instance_colors,
                "gt_label": instance_label,
                "binary_mask": instance_mask,
            })
    
    return instances

def load_scannet_scene_data(scene_name):
    scene_dir = os.path.join(CONFIG.scannet_dir, scene_name)
    ply_path = os.path.join(scene_dir, f"{scene_name}_vh_clean_2.ply")
    ply = plyfile.PlyData.read(ply_path)
    
    # Extract points
    xs = np.array(ply["vertex"].data['x'])[:, None]
    ys = np.array(ply["vertex"].data['y'])[:, None]
    zs = np.array(ply["vertex"].data['z'])[:, None]
    points = np.concatenate((xs, ys, zs), axis=-1)
    
    # Extract colors
    rs = np.array(ply["vertex"].data['red'])[:, None]
    gs = np.array(ply["vertex"].data['green'])[:, None]
    bs = np.array(ply["vertex"].data['blue'])[:, None]
    colors = np.concatenate((rs, gs, bs), axis=-1)
    
    # Load segmentation data
    with open(os.path.join(scene_dir, f"{scene_name}_vh_clean_2.0.010000.segs.json"), 'r') as f:
        segmentation_json = json.load(f)

    with open(os.path.join(scene_dir, f"{scene_name}_vh_clean.aggregation.json"), 'r') as f:
        aggregation_json = json.load(f)
    
    # Map segment IDs to label IDs
    segmentationid_to_label = {}
    for segmentation_group in aggregation_json['segGroups']:
        label = segmentation_group['label']
        segments = segmentation_group['segments']
        label_id = CLASS_LABELS_200.index(label) if label in CLASS_LABELS_200 else -1
        for segmentation_id in segments:
            segmentationid_to_label[segmentation_id] = label_id

    # Assign ground truth labels to points
    segmentation_indices = np.array(segmentation_json['segIndices'])
    gt_labels = np.array([segmentationid_to_label.get(segmentation_idx, -1) for segmentation_idx in segmentation_indices])
    
    # Filter out unlabeled points
    """
    valid_idx = gt_labels != -1
    points = points[valid_idx]
    colors = colors[valid_idx]
    gt_labels = gt_labels[valid_idx]
    """
    
    return points, colors, gt_labels

def load_scannet_gt_instances_open3d(scannet_scene_name):
    scene_path = os.path.join(CONFIG.scannet_dir, scannet_scene_name)

    mesh_file = os.path.join(scene_path, f"{scannet_scene_name}_vh_clean_2.ply")
    mesh = o3d.io.read_triangle_mesh(mesh_file)
    vertices = np.asarray(mesh.vertices)
    colors = np.asarray(mesh.vertex_colors)
    
    agg_file = os.path.join(scene_path, f"{scannet_scene_name}_vh_clean.aggregation.json")
    with open(agg_file, 'r') as f:
        agg_data = json.load(f)
    
    segs_file = os.path.join(scene_path, f"{scannet_scene_name}_vh_clean_2.0.010000.segs.json")
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

            instances.append({
                "points": instance_points,
                "colors": instance_colors,
                "gt_label": instance_label,
                "binary_mask": instance_mask,
            })
    
    return instances

def load_scannet_scene_data_open3d(scene_name):
    scene_dir = os.path.join(CONFIG.scannet_dir, scene_name)

    # Load mesh, segmentations, aggregations 
    mesh = o3d.io.read_triangle_mesh(str(os.path.join(scene_dir, f"{scene_name}_vh_clean_2.ply")))

    with open(os.path.join(scene_dir, f"{scene_name}_vh_clean_2.0.010000.segs.json"), 'r') as f:
        segmentation_json = json.load(f)

    with open(os.path.join(scene_dir, f"{scene_name}_vh_clean.aggregation.json"), 'r') as f:
        aggregation_json = json.load(f)

    # Extract segment indices, segment groups, normals points, colors
    mesh.compute_vertex_normals()
    segmentation_indices = np.array(segmentation_json['segIndices'])
    segmentation_groups = aggregation_json['segGroups']
    points = np.asarray(mesh.vertices)
    colors = np.asarray(mesh.vertex_colors)
    colors = colors * 255.

    # Map segment IDs to label IDs
    segmentationid_to_label = {}
    for segmentation_group in segmentation_groups:
        label = segmentation_group['label']
        segments = segmentation_group['segments']
        label_id = CLASS_LABELS_200.index(label) if label in CLASS_LABELS_200 else -1
        for segmentation_id in segments:
            segmentationid_to_label[segmentation_id] = label_id

    # Assign ground truth labels to points
    gt_labels = np.array([segmentationid_to_label.get(segmentation_idx, -1) for segmentation_idx in segmentation_indices])

    # Filter out unlabeled points
    valid_idx = gt_labels != -1
    points = points[valid_idx]
    colors = colors[valid_idx]
    gt_labels = gt_labels[valid_idx]

    return points, colors, gt_labels

