import sys
import numpy as np
import open3d as o3d
from PyQt5.QtWidgets import QApplication
import torch

from pipeline_tasks.scan_loading import load_scannet_scene
from pipeline_tasks.scan_preprocessing import preprocess_scannet_scene
from pipeline_tasks.scan_segmentation import segment_scannet_scene
from pipeline_tasks.segments_postprocessing import postprocess_scannet_segments
from pipeline_tasks.segments_merging import merge_scannet_segments
from pipeline_tasks.instances_filtering import filter_scannet_instances

from pipeline_models.segmentation_model import InstanceSegmentationModel_UnScene3D
from pipeline_models.feature_extraction_model import FeatureExtractionModel_PointMAE

from pipeline_conf.conf import PATHS

from pipeline_gui.utils.color_utils import generate_distinct_colors
from pipeline_gui.utils.o3d_utils import create_scene_point_clouds
from pipeline_gui.widgets.visualization_window import VisualizationWindow

import os

def random_downsample(points, target_count=8912):
    if len(points) <= target_count:
        return points
    indices = np.random.choice(len(points), target_count, replace=False)
    return points[indices]

def main():
    # Initialize the models
    model_unscene3d = InstanceSegmentationModel_UnScene3D()
    model_pointmae = FeatureExtractionModel_PointMAE() 

    # Find which scannet scenes are download
    folder_path = PATHS.base_dir + "/data/scannet/scannet_scenes"
    downloaded_scannet_scenes = []
    for entry in os.scandir(folder_path):
        if entry.is_dir():
            downloaded_scannet_scenes.append(entry.name)

    print(downloaded_scannet_scenes)

    # Iterate over all available scenes
    for scannet_scene in downloaded_scannet_scenes:
        scannet_scene_name = scannet_scene

        # Use unscene3d to detect instances 
        scannet_scene_mesh, scannet_scene_segs_json, scannet_scene_aggr_json = load_scannet_scene(PATHS.scannet_scenes, scannet_scene_name)
        data, features, inverse_map, coords, colors_normalized, ground_truth_labels = preprocess_scannet_scene(scannet_scene_mesh, scannet_scene_segs_json, scannet_scene_aggr_json)
        outputs = segment_scannet_scene(model_unscene3d, data, features)
        masks_binary, mask_confidences, label_confidences = postprocess_scannet_segments(outputs, inverse_map, coords, colors_normalized, ground_truth_labels)
        masks_binary, mask_confidences, label_confidences = filter_scannet_instances(masks_binary, mask_confidences, label_confidences, 0.9)
        masks_binary, mask_confidences, label_confidences = merge_scannet_segments(masks_binary, mask_confidences, label_confidences)

        instances = []
        # Single out instances using binary mask into list
        for i in range(masks_binary.shape[0]):
            instance_points = coords[masks_binary[i]]
            instances.append(instance_points)
        
        # Iterate over all found instances
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        for i, instance_points in enumerate(instances):
            # Downsample if needed
            points_downsampled_a = random_downsample(instance_points, 8912)

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

            # Save feature vector and instance to file
            instance_filename = f"{scannet_scene_name}_instance_{i}.npz"
            np.savez(os.path.join(PATHS.scannet_instance_output_dir, instance_filename), points=points_downsampled_a, features=feats_np)

if __name__ == '__main__':
    main()