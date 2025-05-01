import os
from pipeline_conf.conf import PATHS

def get_available_scannet_scenes():
    """Get a list of available ScanNet scene names from the instance output directory"""
    scene_names = set()
    
    # Find all .npz files in the instance output directory
    for filename in os.listdir(PATHS.scannet_gt_instance_output_dir):
        if filename.endswith('.npz'):
            # Extract scene name (e.g., "scene0007_00" from "scene0007_00_instance_1.npz")
            parts = filename.split('_')
            if len(parts) >= 2:
                scene_name = f"{parts[0]}_{parts[1]}"
                scene_names.add(scene_name)
    
    return sorted(list(scene_names))
    