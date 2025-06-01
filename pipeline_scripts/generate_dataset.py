# Set cuda environment variables
import os
# Project paths
os.environ["PROJECT_ROOT"] = "/home/shared/"
os.environ["CONDA_ROOT"] = "/home/shared/miniconda3"
os.environ["CONDA_DEFAULT_ENV"] = "classify-unseen-objects"
os.environ["CONDA_PYTHON_EXE"] = f"{os.environ['CONDA_ROOT']}/bin/python"
os.environ["CONDA_PREFIX"] = f"{os.environ['CONDA_ROOT']}/envs/classify-unseen-objects"
os.environ["CONDA_PREFIX_1"] = f"{os.environ['CONDA_ROOT']}/miniconda3"
# CUDA paths and settings
os.environ["CUDA_HOME"] = f"{os.environ['PROJECT_ROOT']}/cuda-11.6/toolkit"
os.environ["LD_LIBRARY_PATH"] = f"{os.environ['PROJECT_ROOT']}/cuda-11.6/toolkit/lib64"
os.environ["CUDNN_LIB_DIR"] = f"{os.environ['PROJECT_ROOT']}/cuda-11.6/toolkit/lib64"
os.environ["TORCH_CUDA_ARCH_LIST"] = "6.0;7.0;7.5"
# Compiler settings
os.environ["CXX"] = "g++-9"
os.environ["CC"] = "gcc-9"
# Update PATH to include CUDA binaries
os.environ["PATH"] = f"{os.environ['CUDA_HOME']}/bin:{os.environ.get('PATH', '')}"

import numpy as np
import sys

from tqdm import tqdm
from pipeline_utils.data_utils import load_scannet_gt_instances, list_available_scannet_scenes, load_scannet_scene_data
from pipeline_utils.pointcloud_utils import random_downsample
from pipeline_models.pointmae_wrapper import PointMAE_Wrapper
from pipeline_models.unscene3d_wrapper import UnScene3D_Wrapper
from pipeline_models.sai3d_wrapper import SAI3D_Wrapper
from pipeline_models.clustering_wrapper import Clustering_Wrapper

from pipeline_conf.conf import CONFIG

if __name__ == '__main__':
    # Check command line arguments
    # Usage: python3 generate_instance_clusters.py <instance_detection_mode> <feature_extraction_mode> <clustering_mode>
    if len(sys.argv) < 2 or \
        sys.argv[1] not in CONFIG.instance_detection_mode_dict or \
        sys.argv[2] not in CONFIG.feature_extraction_mode_dict or \
        sys.argv[3] not in CONFIG.clustering_mode_dict:
        
        print("Usage: python script_dataset.py <instance_detection_mode> <feature_extraction_mode> <clustering_mode>")
        print("Available instance detection modes:", list(CONFIG.instance_detection_mode_dict.keys()))
        print("Available feature extraction modes:", list(CONFIG.feature_extraction_mode_dict.keys()))
        print("Available clustering modes:", list(CONFIG.clustering_mode_dict.keys()))
        sys.exit(1)

    # Set modes and output directory
    instance_detection_mode = CONFIG.instance_detection_mode_dict[sys.argv[1]]["name"]
    feature_extraction_mode = CONFIG.feature_extraction_mode_dict[sys.argv[2]]["name"]
    clustering_mode = CONFIG.clustering_mode_dict[sys.argv[3]]["name"]
    output_dir = CONFIG.instance_detection_mode_dict[sys.argv[1]]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # Initialize model wrappers
    if instance_detection_mode == CONFIG.instance_detection_mode_dict["unscene3d"]["name"]:
        unscene3d_wrapper = UnScene3D_Wrapper()
    elif instance_detection_mode == CONFIG.instance_detection_mode_dict["sai3d"]["name"]:
        sai3d_wrapper = SAI3D_Wrapper()
    pointmae_wrapper = PointMAE_Wrapper() 
    clustering_wrapper = Clustering_Wrapper()

    # Process each scene in ScanNet
    # list_available_scannet_scenes_from_scannet_scenes()
    for scene_name in tqdm(list_available_scannet_scenes(), desc="Processing ScanNet scenes"):

        # Load ScanNet scene data
        points, colors, gt_labels = load_scannet_scene_data(scene_name)

        # Detect instances in scene
        if instance_detection_mode == CONFIG.instance_detection_mode_dict["unscene3d"]["name"]:
            instances = unscene3d_wrapper.detect_instances(points, colors)
        elif instance_detection_mode == CONFIG.instance_detection_mode_dict["sai3d"]["name"]:
            instances = sai3d_wrapper.detect_instances(points, colors, scene_name)
        elif instance_detection_mode == CONFIG.instance_detection_mode_dict["gt"]["name"]:
            instances = load_scannet_gt_instances(scene_name)
            
        # Downsample points, colors, gt_labels
        assert len(points) == len(colors) == len(gt_labels)
        points, colors, gt_labels = random_downsample(min(8912, len(points)), points, colors, gt_labels)

        # Extract features for detected instances and add to instance dictionaries
        for i, (instance) in enumerate(instances):
            features = pointmae_wrapper.extract_features(instance["points"])
            instance["features"] = features

        # Hierachically cluster features of all instances of scene 
        cluster_labels = clustering_wrapper.cluster_instances(output_dir, scene_name, instances)

        # Save instances
        for i, (instance, cluster_label) in enumerate(zip(instances, cluster_labels)):
            np.savez(
                os.path.join(output_dir, f"{scene_name}_instance_{i}.npz"),
                points=instance["points"], 
                colors=instance["colors"], 
                gt_label=instance["gt_label"], 
                features=instance["features"], 
                cluster_label=cluster_label
            )