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

import json
import numpy as np
import matplotlib.pyplot as plt
import umap.umap_ as umap
import torch

from pipeline_conf.conf import CONFIG, DEVICE
from pipeline_models.pointmae_wrapper import PointMAE_Wrapper
from pipeline_utils.pointcloud_utils import random_downsample

# Demo for showcasing UMAP projection of feature extration with PointMAE on ModelNetCore instance pointclouds:

def main():
    model_pointmae = PointMAE_Wrapper()
    
    # Load samples of each class from ModelNetCore
    # ModelNetCore Classes: Airplane, Bag, Cap, Car, Chair, Earphone, Guitar, Knife, Lamp, Laptop
    modelnet_sample_classes = ["Car", "Guitar", "Laptop"]
    modelnet_sample_count = 20
    with open(CONFIG.shapenetcore_test_split_json_path, "r") as f:
        # JSON Format: [ [intCategoryID, stringCategoryName, stringNpyPath], ... ]
        modelnet_data = json.load(f)
    modelnet_samples = []
    for class_name in modelnet_sample_classes:
        samples = [e for e in modelnet_data if e[1] == class_name][:modelnet_sample_count]
        for sample in samples:
            modelnet_samples.append({
                "class_id": sample[0], 
                "class_name": sample[1], 
                "relative_npy_path": sample[2]
            })

    # Iterate over samples and extract features
    sample_features = []
    modelnet_gt_class_labels = []

    for sample in modelnet_samples:
        absolute_path_pointcloud_npy = os.path.join(CONFIG.shapenetcore_base_dir, sample["relative_npy_path"])
        sample_points = np.load(absolute_path_pointcloud_npy).astype(np.float32)
        # PointMAE input size = 8912
        points_downsampled = random_downsample(8912, sample_points)
        points_downsampled = points_downsampled[0].tolist()

        with torch.no_grad():
            features = model_pointmae.extract_features(points_downsampled)
        sample_features.append(features)

        modelnet_gt_class_labels.append(sample["class_name"])

    sample_features = np.array(sample_features)
  
    # Project UMAP
    umap_reducer = umap.UMAP(n_components=2)
    umap_embedding = umap_reducer.fit_transform(sample_features)

    # Plot UMAP
    plt.figure(figsize=(6, 5))
    plt.title("UMAP projection of feature extration with PointMAE on ModelNetCore instance pointclouds")
    plt.legend(loc="best")
    colors = ["blue", "red", "green", "cyan", "magenta", "orange", "purple", "brown", "pink", "gray"]
    for idx, class_name in enumerate(modelnet_sample_classes):
        cat_embedding = umap_embedding[np.array(modelnet_gt_class_labels) == class_name]
        plt.scatter(cat_embedding[:, 0], cat_embedding[:, 1], c=colors[idx], label=class_name)
    plt.show()

if __name__ == "__main__":
    main()