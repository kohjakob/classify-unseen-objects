import os
import json
import numpy as np
import matplotlib.pyplot as plt
import umap.umap_ as umap
import torch
from pipeline_conf.conf import PATHS
from pipeline_models.feature_extraction_model import PointMAE_Wrapper
from pipeline_gui.utils.pointcloud_utils import random_downsample
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Demo for showcasing UMAP projection of feature extration with PointMAE on ModelNetCore instance pointclouds:
# ModelNetCore Classes: Airplane, Bag, Cap, Car, Chair, Earphone, Guitar, Knife, Lamp, Laptop

def main():
    # Load samples of each class from ModelNetCore
    modelnet_sample_classes = ["Airplain", "Bag", "Car", "Chair", "Lamp"]
    modelnet_sample_count = 20
    with open(PATHS.shapenetcore_test_split_json_path, "r") as f:
        # JSON Format: [ [intCategoryID, stringCategoryName, stringNpyPath], ... ]
        modelnet_data = json.load(f)

    modelnet_samples = []
    for class_name in modelnet_sample_classes:
        sample = [e for e in modelnet_data if e[1] == class_name][:modelnet_sample_count]
        modelnet_samples.extend({
            "class_id": sample[0], 
            "class_name": sample[1], 
            "relative_npy_path": sample[2]})

    # Instantiate PointMAE
    pointmae = PointMAE_Wrapper()

    # Iterate over samples and extract features
    features = []
    modelnet_gt_class_labels = []

    for sample in modelnet_samples:
        absolute_path_pointcloud_npy = os.path.join(PATHS.shapenetcore_base_dir, sample["relative_npy_path"])
        sample_points = np.load(absolute_path_pointcloud_npy).astype(np.float32)
        # PointMAE input size = 8912
        points_downsampled = random_downsample(8912, sample_points)
        # Torch tensor with added batch dimension (shape (1, N, 3))
        points_tensor = torch.from_numpy(np.expand_dims(points_downsampled, axis=0)).to(device)

        with torch.no_grad():
            output = pointmae.forward(points_tensor)
        output = output.cpu().numpy().squeeze(0)
        features.append(output)

        modelnet_gt_class_labels.append(sample["class_name"])

    features = np.array(features)
  
    # Project and plot UMAP
    umap_reducer = umap.UMAP(n_components=2)
    umap_embedding = umap_reducer.fit_transform(features)

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