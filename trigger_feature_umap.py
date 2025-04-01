import os
import json
import numpy as np
import matplotlib.pyplot as plt
import umap.umap_ as umap
import torch

from pipeline_conf.conf import PATHS
from pipeline_models.feature_extraction_model import FeatureExtractionModel_PointMAE

def random_downsample(points, target_count=8912):
    """
    Randomly downsample a point cloud to target_count points.
    If the point cloud has fewer points, returns the original array.
    """
    if len(points) <= target_count:
        return points
    indices = np.random.choice(len(points), target_count, replace=False)
    return points[indices]

def main():

    # Adjusted categories of interest (10 classes, 20 objects each)
    categories_of_interest = [
        "Airplane",
        "Bag",
        #"Cap",
        "Car",
        "Chair",
        #"Earphone",
        #"Guitar",
        "Knife",
        "Lamp",
        #"Laptop"
    ]
    num_objects_per_category = 20

    # Load test-split JSON
    with open(PATHS.shapenetcore_test_split_json_path, "r") as f:
        data = json.load(f)

    # Expected structure: [ [intCategoryID, stringCategoryName, stringNpyPath], ... ]
    root_list = data

    # Gather the selected items for each category
    selected_items = []
    for cat in categories_of_interest:
        cat_items = [item for item in root_list if item[1] == cat][:num_objects_per_category]
        selected_items.extend(cat_items)

    model = FeatureExtractionModel_PointMAE()

    feature_vectors = []
    labels = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for obj in selected_items:
        # obj: [intID, categoryName, "catID/points/xxxx.npy"]
        category_name = obj[1]
        relative_npy_path = obj[2]

        npy_path = os.path.join(PATHS.shapenetcore_base_dir, relative_npy_path)
        if os.path.isfile(npy_path):
            points = np.load(npy_path).astype(np.float32)
            # Downsample if needed
            points_downsampled = random_downsample(points, 8912)

            # Add batch dimension => shape (1, N, 3)
            points_downsampled = np.expand_dims(points_downsampled, axis=0)
            points_tensor = torch.from_numpy(points_downsampled).to(device)

            # Forward pass
            with torch.no_grad():
                feats = model.forward(points_tensor)

            # Convert to NumPy for UMAP
            feats_np = feats.cpu().numpy().squeeze(0)
            feature_vectors.append(feats_np)
            labels.append(category_name)
        else:
            print(f"File not found: {npy_path}")

    feature_vectors = np.array(feature_vectors)
    if feature_vectors.shape[0] == 0:
        print("No feature vectors to process.")
        return

    # Apply UMAP to reduce extracted features to 2D
    reducer = umap.UMAP(n_components=2)
    embedding = reducer.fit_transform(feature_vectors)

    # Plot each category in a different color
    plt.figure(figsize=(6, 5))
    colors = ["blue", "red", "green", "cyan", "magenta", "orange", "purple", "brown", "pink", "gray"]
    for idx, cat in enumerate(categories_of_interest):
        cat_embedding = embedding[np.array(labels) == cat]
        plt.scatter(cat_embedding[:, 0], cat_embedding[:, 1], c=colors[idx], label=cat)

    plt.legend(loc="best")
    plt.title("2D UMAP of ShapeNet Feature Vectors")
    plt.savefig("/home/shared/classify-unseen-objects/output/umap_visualization/umap_shapenet_features_3.png")
    plt.show()

if __name__ == "__main__":
    main()