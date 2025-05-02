import os
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
from pipeline_conf.conf import PATHS
from pipeline_tasks.instances_feature_loading import load_instance_features_from_npz


def cluster_instance_features(instance_features, distance_threshold=0.1, metric='cosine'):
    """
    Applies agglomerative clustering to feature vectors using cosine distance.

    Args:
        instance_features (dict): Dictionary => mapping instance names to feature arrays.
        distance_threshold (float): Threshold for clustering.

    Returns:
        np.ndarray: Cluster labels for each instance.
    """
    # Gather all features into a single array for clustering
    all_features = np.vstack(list(instance_features.values()))

    # Compute the cosine distance matrix
    cosine_distances = pairwise_distances(all_features, metric=metric)

    hierarchical_clustering = AgglomerativeClustering(
        n_clusters=None,
        #affinity='precomputed',                    # needed?
        linkage='complete',
        distance_threshold=distance_threshold
    )

    cluster_labels = hierarchical_clustering.fit_predict(cosine_distances)
    print("Clustering successful!")
    return cluster_labels

def save_cluster_labels(instance_features, cluster_labels, folder_path):
    """
    Saves cluster labels (back) into the respective .npz files.

    Args:
        instance_features (dict): Dictionary of instance filenames.
        cluster_labels (np.ndarray): Array of cluster labels.
        folder_path (str): Path to the directory containing .npz files.
    """
    for instance_name, cluster_label in zip(instance_features.keys(), cluster_labels):
        instance_file = os.path.join(folder_path, instance_name)
        data = np.load(instance_file)
        np.savez(
            instance_file,
            points=data['points'],
            features=data['features'],
            colors=data['colors'],
            gt_label=data['gt_label'],
            cluster_label=cluster_label
        )
    print("Labels have been saved back to the corresponding .npz files.")


def run_clustering_pipeline():
    """
    Loads features, runs clustering, and saves labels for scannet instances.
    """
    folder_path = PATHS.scannet_gt_instance_output_dir
    instance_features = load_instance_features_from_npz(folder_path)
    cluster_labels = cluster_instance_features(instance_features, distance_threshold=0.1,metric='cosine')
    save_cluster_labels(instance_features, cluster_labels, folder_path)

    print("Pipeline completed.")
    # Print hierarchical clustering results
    print("Hierarchical Clustering Labels:", cluster_labels)

if __name__ == "__main__":
    run_clustering_pipeline()
