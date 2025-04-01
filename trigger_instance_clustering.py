import numpy as np
from pipeline_conf.conf import PATHS
import os
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
#import hdbscan

# Find detected scannet instances for which features are annotated
folder_path = PATHS.scannet_instance_output_dir
feature_extracted_scannet_instances = []
for entry in os.scandir(folder_path):
    entry = entry.name
    if entry.endswith(".npz"):
        feature_extracted_scannet_instances.append(entry)

instance_dictionary = {}

for scannet_instance in feature_extracted_scannet_instances:
    data = np.load(os.path.join(PATHS.scannet_instance_output_dir, scannet_instance))
    features = data['features']

    print("Features shape:", features.shape)

    # Save features and scannet_instance name to dict. Omit opening and saving points to minimize cache memory usage
    instance_dictionary[scannet_instance] = features

# Gather all features into a single array for clustering
all_features = np.vstack([instance_dictionary[instance] for instance in instance_dictionary])

# Compute the cosine distance matrix
cosine_distances = pairwise_distances(all_features, metric='cosine')

# Agglomerative Clustering
threshold = 0.1 
hierarchical_clustering = AgglomerativeClustering(n_clusters=None, linkage='complete', metric="cosine", distance_threshold=threshold)
hierarchical_labels = hierarchical_clustering.fit_predict(cosine_distances)

for scannet_instance, label in zip(feature_extracted_scannet_instances, hierarchical_labels):
    instance_file = os.path.join(PATHS.scannet_instance_output_dir, scannet_instance)
    data = np.load(instance_file)
    np.savez(instance_file, points=data['points'], features=data['features'], label=label)

print("Labels have been saved back to the corresponding .npz files.")

# Print hierarchical clustering results
print("Hierarchical Clustering Labels:", hierarchical_labels)

