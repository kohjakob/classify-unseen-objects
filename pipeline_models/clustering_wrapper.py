import numpy as np
import os
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances

class Clustering_Wrapper:

    def cluster_instances(self, output_dir, scene_name, instances):    

        instance_features = {}
        for i, (instance) in enumerate(instances):
            instance_features[i] = instance['features']

        # Arrange all features in array for clustering
        instances_features_list = np.vstack([instance_features[instance] for instance in instance_features])

        # Hierachical Clustering
        cosine_distances = pairwise_distances(instances_features_list, metric='cosine')
        cosine_distance_threshold = 0.1 
        hierarchical_clustering = AgglomerativeClustering(n_clusters=None, linkage='complete', metric="cosine", distance_threshold=cosine_distance_threshold)
        hierarchical_labels = hierarchical_clustering.fit_predict(cosine_distances)

        return hierarchical_labels
