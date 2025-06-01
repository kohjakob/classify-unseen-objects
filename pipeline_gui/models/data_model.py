import os
import numpy as np
import umap
from scipy.spatial.distance import cosine

from pipeline_conf.conf import CONFIG
from pipeline_utils.data_utils import list_available_scannet_scenes, load_scannet_scene_data
from pipeline_utils.pointcloud_utils import random_downsample

class DataModel:
    def __init__(self, instance_detection_mode):
        # Config       
        self.instance_detection_mode = instance_detection_mode
        self.instances_dir = CONFIG.instance_detection_mode_dict[instance_detection_mode]["output_dir"]

        # Logic
        self.instance_features = np.array([])
        self.instance_labels = np.array([])
        self.instance_points = []
        self.instance_colors = []
        self.instance_gt_labels = []
        self.scene_points = []
        self.scene_colors = []
        self.umap_embedding = None

        # GUI
        self.available_scannet_scenes = list_available_scannet_scenes()
        self.current_scene_idx = 0
        self.current_cluster = None
        self.current_cluster_index = 0
        self.visited_cluster_indices = []

    def reset(self):
        # Reset all state variables
        self.instance_features = np.array([])
        self.instance_labels = np.array([])
        self.instance_points = []
        self.instance_colors = []
        self.instance_gt_labels = []
        self.scene_points = []
        self.scene_colors = []
        self.umap_embedding = None
        self.current_scene_idx = 0
        self.current_cluster = None
        self.current_cluster_index = 0
        self.visited_cluster_indices = []
        
    def load_scannet_scene_data(self, scene_name):        
        self.reset()

        # Load scene data
        self.scene_points, self.scene_colors, gt_labels = load_scannet_scene_data(scene_name)
        self.scene_points, self.scene_colors = random_downsample(5000, self.scene_points, self.scene_colors)
    
        # Load instances data
        instance_features = []
        instance_labels = []
        for instance in  [e.name for e in os.scandir(self.instances_dir) if e.name.endswith('.npz') and e.name.startswith(scene_name)]:
            instance_data = np.load(os.path.join(self.instances_dir, instance), allow_pickle=True)
            instance_features.append(instance_data['features'])
            instance_labels.append(instance_data['cluster_label'])
            self.instance_points.append(instance_data['points'])
            self.instance_colors.append(instance_data['colors'])
            self.instance_gt_labels.append(instance_data['gt_label'])
        self.instance_features = np.array(instance_features)
        self.instance_labels = np.array(instance_labels)

        # Calculate UMAP embedding
        reducer = umap.UMAP()
        self.umap_embedding = reducer.fit_transform(self.instance_features)

        # Set cluster selection state
        self.current_cluster = np.unique(self.instance_labels)[0]
        self.select_cluster_point()

        return True
    
    def select_cluster_point(self):
        # If no current cluster, select first
        if self.current_cluster is None and len(self.instance_labels) > 0:
            unique_clusters = np.unique(self.instance_labels)
            self.current_cluster = unique_clusters[0]
        
        # If current cluster selected, select element from same cluster with 
        # element closest to current centroid
        if self.current_cluster is not None:
            cluster_indices = np.where(self.instance_labels == self.current_cluster)[0]
            
            if len(cluster_indices) > 0:
                cluster_features = self.instance_features[cluster_indices]
                centroid = np.mean(cluster_features, axis=0)
                
                closest_idx = None
                min_distance = float('inf')
                for idx in cluster_indices:
                    # Cosine distance (1 - cosine similarity)
                    distance = cosine(self.instance_features[idx], centroid)
                    if distance < min_distance:
                        min_distance = distance
                        closest_idx = idx
                
                if closest_idx is not None:
                    self.current_cluster_index = closest_idx
                    self.visited_cluster_indices = [closest_idx]
    
    def find_furthest_point(self):
        # Find the furthest point in the current cluster from the visited points
        cluster_indices = np.where(self.instance_labels == self.current_cluster)[0]
        unvisited = [idx for idx in cluster_indices if idx not in self.visited_cluster_indices]
        
        if not unvisited:
            # Reset visited list but keep the initial point
            self.visited_cluster_indices = [self.visited_cluster_indices[0]] if self.visited_cluster_indices else []
            unvisited = [idx for idx in cluster_indices if idx not in self.visited_cluster_indices]
        
        if not unvisited:
            return None
        
        furthest_idx = None
        max_min_distance = -float('inf')
        
        for idx in unvisited:
            min_distance = float('inf')
            for visited_idx in self.visited_cluster_indices:
                # Cosine distance (1 - cosine similarity)
                distance = cosine(self.instance_features[idx], self.instance_features[visited_idx])
                min_distance = min(min_distance, distance)
            
            if min_distance > max_min_distance:
                max_min_distance = min_distance
                furthest_idx = idx
        
        return furthest_idx
    
    def move_to_next_instance(self):
        """Move to the next instance in the current cluster"""
        next_idx = self.find_furthest_point()
        
        if next_idx is not None:
            self.current_cluster_index = next_idx
            self.visited_cluster_indices.append(next_idx)
            return True
        return False
    
    def move_to_next_cluster(self):
        """Move to the next cluster"""
        unique_clusters = np.unique(self.instance_labels)
        if len(unique_clusters) > 0:
            current_idx = np.where(unique_clusters == self.current_cluster)[0][0]
            next_idx = (current_idx + 1) % len(unique_clusters)
            self.current_cluster = unique_clusters[next_idx]
            self.visited_cluster_indices = []
            self.select_cluster_point()
            return True
        return False
    
    def move_to_next_scan(self):
        """Move to the next scan"""
        if not self.available_scannet_scenes:
            return False
        
        self.current_scene_idx = (self.current_scene_idx + 1) % len(self.available_scannet_scenes)
        return self.load_scannet_scene_data(self.available_scannet_scenes[self.current_scene_idx])
    
    def move_to_previous_scan(self):
        """Move to the previous scan"""
        if not self.available_scannet_scenes:
            return False
        
        self.current_scene_idx = (self.current_scene_idx - 1) % len(self.available_scannet_scenes)
        return self.load_scannet_scene_data(self.available_scannet_scenes[self.current_scene_idx])
    
    def select_point(self, index):
        """Select a point by index"""
        if index < len(self.instance_labels):
            self.current_cluster_index = index
            self.current_cluster = self.instance_labels[index]
            self.visited_cluster_indices = [index]
            return True
        return False
    
    def get_current_scan_name(self):
        """Get the name of the current scan"""
        if self.available_scannet_scenes and 0 <= self.current_scene_idx < len(self.available_scannet_scenes):
            return self.available_scannet_scenes[self.current_scene_idx]
        return "None"
    
    def get_current_point_data(self):
        """Get the point data for the current instance"""
        if 0 <= self.current_cluster_index < len(self.instance_points):
            return np.array(self.instance_points[self.current_cluster_index]), np.array(self.instance_colors[self.current_cluster_index])
        return None, None
    
    def get_current_gt_label(self):
        """Get the ground truth label for the current instance"""
        if 0 <= self.current_cluster_index < len(self.instance_gt_labels):
            return self.instance_gt_labels[self.current_cluster_index]
        return "None"