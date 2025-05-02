import os
import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
import sys

from pipeline_utils.data_utils import list_available_instances_from_output_dir
from pipeline_utils.data_utils import load_scannet_gt_instances, list_available_scannet_scenes_from_scannet_scenes
from pipeline_utils.pointcloud_utils import random_downsample
from pipeline_tasks.scan_loading import load_scannet_scene
from pipeline_tasks.scan_preprocessing import preprocess_scannet_scene
from pipeline_tasks.scan_segmentation import segment_scannet_scene
from pipeline_tasks.segments_postprocessing import postprocess_scannet_segments
from pipeline_tasks.segments_merging import merge_scannet_segments
from pipeline_tasks.instances_filtering import filter_scannet_instances
from pipeline_models.segmentation_model import Unscene3D_Wrapper
from pipeline_models.feature_extraction_model import PointMAE_Wrapper
from pipeline_conf.conf import PATHS

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DatasetGenerator():
    def __init__(self):
        pass

    def generate_from_gt(self):
        os.makedirs(PATHS.scannet_gt_instance_output_dir, exist_ok=True)
        self.output_dir = PATHS.scannet_gt_instance_output_dir
        self.instance_detection_mode = 'gt'
        self._instance_detection()
        self._feature_extraction()
        self._clustering()

    def generate_from_unscene3d(self):
        os.makedirs(PATHS.scannet_instance_output_dir, exist_ok=True)
        self.output_dir = PATHS.scannet_instance_output_dir
        self.instance_detection_mode = 'unscene3d'
        self._instance_detection()
        self._feature_extraction()
        self._clustering()

    def _instance_detection(self):
        model_unscene3d = Unscene3D_Wrapper()

        # ======= Load instances of available scannet scenes =======
        available_scannet_scenes = list_available_scannet_scenes_from_scannet_scenes()

        for scene_name in available_scannet_scenes:
            if self.instance_detection_mode == "unscene3d":

                # ======= Detect instances using UnScene3D =======
                scannet_scene_mesh, scannet_scene_segs_json, scannet_scene_aggr_json = load_scannet_scene(PATHS.scannet_scenes, scene_name)
                data, features, inverse_map, coords, colors_normalized, ground_truth_labels = preprocess_scannet_scene(scannet_scene_mesh, scannet_scene_segs_json, scannet_scene_aggr_json, voxelization=True)
                outputs = segment_scannet_scene(model_unscene3d, data, features)
                masks_binary, mask_confidences, label_confidences = postprocess_scannet_segments(outputs, inverse_map, coords, colors_normalized, ground_truth_labels)
                masks_binary, mask_confidences, label_confidences = filter_scannet_instances(masks_binary, mask_confidences, label_confidences, 0.9)
                masks_binary, mask_confidences, label_confidences = merge_scannet_segments(masks_binary, mask_confidences, label_confidences)

                for i in range(masks_binary.shape[0]):
                    instance_points = coords[masks_binary[i]]
                    instance_colors = colors_normalized[masks_binary[i]]
                    points_downsampled, colors_downsampled = random_downsample(min(8912, len(instance_points)), instance_points, instance_colors)
                    
                    # ======= Save detected instances to file =======
                    instance_name = f"{scene_name}_instance_{i}.npz"
                    np.savez(
                        os.path.join(self.output_dir, instance_name),
                        points=points_downsampled,
                        colors=colors_downsampled,
                        gt_label="Not specified"
                    )
                    print(f"  Saved instance (from unscene3d): {instance_name}")
            
            elif self.instance_detection_mode == "gt":

                # ======= Find gt instances from ScanNet =======
                instances = load_scannet_gt_instances(scene_name)
                
                for i, (gt_instance_points, gt_instance_colors, gt_instance_label) in enumerate(instances):
                    points_downsampled, colors_downsampled = random_downsample(min(8912, len(gt_instance_points)), gt_instance_points, gt_instance_colors)

                    # ======= Save gt instances to file =======
                    instance_name = f"{scene_name}_instance_{i}.npz"
                    np.savez(
                        os.path.join(self.output_dir, instance_name),
                        points=points_downsampled,
                        colors=colors_downsampled,
                        gt_label=gt_instance_label
                    )
                    print(f"  Saved instance (from gt): {instance_name}")

    def _feature_extraction(self):
        model_pointmae = PointMAE_Wrapper() 

        # ======= Load detected instances =======
        scannet_instances = list_available_instances_from_output_dir(self.output_dir)

        # ======= Extract features of instances of available scannet scenes =======
        for instance_name in scannet_instances:
            instance_data = np.load(os.path.join(self.output_dir, instance_name))

            # Add batch dimension => shape (1, N, 3)
            points_downsampled_expanded = np.expand_dims(instance_data['points'], axis=0).astype(np.float32)
            points_tensor = torch.from_numpy(points_downsampled_expanded).to(device)
            points_tensor = points_tensor.float()

            with torch.no_grad():
                output = model_pointmae.forward(points_tensor)
            features = output.cpu().numpy().squeeze(0)

            # ======= Save extracted features to file =======
            np.savez(
                os.path.join(self.output_dir, instance_name),
                points=instance_data['points'],
                colors=instance_data['colors'],
                gt_label=instance_data['gt_label'],
                features=features,
            )
            print(f"  Saved features: {instance_name}")

    def _clustering(self):
        # ======= Load available instance =======
        scannet_instances = list_available_instances_from_output_dir(self.output_dir)
        instance_features = {}

        for instance_name in scannet_instances:
            instance_data = np.load(os.path.join(self.output_dir, instance_name))
            # Omit opening and saving points to minimize cache memory usage
            print(instance_name)
            instance_features[instance_name] = instance_data['features']

        # ======= Cluster instances using features =======
        # Arrange all features in array for clustering
        instances_features_list = np.vstack([instance_features[instance] for instance in instance_features])

        # Hierachical Clustering
        cosine_distances = pairwise_distances(instances_features_list, metric='cosine')
        cosine_distance_threshold = 0.1 
        hierarchical_clustering = AgglomerativeClustering(n_clusters=None, linkage='complete', metric="cosine", distance_threshold=cosine_distance_threshold)
        hierarchical_labels = hierarchical_clustering.fit_predict(cosine_distances)

        # ======= Save clusters to file =======
        for instance_name, cluster_label in zip(scannet_instances, hierarchical_labels):
            instance_file = os.path.join(self.output_dir, instance_name)
            instance_data = np.load(instance_file)
            np.savez(
                instance_file, 
                points=instance_data['points'], 
                colors=instance_data['colors'], 
                gt_label=instance_data['gt_label'], 
                features=instance_data['features'], 
                cluster_label=cluster_label
            )
            print(f"  Saved clusters: {instance_name}")

if __name__ == '__main__':
    dataset_generator = DatasetGenerator()

    if sys.argv[1] == "gt":
        dataset_generator.generate_from_gt()
    elif sys.argv[1] == "unscene3d":
        dataset_generator.generate_from_unscene3d()
    else:
        raise ValueError("Specify instance detection mode!")