import sys
import numpy as np
import open3d as o3d
from PyQt5.QtWidgets import QApplication

from pipeline_tasks.scan_loading import load_scannet_scene
from pipeline_tasks.scan_preprocessing import preprocess_scannet_scene
from pipeline_tasks.scan_segmentation import segment_scannet_scene
from pipeline_tasks.segments_postprocessing import postprocess_scannet_segments
from pipeline_tasks.segments_merging import merge_scannet_segments
from pipeline_tasks.instances_filtering import filter_scannet_instances

from pipeline_models.segmentation_model import InstanceSegmentationModel_UnScene3D

from pipeline_conf.conf import PATHS

from pipeline_gui.utils.color_utils import generate_distinct_colors
from pipeline_gui.utils.o3d_utils import create_scene_point_clouds
from pipeline_gui.widgets.visualization_window import VisualizationWindow

#UnScene3D
def main():
    # Initialize the model
    model = InstanceSegmentationModel_UnScene3D()
    scannet_scene_name = 'scene0000_00'

    # Load and process the scene
    scannet_scene_mesh, scannet_scene_segs_json, scannet_scene_aggr_json = load_scannet_scene(PATHS.scannet_scenes, scannet_scene_name)
    data, features, inverse_map, coords, colors_normalized, ground_truth_labels = preprocess_scannet_scene(scannet_scene_mesh, scannet_scene_segs_json, scannet_scene_aggr_json)
    outputs = segment_scannet_scene(model, data, features)
    masks_binary, mask_confidences, label_confidences = postprocess_scannet_segments(outputs, inverse_map, coords, colors_normalized, ground_truth_labels)
    masks_binary, mask_confidences, label_confidences = merge_scannet_segments(masks_binary, mask_confidences, label_confidences)
    masks_binary, mask_confidences, label_confidences = filter_scannet_instances(masks_binary, mask_confidences, label_confidences, 0.9)

    app = QApplication(sys.argv)
    original_cloud, foreground_cloud, background_cloud = create_scene_point_clouds(coords, colors_normalized, masks_binary)
    window = VisualizationWindow(original_cloud, foreground_cloud, background_cloud, scannet_scene_name)
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()