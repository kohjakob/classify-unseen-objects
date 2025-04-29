import numpy as np
import torch
import MinkowskiEngine as ME

from pipeline_conf.conf import DEVICE, SCANNET_COLOR_NORMALIZE
from external.UnScene3D.datasets.scannet200.scannet200_constants import CLASS_LABELS_200

def preprocess_scannet_scene(scannet_scene_mesh, scannet_scene_segs_json, scannet_scene_aggr_json):
    """
    Preprocess loaded ScanNet scene data including mesh, segmentations, and aggregations.

    Parameters:
        scannet_scene_mesh (open3d.geometry.TriangleMesh): The loaded mesh.
        scannet_scene_segs_json (dict): The loaded segmentations data.
        scannet_scene_aggr_json (dict): The loaded aggregations data.

    Returns:
        ME.SparseTensor: The processed sparse tensor.
    """
    # Load segmentation indices
    seg_indices = np.array(scannet_scene_segs_json['segIndices'])

    # Load segmentation groups
    seg_groups = scannet_scene_aggr_json['segGroups']

    # Load normals, points and colors
    scannet_scene_mesh.compute_vertex_normals()
    points = np.asarray(scannet_scene_mesh.vertices)
    colors = np.asarray(scannet_scene_mesh.vertex_colors)

    # Map segment IDs to label IDs
    segid_to_label = {}
    for seg_group in seg_groups:
        label = seg_group['label']
        segments = seg_group['segments']
        label_id = CLASS_LABELS_200.index(label) if label in CLASS_LABELS_200 else -1
        for segid in segments:
            segid_to_label[segid] = label_id

    # Assign ground truth labels to points
    ground_truth_labels = np.array([segid_to_label.get(segid, -1) for segid in seg_indices])

    # Filter out unlabeled points
    valid_idx = ground_truth_labels != -1
    points = points[valid_idx]
    colors = colors[valid_idx]
    ground_truth_labels = ground_truth_labels[valid_idx]

    # Normalize colors
    colors = colors * 255.
    pseudo_image = colors.astype(np.uint8)[np.newaxis, :, :]
    colors_normalized = np.squeeze(SCANNET_COLOR_NORMALIZE(image=pseudo_image)["image"])

    # Voxelization
    #coords = np.floor(points / 0.02)
    coords = points

    _, _, unique_map, inverse_map = ME.utils.sparse_quantize(
        coordinates=coords, features=colors_normalized, return_index=True, return_inverse=True
    )

    sample_coordinates = coords[unique_map]
    coordinates = [torch.from_numpy(sample_coordinates).int()]
    sample_features = colors_normalized[unique_map]
    features = [torch.from_numpy(sample_features).float()]
    sample_gt_labels = ground_truth_labels[unique_map]

    coordinates, _ = ME.utils.sparse_collate(coords=coordinates, feats=features)
    features = torch.cat(features, dim=0)
    data = ME.SparseTensor(
        coordinates=coordinates,
        features=features,
        device=DEVICE,
    )

    return data, features, inverse_map, coords, colors_normalized, ground_truth_labels