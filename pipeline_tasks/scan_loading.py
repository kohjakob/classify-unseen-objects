import open3d as o3d
import json
import os

def load_scannet_scene(scannet_scene_base_dir, scannet_scene_name):
    """
    Loads a ScanNet scene including the mesh, segmentations, and aggregations.

    Parameters:
        scannet_scene_base_dir (str): The base directory where ScanNet scenes are stored.
        scannet_scene_name (str): The name of the ScanNet scene to load.

    Returns:
        tuple: A tuple containing the loaded mesh (open3d.geometry.TriangleMesh), 
               the segmentations data (dict), 
               and the aggregations data (dict).
    """
    scannet_scene_dir = os.path.join(scannet_scene_base_dir, scannet_scene_name)

    # Load mesh
    scannet_scene_mesh_file = os.path.join(scannet_scene_dir, f"{scannet_scene_name}_vh_clean.ply")
    scannet_scene_mesh = o3d.io.read_triangle_mesh(str(scannet_scene_mesh_file))

    # Load segmentations
    scannet_scene_segs_file = os.path.join(scannet_scene_dir, f"{scannet_scene_name}_vh_clean.segs.json")
    with open(scannet_scene_segs_file, 'r') as f:
        scannet_scene_segs_json = json.load(f)

    # Load aggregations
    scannet_scene_aggr_file = os.path.join(scannet_scene_dir, f"{scannet_scene_name}_vh_clean.aggregation.json")
    with open(scannet_scene_aggr_file, 'r') as f:
        scannet_scene_aggr_json = json.load(f)

    return scannet_scene_mesh, scannet_scene_segs_json, scannet_scene_aggr_json