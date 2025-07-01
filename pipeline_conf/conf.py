import os
from dataclasses import dataclass
from pathlib import Path
import albumentations as A

# !!! Do not change the following lines, they are needed for autmatic updating of the project root path !!!
# PROJECT_ROOT_VARIABLE_MARKER
PROJECT_ROOT = "/usr/people/nfraissl/CUO"
# !!! Do not change the above lines, they are needed for autmatic updating of the project root path !!!

BASE_DIR = PROJECT_ROOT + "/classify-unseen-objects"

@dataclass
class Config:
    base_dir: str = BASE_DIR
    scannet_dir: str = os.path.join(BASE_DIR, "data/scannet/scannet_scenes/")
    
    unscene: str = os.path.join(BASE_DIR, "external/UnScene3D/")
    pointmae: str = os.path.join(BASE_DIR, "external/PointMAE/")
    sai3d: str = os.path.join(BASE_DIR, "external/SAI3D/")
    semanticsam = os.path.join(BASE_DIR, "external/SAI3D/Semantic-SAM/")

    output: str = os.path.join(BASE_DIR, "output/")

    unscene3d_checkpoint: str = os.path.join(BASE_DIR, "data/checkpoints/UnScene3D_DINO_CSC_Pretrained.ckpt")

    # UnScene3D uses hydra and needs split up relative paths to load yaml
    unscene3d_config_dir: str = "../external/UnScene3D/conf"
    unscene3d_config_file: str = "config_base_instance_segmentation.yaml"

    pointmae_checkpoint: str = os.path.join(BASE_DIR, "data/checkpoints/PointMAE_ModelNet40_8k_Pretrained.pth")
    pointmae_config_file_path: str = os.path.join(BASE_DIR, "external/PointMAE/cfgs/finetune_modelnet_8k.yaml")

    # Absolute path to the test-split JSON
    shapenetcore_test_split_json_path = os.path.join(BASE_DIR, "data/shapenetcore/Shapenetcore_benchmark/test_split.json")
    # Base directory where the cat IDs (02691156, 03001627, etc.) reside
    shapenetcore_base_dir = os.path.join(BASE_DIR, "data/shapenetcore/Shapenetcore_benchmark")

    scannet_unscene3d_instance_output_dir = os.path.join(BASE_DIR, "data/scannet/scannet_instances_unscene3d")
    scannet_sai3d_instance_output_dir = os.path.join(BASE_DIR, "data/scannet/scannet_instances_sai3d")
    scannet_gt_instance_output_dir =  os.path.join(BASE_DIR, "data/scannet/scannet_instances_gt")


    sai3d_sam_checkpoint = os.path.join(BASE_DIR, "data/checkpoints/swinl_only_sam_many2many.pth")


    instance_detection_mode_dict = {    
        "gt": {
            "name": "gt",
            "output_dir": scannet_gt_instance_output_dir,
        },
        "unscene3d": {
            "name": "unscene3d",
            "output_dir": scannet_unscene3d_instance_output_dir,
        },
        "sai3d": {
            "name": "sai3d",
            "output_dir": scannet_sai3d_instance_output_dir,
        }
    }

    feature_extraction_mode_dict = {
        "pointmae": {
            "name": "pointmae",
        }
    }

    clustering_mode_dict = {
        "hierarchical": {
            "name": "hierarchical",
        }
    }

CONFIG = Config()

DEVICE = "cuda"
MAX_BATCH_SIZE = 32
DEBUG_MODE = False
SCANNET_COLOR_NORMALIZE = A.Normalize(
    mean=(0.47793125906962, 0.4303257521323044, 0.3749598901421883),
    std=(0.2834475483823543, 0.27566157565723015, 0.27018971370874995)
)
