import os
import sys
import glob
import cv2
import numpy as np
import argparse
import plyfile
from natsort import natsorted

from pipeline_conf.conf import CONFIG
from pipeline_utils.data_utils import load_scannet_scene_data

sys.path.insert(0, CONFIG.semanticsam)
sys.path.insert(0, CONFIG.sai3d)
from helpers.sam_utils import get_sam_by_iou, num_to_natural, viz_mask, my_prepare_image
from semantic_sam import build_semantic_sam, SemanticSamAutomaticMaskGenerator
from sai3d import ScanNet_SAI3D

class SAI3D_Wrapper():
    def __init__(self):       
        self.view_freq=5
        self.level = [3,]  # instance level

        os.chdir(CONFIG.semanticsam)
        self.sam_model = build_semantic_sam(
            model_type='L', 
            ckpt=CONFIG.sai3d_sam_checkpoint
        )
        self.mask_generator = SemanticSamAutomaticMaskGenerator(
            self.sam_model,
            level=self.level # model_type: 'L' / 'T', depends on your checkpoint
        )

    def detect_instances(self, points, colors, scene_name):
        # SAM 2D mask generation
        self.generate_SAM_masks(scene_name)

        # SAI3D 2D mask region growing and 2D-to-3D projection 
        points, mask = self.project_sai3d_mask(points, colors, scene_name)

        # Extract instances
        instances = []
        for mask_label in np.unique(mask):
            instance_colors = colors[np.where(mask == mask_label)[0]]
            instance_points = points[np.where(mask == mask_label)[0]]

            instances.append({
                "points": instance_points,
                "colors": instance_colors,
                "gt_label": "pseudoclass_" + str(mask_label),
                "binary_mask": mask == mask_label,
            })

        return instances

    def generate_SAM_masks(self, scene_name):
        scannet_scene_dir = os.path.join(CONFIG.scannet_dir, scene_name)
        scene_color_input_dir = os.path.join(scannet_scene_dir, "posed_images", "color")
        color_paths = natsorted(glob.glob(os.path.join(scene_color_input_dir, '*.jpg')))
        scene_sam_output_dir = os.path.join(scannet_scene_dir, "posed_images", "semantic-sam")
        os.makedirs(scene_sam_output_dir, exist_ok=True)
    
        for color_path in color_paths:
            color_name = os.path.basename(color_path)
            num = int(color_name[-9:-4])
            if num % self.view_freq != 0:
                continue

            # Check if output files already exist
            mask_color_path = os.path.join(scene_sam_output_dir, f'maskcolor_{color_name[:-4]}.png')
            mask_raw_path = os.path.join(scene_sam_output_dir, f'maskraw_{color_name[:-4]}.png') 
            if os.path.exists(mask_color_path) and os.path.exists(mask_raw_path):
                continue

            original_image, input_image = my_prepare_image(image_pth=color_path)
            labels = get_sam_by_iou(input_image, self.mask_generator)
            # labels = get_sam_by_area(input_image,mask_generator)
            color_mask = viz_mask(labels)
            labels = num_to_natural(labels) + 1  # 0 is background

            cv2.imwrite(os.path.join(
                scene_sam_output_dir, 
                f'maskcolor_{color_name[:-4]}.png'), 
                color_mask
            )

            cv2.imwrite(os.path.join(
                scene_sam_output_dir, 
                f'maskraw_{color_name[:-4]}.png'), 
                labels
            )

    def project_sai3d_mask(self, points, colors, scene_name):
        scannet_scene_dir = os.path.join(CONFIG.scannet_dir, scene_name)

        # Taken from sai3d.py default parameters
        view_freq = 5
        thres_connect = np.linspace(float(0.9), float(0.5), int(5))
        thres_dis = 0.15
        max_neighbor_distance = 2
        similar_metric = '2-norm'
        need_semantic = False
        args = argparse.Namespace(
            scannetpp=False, # Not using ScanNet++
            need_semantic = False, # Only need pseudo-classes
            thres_connect=thres_connect,  
            thres_dis=thres_dis,
            max_neighbor_distance=max_neighbor_distance, 
            similar_metric=similar_metric,
            view_freq=view_freq,
            mask_name="semantic-sam", # Not used in SAI3D_Projection_Wrapper 
            base_dir=scannet_scene_dir,
            scene_id=scene_name,
            test=False, # Not used in SAI3D_Projection_Wrapper
            text=None, # Not used in SAI3D_Projection_Wrapper
            dis_decay=0.5,
            thres_merge=200,
            thres_trunc=0.0,
            from_points_thres=0,
            use_torch=False,
        )

        agent = SAI3D_Projection_Wrapper(points, args)

        agent.init_data(
            scene_id=args.scene_id,
            base_dir=args.base_dir,
            mask_name=args.mask_name,
        )
        
        labels_fine_global = agent.assign_label(
            points, 
            thres_connect=args.thres_connect, 
            vis_dis=args.thres_dis,
            max_neighbor_distance=args.max_neighbor_distance, 
            similar_metric=args.similar_metric
        )

        return points, labels_fine_global

class SAI3D_Projection_Wrapper(ScanNet_SAI3D):
    def __init__(self, points, args=None):
        self.scannetpp = args.scannetpp
        self.view_freq = args.view_freq
        super().__init__(points, args)

    def init_data(self, scene_id, base_dir, mask_name, need_semantic=False):
        self.poses = []
        self.color_intrinsics = []
        self.depth_intrinsics = []
        self.masks = []
        self.depths = []
        self.semantic_masks = []

        scene_color_input_dir = os.path.join(base_dir, "posed_images", "color")
        scene_pose_input_dir = os.path.join(base_dir, "posed_images", "pose")
        scene_depth_input_dir = os.path.join(base_dir, "posed_images", "depth")
        scene_intrinsic_input_dir = os.path.join(base_dir, "posed_images", "intrinsic")
        scene_mask_input_dir = os.path.join(base_dir, "posed_images", "semantic-sam")
        
        color_frames = natsorted(glob.glob(os.path.join(scene_color_input_dir, '*.jpg')))
        for color_path in color_frames:

            frame_name = os.path.basename(color_path)
            frame_number = int(frame_name[-9:-4])
            if frame_number % self.view_freq != 0:
                continue
            frame_number = str(frame_number)

            pose = np.loadtxt(os.path.join(scene_pose_input_dir, frame_number + ".txt")).astype(np.float32)
            self.poses.append(pose)
            
            color_intrinsic = np.loadtxt(os.path.join(scene_intrinsic_input_dir, "intrinsic_color.txt")).astype(np.float32)[:3, :3]
            depth_intrinsic = np.loadtxt(os.path.join(scene_intrinsic_input_dir, "intrinsic_depth.txt")).astype(np.float32)[:3, :3]
            self.color_intrinsics.append(color_intrinsic)
            self.depth_intrinsics.append(depth_intrinsic)

            mask = cv2.imread(os.path.join(scene_mask_input_dir, "maskraw_" + frame_number + ".png"), -1).astype(np.float32)
            self.masks.append(mask)

            depth = cv2.imread(os.path.join(scene_depth_input_dir, frame_number + ".png"), -1).astype(np.float32)
            depth /= 1000.  # Convert to meters
            self.depths.append(depth)

            #semantic_mask = cv2.imread(os.path.join(scene_mask_input_dir, "maskraw_" + frame_number + ".png"), -1).astype(np.float32)
            #self.semantic_masks.append(semantic_mask)

        # Stack all the data
        self.poses = np.stack(self.poses, 0)  # (M, 4, 4)
        self.color_intrinsics = np.stack(self.color_intrinsics, 0)  # (M, 3, 3)
        self.depth_intrinsics = np.stack(self.depth_intrinsics, 0)  # (M, 3, 3)

        self.masks = np.stack(self.masks, 0)  # (M, H, W)
        self.depths = np.stack(self.depths, 0)  # (M, H, W)
        #self.semantic_masks = np.stack(self.semantic_masks, 0)  # (M, H, W)

        self.M = self.masks.shape[0]
        self.CH, self.CW = self.masks.shape[-2:]
        self.DH, self.DW = self.depths.shape[-2:]
        self.base_dir = base_dir
        self.scene_id = scene_id

        #return poses, color_intrinsics, depth_intrinsics, masks, depths, semantic_masks


