
import torch
import sys
import hydra
import numpy as np
import torch
import MinkowskiEngine as ME

from external.UnScene3D.utils.utils import load_checkpoint_with_missing_or_exsessive_keys
from pipeline_conf.conf import DEVICE, CONFIG, SCANNET_COLOR_NORMALIZE

sys.path.insert(0, CONFIG.unscene)

class UnScene3D_Wrapper():
    def __init__(self):
        # Load default configuration
        with hydra.initialize(config_path=CONFIG.unscene3d_config_dir, version_base="1.1"):
            config = hydra.compose(config_name=CONFIG.unscene3d_config_file)

        # Specify checkpoint and set training flags
        config.general.checkpoint = CONFIG.unscene3d_checkpoint
        config.general.train_on_segments = False
        config.general.eval_on_segments = False
        
        # Build InstanceSegmentation model
        self.model = hydra.utils.instantiate(config.model)
        
        # Load pretrained checkpoint
        _, self.model = load_checkpoint_with_missing_or_exsessive_keys(config, self.model)
        self.model = self.model.to(DEVICE)
        self.model.eval()  

    def detect_instances(self, points, colors):
        # Normalize colors
        pseudo_image = colors.astype(np.uint8)[np.newaxis, :, :]
        colors_normalized = np.squeeze(SCANNET_COLOR_NORMALIZE(image=pseudo_image)["image"])

        # Voxelize points
        points_voxelized = np.floor(points / 0.02)

        # Perform sparse quantization using MinkowskiEngine
        _, _, unique_map, inverse_map = ME.utils.sparse_quantize(
            coordinates=points_voxelized,
            features=colors_normalized,
            return_index=True,
            return_inverse=True
        )
        sparse_points = [torch.from_numpy(points_voxelized[unique_map]).int()]
        sparse_color_features = [torch.from_numpy(colors_normalized[unique_map]).float()]

        # Prepare input data using MinkowskiEngine
        sparse_points, _ = ME.utils.sparse_collate(coords=sparse_points, feats=sparse_color_features)
        sparse_color_features = torch.cat(sparse_color_features, dim=0)
        data = ME.SparseTensor(
            coordinates=sparse_points,
            features=sparse_color_features,
            device=DEVICE,
        )

        # Detect instances
        with torch.no_grad():
            outputs = self.model(data, raw_coordinates=sparse_color_features)
        
        # Empty cache to free memory
        del data
        torch.cuda.empty_cache()
        
        # Extract outputs
        logits = outputs["pred_logits"]
        masks = outputs["pred_masks"]
        logits = logits[0].detach().cpu()
        masks = masks[0].detach().cpu()

        mask_confidences = []
        label_confidences = []
        masks_binary = []
        labels = []
        scores = []
        for i in range(len(logits)):
            p_labels = torch.softmax(logits[i], dim=-1)
            p_masks = torch.sigmoid(masks[:, i])
            l = torch.argmax(p_labels, dim=-1)
            c_label = torch.max(p_labels)
            m = p_masks > 0.5
            c_m = p_masks[m].sum() / (m.sum() + 1e-8)
            c = c_m

            if l < 200 and c > 0.9:
                mask_confidences.append(c.item())
                label_confidences.append(c_label.item())
                masks_binary.append(m.numpy()[inverse_map])
                labels.append(l.item())
                scores.append(c_label.item())

        # Filter instances based on confidence
        threshold = 0.9
        valid_indices = np.where(np.array(mask_confidences) > threshold)[0]
        if len(valid_indices) == 0:
            return None, None
        masks_binary = np.array(masks_binary)[valid_indices]
        mask_confidences = np.array(mask_confidences)[valid_indices]
        label_confidences = np.array(label_confidences)[valid_indices]

        # Merge instances based on IoU
        num_instances = len(masks_binary)
        if num_instances <= 1:
            return masks_binary, mask_confidences, label_confidences

        merging = True
        while merging and num_instances > 1:
            merging = False
            iou_matrix = np.zeros((num_instances, num_instances))
            for i in range(num_instances):
                for j in range(i + 1, num_instances):
                    mask_i = masks_binary[i]
                    mask_j = masks_binary[j]
                    intersection = np.logical_and(mask_i, mask_j).sum()
                    union = np.logical_or(mask_i, mask_j).sum()
                    iou_matrix[i, j] = intersection / union if union > 0 else 0.0
                    iou_matrix[j, i] = iou_matrix[i, j]

            pairs_to_merge = np.argwhere(iou_matrix > 0.5) # 0.5?
            pairs_to_merge = pairs_to_merge[pairs_to_merge[:, 0] < pairs_to_merge[:, 1]]

            if len(pairs_to_merge) > 0:
                merging = True
                parent = np.arange(num_instances)

                def find(u):
                    while parent[u] != u:
                        parent[u] = parent[parent[u]]
                        u = parent[u]
                    return u

                def union(u, v):
                    pu, pv = find(u), find(v)
                    if pu != pv:
                        parent[pu] = pv

                for u, v in pairs_to_merge:
                    union(u, v)

                root_to_indices = {}
                for idx in range(num_instances):
                    root = find(idx)
                    if root not in root_to_indices:
                        root_to_indices[root] = []
                    root_to_indices[root].append(idx)

                merged_masks = []
                merged_mask_confidences = []
                merged_label_confidences = []

                for indices in root_to_indices.values():
                    masks_binary = np.array(masks_binary)
                    merged_mask = np.logical_or.reduce(masks_binary[indices, :], axis=0)
                    mask_confidences_np = np.array(mask_confidences)
                    label_confidences_np = np.array(label_confidences)

                    comp_mask_confidences = mask_confidences_np[indices]
                    comp_label_confidences = label_confidences_np[indices]
                    
                    # Find the index of the highest confidence and select the corresponding confidence
                    max_mask_conf_idx = indices[np.argmax(comp_mask_confidences)]
                    max_label_conf_idx = indices[np.argmax(comp_label_confidences)]
                    selected_mask_confidence = mask_confidences[max_mask_conf_idx]
                    selected_label_confidence = label_confidences[max_label_conf_idx]

                    merged_masks.append(merged_mask)
                    merged_mask_confidences.append(selected_mask_confidence)
                    merged_label_confidences.append(selected_label_confidence)

                masks_binary = np.array(merged_masks)
                mask_confidences = np.array(merged_mask_confidences)
                label_confidences = np.array(merged_label_confidences)
                num_instances = len(masks_binary)

        # Save filtered and merged instances
        instances = []
        for i in range(masks_binary.shape[0]):
            instance_points = points[masks_binary[i]]
            instance_colors = colors[masks_binary[i]]
            
            instances.append({
                "points": instance_points,
                "colors": instance_colors,
                "gt_label": "not_specified",
                "binary_mask": masks_binary[i],
            })

        return instances
