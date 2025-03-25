import numpy as np

def merge_scannet_segments(masks_binary, mask_confidences, label_confidences):
    """
    Post-processes the segmentation model outputs.

    Parameters:
        masks_binary (dict): 
        mask_confidences (np.ndarray): 
        label_confidences (np.ndarray): 

    Returns:
        tuple: The processed masks, mask confidences, and label confidences.
    """
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
                masks_binary = np.array(masks_binary)  # Convert to NumPy array
                merged_mask = np.logical_or.reduce(masks_binary[indices, :], axis=0)

                # Convert confidences to a NumPy array for proper indexing
                mask_confidences_np = np.array(mask_confidences)
                label_confidences_np = np.array(label_confidences)

                # Select confidences corresponding to the indices
                comp_mask_confidences = mask_confidences_np[indices]  # Now you can index with 'indices'
                comp_label_confidences = label_confidences_np[indices]  # Now you can index with 'indices'
                
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

    return masks_binary, mask_confidences, label_confidences