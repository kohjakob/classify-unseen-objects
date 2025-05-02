import numpy as np

def filter_scannet_instances(masks_binary, mask_confidences, label_confidences, threshold=0.9):
    """
    Filter the merged segmentation instances.

    Parameters:
        masks_binary (dict): 
        mask_confidences (np.ndarray): 
        label_confidences (np.ndarray): 

    Returns:
        tuple: The filtered masks, mask confidences, and label confidences.
    """
    valid_indices = np.where(np.array(mask_confidences) > threshold)[0]
    if len(valid_indices) == 0:
        return None, None
    masks_binary = np.array(masks_binary)[valid_indices]
    mask_confidences = np.array(mask_confidences)[valid_indices]
    label_confidences = np.array(label_confidences)[valid_indices]

    return masks_binary, mask_confidences, label_confidences
