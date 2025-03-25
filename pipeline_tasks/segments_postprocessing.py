import torch

def postprocess_scannet_segments(outputs, inverse_map, coords, colors_normalized, ground_truth_labels):
    """
    Post-processes the segmentation model outputs.

    Parameters:
        outputs (dict): The model outputs containing logits and masks.
        inverse_map (np.ndarray): The inverse map from the preprocessing step.
        coords (np.ndarray): The coordinates from the preprocessing step.
        colors_normalized (np.ndarray): The normalized colors from the preprocessing step.
        ground_truth_labels (np.ndarray): The ground truth labels from the preprocessing step.

    Returns:
        tuple: The processed masks, mask confidences, and label confidences.
    """
    logits = outputs["pred_logits"]
    masks = outputs["pred_masks"]

    # Reformat predictions
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

    return masks_binary, mask_confidences, label_confidences