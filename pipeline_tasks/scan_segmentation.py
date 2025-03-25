import torch

def segment_scannet_scene(model, data, features):
    """
    Segments the preprocessed ScanNet scene using the given model.

    Parameters:
        model (torch.nn.Module): The segmentation model.
        data (ME.SparseTensor): The preprocessed sparse tensor.
        features (torch.Tensor): The features used for the segmentation.

    Returns:
        tuple: A tuple containing the model outputs, inverse map, coordinates, 
               normalized colors, and ground truth labels.
    """
    with torch.no_grad():
        outputs = model(data, raw_coordinates=features)
    del data
    torch.cuda.empty_cache()
        
    return outputs