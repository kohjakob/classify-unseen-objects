import os
import numpy as np
from pipeline_conf.conf import PATHS

def load_instance_features_from_npz(folder_path):
    """
    Loads features from all .npz files in a directory.

    Args:
        folder_path (str): Path to the directory (containing .npz files)

    Returns:
        dict: A dictionary mapping instance filenames to feature arrays.
    """
    
    instance_dict = {}
    for entry in os.scandir(folder_path):
        if entry.name.endswith(".npz"):
            data = np.load(entry.path)
            instance_dict[entry.name] = data['features']
            print(f"Loaded features from {entry.name}, shape: {data['features'].shape}")
    return instance_dict

