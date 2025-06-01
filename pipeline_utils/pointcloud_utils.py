import numpy as np

def random_downsample(target_array_length=1000, *arrays):
    """
    Randomly downsample multiple arrays to a target number of points
    while maintaining correspondence across all arrays.
    
    Args:
        target_count: Desired number of points after downsampling
        *arrays: Variable number of arrays, all must have the same length in the first dimension
        
    Returns:
        List of downsampled arrays with the same random indices applied to all
    """
    np.random.seed(42)  # For reproducibility

    if not arrays:
        return []
    
    # Verify all arrays have the same length (shape in the first dimension)
    input_array_length = arrays[0].shape[0]
    for i, arr in enumerate(arrays):
        if arr.shape[0] != input_array_length:
            raise ValueError(f"Array at index {i} has different length ({arr.shape[0]}) than the first array ({input_array_length})")
    
    # Downsample if necessary
    if input_array_length <= target_array_length:
        return arrays
    else:
        indices = np.random.choice(input_array_length, target_array_length, replace=False)
        downsampled_arrays = [arr[indices] for arr in arrays]
        return downsampled_arrays