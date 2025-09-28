import numpy as np
 
def xavier(
    shape: tuple[int, int], 
    uniform: bool = True
)-> np.ndarray:
    """Xavier weights initialization"""
    # Get dims
    in_features, out_features = shape
    # Calculate xavier uniform or xavier normal
    if uniform:
        limit = np.sqrt(6 / (in_features + out_features))
        return np.random.uniform(-limit, limit, size=shape)
    else:
        std = np.sqrt(2 / (in_features + out_features))
        return np.random.normal(0, std, size=shape)