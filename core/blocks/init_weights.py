import numpy as np
 
def xavier(
    shape: tuple[int, int], 
    uniform: bool = True
)-> np.ndarray:
    """Xavier weights initialization"""
    # Get dims
    if len(shape) == 2:
        in_features, out_features = shape
    elif len(shape) == 4:
        in_features = shape[1] * shape[3]
        out_features = shape[0] * shape[2]

    # Calculate xavier uniform or xavier normal
    if uniform:
        limit = np.sqrt(6 / (in_features + out_features))
        return np.random.uniform(-limit, limit, size=shape)
    else:
        std = np.sqrt(2 / (in_features + out_features))
        return np.random.normal(0, std, size=shape)