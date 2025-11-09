from core.data import Tensor
 
def xavier(
    shape: tuple[int, int], 
    dtype: str = "fp32",
    uniform: bool = True
)-> Tensor:
    """Xavier weights initialization"""
    # Get dims
    if len(shape) == 2:
        in_features, out_features = shape
    elif len(shape) == 4:
        in_features = shape[1] * shape[3]
        out_features = shape[0] * shape[2]

    # Calculate xavier uniform or xavier normal
    if uniform:
        limit = (6 / (in_features + out_features))**0.5
        return Tensor.random_uniform(-limit, limit, size = shape, dtype = dtype)
    else:
        std = (2 / (in_features + out_features))**0.5
        return Tensor.random_normal(0, std, size = shape, dtype = dtype)