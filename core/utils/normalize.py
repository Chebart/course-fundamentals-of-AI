import numpy as np

def standard_normalization(
    x: np.ndarray,
    mean: float,
    std: float,
    eps: float = 1e-8
)->np.ndarray:
    """Normalize input values by mean and std"""
    return (x - mean) / (std + eps) 

def min_max_normalization(
    x: np.ndarray,
    eps: float = 1e-8
)->np.ndarray:
    """Normalize input values by min and max"""
    vmin, vmax = x.min(), x.max()
    return (x - vmin) / (vmax - vmin + eps)