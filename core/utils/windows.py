import numpy as np

def pad_2d_data(
    x: np.ndarray,
    pad: int
)-> np.ndarray:
    """Adds zero pad to 2d input that pooling or convolution operations can handle borders properly"""
    return np.pad(
        x, 
        ((0, 0), (0, 0), (pad, pad), (pad, pad)), 
        mode='constant'
    )
    
def split_2d_data_on_windows(
    x: np.ndarray, 
    k_size: int, 
    stride: int,
    pad: int
)-> tuple[np.ndarray, np.ndarray]:
    """Splits a 4D input (N, C, H, W) into smaller 2D sliding windows of size (k_size, k_size) 
       according to the specified stride and padding"""
    # Apply padding
    if pad > 0:
        x = pad_2d_data(x, pad)
    N, C, H_in, W_in = x.shape
    # Output dimensions
    H_out = (H_in - k_size) // stride + 1
    W_out = (W_in - k_size) // stride + 1

    # Create sliding windows
    shape = (N, C, H_out, W_out, k_size, k_size)
    strides = (
        x.strides[0],
        x.strides[1],
        x.strides[2] * stride,
        x.strides[3] * stride,
        x.strides[2],
        x.strides[3],
    )
    windows = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)

    return windows, x.shape

def get_2d_data_from_windows(
    windows: np.ndarray, 
    stride: int,
    pad: int
)-> tuple[np.ndarray, np.ndarray]:
    """Union small 2D sliding windows of size (k_size, k_size) into a 4D input (N, C, H, W)
       according to the specified stride and padding"""
    # Output dimensions
    N, C, H_out, W_out, k_size, _ = windows.shape
    # Input dimensions
    H_in = (H_out - 1) * stride + k_size
    W_in = (W_out - 1) * stride + k_size
    # Get padded input_data
    input_data = np.zeros((N, C, H_in, W_in), dtype=windows.dtype)
    s, k = stride, k_size
    for i in range(H_out):
        for j in range(W_out):
            input_data[:, :, i*s:i*s+k, j*s:j*s+k] += windows[:, :, i, j, :, :]

    # Remove padding
    if pad > 0:
        input_data = input_data[:, :, pad:-pad, pad:-pad]

    return input_data