import numpy as np

from .abstract_block import AbstractBlock
from ..utils import split_2d_data_on_windows

class AvgPool2D(AbstractBlock):
    """ 
    out[N, C, h, w] = mean_{0 ≤ m < kernel_height, 0 ≤ n < kernel_width} 
                      input[N, C, stride[0]*h + m, stride[1]*w + n]
    """
    
    def __init__(
        self,
        kernel_size: int,
        stride: int = None,
        padding: int = 0
    ):
        # Init attributes
        self._k = kernel_size
        self._s = stride if stride is not None else 1
        self._p = padding

    def forward(self, x):
        # Check input dims
        if x.ndim != 4:
            raise RuntimeError("x must be 4d array with shape (N, B, H, W)")

        # Get windows
        windows, x_shape = split_2d_data_on_windows(x, k_size = self._k, stride = self._s, pad = self._p)
        self.x_shape = x_shape

        return windows.mean(axis=(4, 5))

    def parameters(self):
        return []

    def backward(self, dLdy):
        # Accumulate gradients
        s, k = self._s, self._k
        grad_per_element = dLdy[:, :, :, :, None, None] / (k*k)
        dx_padded = np.zeros(self.x_shape, dtype=dLdy.dtype)
        H_out, W_out = dLdy.shape[2:]
        for i in range(k):
            for j in range(k):
                dx_padded[:, :, i:s*H_out+i:s, j:s*W_out+j:s] += grad_per_element

        # Remove padding if needed
        if self._p > 0:
            dx_padded = dx_padded[:, :, self._p:-self._p, self._p:-self._p]

        return dx_padded