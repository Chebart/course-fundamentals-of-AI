from ..utils import split_2d_data_on_windows
from .abstract_block import AbstractBlock
from core.data import Tensor

class MaxPool2D(AbstractBlock):
    """ 
    out[N, C, h, w] = max_{0 ≤ m < kernel_height, 0 ≤ n < kernel_width} 
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
        # Get only first max value in each window
        N, C, H_out, W_out, k, _ = windows.shape
        windows_flat = windows.reshape(N, C, H_out, W_out, k*k)
        max_idx = windows_flat.argmax(axis=4)
        
        # Save mask for backward pass
        self.max_mask = Tensor.zeros(self.x_shape, dtype="bool", device=windows_flat.device)
        self.max_mask.put_along_axis(max_idx[..., None], True, axis=4)
        self.max_mask = self.max_mask.reshape(N, C, H_out, W_out, k, k)

        return windows.max(axis=(4, 5))

    def parameters(self):
        return []

    def backward(self, dLdy):
        # Apply mask on dLdy
        masked_dLdy = self.max_mask * dLdy[:, :, :, :, None, None]

        # Accumulate gradients
        dLdx = Tensor.zeros(self.x_shape, dtype=dLdy.dtype, device=dLdy.device)
        H_out, W_out = dLdy.shape[2:]
        s, k = self._s, self._k
        for i in range(k):
            for j in range(k):
                dLdx[:, :, i:s*H_out+i:s, j:s*W_out+j:s] += masked_dLdy[:, :, :, :, i, j]

        # Remove padding if needed
        if self._p > 0:
            dLdx = dLdx[:, :, self._p:-self._p, self._p:-self._p]

        return dLdx