from __future__ import annotations
import operator

from .backend import AVAILABLE_BACKENDS

class Tensor:
    """
    Custom data class, supporting automatic backend selection (NumPy/CuPy) 
    and seamless CPU/GPU device tracking.
    """

    def __init__(self, data, device: str = "cpu"):
        self._backend = self.define_backend(device)

        if self.backend.__name__ == "cupy":
            self._device = device
            with self.backend.cuda.Device(self.device_id):
                self._data = self.backend.asarray(data)

        elif self.backend.__name__ == "numpy":
            self._device = device
            self._data = self.backend.array(data)

    # ---------------------
    # Properties
    # ---------------------

    @property
    def backend(self):
        return self._backend
    
    @property
    def data(self):
        return self._data

    @property
    def device(self):
        return self._device
    
    @property
    def device_id(self):
        if self.device.startswith("cuda"): 
            return int(self.device.split(':')[-1])
        else:
            return None
    
    @property
    def shape(self):
        return self._data.shape

    # ---------------------
    # Device transfer methods
    # ---------------------

    def to_device(self, device: str):
        new_backend = self.define_backend(device)
        self._device = device

        if device.startswith("cuda"):
            if self.backend.__name__ == "numpy":
                self._backend = new_backend

            with self.backend.cuda.Device(self.device_id):
                self._data = self.backend.asarray(self.data)
        else:
            if self.backend.__name__ == "cupy":
                self._data = self.data.get()
                self._backend = new_backend

        return self

    # ---------------------
    # Verification methods
    # ---------------------

    @staticmethod
    def define_backend(device: str):
        if device.startswith("cuda"):
            if "cupy" not in AVAILABLE_BACKENDS:
                raise ValueError(f"CuPy is not installed. Cannot move tensor to {device}.")
            return AVAILABLE_BACKENDS["cupy"]
        
        return AVAILABLE_BACKENDS["numpy"]

    def same_device(self, other: Tensor):
        return self.device == other.device

    # ---------------------
    # Binary operation methods
    # ---------------------

    def _binary_op(op_func):
        def method(self, other):
            if isinstance(other, Tensor):
                if not self.same_device(other):
                    raise ValueError(
                        f"Expected all tensors to be on the same device, "
                        f"but found {self.device} and {other.device}!"
                    )
                result = op_func(self.data, other.data)
            else:
                result = op_func(self.data, other)

            return Tensor(result, self.device)
        
        return method
    
    __add__ = _binary_op(operator.add)
    __sub__ = _binary_op(operator.sub)
    __mul__ = _binary_op(operator.mul)
    __truediv__ = _binary_op(operator.truediv)
    __pow__ = _binary_op(operator.pow)

    __radd__ = __add__
    __rsub__ = _binary_op(lambda x, y: operator.sub(y, x))
    __rmul__ = __mul__
    __rtruediv__ = _binary_op(lambda x, y: operator.truediv(y, x))
    __rpow__ = _binary_op(lambda x, y: operator.pow(y, x))

    # ---------------------
    # Unary operation methods
    # ---------------------

    def __neg__(self):
        return Tensor(-self.data, self.device)

    # ---------------------
    # Elementwise methods
    # ---------------------

    def exp(self):
        return Tensor(self.backend.exp(self.data), self.device)

    def log(self):
        return Tensor(self.backend.log(self.data), self.device)

    def sign(self):
        return Tensor(self.backend.sign(self.data), self.device)

    # ---------------------
    # Views methods
    # ---------------------

    def __getitem__(self, index):
        return Tensor(self.data[index], self.device)

    def reshape(self, *shape):
        return Tensor(self.data.reshape(*shape), self.device)

    def transpose(self, *axes):
        return Tensor(self.data.transpose(*axes), self.device)

    def as_strided(self, shape, strides):
        return Tensor(self.backend.lib.stride_tricks.as_strided(self.data, shape=shape, strides=strides), self.device)

    # ---------------------
    # Aggregation methods
    # ---------------------

    def sum(self, axis=None, keepdims=False):
        return Tensor(self.data.sum(axis=axis, keepdims=keepdims), self.device)

    def mean(self, axis=None, keepdims=False):
        return Tensor(self.data.mean(axis=axis, keepdims=keepdims), self.device)

    # ---------------------
    # Utility methods
    # ---------------------

    def fill(self, value: int):
        self._data.fill(value)
        return self
    
    def clone(self):
        return Tensor(self.data.copy(), device=self.device)

    @staticmethod
    def eye(n: int, m: int = None, device: str = "cpu"):
        backend = Tensor.define_backend(device)
        return Tensor(backend.eye(n, m), device = device)

    @staticmethod
    def random_uniform(low: float = 0.0, high: float = 1.0, size: tuple = (2, 2), device: str = "cpu"):
        backend = Tensor.define_backend(device)
        return Tensor(backend.random.uniform(low, high, size = size), device = device)

    @staticmethod
    def random_normal(mean: float = 0.0, std: float = 1.0, size: tuple = (2, 2), device: str = "cpu"):
        backend = Tensor.define_backend(device)
        return Tensor(backend.random.normal(mean, std, size = size), device = device)

    # ---------------------
    # Representation methods
    # ---------------------

    def __repr__(self):
        return f"Tensor({self.data}, device={self.device})"