from __future__ import annotations
import operator

from .backend import AVAILABLE_BACKENDS

class Tensor:
    """
    Custom data class, supporting automatic backend selection (NumPy/CuPy) 
    and seamless CPU/GPU device tracking.
    """

    def __init__(self, data, dtype: str = "fp16", device: str = "cpu"):
        self._backend = Tensor.define_backend(device)
        self._dtype = Tensor.get_backend_dtype(self.backend, dtype)

        if self.backend.__name__ == "cupy":
            self._device = device
            with self.backend.cuda.Device(self.device_id):
                self._data = self.backend.asarray(data, dtype = self._dtype)

        elif self.backend.__name__ == "numpy":
            self._device = device
            self._data = self.backend.array(data, dtype = self._dtype)

        # change dtype on string value
        self._dtype = dtype

    # ---------------------
    # Verification methods
    # ---------------------

    def same_device(self, other: Tensor):
        return self.device == other.device

    @staticmethod
    def define_backend(device: str):
        if device.startswith("cuda"):
            if "cupy" not in AVAILABLE_BACKENDS:
                raise ValueError(f"CuPy is not installed. Cannot move tensor to {device}.")
            return AVAILABLE_BACKENDS["cupy"]
        
        return AVAILABLE_BACKENDS["numpy"]

    @staticmethod
    def get_backend_dtype(backend, dtype: str):
        dtype_map = {
            "fp16": backend.float16,
            "fp32": backend.float32,
            "fp64": backend.float64,
            "int16": backend.int16,
            "int32": backend.int32,
            "int64": backend.int64,
            "bool": backend.bool_,
        }
        if dtype not in dtype_map:
            raise ValueError(f"Unsupported dtype: '{dtype}'")
        
        return dtype_map[dtype]
    
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
    def T(self):
        return Tensor(self.data.T, self.dtype, self.device)

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

    @property
    def dtype(self):
        return self._dtype

    # ---------------------
    # Device transfer methods
    # ---------------------

    def to_device(self, device: str)-> Tensor:
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

            return Tensor(result, self.dtype, self.device)
        
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

    @staticmethod
    def _inplace_binary_op(op_func):
        def method(self, other):
            if isinstance(other, Tensor):
                if not self.same_device(other):
                    raise ValueError(
                        f"Expected all tensors to be on the same device, "
                        f"but found {self.device} and {other.device}!"
                    )
                self._data = op_func(self.data, other.data)
            else:
                self._data = op_func(self.data, other)

            return self
        
        return method

    __iadd__ = _inplace_binary_op(operator.iadd)
    __isub__ = _inplace_binary_op(operator.isub)
    __imul__ = _inplace_binary_op(operator.imul)
    __itruediv__ = _inplace_binary_op(operator.itruediv)
    __ipow__ = _inplace_binary_op(operator.ipow)

    def _matmul_op():
        def method(self, other):
            if not isinstance(other, Tensor):
                raise TypeError(f"Matrix multiplication requires a Tensor, got {type(other)}")

            if not self.same_device(other):
                raise ValueError(
                    f"Expected all tensors to be on the same device, "
                    f"but found {self.device} and {other.device}"
                )

            result = self.backend.matmul(self.data, other.data)
            return Tensor(result, self.dtype, self.device)
        
        return method

    __matmul__ = _matmul_op()
    __rmatmul__ = _matmul_op()

    # ---------------------
    # Unary operation methods
    # ---------------------

    def __neg__(self)-> Tensor:
        return Tensor(-self.data, self.dtype, self.device)

    # ---------------------
    # Elementwise methods
    # ---------------------

    def exp(self)-> Tensor:
        return Tensor(self.backend.exp(self.data), self.dtype, self.device)

    def log(self)-> Tensor:
        return Tensor(self.backend.log(self.data), self.dtype, self.device)

    def sign(self)-> Tensor:
        return Tensor(self.backend.sign(self.data), self.dtype, self.device)

    # ---------------------
    # Views methods
    # ---------------------

    def __getitem__(self, index)-> Tensor:
        return Tensor(self.data[index], self.dtype, self.device)

    def reshape(self, *shape)-> Tensor:
        return Tensor(self.data.reshape(*shape), self.dtype, self.device)

    def transpose(self, *axes)-> Tensor:
        return Tensor(self.data.transpose(*axes), self.dtype, self.device)

    def as_strided(self, shape, strides)-> Tensor:
        return Tensor(self.backend.lib.stride_tricks.as_strided(self.data, shape=shape, strides=strides), 
                      self.dtype, self.device)

    # ---------------------
    # Aggregation methods
    # ---------------------

    def max(self, axis=None, keepdims=False)-> Tensor:
        return Tensor(self.data.max(axis=axis, keepdims=keepdims), self.dtype, self.device)

    def min(self, axis=None, keepdims=False)-> Tensor:
        return Tensor(self.data.min(axis=axis, keepdims=keepdims), self.dtype, self.device)

    def sum(self, axis=None, keepdims=False)-> Tensor:
        return Tensor(self.data.sum(axis=axis, keepdims=keepdims), self.dtype, self.device)

    def mean(self, axis=None, keepdims=False)-> Tensor:
        return Tensor(self.data.mean(axis=axis, keepdims=keepdims), self.dtype, self.device)

    # ---------------------
    # Utility methods
    # ---------------------

    def __len__(self):
        return self.shape[0]

    def fill(self, value: int):
        self._data.fill(value)
        return self
    
    def clone(self)-> Tensor:
        return Tensor(self.data.copy(), self.dtype, self.device)

    @staticmethod
    def zeros(shape, dtype = "fp16", device="cpu")-> Tensor:
        backend = Tensor.define_backend(device)
        backend_dtype = Tensor.get_backend_dtype(backend, dtype)
        return Tensor(backend.zeros(shape, dtype = backend_dtype), dtype, device)
    
    @staticmethod
    def eye(
        n: int, 
        m: int = None, 
        dtype = "fp16", 
        device: str = "cpu"
    )-> Tensor:
        backend = Tensor.define_backend(device)
        backend_dtype = Tensor.get_backend_dtype(backend, dtype)
        return Tensor(backend.eye(n, m, dtype = backend_dtype), dtype, device)

    @staticmethod
    def random_uniform(
        low: float = 0.0, 
        high: float = 1.0, 
        size: tuple = (2, 2),
        dtype = "fp16",
        device: str = "cpu"
    )-> Tensor:
        backend = Tensor.define_backend(device)
        return Tensor(backend.random.uniform(low, high, size = size), dtype, device)

    @staticmethod
    def random_normal(
        mean: float = 0.0, 
        std: float = 1.0, 
        size: tuple = (2, 2), 
        dtype = "fp16", 
        device: str = "cpu"
    )-> Tensor:
        backend = Tensor.define_backend(device)
        return Tensor(backend.random.normal(mean, std, size = size), dtype, device)

    # ---------------------
    # Data type conversion methods
    # ---------------------

    def to_numpy(self):
        if self.backend.__name__ == "cupy":
            return self.data.get()
        elif self.backend.__name__ == "numpy":
            return self.data

    # ---------------------
    # Representation methods
    # ---------------------

    def __repr__(self):
        return f"Tensor({self.data}, device={self.device})"