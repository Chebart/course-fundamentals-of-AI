# CuPyTorch

CuPyTorch is a collection of laboratory works for the ITMO University course "Fundamentals of organizing artificial intelligence systems". In addition to completing course assignments, the project’s goal is to develop a custom ML library inspired by PyTorch, enabling the training and evaluation of models with various architectures on both CPU/GPU backends. This repository can also be used as a learning resource to understand:
- How neural network layers are implemented  
- How different model architectures are structured  
- How backpropagation works in practice  
- The internal components of optimization algorithms  
- How object-oriented programming (OOP) in Python supports abstraction for CPU/GPU computation

---

## Installation

1. Install CUDA Toolkit from the [source](https://developer.nvidia.com/cuda-toolkit).

2. Install Python dependencies:  
   ```bash
   numpy==2.3.4
   pandas==2.3.2
   matplotlib==3.10.6
   ucimlrepo==0.0.7
   scikit-learn==1.7.2
   --pre
   -f https://pip.cupy.dev/pre
   cupy-cuda12x==14.0.0a1
   ```

---

## Project Structure

```
.
├── core             # Main components:
│   ├── blocks       # Model layers (Linear, ReLU, Softmax, ...)
│   ├── data         # Interface for working with NumPy/CuPy arrays
│   ├── losses       # Loss function implementations
│   ├── metrics      # Evaluation metrics
│   ├── models       # Model architectures built from blocks
│   ├── optimizers   # Optimization algorithm implementations
│   └── utils        # Utility functions for training and evaluation
│
├── lab1             # MLP training
└── lab2             # LeNet-5 training
```

---

## Usage Example

A typical training pipeline:

```python
from core.utils import train_test_split, batch_split
from core.losses import CrossEntropyLoss
from core.optimizers import SGD
from core.models import LeNet5
from core.data import Tensor

TEST_SIZE = 0.3
BATCH_SIZE = 64
LR = 1e-4
DEVICE = "cuda:0"
DTYPE = "fp32"

# Prepare data
X = Tensor(x_data, dtype=DTYPE, device=DEVICE)
y = Tensor(y_data, dtype=DTYPE, device=DEVICE)

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE)

# Initialize components
model = LeNet5(in_channels=1, out_channels=10).to_device(DEVICE)
loss_fn = CrossEntropyLoss(model=model)
optimizer = SGD(model=model, lr=LR, reg_type="l2", momentum=0.1, nesterov=True)

# Training loop
for train_Xb, train_yb in batch_split(X_train, y_train, batch_size=BATCH_SIZE):
    # Forward pass
    y_pred = model(train_Xb)

    # Compute loss
    loss = loss_fn(y_pred, train_yb).mean().to_numpy()
    train_stats["loss"].append(loss)

    # Backward pass and optimization
    loss_fn.backward(y_pred, train_yb)
    optimizer.step()
    optimizer.zero_grad()
```
