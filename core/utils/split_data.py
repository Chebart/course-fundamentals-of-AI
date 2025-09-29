import numpy as np

def train_test_split(
    X: np.ndarray, 
    y: np.ndarray, 
    test_size: float = 0.2, 
    shuffle: bool = True, 
    random_state: int = 42
)-> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split arrays into random train and test subsets"""
    # Get objects count
    n_samples = X.shape[0]
    # Handle test_size
    n_test = int(n_samples * test_size)
    # Shuffle indices
    indices = np.arange(n_samples)
    if shuffle:
        rng = np.random.default_rng(seed=random_state)
        rng.shuffle(indices)

    # Split data
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    
    return X_train, X_test, y_train, y_test

def batch_split(
    X: np.ndarray, 
    y: np.ndarray, 
    batch_size: int = 8, 
    shuffle: bool = True, 
    random_state: int = 42
):
    """Split data into mini-batches"""
    # Get objects count
    n_samples = X.shape[0]
    # Shuffle indices
    indices = np.arange(n_samples)
    if shuffle:
        rng = np.random.default_rng(seed=random_state)
        rng.shuffle(indices)

    # Create generator
    for start in range(0, n_samples, batch_size):
        end = start + batch_size
        batch_idx = indices[start:end]
        yield X[batch_idx], y[batch_idx]