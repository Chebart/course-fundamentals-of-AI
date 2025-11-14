import numpy as np

def mse(
    y_pred: np.ndarray,
    y_true: np.ndarray
)->np.ndarray:
    """mean((y_true - y_pred)**2)"""
    return np.mean((y_true - y_pred)**2)

def rmse(
    y_pred: np.ndarray,
    y_true: np.ndarray
)->np.ndarray:
    """sqrt(mean((y_true - y_pred)**2))"""
    return np.sqrt(mse(y_pred, y_true))

def r2(
    y_pred: np.ndarray,
    y_true: np.ndarray,
)->np.ndarray:
    """1 - sum((y_true - y_pred)**2) / sum((y_true - y_true.mean())**2)"""
    return 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)