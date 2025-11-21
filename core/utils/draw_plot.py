import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

from .normalize import min_max_normalization

def plot_curves(
    x: np.ndarray, 
    y: np.ndarray, 
    title: str, 
    xlabel: str, 
    ylabel: str, 
    save_path: str
):
    """Draws multiple curves on the same plot"""
    plt.figure(figsize=(10, 6))

    y = np.atleast_2d(y)
    if y.shape[1] != x.shape[0]:
        y = y.T
    for i in range(y.shape[0]):
        plt.plot(x, y[i], linewidth=2, label=f"Curve {i+1}")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_tsne(
    x: np.ndarray, 
    y: np.ndarray, 
    colors: np.ndarray,
    title: str, 
    xlabel: str, 
    ylabel: str, 
    save_path: str
):
    """Draws multiple curves on the same plot"""
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(x, y, c=colors, cmap='tab10', s=10, alpha=0.7)
    plt.colorbar(scatter, ticks=range(10))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

def save_restoration_grid(
    recon: np.ndarray,
    init: np.ndarray, 
    save_path: str
):
    """Draws reconstruction results"""
    # normalize reconstructed images
    recon = (255 * min_max_normalization(recon)).astype(np.uint8)
    init = (255 * min_max_normalization(init)).astype(np.uint8)

    # get array dims
    B, H, W = recon.shape
    # combine preds and targets
    pairs = np.concatenate([recon, init], axis=2)

    # Compute grid size
    grid_cols = int(np.ceil(np.sqrt(B)))
    grid_rows = int(np.ceil(B / grid_cols))
    total_cells = grid_rows * grid_cols
    if total_cells > B:
        pad = total_cells - B
        pairs = np.concatenate(
            [pairs, np.zeros((pad, H, 2*W), dtype = pairs.dtype)], 
            axis=0
        )

    # reshape grid to quadratic shape
    grid = pairs.reshape(grid_rows, grid_cols, H, 2 * W)
    grid = grid.swapaxes(1, 2).reshape(grid_rows * H, grid_cols * 2 * W)
    # save as image
    img = Image.fromarray(grid)
    img.save(save_path)