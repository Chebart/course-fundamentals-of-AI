import matplotlib.pyplot as plt
import numpy as np

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

def save_mnist_grid(
    recon: np.ndarray,
    init: np.ndarray, 
    save_path: str
):
    # get array dims
    N, H, W = recon.shape
    # combine preds and targets
    pairs = np.concatenate([recon, init], axis=2)

    # Compute grid size
    grid_cols = int(np.ceil(np.sqrt(N)))
    grid_rows = int(np.ceil(N / grid_cols))
    total_cells = grid_rows * grid_cols
    if total_cells > N:
        pad = total_cells - N
        pairs = np.concatenate(
            [pairs, np.zeros((pad, H, 2*W), dtype = pairs.dtype)], 
            axis=0
        )

    # reshape grid to quadratic shape
    grid = pairs.reshape(grid_rows, grid_cols, H, 2 * W)
    grid = grid.swapaxes(1, 2).reshape(grid_rows * H, grid_cols * 2 * W)
    np.save(save_path, grid)