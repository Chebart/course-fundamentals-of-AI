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
    plt.plot(x, y, linewidth=2)
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