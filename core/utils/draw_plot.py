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
