# visualization/plotter.py
import matplotlib.pyplot as plt
import numpy as np

def plot_hyperbolic_lattice(points_2d: np.ndarray, edges=None, title: str = "Hyperbolic Lattice", save_path: str = None):
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.scatter(points_2d[:, 0], points_2d[:, 1], c='cyan', s=10, alpha=0.8)
    
    if edges is not None:
        for i, neighbors in enumerate(edges):
            for j in neighbors:
                if j > i:
                    ax.plot(*zip(points_2d[i], points_2d[j]), color='magenta', lw=0.5, alpha=0.5)
    
    # Poincaré disk boundary
    circle = plt.Circle((0, 0), 1, color='white', fill=False, ls='--', lw=2)
    ax.add_patch(circle)
    
    ax.set_facecolor('black')
    ax.axis('equal')
    ax.axis('off')
    plt.title(title, color='white')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='black')
    plt.show()
