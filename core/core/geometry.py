# core/geometry.py
import numpy as np
from scipy.spatial.distance import pdist, squareform

PHI = (1 + np.sqrt(5)) / 2
GOLDEN_ANGLE = np.pi * (3 - np.sqrt(5))

def golden_spiral_points(n_points: int, dim: int = 2, radius_scale: float = 0.99):
    """Generate points along dual counter-rotating golden spirals in nD (projected to disk)."""
    indices = np.arange(n_points)
    angles = indices * GOLDEN_ANGLE
    radii = np.exp(-0.3 * np.log(indices + 1) / np.log(PHI)) * radius_scale
    
    points = np.zeros((n_points, dim))
    for i in range(n_points):
        r = radii[i]
        if dim >= 2:
            points[i, 0] = r * np.cos(angles[i])
            points[i, 1] = r * np.sin(angles[i])
        if dim >= 3:
            points[i, 2] = r * np.cos(-angles[i])  # Counter-rotating component
    return points

def poincare_disk_project(points: np.ndarray, curvature: float = -1.0):
    """Simple projection to ensure points stay inside the Poincaré disk."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    scaled = points / (1 + np.sqrt(1 + curvature * norms**2))
    return scaled

def build_hyperbolic_graph(points: np.ndarray, k_neighbors: int = 8):
    """Build nearest-neighbor graph in hyperbolic space."""
    dists = squareform(pdist(points))
    neighbors = np.argpartition(dists, k_neighbors + 1, axis=1)[:, :k_neighbors + 1]
    return neighbors
