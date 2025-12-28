# core/geometry.py
"""
Core geometric utilities for the Radial Hyperbolic Architecture.
Includes golden-ratio spiral generation, Poincaré ball operations,
hyperbolic distance, and graph construction in high-dimensional hyperbolic space.
"""

import numpy as np
import torch
from scipy.spatial.distance import pdist, squareform

# Constants
PHI = (1 + np.sqrt(5)) / 2
GOLDEN_ANGLE = np.pi * (3 - np.sqrt(5))  # ~137.50776 degrees in radians


def golden_spiral_points(n_points: int, dim: int = 37, radius_scale: float = 0.99) -> np.ndarray:
    """
    Generate points along dual counter-rotating golden-angle spirals in n-dimensional space.
    Naturally produces optimal packing and phyllotaxis-like patterns.
    """
    indices = np.arange(n_points)
    angles = indices * GOLDEN_ANGLE
    # Logarithmic radial scaling modulated by golden ratio for bounded growth
    radii = radius_scale * np.exp(-0.3 * np.log(indices + 1) / np.log(PHI))

    points = np.zeros((n_points, dim))
    
    if dim >= 2:
        points[:, 0] = radii * np.cos(angles)
        points[:, 1] = radii * np.sin(angles)
    
    if dim >= 3:
        # Counter-rotating spiral in higher dimensions
        points[:, 2] = radii * np.cos(-angles + np.pi / 3)
    
    # Fill remaining dimensions with smaller perturbations for richness
    if dim > 3:
        for d in range(3, dim):
            points[:, d] = radii * np.sin(angles * (d + 1)) * 0.5

    return points


def poincare_disk_project(points: np.ndarray, curvature: float = -1.0) -> np.ndarray:
    """
    Simple projection to keep points inside the Poincaré disk (unit ball).
    Works on numpy arrays.
    """
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    factor = np.sqrt(1 + curvature * norms**2)
    factor = np.maximum(factor, 1e-8)  # Avoid division by zero
    return points / factor


def build_hyperbolic_graph(points: np.ndarray, k_neighbors: int = 8):
    """
    Build k-nearest neighbor graph using Euclidean distance as approximation
    (sufficient for visualization and local structure in Poincaré disk).
    Returns neighbor indices array of shape (n_points, k_neighbors + 1) including self.
    """
    dists = squareform(pdist(points))
    neighbors = np.argpartition(dists, k_neighbors + 1, axis=1)[:, :k_neighbors + 1]
    return neighbors


# ==================== Torch-based Hyperbolic Operations (Poincaré Ball Model) ====================

def mobius_add(x: torch.Tensor, y: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """
    Möbius addition in the Poincaré ball model with curvature -c.
    x, y: tensors of shape (... , dim)
    """
    x2 = torch.sum(x ** 2, dim=-1, keepdim=True)
    y2 = torch.sum(y ** 2, dim=-1, keepdim=True)
    xy = torch.sum(x * y, dim=-1, keepdim=True)
    
    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    denom = 1 + 2 * c * xy + c ** 2 * x2 * y2
    denom = denom.clamp_min(1e-15)
    
    return num / denom


def expmap0(v: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """
    Exponential map at the origin in the Poincaré ball.
    Maps tangent vector v at 0 to the manifold.
    """
    sqrt_c = c ** 0.5
    v_norm = torch.norm(v, p=2, dim=-1, keepdim=True).clamp_min(1e-15)
    result = torch.tanh(sqrt_c * v_norm) * (v / (sqrt_c * v_norm))
    return result


def hyperbolic_distance(x: torch.Tensor, y: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """
    Hyperbolic distance between points x and y in the Poincaré ball of curvature -c.
    Returns scalar distances.
    """
    m = mobius_add(-x, y, c=c)
    m_norm = torch.norm(m, p=2, dim=-1).clamp_min(1e-15)
    sqrt_c = c ** 0.5
    dist = 2 / sqrt_c * torch.artanh(sqrt_c * m_norm)
    return dist


# Optional: Log map (inverse of expmap0) if needed later
def logmap0(p: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    sqrt_c = c ** 0.5
    p_norm = torch.norm(p, p=2, dim=-1, keepdim=True).clamp_min(1e-15)
    scale = 1 / (sqrt_c * torch.tanh(sqrt_c * p_norm))
    return scale * p
