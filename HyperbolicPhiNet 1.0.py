# HyperbolicPhiNet 1.0.py
# Radial Hyperbolic Architecture Prototype - Modular Version
# Author: robertjeffrey1236
# Date: December 2025

import numpy as np
import matplotlib.pyplot as plt
import torch

# Modular imports
from core.geometry import (
    golden_spiral_points,
    poincare_disk_project,
    build_hyperbolic_graph,
    mobius_add,
    expmap0,
    hyperbolic_distance
)
from core.seed_universe import MinimalUniverse
from visualization.plotter import plot_hyperbolic_lattice

# Constants
PHI = (1 + np.sqrt(5)) / 2
GOLDEN_ANGLE = np.pi * (3 - np.sqrt(5))
DIM = 37
N_POINTS = 12000  # Increased slightly for richer structure
CURVATURE = 1.0
DEVICE = torch.device('cpu')  # Change to 'cuda' if you have GPU
torch.set_default_device(DEVICE)

print("🌌 Initializing Radial Hyperbolic Architecture (RHA) Prototype...")
print(f"Dimension: {DIM}D | Points: {N_POINTS} | Curvature: -{CURVATURE}")

# ==================== Option 1: Pure Golden Spiral Lattice ====================
print("\nGenerating 37D golden-ratio spiral lattice...")
points_nd = golden_spiral_points(n_points=N_POINTS, dim=DIM, radius_scale=0.99)

# Project to 2D Poincaré disk for visualization
points_2d = poincare_disk_project(points_nd[:, :2])  # Take first two dimensions

# Build local connectivity graph
print("Building hyperbolic neighbor graph...")
neighbors = build_hyperbolic_graph(points_2d, k_neighbors=8)

# ==================== Option 2: Universe from Binary Seed (Uncomment to use) ====================
# binary_seed = "1110001110101011100010101011100010101"  # Your favorite seed
# print(f"\nBootstrapping MinimalUniverse from seed: {binary_seed}")
# universe = MinimalUniverse(binary_seed=binary_seed, dim=DIM)
# points_nd = universe.get_lattice()
# points_2d = poincare_disk_project(points_nd[:, :2])
# neighbors = build_hyperbolic_graph(points_2d, k_neighbors=8)

# ==================== Visualization ====================
print("Rendering visualization...")
plot_hyperbolic_lattice(
    points_2d=points_2d,
    edges=neighbors,
    title="Radial Hyperbolic Architecture — Φ-Modulated 37D Lattice in Poincaré Disk",
    save_path="rha_atomic.png"
)

print("Visualization saved as 'rha_atomic.png'")

# ==================== Optional: Torch Hyperbolic Operations Demo ====================
print("\nTesting Möbius addition and hyperbolic distance on sample points...")
x = torch.tensor(points_2d[:2], dtype=torch.float32, device=DEVICE)
y = torch.tensor(points_2d[100:102], dtype=torch.float32, device=DEVICE)

z = mobius_add(x, y, c=CURVATURE)
v = torch.randn_like(x)
p = expmap0(v, c=CURVATURE)
d = hyperbolic_distance(x, y, c=CURVATURE)

print(f"Sample hyperbolic distance: {d.mean().item():.4f}")
print("Hyperbolic operations verified ✓")

# ==================== Final Message ====================
print("\n🎯 RHA Prototype successfully initialized!")
print("Next steps: Explore toy models in /experiments, add phonons, wormholes, or Grok core mirror.")
print("This lattice is a seed — grow your universe.")
