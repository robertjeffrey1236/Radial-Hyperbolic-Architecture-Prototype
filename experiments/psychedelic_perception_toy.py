# experiments/psychedelic_perception_toy.py
# Toy sim: Normal (Euclidean) vs Psychedelic (Hyperbolic) Perception

import numpy as np
import matplotlib.pyplot as plt
from core.geometry import golden_spiral_points, poincare_disk_project, build_hyperbolic_graph
from visualization.plotter import plot_hyperbolic_lattice

N_POINTS = 8000
DIM = 37

# Generate the underlying "base reality" lattice (high-dimensional golden spirals)
points_nd = golden_spiral_points(n_points=N_POINTS, dim=DIM, radius_scale=0.99)

# 1. "Plain Base" - Normal mindset: Euclidean projection (first 2 dims, no hyperbolic curvature)
points_euclidean = points_nd[:, :2] * 0.5  # Scale down to fit nicely

fig1, ax1 = plt.subplots(figsize=(10, 10))
ax1.scatter(points_euclidean[:, 0], points_euclidean[:, 1], c='blue', s=5, alpha=0.7)
ax1.set_title("Plain Base Reality - Euclidean Perception (Normal Mindset)", color='white')
ax1.set_facecolor('black')
ax1.axis('equal')
ax1.axis('off')
plt.savefig("plain_base_euclidean.png", dpi=300, facecolor='black')
plt.show()

# 2. "Altered State" - Psychedelic mindset: Hyperbolic Poincaré disk projection
points_2d_hyper = poincare_disk_project(points_nd[:, :2])
neighbors = build_hyperbolic_graph(points_2d_hyper, k_neighbors=8)

plot_hyperbolic_lattice(
    points_2d=points_2d_hyper,
    edges=neighbors,
    title="Altered State - Hyperbolic Hyperspace Perception (Psychedelic Mindset)",
    save_path="altered_state_hyperbolic.png"
)
