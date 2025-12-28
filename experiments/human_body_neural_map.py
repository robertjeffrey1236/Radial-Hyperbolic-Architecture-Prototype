# experiments/human_body_neural_map.py
# Full Human Bio-Architecture: Body + Vascular + Nervous + Neural Map (Connectome)
# Brain as hyperbolic sub-manifold with cortical clusters and long-range tracts

import numpy as np
import matplotlib.pyplot as plt
from core.geometry import golden_spiral_points, poincare_disk_project, build_hyperbolic_graph

N_POINTS = 18000
DIM = 37
BRAIN_POINTS = 4000  # Dense neural substrate in head region

# Global substrate
points_nd = golden_spiral_points(n_points=N_POINTS, dim=DIM, radius_scale=0.99)
points_2d = poincare_disk_project(points_nd[:, :2])
neighbors = build_hyperbolic_graph(points_2d, k_neighbors=10)

# Body landmarks (focused on head for neural map)
body_landmarks = {
    'heart': np.array([0.0, 0.0]),
    'crown': np.array([0.0, 0.68]),
    'third_eye': np.array([0.0, 0.50]),
    'brain_center': np.array([0.0, 0.59]),
}

fig, ax = plt.subplots(figsize=(16, 18))
ax.set_facecolor('black')

# Faint full-body substrate (tissue + capillary)
ax.scatter(points_2d[:, 0], points_2d[:, 1], c='darkgray', s=1, alpha=0.1)

# === Neural Map: Brain Connectome in Hyperbolic Sub-Manifold ===
# Generate dense brain lattice centered at crown
brain_offsets = golden_spiral_points(n_points=BRAIN_POINTS, dim=2, radius_scale=0.3)
brain_points = brain_offsets + body_landmarks['brain_center']
brain_points = np.clip(brain_points, -0.95, 0.95)  # Keep inside disk

# Color by cortical region (simplified lobes + DMN)
colors = []
for p in brain_offsets:
    angle = np.arctan2(p[1], p[0])
    r = np.linalg.norm(p)
    if r < 0.1:
        colors.append('gold')          # Default Mode Network core
    elif abs(angle) < np.pi/4:
        colors.append('violet')        # Frontal lobe
    elif angle > np.pi/2:
        colors.append('indigo')        # Temporal
    elif angle < -np.pi/2:
        colors.append('blue')          # Parietal
    else:
        colors.append('purple')        # Occipital / other

ax.scatter(brain_points[:, 0], brain_points[:, 1], c=colors, s=15, alpha=0.9, edgecolor='white', linewidth=0.3, zorder=10)

# Long-range white matter tracts (wormhole-like shortcuts)
tracts = [
    ([0.0, 0.68], [-0.3, 0.55]),  # Left hemisphere connection
    ([0.0, 0.68], [0.3, 0.55]),   # Right
    ([-0.2, 0.62], [0.2, 0.62]),   # Corpus callosum analog
    ([0.0, 0.50], [0.0, 0.68]),    # Third eye to crown vertical tract
