# experiments/human_body_nervous_system.py
# Human Body + Nervous System in Radial Hyperbolic Architecture
# Fractal branching nerves, chakra nodes, golden-ratio proportions

import numpy as np
import matplotlib.pyplot as plt
from core.geometry import golden_spiral_points, poincare_disk_project, build_hyperbolic_graph

N_POINTS = 12000
DIM = 37

# Generate substrate lattice
points_nd = golden_spiral_points(n_points=N_POINTS, dim=DIM, radius_scale=0.99)
points_2d = poincare_disk_project(points_nd[:, :2])
neighbors = build_hyperbolic_graph(points_2d, k_neighbors=10)

# === Key Body Landmarks (same as before, refined) ===
body_landmarks = {
    'root': np.array([0.0, -0.65]),
    'sacral': np.array([0.0, -0.45]),
    'solar_plexus': np.array([0.0, -0.20]),
    'heart': np.array([0.0, 0.0]),            # Central core
    'throat': np.array([0.0, 0.20]),
    'third_eye': np.array([0.0, 0.40]),
    'crown': np.array([0.0, 0.60]),
    'left_shoulder': np.array([-0.35, 0.05]),
    'right_shoulder': np.array([0.35, 0.05]),
    'left_elbow': np.array([-0.55, -0.05]),
    'right_elbow': np.array([0.55, -0.05]),
    'left_hand': np.array([-0.65, -0.15]),
    'right_hand': np.array([0.65, -0.15]),
    'left_knee': np.array([-0.15, -0.85]),
    'right_knee': np.array([0.15, -0.85]),
    'left_foot': np.array([-0.25, -1.0]),
    'right_foot': np.array([0.25, -1.0]),
}

chakra_colors = {
    'root': 'red', 'sacral': 'orange', 'solar_plexus': 'yellow',
    'heart': 'green', 'throat': 'cyan', 'third_eye': 'blue', 'crown': 'violet'
}

fig, ax = plt.subplots(figsize=(14, 16))
ax.set_facecolor('black')

# Faint background lattice (neural substrate)
ax.scatter(points_2d[:, 0], points_2d[:, 1], c='gray', s=2, alpha=0.2)

# === Central Nervous System: Spine + Brain Projection ===
spine_path = np.array([body_landmarks[k] for k in ['root', 'sacral', 'solar_plexus', 'heart', 'throat', 'third_eye', 'crown']])
ax.plot(spine_path[:, 0], spine_path[:, 1], c='white', lw=6, alpha=0.9, label='Central Nervous System')

# Brain "aura" at crown/third eye
brain_center = (body_landmarks['crown'] + body_landmarks['third_eye']) / 2
brain_radius = 0.25
brain_circle = plt.Circle(brain_center, brain_radius, color='blue', fill=False, ls='-', lw=3, alpha=0.6)
ax.add_patch(brain_circle)
ax.scatter(brain_center[0], brain_center[1], c='indigo', s=300, alpha=0.7, edgecolor
