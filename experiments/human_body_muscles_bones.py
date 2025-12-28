# experiments/human_body_muscles_bones.py
# Wholesome Human: Muscles + Bones + Previous Layers
# Skeletal framework and muscular power in hyperbolic space

import numpy as np
import matplotlib.pyplot as plt
from core.geometry import golden_spiral_points, poincare_disk_project

N_POINTS = 30000
DIM = 37

# Substrate
points_nd = golden_spiral_points(n_points=N_POINTS, dim=DIM, radius_scale=0.99)
points_2d = poincare_disk_project(points_nd[:, :2])

fig, ax = plt.subplots(figsize=(18, 26))
ax.set_facecolor('black')

# Faint substrate
ax.scatter(points_2d[:, 0], points_2d[:, 1], c='gray', s=0.5, alpha=0.05)

# === Bones: Crystalline Skeletal Framework ===
bone_segments = [
    # Spine
    ([0, 0], [-0.7, 0.7]),
    # Skull
    ([0, 0.7], [0, 0.8]), 
    ([ -0.15, 0.75 ], [ 0.15, 0.75 ]),
    # Rib cage
    ([ -0.25, 0.1 ], [ 0.25, 0.1 ]), ([ -0.22, 0.0 ], [ 0.22, 0.0 ]), ([ -0.2, -0.1 ], [ 0.2, -0.1 ]),
    # Arms
    ([0.0, 0.1], [-0.4, -0.1]), ([0.0, 0.1], [0.4, -0.1]),  # Shoulders to elbows
    ([-0.4, -0.1], [-0.6, -0.3]), ([0.4, -0.1], [0.6, -0.3]),  # Elbows to hands
    # Legs
    ([0.0, -0.2], [-0.25, -0.9]), ([0.0, -0.2], [0.25, -0.9]),  # Hips to knees
    ([-0.25, -0.9], [-0.3, -1.1]), ([0.25, -0.9], [0.3, -1.1]),  # Knees to feet
]

for start, end in bone_segments:
    ax.plot([start[0], end[0]], [start[1], end[1]], c='cyan', lw=6, alpha=0.9, solid_capstyle='round')
    # Mineral crystal texture
    for i in np.linspace(0, 1, 8):
        mid = np.array(start) * (1 - i) + np.array(end) * i
        offset = np.random.normal(0, 0.02, 2)
        ax.scatter(mid[0] + offset[0], mid[1] + offset[1], c='white', s=10, alpha=0.6)

# Skull glow
ax
