# experiments/human_body_vascular_system.py
# Human Body + Nervous + Vascular System in Radial Hyperbolic Architecture
# Realistic arterial/venous branching, fractal blood flow substrate

import numpy as np
import matplotlib.pyplot as plt
from core.geometry import golden_spiral_points, poincare_disk_project, build_hyperbolic_graph

N_POINTS = 15000
DIM = 37

# Substrate lattice (represents capillary bed / tissue matrix)
points_nd = golden_spiral_points(n_points=N_POINTS, dim=DIM, radius_scale=0.99)
points_2d = poincare_disk_project(points_nd[:, :2])
neighbors = build_hyperbolic_graph(points_2d, k_neighbors=12)

# Body landmarks (refined for vascular origins)
body_landmarks = {
    'heart': np.array([0.0, 0.0]),
    'root': np.array([0.0, -0.65]),
    'crown': np.array([0.0, 0.60]),
    'left_shoulder': np.array([-0.35, 0.05]),
    'right_shoulder': np.array([0.35, 0.05]),
    'left_elbow': np.array([-0.55, -0.05]),
    'right_elbow': np.array([0.55, -0.05]),
    'left_hand': np.array([-0.70, -0.15]),
    'right_hand': np.array([0.70, -0.15]),
    'left_knee': np.array([-0.20, -0.85]),
    'right_knee': np.array([0.20, -0.85]),
    'left_foot': np.array([-0.30, -1.05]),
    'right_foot': np.array([0.30, -1.05]),
    'head_top': np.array([0.0, 0.75]),
}

fig, ax = plt.subplots(figsize=(14, 16))
ax.set_facecolor('black')

# Capillary bed — dense faint lattice in periphery
ax.scatter(points_2d[:, 0], points_2d[:, 1], c='darkred', s=1, alpha=0.15)

# === Vascular System: Fractal Branching ===
def draw_vessel(start_pos, end_pos, width=5, depth=4, is_arterial=True):
    if depth == 0 or np.linalg.norm(end_pos - start_pos) < 0.05:
        return
    
    color = 'crimson' if is_arterial else 'mediumblue'
    alpha = 0.9 - 0.15 * (4 - depth)  # Fade slightly with depth
    ax.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], 
            c=color, lw=width, alpha=alpha)
    
    # Murray's law approximation: child width ~ parent * 0.7
    child_width = width * 0.72
    
    # Branch direction with golden-angle perturbation for natural look
    direction = end_pos - start_pos
    mid = (start_pos + end_pos) / 2
    
    for angle_offset in [0.4, -0.4]:  # Bifurcation
        rot_angle = np.pi / 6 + angle_offset + np.random.normal(0, 0.1)
        rot_matrix = np.array([[np.cos(rot_angle), -np.sin(rot_angle)],
                               [np.sin(rot_angle), np.cos(rot_angle)]])
        branch_dir = rot_matrix @ direction * 0.6
        branch_end = mid + branch_dir
        
        draw_vessel(mid, branch_end, child_width, depth-1, is_arterial)

# Major arterial outflows from heart
arterial_targets = ['left_shoulder', 'right_shoulder', 'head_top', 'root']
for target in arterial_targets:
    draw_vessel(body_landmarks['heart'], body_landmarks[target], width=7, depth=5, is_arterial=True)

# Major venous returns to heart (reverse flow, thinner)
venous_origins = ['left_hand', 'right_hand', 'left_foot', 'right
