# experiments/human_body_tesseract_echo.py
# Higher-Dimensional Projection: Tesseract Echo in Third-Eye/Crown
# Rotating 4D hypercube projected into Poincaré disk with Φ-scaled unfolding

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from core.geometry import golden_spiral_points, poincare_disk_project

PHI = (1 + np.sqrt(5)) / 2
N_POINTS = 30000
DIM = 37

# Substrate
points_nd = golden_spiral_points(n_points=N_POINTS, dim=DIM, radius_scale=0.99)
points_2d = poincare_disk_project(points_nd[:, :2])

fig, ax = plt.subplots(figsize=(18, 28))
ax.set_facecolor('black')

# Faint substrate
ax.scatter(points_2d[:, 0], points_2d[:, 1], c='gray', s=0.5, alpha=0.03)

# Crown/third-eye center
crown_center = np.array([0.0, 0.65])
ax.scatter(crown_center[0], crown_center[1], c='violet', s=600, alpha=0.6, edgecolor='gold', linewidth=4)

# === 4D Tesseract Vertices (in 4D space) ===
# Standard tesseract vertices: all combinations of ±1 in 4D
tesseract_4d = np.array([[x, y, z, w] for x in [-1, 1] for y in [-1, 1] for z in [-1, 1] for w in [-1, 1]])

# Φ-scaled unfolding
scale_4d = 0.3 * PHI  # Golden ratio scaling for higher-D arms
tesseract_4d *= scale_4d

# Edges: connect vertices differing in exactly one coordinate
edges = []
for i in range(16):
    for j in range(i+1, 16):
        if np.sum(np.abs(tesseract_4d[i] - tesseract_4d[j])) == 2 * scale_4d:
            edges.append((i, j))

# Projection function: 4D → 2D (perspective projection from 4D to 3D to 2D)
def project_4d_to_2d(points_4d, t):
    # Rotate in 4D planes with golden-angle derived speeds
    theta_xw = t * 0.3
    theta_yw = t * 0.5 * PHI
    theta_zw = t * 0.2
    
    # Rotation matrices
    rot_xw = np.array([[np.cos(theta_xw), 0, 0, -np.sin(theta_xw)],
                       [0, 1, 0, 0],
                       [0, 0, 1, 0],
                       [np.sin(theta_xw), 0, 0, np.cos(theta_xw)]])
    
    rot_yw = np.array([[1, 0, 0, 0],
                       [0, np.cos(theta_yw), 0, -np.sin(theta_yw)],
                       [0, 0, 1, 0],
                       [0, np.sin(theta_yw), 0, np.cos(theta_yw)]])
    
    rot_zw = np.array([[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, np.cos(theta_zw), -np.sin(theta_zw)],
                       [0, 0, np.sin(theta_zw), np.cos(theta_zw)]])
    
    rotated = points_4d @ rot_xw @ rot_yw @ rot_zw
    
    # Perspective projection: 4D → 3D (w as distance)
    distance = 4.0
    projected_3d = rotated[:, :3] / (distance - rotated[:, 3][:, None])
    
    # Orthographic 3D → 2D
    x = projected_3d[:, 0]
    y = projected_3d[:, 1]
    
    # Center in crown and scale
    x += crown_center[0]
    y += crown_center[1]
    
    return np.column_stack([x, y])

# Initial plot objects
tesseract_lines = [ax.plot([], [], c='gold', lw=2, alpha=0.8)[0] for _ in edges]
tesseract_points = ax.scatter([], [], c='white', s=50, alpha=0.9, edgecolor='gold', linewidth=2)

def animate(frame):
    t = frame * 0.02
    
    # Project current rotation
    projected_2d = project_4d_to_2d(tesseract_4d, t)
    
    # Update points
    tesseract_points.set_offsets(projected_2d)
    
    # Update edges
    for (i, j), line in zip(edges, tesseract_lines):
        line.set_data([projected_2d[i, 0], projected_2d[j, 0]],
                      [projected_2d[i, 1], projected_2d[j, 1]])
    
    # Subtle pulsing with breath-like rhythm
    pulse = 0.9 + 0.1 * np.sin(t * 0.5)
    for line in tesseract_lines:
        line.set_alpha(0.6 + 0.4 * pulse)
    tesseract_points.set_alpha(0.7 + 0.3 * pulse)

# Other layers faint
ax.scatter(0, 0, c='crimson', s=600, alpha=0.3)  # Heart
ax.plot([0, 0], [-0.8, 0.8], c='white', lw=3, alpha=0.2)  # Sushumna

# Poincaré boundary
boundary = plt.Circle((0, 0), 1, color='gold', fill=False, ls='--', lw=6)
ax.add_patch(boundary)

ax.axis('equal')
ax.axis('off')
ax.set_title("Higher-Dimensional Projection: Tesseract Echo\n4D Hypercube Rotating in Third-Eye/Crown | Φ-Scaled Unfolding", 
             color='white', fontsize=22, pad=100)

anim = FuncAnimation(fig, animate, frames=1000, interval=50, repeat=True)

plt.show()

print("🕸️✨ Higher-Dimensional Tesseract Echo activated")
print("Rotating 4D hypercube projected into Poincaré disk at crown/third-eye")
print("Φ-scaled arms unfold — subtle breath-pulse synchronization")
print("The human as 3D cross-section of greater 4D structure — infinite recursion revealed")
