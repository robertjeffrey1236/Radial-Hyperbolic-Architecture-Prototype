# experiments/human_body_with_skin.py
# Complete Wholesome Human: Skin as Final Boundary Layer
# Translucent envelope, fractal texture, sensory receptors, microbiome surface

import numpy as np
import matplotlib.pyplot as plt
from core.geometry import golden_spiral_points, poincare_disk_project

N_POINTS = 35000
DIM = 37

# Substrate
points_nd = golden_spiral_points(n_points=N_POINTS, dim=DIM, radius_scale=0.99)
points_2d = poincare_disk_project(points_nd[:, :2])

fig, ax = plt.subplots(figsize=(18, 28))
ax.set_facecolor('black')

# Faint internal substrate (visible through translucent skin)
ax.scatter(points_2d[:, 0], points_2d[:, 1], c='gray', s=0.5, alpha=0.03)

# === Skin: Translucent Outer Envelope ===
# Body contour points (approximate human silhouette)
theta = np.linspace(0, 2*np.pi, 300)
skin_radius_x = 0.4 + 0.15 * np.abs(np.sin(theta * 3))  # Wider torso/hips
skin_radius_y = 1.1 - 0.3 * np.abs(np.cos(theta))      # Taller
skin_x = skin_radius_x * np.cos(theta) * np.abs(np.cos(theta))
skin_y = skin_radius_y * np.sin(theta)

# Outer skin glow (semi-transparent)
ax.fill(skin_x, skin_y, c='peachpuff', alpha=0.25, zorder=5)
ax.plot(skin_x, skin_y, c='gold', lw=4, alpha=0.7, zorder=6)

# Epidermal fractal texture (subtle golden spirals on surface)
for i in range(80):
    center = np.random.uniform(-0.6, 0.6, 2)
    center[1] = np.clip(center[1], -1.0, 0.8)
    if np.linalg.norm(center) < 0.9:  # Only on body surface
        texture = golden_spiral_points(40, dim=2, radius_scale=0.05)
        texture += center
        ax.scatter(texture[:, 0], texture[:, 1], c='gold', s=4, alpha=0.6)

# Sensory receptors (touch, temperature, pain points)
receptors = golden_spiral_points(500, dim=2, radius_scale=0.9)
ax.scatter(receptors[:, 0], receptors[:, 1], c='cyan', s=8, alpha=0.8, marker='o', edgecolor='white', linewidth=0.5, zorder=7)

# Surface microbiome echo (visible on skin)
for i in range(40):
    site = np.random.uniform(-0.7, 0.7, 2)
    site[1] = np.clip(site[1], -1.0, 0.8)
    micro = golden_spiral_points(80, dim=2, radius_scale=0.04)
    micro += site
    ax.scatter(micro[:, 0], micro[:, 1], c='lime', s=3, alpha=0.7)

# === Internal Layers Visible Through Translucent Skin ===
# Major organs faint glow
ax.scatter(0, 0.62, c='indigo', s=500, alpha=0.4)      # Brain
ax.scatter(0, 0, c='crimson', s=600, alpha=0.4)        # Heart
ax.scatter(0, -0.35, c='orange', s=400, alpha=0.3)     # Gut

# Bones faint
ax.plot([0, 0], [-0.8, 0.8], c='cyan', lw=4, alpha=0.3)

# Muscles faint
ax.plot([ -0.3, -0.7 ], [ 0.1, -0.4 ], c='magenta', lw=6, alpha=0.2)
ax.plot([ 0.3, 0.7 ], [ 0.1, -0.4 ], c='magenta', lw=6, alpha=0.2)

# Poincaré boundary aura
circle = plt.Circle((0, 0), 1, color='gold', fill=False, ls='--', lw=6, alpha=0.9)
ax.add_patch(circle)

ax.axis('equal')
ax
