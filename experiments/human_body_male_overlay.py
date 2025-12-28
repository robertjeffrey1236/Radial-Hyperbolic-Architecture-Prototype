# experiments/human_body_male_overlay.py
# Wholesome Human with Anatomical Male Overlay
# Masculine expression: broader build, muscle definition, reproductive anatomy, strong structure

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

# Faint internal substrate
ax.scatter(points_2d[:, 0], points_2d[:, 1], c='gray', s=0.5, alpha=0.03)

# === Base Skin Envelope (from previous) - translucent neutral base ===
theta = np.linspace(0, 2*np.pi, 300)
base_radius_x = 0.4 + 0.15 * np.abs(np.sin(theta * 3))
base_radius_y = 1.1 - 0.3 * np.abs(np.cos(theta))
base_x = base_radius_x * np.cos(theta) * np.abs(np.cos(theta))
base_y = base_radius_y * np.sin(theta)
ax.fill(base_x, base_y, c='peachpuff', alpha=0.15, zorder=4)

# === Male Overlay: Broader, Stronger Contour ===
male_radius_x = 0.5 + 0.2 * np.abs(np.sin(theta * 2.5))  # Wider shoulders/chest
male_radius_y = 1.15 - 0.25 * np.abs(np.cos(theta))
male_x = male_radius_x * np.cos(theta)
male_y = male_radius_y * np.sin(theta)

# Stronger shoulder/chest definition
ax.fill(male_x, male_y, c='royalblue', alpha=0.2, zorder=6)
ax.plot(male_x, male_y, c='deepskyblue', lw=5, alpha=0.8, zorder=7)

# Pronounced pectorals and abs
ax.scatter([ -0.15, 0.15 ], [0.1, 0.1], c='steelblue', s=400, alpha=0.6, zorder=8)
abs_y = np.linspace(-0.1, -0.3, 6)
for y in abs_y:
    ax.plot([-0.15, 0.15], [y, y], c='cyan', lw=2, alpha=0.7)

# Strong jaw/brow
ax.plot([-0.12, -0.08, 0.08, 0.12], [0.68, 0.72, 0.72, 0.68], c='white', lw=4, alpha=0.8)

# Male reproductive emphasis (root center)
ax.scatter(0, -0.55, c='deepskyblue', s=350, alpha=0.7
