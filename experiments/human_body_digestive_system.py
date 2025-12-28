# experiments/human_body_digestive_system.py
# Wholesome Human with Complete Digestive System
# Full tract: esophagus → stomach → small/large intestine → absorption + microbiome integration

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

# Faint substrate
ax.scatter(points_2d[:, 0], points_2d[:, 1], c='gray', s=0.5, alpha=0.03)

# === Digestive Tract Pathway ===
# Esophagus (throat to stomach)
ax.plot([0, 0], [0.25, -0.1], c='peachpuff', lw=8, alpha=0.8, solid_capstyle='round')

# Stomach (upper left curve)
stomach_center = np.array([ -0.1, -0.1 ])
stomach_theta = np.linspace(0, 1.5*np.pi, 100)
stomach_x = stomach_center[0] + 0.15 * np.cos(stomach_theta)
stomach_y = stomach_center[1] + 0.12 * np.sin(stomach_theta) - 0.05
ax.plot(stomach_x, stomach_y, c='coral', lw=10, alpha=0.9)
ax.scatter(-0.1, -0.1, c='orange', s=400, alpha=0.8, edgecolor='gold', linewidth=3)
ax.text(-0.1, 0.05, 'Stomach', color='white', fontsize=12, ha='center')

# Small Intestine — long coiled fractal path (absorption center)
small_intest = []
current = np.array([-0.1, -0.2])
for i in range(60):
    angle = i * 0.4 + np.pi/4
    step = 0.06 + 0.01 * np.sin(i * 0.5)
    current += step * np.array([np.cos(angle), np.sin(angle)])
    small_intest.append(current.copy())

small_intest = np.array(small_intest)
ax.plot(small_intest[:, 0], small_intest[:, 1], c='gold', lw=6, alpha=0.8)
# Villi — micro absorption spirals
for i in range(0, len(small_intest), 8):
    pos = small_intest[i]
    villi = golden_spiral_points(30, dim=2, radius_scale=0.03) * 0.7
    villi += pos
    ax.scatter(villi[:, 0], villi[:, 1], c='yellow', s=4, alpha=0.9)

# Large Intestine — framing lower body
large_intest = [
    [-0.3, -0.3], [-0.3, -0.7], [0.3, -0.7], [0.3, -0.3],
    [0.2, -0.5], [0.0, -0.6], [-0.2, -0.5]
]
for seg in large_intest:
    ax.scatter(seg[0], seg[1], c='sandybrown', s=300, alpha=0.7)
ax.plot([-0.3, -0.3, 0.3, 0.3, -0.3], [-0.3, -0.7, -0.7, -0.3, -0.3], 
        c='peru', lw=8, alpha=0.8)

# Rectum / elimination point
ax.scatter(0, -0.75, c='sienna', s=250, alpha=0.8)

# Liver (detox partner, right upper abdomen)
ax.scatter(0.22, -0.05, c='darkred', s=450, alpha=0.8, edgecolor='gold', linewidth=3)
ax.text(0.22, 0.1, 'Liver', color='white', fontsize=12, ha='center')

# Microbiome dense in intestines
for center in [[0, -0.4], [-0.15, -0.5], [0.15, -0.5]]:
    micro = golden_spiral_points(400, dim=2, radius_scale=0.12)
    micro += center
    ax.scatter(micro[:, 0], micro[:, 1], c='lime', s=5, alpha=0.8)

# === Other Systems Faint Overlay ===
ax.scatter(0, 0.62, c='indigo', s=500, alpha=0.4)      # Brain
ax.scatter(0, 0, c='crimson', s=600, alpha=0.4)        # Heart
ax.plot([0, 0], [-0.8, 0.8], c='cyan', lw=4, alpha=0.3)  # Spine
ax.fill(base_x, base_y, c='peachpuff', alpha=0.1)       # Skin echo

# Poincaré boundary
circle = plt.Circle((0, 0), 1, color='orange', fill=False, ls='--', lw=6, alpha=0.9)
ax.add_patch(circle)

ax.axis('equal')
ax.axis('
