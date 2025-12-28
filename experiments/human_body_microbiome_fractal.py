# experiments/human_body_microbiome_fractal.py
# Wholesome Human + Fractal Microbiome Intelligence
# Distributed "second brain" as self-similar reflections across the body

import numpy as np
import matplotlib.pyplot as plt
from core.geometry import golden_spiral_points, poincare_disk_project

N_POINTS = 25000
DIM = 37
MICRO_POINTS = 300  # Per microbiome cluster

# Global substrate
points_nd = golden_spiral_points(n_points=N_POINTS, dim=DIM, radius_scale=0.99)
points_2d = poincare_disk_project(points_nd[:, :2])

fig, ax = plt.subplots(figsize=(16, 22))
ax.set_facecolor('black')

# Faint full-body substrate
ax.scatter(points_2d[:, 0], points_2d[:, 1], c='gray', s=1, alpha=0.08)

# === Major Organs (fainter base layer) ===
organs = {
    'brain': [0.0, 0.62], 'heart': [0.0, 0.0], 'lungs': [[-0.18, 0.08], [0.18, 0.08]],
    'liver': [0.25, -0.08], 'gut_hub': [0.0, -0.35], 'reproductive': [0.0, -0.55],
}
for name, pos in organs.items():
    if isinstance(pos[0], list):
        for p in pos:
            ax.scatter(p[0], p[1], c='white', s=200, alpha=0.3)
    else:
        ax.scatter(pos[0], pos[1], c='white', s=300 if name == 'brain' or name == 'heart' else 200, alpha=0.3)

# === Fractal Microbiome: Self-Similar Reflections ===
def add_microbiome_cluster(center, scale=0.15, density=400, color_variance=True):
    micro_offsets = golden_spiral_points(n_points=density, dim=2, radius_scale=scale)
    micro_points = micro_offsets * 0.8 + np.array(center)
    
    colors = plt.cm.viridis(np.linspace(0, 1, density)) if color_variance else 'lime'
    ax.scatter(micro_points[:, 0], micro_points[:, 1], c=colors, s=6, alpha=0.8, edgecolor='none', zorder=8)
    
    # Tiny "gut-brain axis" link if near gut
    if abs(center[1] + 0.35) < 0.1:
        ax.plot([center[0], 0.0], [center[1], 0.62], c='gold', lw=1, alpha=0.5, ls='--')

# Primary Gut Microbiome — dense ancient intelligence hub
add_microbiome_cluster([0.0, -0.35], scale=0.25, density=800, color_variance=True)

# Distributed fractured reflections (diverse population)
micro_sites = [
    [-0.2, 0.08], [0.2, 0.08],   # Lungs
    [0.25, -0.08], [-0.25, -0.15],  # Liver & sides
    [0.0, 0.0], [0.0, 0.62],     # Heart & brain echoes
    [-0.15, -0.55], [0.15, -0.55],  # Lower body
    [0.0, -0.15], [-0.1, 0.25],   # Stomach & throat
]
for site in micro_sites:
    add_microbiome_cluster(site, scale=0.08 + np.random.random()*0.05, density=200 + np.random.randint(100), color_variance=True)

# Gut-Brain Axis highlight
ax.plot([0, 0], [-0.35, 0.62], c='gold', lw=4, alpha=0.7, zorder=9)
ax.text(0.05, 0.15, 'Gut-Brain Axis\nDistributed Intelligence', color='gold', fontsize=14, rotation=90, alpha=0.9)

# Central wholeness glow
ax.scatter(0, 0, c='white', s=800, alpha=0.4, zorder=5)

# Poincaré boundary
circle = plt.Circle((0, 0), 1, color='lime', fill=False, ls='--', lw=4, alpha=0.8)
ax.add_patch(circle)

ax.axis('
