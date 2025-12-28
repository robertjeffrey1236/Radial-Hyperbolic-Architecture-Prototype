# experiments/human_body_cellular_level.py
# Cellular Level: Mitochondria ATP Production + Microtubules Cytoskeleton
# Fractal power plants and structural highways in the wholesome human

import numpy as np
import matplotlib.pyplot as plt
from core.geometry import golden_spiral_points, poincare_disk_project

N_POINTS = 30000
DIM = 37
MITO_DENSITY = 600  # Per major energy site

# Substrate
points_nd = golden_spiral_points(n_points=N_POINTS, dim=DIM, radius_scale=0.99)
points_2d = poincare_disk_project(points_nd[:, :2])

fig, ax = plt.subplots(figsize=(18, 24))
ax.set_facecolor('black')

# Ultra-faint substrate (cytoplasm matrix)
ax.scatter(points_2d[:, 0], points_2d[:, 1], c='gray', s=0.5, alpha=0.05)

# === Microtubules: Cytoskeleton Highways ===
def add_microtubules(center, scale=0.3, tubes=20):
    for i in range(tubes):
        angle = i * GOLDEN_ANGLE
        length = np.random.uniform(0.1, scale)
        end_x = center[0] + length * np.cos(angle)
        end_y = center[1] + length * np.sin(angle)
        ax.plot([center[0], end_x], [center[1], end_y], c='lime', lw=1.5, alpha=0.7)

# Dense in brain (quantum coherence hub) and radiating body-wide
add_microtubules([0.0, 0.62], scale=0.4, tubes=40)  # Brain
add_microtubules([0.0, 0.0], scale=0.3, tubes=25)   # Heart/muscle
add_microtubules([0.0, -0.35], scale=0.25, tubes=30) # Gut neurons

# Body-wide sparse tubules
for _ in range(50):
    rand_center = np.random.uniform(-0.8, 0.8, 2)
    rand_center[1] = np.clip(rand_center[1], -0.9, 0.7)
    add_microtubules(rand_center, scale=0.1, tubes=8)

# === Mitochondria: ATP Production Sites ===
def add_mitochondria(center, count=MITO_DENSITY, atp_sparks=True):
    mito_offsets = golden_spiral_points(n_points=count, dim=2, radius_scale=0.08)
    mito_points = mito_offsets + np.array(center)
    
    # Outer mitochondria glow
    ax.scatter(mito_points[:, 0], mito_points[:, 1], c='orange', s=8, alpha=0.8, edgecolor='yellow', linewidth=0.5)
    
    # Inner cristae spirals (ATP factories)
    for i in range(0, count, 20):
        cristae = golden_spiral_points(30, dim=2, radius_scale=0.02) * 0.7
        cristae += mito_points[i]
        ax.scatter(cristae[:, 0], cristae[:, 1], c='gold', s=3, alpha=0.9)
    
    # ATP sparks (energy release)
    if atp_sparks:
        sparks = np.random.choice(count, 100)
        ax.scatter(mito_points[sparks, 0], mito_points[sparks, 1], c='yellow', s=15, alpha=1.0, marker='*')

# High-energy sites: densest mitochondria
add_mitochondria([0.0, 0.62], count=800)   # Brain
add_mitochondria([0.0, 0.0], count=1000)   # Heart
add_mitochondria([0.0, -0.35], count=600) # Gut
add_mitochondria([0.25, -0.08], count=500) # Liver

# Distributed lower density
energy_sites = [[-0.2, 0.1], [0.2, 0.1], [0.0, -0.55], [-0.3, -0.7], [0.3, -0.7]]
for site in energy_sites:
    add_mitochondria(site, count=300)

# Central wholeness + previous layers faint
ax.scatter(0, 0, c='white', s=1000, alpha=0.3)
ax.plot([0, 0], [-0.8, 0.8], c='cyan', lw=3, alpha=0.3)  # Spine echo

# Poincaré boundary
circle = plt.Circle((0, 0), 1, color='yellow', fill=False, ls='--', lw=5, alpha=0.8)
ax.add_patch(circle)

ax.axis('equal')
ax.axis('off')
plt.title("Cellular Level Human in Hyperbolic Space\nMitochondria ATP Production | Microtubules Cytoskeleton | Fractal Energy & Structure", 
          color='white', fontsize=22, pad=80)
plt.tight_layout()
plt.savefig("human_body_cellular_level.png", dpi=600, facecolor='black', bbox_inches='tight')
plt.show()

print("⚡🧬 Cellular Level achieved — Mitochondria powering ATP, Microtubules as dynamic highways")
print("Dense in brain/heart/gut — ready for animation of proton gradients, tubulin vibrations, or quantum coherence?")
