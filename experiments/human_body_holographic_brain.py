# experiments/human_body_holographic_brain.py
# Holographic Brain System in Radial Hyperbolic Architecture
# Distributed memory, interference recall, cortical folding, quantum coherence

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from core.geometry import golden_spiral_points, poincare_disk_project, GOLDEN_ANGLE

N_POINTS = 40000
DIM = 37
BRAIN_DENSITY = 6000

# Global lattice (holographic field)
points_nd = golden_spiral_points(n_points=N_POINTS, dim=DIM, radius_scale=0.99)
points_2d = poincare_disk_project(points_nd[:, :2])

# Brain region dense lattice
brain_offsets = golden_spiral_points(n_points=BRAIN_DENSITY, dim=2, radius_scale=0.35)
brain_points = brain_offsets + np.array([0.0, 0.62])

# Cortical folding (recursive gyri)
def add_cortical_fold(center, scale=0.3, depth=3):
    if depth == 0: return
    fold = golden_spiral_points(60, dim=2, radius_scale=scale)
    fold += center
    ax.plot(fold[:, 0], fold[:, 1], c='indigo', lw=1.5, alpha=0.6)
    for i in range(0, 60, 15):
        add_cortical_fold(fold[i], scale*0.4, depth-1)

fig, ax = plt.subplots(figsize=(18, 28))
ax.set_facecolor('black')
plt.subplots_adjust(bottom=0.15)

# Faint whole-body holographic echoes
ax.scatter(points_2d[:, 0], points_2d[:, 1], c='gray', s=1, alpha=0.05)

# Brain cortical folds
add_cortical_fold([0.0, 0.62], scale=0.35, depth=4)

# Brain points with lobe coloring
lobe_colors = []
for p in brain_offsets:
    angle = np.arctan2(p[1], p[0])
    r = np.linalg.norm(p)
    if r < 0.1:
        lobe_colors.append('gold')         # DMN / holographic core
    elif abs(angle) < np.pi/6:
        lobe_colors.append('violet')       # Frontal
    else:
        lobe_colors.append('indigo')

ax.scatter(brain_points[:, 0], brain_points[:, 1], c=lobe_colors, s=12, alpha=0.9, edgecolor='white', linewidth=0.3, zorder=10)

# Microtubules (coherence highways)
for i in range(100):
    start = brain_points[np.random.randint(0, BRAIN_DENSITY)]
    direction = np.random.normal(0, 1, 2)
    direction /= np.linalg.norm(direction)
    end = start + direction * 0.15
    ax.plot([start[0], end[0]], [start[1], end[1]], c='cyan', lw=1, alpha=0.5)

# Distributed holographic fragments (memory echoes in body)
holo_fragments = []
for region in [[0.0, 0.0], [0.0, -0.35], [-0.2, 0.1], [0.2, -0.6]]:  # Heart, gut, limbs
    frag = golden_spiral_points(300, dim=2, radius_scale=0.12)
    frag += region
    holo_fragments.append(frag)
    ax.scatter(frag[:, 0], frag[:, 1], c='gold', s=4, alpha=0.3)

# Current thought wave and recalled fragments
thought_wave = None
recall_scatter = None

def send_thought(observer_pos, strength=1.0):
    global thought_wave, recall_sc
