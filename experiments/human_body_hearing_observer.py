# experiments/human_body_hearing_observer.py
# Wholesome Human with Hearing as Observer Feature
# Ears as draggable receptors — focused listening illuminates resonant patterns

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from core.geometry import golden_spiral_points, poincare_disk_project, GOLDEN_ANGLE

N_POINTS = 20000
DIM = 37

# Lattice
points_nd = golden_spiral_points(n_points=N_POINTS, dim=DIM, radius_scale=0.99)
points_2d = poincare_disk_project(points_nd[:, :2])

# Initial ear positions
left_ear = np.array([-0.20, 0.68])
right_ear = np.array([0.20, 0.68])
inner_ear = np.array([0.0, 0.70])  # Subtle third-ear intuition

fig, ax = plt.subplots(figsize=(16, 20))
ax.set_facecolor('black')
plt.subplots_adjust(bottom=0.15)

# Background lattice (silent)
silent_scatter = ax.scatter(points_2d[:, 0], points_2d[:, 1], c='gray', s=2, alpha=0.1)

# Current resonant points and waves
resonant_scatter = None
wave_lines = []

def update_hearing(left_pos, right_pos, sensitivity=1.0, inner_active=True):
    global resonant_scatter, wave_lines
    
    # Clear previous
    if resonant_scatter:
        resonant_scatter.remove()
    for line in wave_lines:
        line.remove()
    wave_lines.clear()
    
    # Combined hearing field
    dist_left = np.linalg.norm(points_2d - left_pos, axis=1)
    dist_right = np.linalg.norm(points_2d - right_pos, axis=1)
    combined_dist = np.minimum(dist_left, dist_right)
    
    in_range = combined_dist < sensitivity
    resonance = np.exp(-combined_dist * 2.5 / sensitivity)
    
    global resonant_scatter
    resonant_scatter = ax.scatter(points_2d[in_range, 0], points_2d[in_range, 1],
                                  c='lime', s=15 * resonance[in_range], alpha=0.9, zorder=10)
    
    # Sound waves — golden spirals from each ear
    for ear_pos, color in zip([left_pos, right_pos], ['cyan', 'magenta']):
        for r in np.linspace(0.05, sensitivity, 8):
            wave = golden_spiral_points(60, dim=2, radius_scale=r)
            wave += ear_pos
            line = ax.plot(wave[:,
