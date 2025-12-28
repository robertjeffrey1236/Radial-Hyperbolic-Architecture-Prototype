# experiments/human_body_sight_observer.py
# Wholesome Human with Sight as Observer Feature
# Eyes as draggable portals — perception renders the hyperbolic universe dynamically

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from core.geometry import golden_spiral_points, poincare_disk_project

N_POINTS = 20000
DIM = 37

# Generate lattice
points_nd = golden_spiral_points(n_points=N_POINTS, dim=DIM, radius_scale=0.99)
points_2d = poincare_disk_project(points_nd[:, :2])

# Initial eye positions (on face)
left_eye = np.array([-0.12, 0.72])
right_eye = np.array([0.12, 0.72])
third_eye = np.array([0.0, 0.75])

fig, ax = plt.subplots(figsize=(16, 20))
ax.set_facecolor('black')
plt.subplots_adjust(bottom=0.15)

# Faint full lattice (unperceived background)
base_scatter = ax.scatter(points_2d[:, 0], points_2d[:, 1], c='gray', s=2, alpha=0.1)

# Sight cones (field of view)
def update_sight(left_pos, right_pos, fov=0.8, third_active=True):
    # Clear previous perceived points
    if 'perc_scatter' in globals():
        perc_scatter.remove()
    if 'cone_left' in globals():
        cone_left.remove()
        cone_right.remove()
    if 'third_cone' in globals():
        third_cone.remove()
    
    # Distance from eyes
    dist_left = np.linalg.norm(points_2d - left_pos, axis=1)
    dist_right = np.linalg.norm(points_2d - right_pos, axis=1)
    
    # Perceived: closer to either eye + angle within cone
    in_view = (dist_left < fov) | (dist_right < fov)
    intensity = np.exp(-np.minimum(dist_left, dist_right) * 3)
    
    global perc_scatter
    perc_scatter = ax.scatter(points_2d[in_view, 0], points_2d[in_view, 1], 
                              c='cyan', s=10 * intensity[in_view], alpha=0.9, zorder=10)
    
    # Sight cones
    global cone_left, cone_right
    cone_angle = np.linspace(-np.pi/4, np.pi/4, 20)
    cone_left_x = left_pos[0] + fov * np.cos(cone_angle)
    cone_left_y = left_pos[1] + fov * np.sin(cone_angle)
    cone_left = ax.plot(np.append(cone_left_x, left_pos[0]), 
                        np.append(cone_left_y, left_pos[1]), c='cyan', lw=2, alpha=0.5)[0]
    
    cone_right_x = right_pos[0] + fov * np.cos(cone_angle)
    cone_right_y = right_pos[1] + fov * np.sin(cone_angle)
    cone_right = ax.plot(np.append(cone_right_x, right_pos[0]), 
                         np.append(cone_right_y, right_pos[1]), c='cyan', lw=2, alpha=0.5)[0]
    
    # Third eye expansion
    if
