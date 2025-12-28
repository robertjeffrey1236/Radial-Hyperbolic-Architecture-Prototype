# experiments/human_body_ego_dissolution.py
# Ego-Dissolution / Psychedelic State Mode
# Toggle: boundaries dissolve • lattice expands • colors hypersaturate • observer multiplies

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from core.geometry import golden_spiral_points, poincare_disk_project

N_POINTS = 40000
DIM = 37

# Base lattice
points_nd = golden_spiral_points(n_points=N_POINTS, dim=DIM, radius_scale=0.99)
points_2d = poincare_disk_project(points_nd[:, :2])

fig, ax = plt.subplots(figsize=(18, 28))
ax.set_facecolor('black')
plt.subplots_adjust(bottom=0.1)

# Normal state elements
normal_lattice = ax.scatter(points_2d[:, 0], points_2d[:, 1], c='cyan', s=8, alpha=0.7)
skin = plt.Circle((0, 0), 0.9, color='white', fill=False, lw=4, alpha=0.6)
ax.add_patch(skin)
chakras = []
for y, color in zip(np.linspace(-0.6, 0.65, 7), ['red','orange','yellow','green','cyan','indigo','violet']):
    c = ax.scatter(0, y, c=color, s=400, alpha=0.7, edgecolor='white', linewidth=2)
    chakras.append(c)

# Psychedelic state elements (initially hidden)
psy_lattice = ax.scatter([], [], c='magenta', s=15, alpha=1.0, edgecolor='white', linewidth=1)
multi_observers = ax.scatter([], [], c='white', s=300, marker='*', alpha=0.9)
hypersaturated_aura = plt.Circle((0, 0), 1.5, color='rainbow', fill=False, lw=8, alpha=0)

psy_mode = False

def toggle_psychedelic(event):
    global psy_mode
    psy_mode = not psy_mode
    
    if psy_mode:
        # Dissolve boundaries
        skin.set_alpha(0)
        skin.set_radius(2.0)
        
        # Expand & hypersaturate lattice
        expanded = points_2d * (1.5 + 0.5 * np.random.random(N_POINTS)[:, None])
        colors = plt.cm.psychedelic(np.random.random(N_POINTS))
        normal_lattice.set_offsets(expanded)
        normal_lattice.set_color(colors)
        normal_lattice.set_sizes(20 + 30 * np.random.random(N_POINTS))
        normal_lattice.set_alpha(1.0)
        
        # Multiply observers (non-local awareness)
        obs_x = np.random.uniform(-0.8, 0.8, 12)
        obs_y = np.random.uniform(-0.9, 0.9, 12)
        multi_observers.set_offsets(np.c_[obs_x, obs_y])
        
        # Bloom chakras & aura
        for c in chakras:
            c.set_alpha(1.0)
            c.set_sizes([800])
        ax.add_patch(hypersaturated_aura)
        hypersaturated_aura.set_alpha(0.8)
        hypersaturated_aura.set_linewidth(12)
        
        ax.set_title("EGO-DISSOLUTION STATE\nBoundaries Dissolved • Infinite Recursion • Non-Local Awareness", 
                     color='white', fontsize=24, pad=100)
        
    else:
        # Return to embodied state
        skin.set_alpha(0.6)
        skin.set_radius(0.9)
        
        normal_lattice.set_offsets(points_2d)
        normal_lattice.set_color('cyan')
        normal_lattice.set_sizes(8)
        normal_lattice.set_alpha(0
