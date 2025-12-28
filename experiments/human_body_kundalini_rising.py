# experiments/human_body_kundalini_rising.py
# Kundalini Rising Simulation in Radial Hyperbolic Architecture
# Coiled root energy awakens → rises → activates chakras → crown unity

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from core.geometry import golden_spiral_points, poincare_disk_project

N_POINTS = 30000
DIM = 37

# Substrate
points_nd = golden_spiral_points(n_points=N_POINTS, dim=DIM, radius_scale=0.99)
points_2d = poincare_disk_project(points_nd[:, :2])

fig, ax = plt.subplots(figsize=(18, 30))
ax.set_facecolor('black')

# Faint substrate
ax.scatter(points_2d[:, 0], points_2d[:, 1], c='gray', s=0.5, alpha=0.03)

# Chakra positions and colors
chakras = [
    ('root',      [0.0, -0.60], 'red',     0),
    ('sacral',    [0.0, -0.40], 'orange',  0),
    ('solar',     [0.0, -0.15], 'yellow',  0),
    ('heart',     [0.0,  0.00], 'green',   0),
    ('throat',    [0.0,  0.20], 'cyan',    0),
    ('third_eye', [0.0,  0.45], 'indigo',  0),
    ('crown',     [0.0,  0.65], 'violet',  0),
]

chakra_scatters = []
chakra_spirals = []
for name, pos, color, _ in chakras:
    pos = np.array(pos)
    scatter = ax.scatter(pos[0], pos[1], c=color, s=300, alpha=0.4, edgecolor='white', linewidth=2)
    chakra_scatters.append(scatter)
    # Initial dormant spiral
    spiral = golden_spiral_points(50, dim=2, radius_scale=0.05)
    spiral += pos
    line = ax.plot(spiral[:, 0], spiral[:, 1], c=color, lw=1, alpha=0.3)[0]
    chakra_spirals.append(line)

# Kundalini coil at root (dormant)
coil = golden_spiral_points(200, dim=2, radius_scale=0.15)
coil[:, 0] *= 0.5  # Tighter coil
coil += np.array([0.0, -0.60])
kundalini_coil = ax.plot(coil[:, 0], coil[:, 1], c='darkred', lw=4, alpha=0.8)[0]

# Rising energy trail
rising_trail = ax.plot([], [], c='white', lw=6, alpha=0.9)[0]
trail_points = []

# Aura expansion
aura = plt.Circle((0, 0), 1.1, color='white', fill=False, lw=5, alpha=0.2)
ax.add_patch(aura)

# Brain coherence
brain_glow = ax.scatter(0, 0.62, c='gold', s=400, alpha=0.3)

# Poincaré boundary
boundary = plt.Circle((0, 0), 1, color='gold', fill=False, ls='--', lw=6)
ax.add_patch(boundary)

ax.axis('equal')
ax.axis('off')
title = ax.set_title("Kundalini Rising — The Awakening\nDormant → Activation → Unity", color='white', fontsize=24, pad=100)

# Animation
FRAMES = 600
kundalini_y
