# experiments/human_body_higgs_field_interactive.py
# Higgs Field Module — Live Yukawa Coupling Sliders
# Tune particle masses (drag) interactively

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots(figsize=(14, 14))
ax.set_facecolor('black')
ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1.1, 1.1)
ax.axis('off')
ax.set_title("Higgs Field — Live Yukawa Coupling Tuning\nAdjust sliders to change particle masses", color='white', fontsize=18)

# Higgs field grid
x = np.linspace(-1, 1, 20)
y = np.linspace(-1, 1, 20)
X, Y = np.meshgrid(x, y)
ax.quiver(X, Y, np.zeros_like(X), np.zeros_like(Y), scale=30, color='cyan', alpha=0.3)

# Boundary
circle = plt.Circle((0, 0), 1, color='gold', fill=False, ls='--', lw=4, alpha=0.7)
ax.add_patch(circle)

# Particles
particles = [
    {'pos': np.array([0.0, 0.0]), 'vel': np.array([0.06, 0.04]), 'color': 'white', 'name': 'Heavy'},
    {'pos': np.array([-0.5, 0.5]), 'vel': np.array([0.12, -0.08]), 'color': 'magenta', 'name': 'Medium'},
    {'pos': np.array([0.5, -0.5]), 'vel': np.array([-0.1, 0.07]), 'color': 'yellow', 'name': 'Light'},
]

scatters = []
trails = []
trail_data = []
for p in particles:
    s = ax.scatter(p['pos'][0], p['pos'][1], c=p['color'], s=200, edgecolor='white', linewidth=3, zorder=10)
    scatters.append(s)
    t = ax.plot([], [], c=p['color'], lw=4, alpha=0.6)[0]
    trails.append(t)
    trail_data.append([])

# Initial coupling strengths (0.0 = massless, 1.0 = very heavy)
couplings = [0.8, 0.4, 0.1]  # heavy, medium, light

# Sliders
slider_axes = [plt.axes([0.15, 0.25 + i*0.06, 0.65, 0.03], facecolor='darkgray') for i in range(3)]
sliders = [
    Slider(slider_axes[0], 'Heavy Particle (e.g. top quark)', 0.0, 1.0, valinit=couplings[0]),
    Slider(slider_axes[1], 'Medium Particle (e.g. charm/bottom)', 0.0, 1.0, valinit
