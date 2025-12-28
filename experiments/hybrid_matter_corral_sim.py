# experiments/hybrid_matter_corral_sim.py
# Hybrid Matter Corral Simulation
# Supercooled liquid trapped in atomic ring — solid/liquid duality

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import CheckButtons

fig, ax = plt.subplots(figsize=(12, 12))
ax.set_facecolor('black')
ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1.1, 1.1)
ax.axis('off')

# Poincaré boundary
boundary = plt.Circle((0, 0), 1, color='gold', fill=False, lw=4, alpha=0.7)
ax.add_patch(boundary)

# Atomic corral ring (pinned defects)
ring_theta = np.linspace(0, 2*np.pi, 30)
ring_x = 0.6 * np.cos(ring_theta)
ring_y = 0.6 * np.sin(ring_theta)
corral = ax.scatter(ring_x, ring_y, c='white', s=200, alpha=0.9, edgecolor='cyan', linewidth=2)

# Trapped liquid particles (supercooled, fluid)
N_TRAPPED = 200
trapped_x = np.random.uniform(-0.5, 0.5, N_TRAPPED)
trapped_y = np.random.uniform(-0.5, 0.5, N_TRAPPED)
trapped = ax.scatter(trapped_x, trapped_y, c='deepskyblue', s=50, alpha=0.8)

# Measurement toggle (observer "looking" = which-path info)
measurement_on = False

def animate(frame):
    t = frame * 0.05
    
    if measurement_on:
        # Collapse to glass-like solid (unstable structure)
        grid_x, grid_y = np.meshgrid(np.linspace(-0.4, 0.4, 15), np.linspace(-0.4, 0.4, 15))
        trapped.set_offsets(np.c_[grid_x.ravel() + 0.05*np.random.randn(225), grid_y.ravel() + 0.05*np.random.randn(225)])
        trapped.set_color('lightgray')
        trapped.set_alpha(0.6)
        ax.set_title("Measurement ON — Collapses to Unstable Glass-Like Solid", color='yellow')
    else:
        # Fluid supercooled liquid — free motion inside corral
        trapped_x = trapped.get_offsets()[:, 0] + 0.02 * np.sin(t + np.random.randn(N_TRAPPED))
        trapped_y = trapped.get_offsets()[:, 1] + 0.02 * np.cos(t + np.random.randn(N_TRAPPED))
        # Bounce inside ring
        norm = np.sqrt(trapped_x**2 + trapped_y**2)
        outside = norm > 0.55
        trapped_x[outside] *= 0.55 / norm[outside]
        trapped_y[outside] *= 0.55 / norm[outside]
        trapped.set_offsets(np.c_[trapped_x, trapped_y])
        trapped.set_color('deepskyblue')
        trapped.set_alpha(0.8 + 0.2 * np.sin(t))
        ax.set_title("Eraser ON — Stable Supercooled Liquid (Solid + Liquid Duality)", color='cyan')

anim = FuncAnimation(fig, animate, interval=80, repeat=True)

# Toggle measurement (which-path info)
rax = plt.axes([0.05, 0.5, 0.2, 0.1], facecolor='black')
check = CheckButtons(rax, ['Measurement ON (Collapse)'], [False])

def toggle_measurement(label):
    global measurement_on
    measurement_on = not measurement_on
    plt.draw()

check.on_clicked(toggle_measurement)

ax.set_title("Hybrid State of Matter Corral\nToggle Measurement — Watch Duality Emerge", color='white')

plt.show()

print("🧪 Hybrid Matter Corral Simulation activated")
print("Supercooled liquid trapped in atomic ring")
print("Toggle 'Measurement ON' — collapses to unstable glass")
print("Eraser (no measurement) → stable solid-liquid duality")
print("Direct sim of the new discovered state")
