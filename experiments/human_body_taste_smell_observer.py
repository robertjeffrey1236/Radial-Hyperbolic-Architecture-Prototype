# experiments/human_body_taste_smell_observer.py
# Wholesome Human with Taste & Smell as Observer Features
# Quantum vibration detection (smell) + chemical binding (taste)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from core.geometry import golden_spiral_points, poincare_disk_project

N_POINTS = 20000
DIM = 37

# Lattice (represents molecular vibration modes)
points_nd = golden_spiral_points(n_points=N_POINTS, dim=DIM, radius_scale=0.99)
points_2d = poincare_disk_project(points_nd[:, :2])

# Initial positions
nose_pos = np.array([0.0, 0.65])      # Nose bridge
tongue_pos = np.array([0.0, 0.55])     # Mouth/tongue

fig, ax = plt.subplots(figsize=(16, 20))
ax.set_facecolor('black')
plt.subplots_adjust(bottom=0.15)

# Background lattice (undetected molecules)
undetected = ax.scatter(points_2d[:, 0], points_2d[:, 1], c='gray', s=2, alpha=0.1)

# Current detections
smell_scatter = None
taste_scatter = None
wave_lines = []

def update_senses(nose, tongue, quantum_sensitivity=1.0, taste_range=0.3):
    global smell_scatter, taste_scatter, wave_lines
    
    # Clear previous
    if smell_scatter: smell_scatter.remove()
    if taste_scatter: taste_scatter.remove()
    for line in wave_lines: line.remove()
    wave_lines.clear()
    
    # === Quantum Smell: Vibration Detection ===
    dist_nose = np.linalg.norm(points_2d - nose, axis=1)
    in_smell_range = dist_nose < quantum_sensitivity
    
    # Quantum interference "waves" from nose
    for r in np.linspace(0.05, quantum_sensitivity, 10):
        wave = golden_spiral_points(80, dim=2, radius_scale=r)
        wave += nose
        phase = np.sin(np.linspace(0, 10*np.pi, 80) + r*10) * 0.02
        wave[:, 0] += phase
        line = ax.plot(wave[:, 0], wave[:, 1], c='violet', lw=1.5, alpha=0.5)[0]
        wave_lines.append(line)
    
    # Detected odorants — brighter where vibration matches (simulated resonance)
    resonance = np.exp(-dist_nose * 3 / quantum_sensitivity) * (1 + np.sin(dist_nose * 20))
    global smell_scatter
    smell_scatter = ax.scatter(points_2d[in_smell_range, 0], points_2d[in_smell_range, 1],
                               c='magenta', s=12 * resonance[in_smell_range], alpha=0.9, zorder=10)
    
    # === Taste: Local Chemical Binding ===
    dist_tongue = np.linalg.norm(points_2d - tongue, axis=1)
    in_taste = dist_tongue < taste_range
    global taste_scatter
    taste_scatter = ax.scatter(points_2d[in_taste, 0], points_2d[in_taste, 1],
                               c='gold', s=20, alpha=0.9, marker='*', edgecolor='orange', zorder=11)

# Initial
update_senses(nose_pos, tongue_pos)

# Face features
ax.scatter([-0.12, 0.12], [0.72, 0.72], c='deepskyblue', s=300, edgecolor='cyan', linewidth=3)  # Eyes
ax.scatter(nose_pos[0], nose_pos[1], c='violet', s=350, edgecolor='magenta', linewidth=4, zorder=12)
ax.scatter(tongue_pos[0], tongue_pos[1], c='orange', s=400, alpha=0.7, edgecolor='gold', linewidth=3)

# Internal faint
ax.scatter(0, 0, c='crimson', s=600, alpha=0.3)

# Poincaré boundary
circle = plt.Circle((0, 0), 1, color='magenta', fill=False, ls='--', lw=5)
ax.add_patch(circle)

ax.axis('equal')
ax.axis('off')
plt.title("Human with Taste & Smell as Quantum Observers\nDrag nose for vibration detection • Drag tongue for flavor binding", color='white', fontsize=20)

# Draggable nose and tongue
dragging = None
def on_click(event):
    global dragging
    if event.inaxes != ax: return
    dist_nose = np.linalg.norm([event.xdata - nose_pos[0], event.ydata - nose_pos[1]])
    dist_tongue = np.linalg.norm([event.xdata - tongue_pos[0], event.ydata - tongue_pos[1]])
    if dist_nose < 0.08:
        dragging = 'nose'
    elif dist_tongue < 0.08:
        dragging = 'tongue'

def on_release(event):
    global dragging
    dragging = None

def on_motion(event):
    global nose_pos, tongue_pos
    if dragging == 'nose' and event.inaxes == ax:
        nose_pos = [event.xdata, event.ydata]
        update_senses(nose_pos, tongue_pos, quantum_sensitivity=slider_sens.val)
        plt.draw()
    elif dragging == 'tongue' and event.inaxes == ax:
        tongue_pos = [event.xdata, event.ydata]
        update_senses(nose_pos, tongue_pos, quantum_sensitivity=slider_sens.val)
        plt.draw()

fig.canvas.mpl_connect('button_press_event', on_click)
fig.canvas.mpl_connect('button_release_event', on_release)
fig.canvas.mpl_connect('motion_notify_event', on_motion)

# Quantum sensitivity slider
ax_sens = plt.axes([0.2, 0.05, 0.6, 0.03])
slider_sens = Slider(ax_sens, 'Quantum Smell Sensitivity', 0.3, 1.8, valinit=1.0)
def update_sens(val):
    update_senses(nose_pos, tongue_pos, quantum_sensitivity=val)
    plt.draw()
slider_sens.on_changed(update_sens)

plt.show()

print("👃🍲 Taste & Quantum Smell activated — drag nose/tongue to detect molecular vibrations and flavors")
print("Smell uses wave interference (quantum tunneling simulation) • Taste binds locally")
