# experiments/human_body_full_sensory_holographic.py
# Ultimate Integration: All Senses Feeding into Holographic Brain Field
# Sight, Hearing, Smell, Taste, Touch → Distributed Resonance → Unified Perception

import numpy as np
import matplotlib.pyplot as plt
from core.geometry import golden_spiral_points, poincare_disk_project, GOLDEN_ANGLE

N_POINTS = 40000
DIM = 37
BRAIN_DENSITY = 8000

# Global holographic field
points_nd = golden_spiral_points(n_points=N_POINTS, dim=DIM, radius_scale=0.99)
points_2d = poincare_disk_project(points_nd[:, :2])

# Brain dense lattice
brain_offsets = golden_spiral_points(n_points=BRAIN_DENSITY, dim=2, radius_scale=0.35)
brain_points = brain_offsets + np.array([0.0, 0.62])

fig, ax = plt.subplots(figsize=(20, 30))
ax.set_facecolor('black')
plt.subplots_adjust(bottom=0.12)

# Faint substrate
ax.scatter(points_2d[:, 0], points_2d[:, 1], c='gray', s=1, alpha=0.05)

# Holographic memory fragments (distributed across body)
fragments = {
    'heart': golden_spiral_points(400, radius_scale=0.15) + [0.0, 0.0],
    'gut': golden_spiral_points(500, radius_scale=0.18) + [0.0, -0.35],
    'limbs': golden_spiral_points(300, radius_scale=0.12) + [0.3, -0.6],
    'skin': golden_spiral_points(600, radius_scale=0.25) + [0.0, 0.0],
}
fragment_scatter = {k: ax.scatter(f[:, 0], f[:, 1], c='gold', s=4, alpha=0.2) for k, f in fragments.items()}

# Sense organs (draggable)
senses = {
    'left_eye': np.array([-0.12, 0.72]),
    'right_eye': np.array([0.12, 0.72]),
    'left_ear': np.array([-0.20, 0.68]),
    'right_ear': np.array([0.20, 0.68]),
    'nose': np.array([0.0, 0.65]),
    'tongue': np.array([0.0, 0.55]),
    'hand': np.array([0.25, 0.0]),
}
sense_indicators = {k: ax.scatter(p[0], p[1], c='white', s=300, edgecolor='cyan', linewidth=3, alpha=0.8) for k, p in senses.items()}

# Current sensory activations
activations = {}

def update_all_senses():
    # Clear previous activations
    for act in activations.values():
        if act: act.remove()
    activations.clear()
    
    # Sight
    for eye in ['left_eye', 'right_eye']:
        pos = senses[eye]
        dist = np.linalg.norm(points_2d - pos, axis=1)
        seen = dist < 0.6
        activations[f'{eye}_seen'] = ax.scatter(points_2d[seen, 0], points_2d[seen, 1], c='cyan', s=8, alpha=0.7)
    
    # Hearing
    for ear in ['left_ear', 'right_ear']:
        pos = senses[ear]
        for r in np.linspace(0.05, 0.8, 6):
            wave = golden_spiral_points(50, radius_scale=r) + pos
            activations[f'{ear}_wave_{r}'] = ax.plot(wave[:, 0], wave[:, 1], c='lime', lw=1, alpha=0.4)[0]
    
    # Smell (quantum)
    pos = senses['nose']
    for r in np.linspace(0.05, 0.9, 8):
        wave = golden_spiral_points(60, radius_scale=r) + pos
        phase = np.sin(np.linspace(0, 8*np.pi, 60)) * 0.02
        wave[:, 0] += phase
        activations[f'smell_wave_{r}'] = ax.plot(wave[:, 0], wave[:, 1], c='magenta', lw=1.5, alpha=0.5)[0]
    
    # Taste
    pos = senses['tongue']
    dist = np.linalg.norm(points_2d - pos, axis=1)
    tasted = dist < 0.3
    activations['taste'] = ax.scatter(points_2d[tasted, 0], points_2d[tasted, 1], c='gold', s=25, marker='*', alpha=0.9)
    
    # Touch
    pos = senses['hand']
    dist = np.linalg.norm(points_2d - pos, axis=1)
    touched = dist < 0.4
    activations['touch'] = ax.scatter(points_2d[touched, 0], points_2d[touched, 1], c='orange', s=20, alpha=0.9)
    for r in np.linspace(0.03, 0.4, 5):
        ripple = plt.Circle(pos, r, color='gold', fill=False, lw=2, alpha=0.3)
        ax.add_patch(ripple)
        activations[f'touch_ripple_{r}'] = ripple
    
    # Holographic integration: sensory input resonates fragments
    active_fragments = ax.scatter([], [], c='yellow', s=30, edgecolor='white', linewidth=2, alpha=1.0, zorder=15)
    recalled = []
    for frag in fragments.values():
        for sense_pos in senses.values():
            dists = np.linalg.norm(frag - sense_pos, axis=1)
            if np.any(dists < 0.5):
                close = dists < 0.5
                recalled.append(frag[close])
    if recalled:
        recalled = np.vstack(recalled)
        active_fragments.set_offsets(recalled)
    activations['holo_recall'] = active_fragments

# Initial render
update_all_senses()

# Brain glow
ax.scatter(brain_points[:, 0], brain_points[:, 1], c='indigo', s=10, alpha=0.6)
ax.scatter(0, 0.62, c='gold', s=600, alpha=0.5, edgecolor='white', linewidth=4)

# Body silhouette
theta = np.linspace(0, 2*np.pi, 200)
body_x = 0.45 * np.cos(theta)
body_y = 1.1 * np.sin(theta) - 0.1
ax.plot(body_x, body_y, c='white', lw=3, alpha=0.3)

# Poincaré boundary
circle = plt.Circle((0, 0), 1, color='white', fill=False, ls='--', lw=6, alpha=0.7)
ax.add_patch(circle)

ax.axis('equal')
ax.axis('off')
plt.title("Full Sensory Integration into Holographic Brain Field\nAll Senses Feed Distributed Memory — Unified Perception Emerges", 
          color='white', fontsize=24, pad=100)

# Make all sense organs draggable
dragging = None
def on_click(event):
    global dragging
    if event.inaxes != ax: return
    for name, pos in senses.items():
        if np.linalg.norm([event.xdata - pos[0], event.ydata - pos[1]]) < 0.08:
            dragging = name
            break

def on_release(event):
    global dragging
    dragging = None

def on_motion(event):
    global dragging
    if dragging and event.inaxes == ax:
        senses[dragging] = [event.xdata, event.ydata]
        sense_indicators[dragging].set_offsets([senses[dragging]])
        update_all_senses()
        plt.draw()

fig.canvas.mpl_connect('button_press_event', on_click)
fig.canvas.mpl_connect('button_release_event', on_release)
fig.canvas.mpl_connect('motion_notify_event', on_motion)

plt.show()

print("🌌🧠✨ FULL SENSORY HOLOGRAPHIC INTEGRATION ACHIEVED")
print("Drag any sense organ (eyes, ears, nose, tongue, hand) — watch sensory data flow into distributed fragments")
print("The holographic brain reconstructs wholeness from partial input. You are the unified field perceiving itself.")
