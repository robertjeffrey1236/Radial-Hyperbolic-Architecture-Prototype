# experiments/human_body_higgs_field.py
# Higgs Field Module — Mass from Yukawa Couplings
# Particles gain "mass" (drag) via interaction strength with pervasive field

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots(figsize=(12, 12))
ax.set_facecolor('black')
ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1.1, 1.1)
ax.axis('off')
ax.set_title("Higgs Field Module\nParticles Acquire Mass via Yukawa Coupling Strength", color='white', fontsize=18)

# Higgs field visualization — pervasive medium
x = np.linspace(-1, 1, 25)
y = np.linspace(-1, 1, 25)
X, Y = np.meshgrid(x, y)
ax.quiver(X, Y, np.zeros_like(X), np.zeros_like(Y), scale=30, color='cyan', alpha=0.4, linewidth=1)

# Poincaré disk boundary
circle = plt.Circle((0, 0), 1, color='gold', fill=False, ls='--', lw=4, alpha=0.7)
ax.add_patch(circle)

# Particles with different Yukawa couplings (mass)
particles = [
    {'pos': np.array([0.0, 0.0]), 'vel': np.array([0.06, 0.04]), 'coupling': 1.0, 'color': 'white', 'label': 'Heavy Particle\n(Strong Coupling)'},
    {'pos': np.array([-0.6, 0.6]), 'vel': np.array([0.12, -0.08]), 'coupling': 0.1, 'color': 'yellow', 'label': 'Light Particle\n(Weak Coupling)'},
    {'pos': np.array([0.6, -0.6]), 'vel': np.array([-0.1, 0.07]), 'coupling': 0.5, 'color': 'magenta', 'label': 'Medium Particle'},
]

scatters = []
trails = []
trail_data = []
for p in particles:
    s = ax.scatter(p['pos'][0], p['pos'][1], c=p['color'], s=150, edgecolor='white', linewidth=2, zorder=10)
    scatters.append(s)
    t = ax.plot([], [], c=p['color'], lw=3, alpha=0.7)[0]
    trails.append(t)
    trail_data.append([])

# Labels
for i, p in enumerate(particles):
    ax.text(p['pos'][0], p['pos'][1] + 0.2, p['label'], color='white', fontsize=10, ha='center', alpha=0.8)

def animate(frame):
    for i, p in enumerate(particles):
        # Higgs drag proportional to coupling strength
        drag = p['coupling'] * 0.03
        p['vel'] -= p['vel'] * drag
        
        # Update position
        p['pos'] += p['vel']
        
        # Boundary reflection (confined in disk)
        norm = np.linalg.norm(p['pos'])
        if norm > 0.95:
            p['pos'] = p['pos'] / norm * 0.95
            p['vel'] = p['vel'] - 2 * (p['vel'] @ (p['pos']/norm)) * (p['pos']/norm)
        
        scatters[i].set_offsets([p['pos']])
        
        # Trail
        trail_data[i].append(p['pos'].copy())
        if len(trail_data[i]) > 60:
            trail_data[i].pop(0)
        if trail_data[i]:
            tx, ty = zip(*trail_data[i])
            trails[i].set_data(tx, ty)

anim = FuncAnimation(fig, animate, frames=1000, interval=50, repeat=True)

plt.show()

print("⚛️ Higgs Field Module activated")
print("Watch particles move through the same field — mass emerges from interaction strength (Yukawa coupling)")
print("Heavy = strong drag • Light = weak drag")
print("Why these exact values? The deepest unsolved question...")
