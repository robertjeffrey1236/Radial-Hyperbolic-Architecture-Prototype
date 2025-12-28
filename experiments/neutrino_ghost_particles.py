# experiments/neutrino_ghost_particles.py
# Neutrino Oscillation & Ghost Particle Simulation
# Trillions pass through — barely interact — oscillate flavors

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random

fig, ax = plt.subplots(figsize=(16, 24))
ax.set_facecolor('black')
ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1.2, 1.0)
ax.axis('off')

# Human body faint
body = plt.Circle((0, 0), 0.9, color='white', fill=False, lw=3, alpha=0.2)
ax.add_patch(body)

# Title
title = ax.set_title("Neutrino Ghost Particles — Trillions Pass Through You Every Second", color='white', fontsize=20)

# Neutrino flavors/colors
flavors = {'electron': 'cyan', 'muon': 'magenta', 'tau': 'yellow'}

# Spawn many neutrinos streaming through
N_NEUTRINOS = 300
neutrinos = []
for _ in range(N_NEUTRINOS):
    start_x = random.uniform(-1.1, 1.1)
    start_y = -1.2
    vel_y = random.uniform(0.02, 0.05)
    flavor = random.choice(list(flavors.keys()))
    phase = random.uniform(0, 2*np.pi)
    neutrinos.append({'pos': np.array([start_x, start_y]), 
                      'vel': np.array([0, vel_y]), 
                      'flavor': flavor,
                      'phase': phase})

neutrino_scatters = ax.scatter([], [], s=30, alpha=0.8)

def animate(frame):
    t = frame * 0.05
    
    # Update positions
    positions = []
    colors = []
    for n in neutrinos:
        n['pos'] += n['vel']
        n['phase'] += 0.1  # Oscillation rate
        
        # Flavor oscillation (simple 2-flavor model for visual)
        prob_muon = np.sin(n['phase'])**2
        current_flavor = 'muon' if random.random() < prob_muon else 'electron'
        
        positions.append(n['pos'])
        colors.append(flavors[current_flavor])
        
        # Reset if passed through
        if n['pos'][1] > 1.0:
            n['pos'][1] = -1.2
            n['pos'][0] = random.uniform(-1.1, 1.1)
    
    neutrino_scatters.set_offsets(positions)
    neutrino_scatters.set_color(colors)
    
    # Rare interaction flash (1 in a million chance)
    if random.random() < 0.001:
        flash_x = random.uniform(-0.8, 0.8)
        flash_y = random.uniform(-0.8, 0.8)
        ax.scatter(flash_x, flash_y, c='white', s=300, marker='*', alpha=1.0)
        title.set_text("RARE INTERACTION! Neutrino Detected")
    else:
        title.set_text("Neutrino Ghost Particles — Oscillating Flavors\nTrillions Pass Unnoticed")

anim = FuncAnimation(fig, animate, interval=50, repeat=True)

# Boundary
boundary = plt.Circle((0, 0), 1, color='cyan', fill=False, lw=5)
ax.add_patch(boundary)

plt.show()

print("👻 Neutrino Ghost Particle Simulation activated")
print("300 neutrinos streaming through body — oscillating between flavors")
print("Rare interaction flash — like real detectors (IceCube, Super-K)")
print("CP violation hint: slight flavor bias possible in future versions")
