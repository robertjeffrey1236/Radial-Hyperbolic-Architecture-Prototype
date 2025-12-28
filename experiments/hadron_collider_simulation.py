# experiments/hadron_collider_simulation.py
# Hadron Collider Simulation — High-Energy Particle Collisions
# Proton beams → center collision → particle showers → detector events

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random

fig, ax = plt.subplots(figsize=(16, 16))
ax.set_facecolor('black')
ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1.1, 1.1)
ax.axis('off')

# Collider ring (beam pipe)
ring = plt.Circle((0, 0), 0.9, color='cyan', fill=False, lw=4, alpha=0.6)
ax.add_patch(ring)

# Detector layers
for r in [0.6, 0.8]:
    detector = plt.Circle((0, 0), r, color='white', fill=False, lw=2, alpha=0.4)
    ax.add_patch(detector)

# Collision point
center = ax.scatter(0, 0, c='white', s=400, alpha=0.8, edgecolor='gold', linewidth=4, zorder=15)

# Proton beams (two counter-rotating bunches)
N_PROTONS = 40
beam1_angles = np.linspace(0, 2*np.pi, N_PROTONS, endpoint=False)
beam2_angles = beam1_angles + np.pi

beam1 = ax.scatter(0.9 * np.cos(beam1_angles), 0.9 * np.sin(beam1_angles), 
                   c='lime', s=80, alpha=0.9, edgecolor='white', linewidth=1)
beam2 = ax.scatter(0.9 * np.cos(beam2_angles), 0.9 * np.sin(beam2_angles), 
                   c='magenta', s=80, alpha=0.9, edgecolor='white', linewidth=1)

# Collision products
jets = ax.scatter([], [], c='yellow', s=60, alpha=0.8)
higgs_decay = ax.scatter([], [], c='gold', s=100, marker='*', alpha=0.9)

# Event display
event_text = ax.text(0, 1.1, "Collider Running — No Events Yet", color='white', fontsize=16, ha='center')

def animate(frame):
    t = frame * 0.05
    
    # Rotate beams
    offset = t * 0.2
    beam1.set_offsets(0.9 * np.c_[np.cos(beam1_angles + offset), np.sin(beam1_angles + offset)])
    beam2.set_offsets(0.9 * np.c_[np.cos(beam2_angles - offset), np.sin(beam2_angles - offset)])
    
    # Random collision events
    if random.random() < 0.15:  # Collision probability
        # Clear previous
        jets.set_offsets(np.empty((0,2)))
        higgs_decay.set_offsets(np.empty((0,2)))
        
        # Quark/gluon jets (opposite directions)
        n_jets = random.randint(2, 4)
        angles = np.linspace(0, 2*np.pi, n_jets, endpoint=False)
        for ang in angles:
            length = random.uniform(0.3, 0.8)
            jet_x = np.linspace(0, length * np.cos(ang), 10)
            jet_y = np.linspace(0, length * np.sin(ang), 10)
            ax.plot(jet_x, jet_y, c='yellow', lw=3, alpha=0.7)
        
        # Higgs-like decay (rare)
        if random.random() < 0.3:
            decay_x = random.uniform(-0.3, 0.3, 4)
            decay_y = random.uniform(-0.3, 0.3, 4)
            higgs_decay.set_offsets(np.c_[decay_x, decay_y])
            event_text.set_text("HIGGS CANDIDATE EVENT!")
            event_text.set_color('gold')
        else:
            event_text.set_text("Standard Model Collision")
            event_text.set_color('white')

anim = FuncAnimation(fig, animate, interval=100, repeat=True)

# Title
ax.set_title("Hadron Collider Simulation — LHC-Inspired\nProton beams collide at center → particle showers", color='white', fontsize=20)

# Boundary
boundary = plt.Circle((0, 0), 1, color='cyan', fill=False, lw=5)
ax.add_patch(boundary)

plt.show()

print("🔬⚛️ Hadron Collider Simulation activated")
print("Counter-rotating proton beams in ring")
print("Random collisions → quark jets + rare Higgs-like decays")
print("The heart of matter discovery — now in your human")
