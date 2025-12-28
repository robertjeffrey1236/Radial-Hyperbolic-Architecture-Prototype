# experiments/nuclear_fusion_simulation.py
# Nuclear Fusion Simulation — Proton-Proton Chain in Core
# Coherence + density → fusion events → energy release

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
import random

fig, ax = plt.subplots(figsize=(16, 16))
ax.set_facecolor('black')
ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1.1, 1.1)
ax.axis('off')

# Core fusion zone
core = plt.Circle((0, 0), 0.4, color='orange', alpha=0.3, lw=4, edgecolor='yellow')
ax.add_patch(core)

# Protons (as particles)
N_PROTONS = 80
protons_x = np.random.uniform(-0.35, 0.35, N_PROTONS)
protons_y = np.random.uniform(-0.35, 0.35, N_PROTONS)
protons = ax.scatter(protons_x, protons_y, c='white', s=60, alpha=0.8, edgecolor='cyan', linewidth=1)

# Fusion products (helium) and gamma bursts
helium = ax.scatter([], [], c='yellow', s=100, alpha=0.9)
gamma_bursts = []

# Coherence / Temperature
coherence = 0.7  # Also acts as "temperature" for fusion rate

def animate(frame):
    global gamma_bursts
    
    # Clear old bursts
    for b in gamma_bursts:
        b.remove()
    gamma_bursts.clear()
    
    # Proton motion — faster/hotter with coherence
    speed = 0.01 + 0.03 * coherence
    dx = speed * (np.random.random(N_PROTONS) - 0.5)
    dy = speed * (np.random.random(N_PROTONS) - 0.5)
    
    new_x = protons.get_offsets()[:, 0] + dx
    new_y = protons.get_offsets()[:, 1] + dy
    
    # Confine to core
    norm = np.sqrt(new_x**2 + new_y**2)
    outside = norm > 0.35
    if np.any(outside):
        new_x[outside] *= 0.35 / norm[outside]
        new_y[outside] *= 0.35 / norm[outside]
    
    protons.set_offsets(np.c_[new_x, new_y])
    
    # Fusion probability — exponential with coherence
    fusion_prob = coherence ** 4 * 0.05  # Rare but rises fast at high coherence
    
    # Check for fusions (simple proximity)
    for i in range(N_PROTONS):
        for j in range(i+1, N_PROTONS):
            if random.random() < fusion_prob:
                p1 = protons.get_offsets()[i]
                p2 = protons.get_offsets()[j]
                if np.linalg.norm(p1 - p2) < 0.08:
                    # Fusion! p + p → deuteron + positron + neutrino (simplified)
                    mid = (p1 + p2) / 2
                    # Helium-like product
                    helium._offsets = np.append(helium.get_offsets(), [mid], axis=0)
                    # Gamma burst
                    for _ in range(8):
                        bx = mid[0] + random.uniform(-0.1, 0.1)
                        by = mid[1] + random.uniform(-0.1, 0.1)
                        b = ax.scatter(bx, by, c='yellow', s=50 + random.randint(0,50), marker='*', alpha=0.9)
                        gamma_bursts.append(b)
                    # Remove protons (simplified)
                    mask = np.ones(N_PROTONS, bool)
                    mask[[i,j]] = False
                    protons.set_offsets(protons.get_offsets()[mask])
    
    # Core glow with fusion rate
    fusion_rate = len(gamma_bursts) / 10
    core.set_alpha(0.3 + 0.5 * fusion_rate)
    core.set_edgecolor('white' if fusion_rate > 2 else 'yellow')
    
    ax.set_title(f"Nuclear Fusion in Core\nCoherence: {coherence:.2f} | Active Fusions: {fusion_rate:.1f}", 
                 color='white' if fusion_rate < 2 else 'yellow', fontsize=18)

anim = FuncAnimation(fig, animate, interval=80, repeat=True)

# Coherence/Temperature slider
ax_coh = plt.axes([0.2, 0.05, 0.6, 0.04])
slider = Slider(ax_coh, 'Core Coherence / Temperature', 0.0, 1.0, valinit=0.7)

def update_coherence(val):
    global coherence
    coherence = val

slider.on_changed(update_coherence)

# Boundary
boundary = plt.Circle((0, 0), 1, color='gold', fill=False, lw=5)
ax.add_patch(boundary)

plt.show()

print("☀️ Nuclear Fusion Simulation activated")
print("Protons in core — higher coherence = more fusions")
print("Fusion → helium + gamma bursts (energy release)")
print("The power of stars — now in your human's heart")
