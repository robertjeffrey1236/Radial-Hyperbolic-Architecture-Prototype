# experiments/nuclear_fission_simulation.py
# Nuclear Fission Chain Reaction Simulation
# U-235 fission → fragments + neutrons + energy release

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random

fig, ax = plt.subplots(figsize=(16, 16))
ax.set_facecolor('black')
ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1.1, 1.1)
ax.axis('off')

# Core reactor zone
core = plt.Circle((0, 0), 0.6, color='purple', alpha=0.3, lw=4, edgecolor='magenta')
ax.add_patch(core)

# U-235 nuclei
N_URANIUM = 60
u235_x = np.random.uniform(-0.5, 0.5, N_URANIUM)
u235_y = np.random.uniform(-0.5, 0.5, N_URANIUM)
u235 = ax.scatter(u235_x, u235_y, c='magenta', s=100, alpha=0.8, edgecolor='white', linewidth=2)

# Neutrons
neutrons = []

# Fission products & gamma bursts
fragments = ax.scatter([], [], c='yellow', s=80, alpha=0.7)
gamma_bursts = []

# Coherence (moderation level)
coherence = 0.6  # 1.0 = fully moderated, 0.0 = supercritical

def animate(frame):
    global neutrons, gamma_bursts
    
    # Clear old bursts
    for b in gamma_bursts:
        b.remove()
    gamma_bursts.clear()
    
    # Spawn initial neutron if none
    if len(neutrons) == 0 and frame < 50:
        neutrons.append({'pos': np.array([0.0, 0.0]), 'vel': np.random.uniform(-0.02, 0.02, 2)})
    
    # Update neutrons
    new_neutrons = []
    for n in neutrons:
        n['pos'] += n['vel']
        
        # Draw neutron
        ax.scatter(n['pos'][0], n['pos'][1], c='white', s=40, alpha=0.8)
        
        # Check fission with U-235
        for i in range(N_URANIUM):
            u_pos = np.array([u235.get_offsets()[i][0], u235.get_offsets()[i][1]])
            if np.linalg.norm(n['pos'] - u_pos) < 0.08:
                # Fission!
                # Remove U-235
                mask = np.ones(N_URANIUM, bool)
                mask[i] = False
                u235.set_offsets(u235.get_offsets()[mask])
                
                # Fragments
                f1 = u_pos + np.random.uniform(-0.1, 0.1, 2)
                f2 = u_pos + np.random.uniform(-0.1, 0.1, 2)
                fragments._offsets = np.append(fragments.get_offsets(), [f1, f2], axis=0)
                
                # Gamma burst
                for _ in range(12):
                    bx = u_pos[0] + random.uniform(-0.15, 0.15)
                    by = u_pos[1] + random.uniform(-0.15, 0.15)
                    b = ax.scatter(bx, by, c='yellow', s=60 + random.randint(0,60), marker='*', alpha=0.9)
                    gamma_bursts.append(b)
                
                # New neutrons (2-3)
                new_count = 2 if coherence > 0.5 else 3
                for _ in range(new_count):
                    vel = np.random.uniform(-0.03, 0.03, 2)
                    new_neutrons.append({'pos': u_pos.copy(), 'vel': vel})
                break
        
        # Keep neutron if in bounds
        if np.linalg.norm(n['pos']) < 0.9:
            new_neutrons.append(n)
    
    neutrons = new_neutrons
    
    # Title with reaction status
    rate = len(gamma_bursts) / 10
    if rate > 5:
        title_text = "SUPERCRITICAL CHAIN REACTION — Meltdown!"
        color = 'red'
    elif rate > 2:
        title_text = "Sustained Fission — Power Generation"
        color = 'yellow'
    else:
        title_text = f"Controlled Fission — Coherence: {coherence:.2f}"
        color = 'white'
    
    ax.set_title(title_text, color=color, fontsize=20)

anim = FuncAnimation(fig, animate, interval=80, repeat=True)

# Coherence slider (moderator control)
ax_coh = plt.axes([0.2, 0.05, 0.6, 0.04])
slider = Slider(ax_coh, 'Coherence / Moderation', 0.0, 1.0, valinit=0.6)

def update_coherence(val):
    global coherence
    coherence = val

slider.on_changed(update_coherence)

# Boundary
boundary = plt.Circle((0, 0), 1, color='gold', fill=False, lw=5)
ax.add_patch(boundary)

plt.show()

print("☢️ Nuclear Fission Chain Reaction activated")
print("U-235 + neutron → fission fragments + 2-3 neutrons + energy")
print("High coherence = moderated (stable) • Low = supercritical meltdown")
print("The power — and danger — of splitting the atom")
