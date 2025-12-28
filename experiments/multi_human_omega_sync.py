# experiments/multi_human_omega_sync.py
# 12-Human Omega Coherence — The Unity Threshold
# When 12 humans sync → global field becomes ONE

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random

NUM_HUMANS = 12
COHERENCE_THRESHOLD = 0.85

class Human:
    def __init__(self, id):
        self.id = id
        self.pos = np.random.uniform(-0.7, 0.7, 2)
        self.coherence = random.uniform(0.3, 0.8)
        self.breath_phase = random.uniform(0, 2*np.pi)
        self.color = plt.cm.plasma(self.coherence)
        
    def update(self, t):
        # Breath drives coherence fluctuation
        self.breath_phase += 0.05
        base = 0.5 + 0.4 * np.sin(self.breath_phase)
        noise = 0.1 * np.sin(t * 3 + self.id)
        self.coherence = np.clip(base + noise, 0.1, 1.0)
        self.color = plt.cm.plasma(self.coherence)

humans = [Human(i) for i in range(NUM_HUMANS)]

fig, ax = plt.subplots(figsize=(16, 16))
ax.set_facecolor('black')
ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1.1, 1.1)
ax.axis('off')

# Individual human scatters
scatters = [ax.scatter(h.pos[0], h.pos[1], c=h.color, s=400, alpha=0.8, edgecolor='white', linewidth=2) for h in humans]

# Binding lines (appear in high sync)
binding_lines = []

# Global unity glow
unity_glow = plt.Circle((0, 0), 0.1, color='white', alpha=0)
ax.add_patch(unity_glow)

# Poincaré boundary
boundary = plt.Circle((0, 0), 1, color='gold', fill=False, lw=5, alpha=0.7)
ax.add_patch(boundary)

def animate(frame):
    t = frame * 0.05
    
    # Update humans
    for h, s in zip(humans, scatters):
        h.update(t)
        s.set_offsets([h.pos])
        s.set_color(h.color)
        s.set_alpha(0.6 + 0.4 * h.coherence)
    
    # Check for 12-human sync
    avg_coherence = np.mean([h.coherence for h in humans])
    all_high = all(h.coherence >= COHERENCE_THRESHOLD for h in humans)
    
    # Clear old lines
    for line in binding_lines:
        line.remove()
    binding_lines.clear()
    
    if all_high:
        # OMEGA SYNC — golden binding
        for i, h1 in enumerate(humans):
            for h2 in humans[i+1:]:
                line = ax.plot([h1.pos[0], h2.pos[0]], [h1.pos[1], h2.pos[1]], 
                               c='gold', lw=2, alpha=0.8)[0]
                binding_lines.append(line)
        
        # Expand unity glow
        unity_glow.set_radius(1.5 + 0.5 * np.sin(t * 5))
        unity_glow.set_alpha(0.8)
        
        ax.set_title("OMEGA SYNC ACHIEVED\nThe 12 Have Become the ONE\n100% Coherence — Love as Phase-Locked Resonance", 
                     color='gold', fontsize=22)
    else:
        unity_glow.set_radius(0.1)
        unity_glow.set_alpha(0.2)
        
        ax.set_title(f"Multi-Human Coherence Field\nAverage Coherence: {avg_coherence:.2f} — Building Toward Unity", 
                     color='white', fontsize=18)
    
    return scatters + binding_lines

anim = FuncAnimation(fig, animate, interval=80, repeat=True)

plt.show()

print("🌌👥 12-Human Omega Sync activated")
print("When all 12 reach high coherence → golden binding light connects everything")
print("The 12 become the ONE — direct simulation of collective unity")
print("Love as the stabilizer — exactly as theorized")
