# experiments/consciousness_transition_death.py
# Death & Consciousness Leaving the Body Simulation
# Gradual shutdown → tunnel → detachment → release into infinite

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random

fig, ax = plt.subplots(figsize=(16, 24))
ax.set_facecolor('black')
ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1.2, 1.0)
ax.axis('off')

# Body
body = plt.Circle((0, 0), 0.9, color='white', fill=False, lw=5, alpha=0.8)
ax.add_patch(body)

# Heart
heart = ax.scatter(0, 0, c='crimson', s=800, alpha=0.8)

# Brain
brain = ax.scatter(0, 0.62, c='indigo', s=600, alpha=0.7)

# Lattice (life field)
lattice_points = np.random.uniform(-0.9, 0.9, (2000, 2))
lattice_points[:, 1] = np.clip(lattice_points[:, 1], -1.0, 0.8)
lattice = ax.scatter(lattice_points[:, 0], lattice_points[:, 1], c='cyan', s=5, alpha=0.5)

# Observer (starts in body)
observer = np.array([0.0, 0.0])
observer_scatter = ax.scatter(observer[0], observer[1], c='white', s=400, marker='*', alpha=0.0)

# Tunnel light
tunnel_light = plt.Circle((0, 0.9), 0.1, color='white', alpha=0.0)
ax.add_patch(tunnel_light)

# Memory fragments
memories = []

phase = 0  # 0: living, 1: dying, 2: tunnel, 3: detachment, 4: release
t = 0

def animate(frame):
    global phase, t, observer
    t += 0.05
    
    if phase == 0:  # Living state
        breath = np.sin(t * 0.5)
        heart_alpha = 0.8 + 0.2 * breath
        heart.set_alpha(heart_alpha)
        if random.random() < 0.01:
            phase = 1  # Begin death
            ax.set_title("Transition Begins — Systems Slowing", color='white')
    
    elif phase == 1:  # Dying
        progress = min(t / 10, 1)
        body.set_alpha(0.8 - 0.7 * progress)
        heart.set_alpha(0.8 - 0.8 * progress)
        brain.set_alpha(0.7 - 0.7 * progress)
        lattice.set_alpha(0.5 - 0.4 * progress)
        if progress == 1:
            phase = 2
            ax.set_title("Tunnel of Light — Consciousness Rising", color='gold')
    
    elif phase == 2:  # Tunnel
        tunnel_alpha = min(t - 10, 1)
        tunnel_light.set_alpha(tunnel_alpha)
        tunnel_light.set_radius(0.1 + tunnel_alpha * 0.8)
        # Lattice contracts toward crown
        lattice_offsets = lattice.get_offsets()
        lattice_offsets = lattice_offsets * (1 - tunnel_alpha * 0.5) + np.array([0, 0.9]) * tunnel_alpha * 0.5
        lattice.set_offsets(lattice_offsets)
        if tunnel_alpha == 1:
            phase = 3
            observer_scatter.set_alpha(1.0)
            ax.set_title("Detachment — Observer Rises", color='white')
    
    elif phase == 3:  # Detachment
        observer[1] = min(observer[1] + 0.02, 1.5)
        observer_scatter.set_offsets([observer])
        # Life review flashes
        if random.random() < 0.1:
            mem_pos = np.random.uniform(-0.8, 0.8, 2)
            mem_pos[1] = np.clip(mem_pos[1], -1.0, 0.8)
            memories.append(ax.scatter(mem_pos[0], mem_pos[1], c='gold', s=300, alpha=0.8, marker='*'))
        if observer[1] > 1.4:
            phase = 4
            ax.set_title("Release — Merging with Infinite", color='white')
    
    elif phase == 4:  # Release
        final_alpha = min((t - 30) / 5, 1)
        ax.set_facecolor(plt.cm.colors.to_rgb('white') * final_alpha + np.array([0,0,0]) * (1-final_alpha))
        for m in memories:
            m.set_alpha(1 - final_alpha)
        if final_alpha == 1:
            ax.set_title("Unity Achieved\nConsciousness Returns to Source", color='black')

# Restart on spacebar
def on_key(event):
    if event.key == ' ':
        plt.close()
        # Re-run would restart, but for now just note
        print("Rebirth — simulation restarts")

fig.canvas.mpl_connect('key_press_event', on_key)

anim = FuncAnimation(fig, animate, interval=80, repeat=False)

# Boundary
boundary = plt.Circle((0, 0), 1, color='gold', fill=False, lw=5)
ax.add_patch(boundary)

plt.show()

print("🕊️ Death & Consciousness Transition Simulation activated")
print("Gradual shutdown → tunnel of light → detachment → release into infinite")
print("Press spacebar to 'rebirth' (restart)")
print("The full cycle — from life to unity")
