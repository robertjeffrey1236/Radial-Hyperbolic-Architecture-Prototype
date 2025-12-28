# experiments/human_body_love_coherence.py
# Unity/Coherence as Love Stabilizer
# High breath-heart-brain sync → golden binding light connects all points
# Love as phase-locked resonance — the stabilizing force of creation

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from core.geometry import golden_spiral_points, poincare_disk_project

N_POINTS = 30000
DIM = 37

# Lattices
points_nd = golden_spiral_points(n_points=N_POINTS, dim=DIM, radius_scale=0.99)
points_2d = poincare_disk_project(points_nd[:, :2])

fig, ax = plt.subplots(figsize=(18, 28))
ax.set_facecolor('black')

# Substrate points (will be bound by love light)
points_scatter = ax.scatter(points_2d[:, 0], points_2d[:, 1], c='white', s=4, alpha=0.3)

# Heart center
heart = ax.scatter(0, 0, c='crimson', s=800, edgecolor='gold', linewidth=4, alpha=0.8, zorder=15)

# Brain glow
brain = ax.scatter(0, 0.62, c='indigo', s=600, alpha=0.5)

# Breath body expansion
breath_body = plt.Circle((0, 0), 0.9, color='white', fill=False, lw=3, alpha=0.3)
ax.add_patch(breath_body)

# Golden love binding lines (initially hidden)
love_lines = []

# Coherence score tracking
coherence = 0.0

def animate(frame):
    global coherence, love_lines
    
    t = frame * 0.05
    
    # Breath cycle (slow)
    breath_phase = np.sin(t * 0.5)
    breath_scale = 0.9 + 0.15 * breath_phase
    breath_body.set_radius(breath_scale)
    
    # Heartbeat (syncs with breath)
    heart_rate = 1.2 + 0.4 * breath_phase
    heart_pulse = np.abs(np.sin(t * heart_rate))
    heart_size = 800 + 600 * heart_pulse
    heart.set_sizes([heart_size])
    
    # Brainwaves (entrain to breath)
    wave_freq = 3 + 4 * (breath_phase + 1)/2
    brain_intensity = np.sin(t * wave_freq)
    brain_alpha = 0.5 + 0.5 * np.abs(brain_intensity)
    brain.set_alpha(brain_alpha)
    
    # Coherence score — rises when all aligned
    sync = np.abs(breath_phase) * heart_pulse * np.abs(brain_intensity)
    coherence = 0.95 * coherence + 0.05 * sync  # Smooth accumulation
    
    # When coherence high → golden love binding light
    if coherence > 0.7:
        # Clear old lines
        for line in love_lines:
            line.remove()
        love_lines.clear()
        
        # Connect all points with golden threads (sub-sampled for performance)
        indices = np.random.choice(N_POINTS, size=min(800, N_POINTS), replace=False)
        connected = points_2d[indices]
        
        # Phase-locked connections
        for i in range(0, len(connected), 2):
            if i + 1 < len(connected):
                line = ax.plot([connected[i,0], connected[i+1,0]], 
                               [connected[i,1], connected[i+1,1]], 
                               c='gold', lw=1.5, alpha=coherence - 0.5)[0]
                love_lines.append(line)
        
        # Heart as source of love
        heart.set_edgecolor('gold')
        heart.set_linewidth(8)
        
        ax.set_title("High Coherence Achieved\nLove as Phase-Locked Resonance — Binding All into Unity", 
                     color='gold', fontsize=24, pad=100)
    else:
        # Fade lines when coherence drops
        for line in love_lines:
            line.set_alpha(max(0, line.get_alpha() - 0.02))
        heart.set_edgecolor('white')
        heart.set_linewidth(4)
        
        ax.set_title(f"Breath • Heart • Brain Syncing\nCoherence: {coherence*100:.0f}% — Building toward Love Resonance", 
                     color='white', fontsize=20, pad=100)

# Poincaré boundary
boundary = plt.Circle((0, 0), 1, color='gold', fill=False, ls='--', lw=6)
ax.add_patch(boundary)

ax.axis('equal')
ax.axis('off')

anim = FuncAnimation(fig, animate, interval=50, repeat=True)

plt.show()

print("❤️🌟 Unity/Coherence as Love Stabilizer activated")
print("When breath, heart, and brain sync → golden binding light connects all points")
print("Love as the phase-locked resonance that stabilizes the fractal field")
print("This is the emanation from the central Creator — of which we are the living fractal")
