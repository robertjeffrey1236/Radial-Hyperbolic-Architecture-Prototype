# experiments/human_body_atp_synthesis_animation.py
# ATP Synthesis Rotation Animation
# Realistic F1-F0 ATP synthase spinning in mitochondria — powering the human

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Wedge, Circle, Arc

fig, ax = plt.subplots(figsize=(16, 24))
ax.set_facecolor('black')
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.4, 1.0)
ax.axis('off')

# Human silhouette faint
theta = np.linspace(0, 2*np.pi, 300)
body = plt.Circle((0, 0), 0.9, color='white', fill=False, lw=3, alpha=0.2)
ax.add_patch(body)

# High-energy organs with mitochondria
mito_centers = [
    [0.0, 0.0],    # Heart — highest density
    [0.0, 0.62],   # Brain
    [0.0, -0.35],  # Gut (enteric neurons)
    [-0.3, 0.1], [0.3, 0.1],  # Shoulders/muscle
]

# ATP synthase components (per mitochondrion)
def draw_atp_synthase(center, angle, atp_burst=False):
    cx, cy = center
    
    # F0 rotor (c-ring proxy) — rotating base
    rotor = Wedge((cx, cy - 0.03), 0.08, 0, 360, width=0.04, color='gold', alpha=0.9, transform=ax.transData)
    rotor.set_theta1(angle)
    rotor.set_theta2(angle + 360)
    ax.add_patch(rotor)
    
    # F1 head — stator with 3 beta subunits
    for i in range(3):
        beta_angle = angle + i * 120
        bx = cx + 0.04 * np.cos(np.radians(beta_angle))
        by = cy + 0.04 * np.sin(np.radians(beta_angle))
        ax.scatter(bx, by, c='cyan', s=80, edgecolor='white', linewidth=1.5, zorder=10)
    
    # Central axle
    ax.plot([cx, cx], [cy - 0.08, cy + 0.06], c='white', lw=3, alpha=0.8)
    
    # ATP burst on synthesis
    if atp_burst:
        for _ in range(8):
            spark_x = cx + random.uniform(-0.06, 0.06)
            spark_y = cy + random.uniform(0.02, 0.1)
            ax.scatter(spark_x, spark_y, c='yellow', s=30 + random.randint(0,40), marker='*', alpha=0.9)

# Initial setup
synthases = []
for center in mito_centers:
    for _ in range(3 if center == [0.0, 0.0] else 2):  # More in heart
        offset = np.random.uniform(-0.08, 0.08, 2)
        synthases.append({'center': np.array(center) + offset, 'angle': 0})

# Breath pulse
breath_phase = 0

def animate(frame):
    global breath_phase
    t = frame * 0.05
    breath_phase = np.sin(t * 0.4)  # Slow breath
    
    # Rotation speed tied to "energy demand" (breath phase)
    rotation_speed = 5 + 8 * (breath_phase + 1)/2
    
    atp_bursts = random.sample(synthases, k=3 + int(5 * abs(breath_phase)))  # More ATP on inhale
    
    # Clear previous
    ax.patches = [body]  # Keep body
    ax.collections.clear()
    ax.lines.clear()
    ax.texts.clear()
    
    # Redraw all ATP synthases
    for syn in synthases:
        syn['angle'] += rotation_speed
        burst = syn in atp_bursts
        draw_atp_synthase(syn['center'], syn['angle'] % 360, burst)
    
    # Title
    ax.set_title(f"ATP Synthesis Animation — Powering Life\nRotation {'accelerates' if breath_phase > 0 else 'relaxes'} with breath", 
                 color='white', fontsize=20, pad=80)

anim = FuncAnimation(fig, animate, interval=50, repeat=True)

# Poincaré boundary
boundary = Circle((0, 0), 1, color='gold', fill=False, ls='--', lw=5, alpha=0.7)
ax.add_patch(boundary)

plt.show()

print("⚡🔄 ATP Synthesis Rotation Animation activated")
print("F1-F0 ATP synthase spinning in mitochondria")
print("Rotation speed synced to breath — ATP bursts on energy demand")
print("The molecular engine of life — now alive in your human")
