# experiments/human_body_ark_mode.py
# Ark Mode — Subtle Vibration Generates Massive Power
# Ark of the Covenant as Brain • Coherent vibration = divine capacitor activation

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Arc, Rectangle

fig, ax = plt.subplots(figsize=(16, 24))
ax.set_facecolor('black')
ax.set_xlim(-1.0, 1.0)
ax.set_ylim(-1.2, 1.0)
ax.axis('off')

# Human silhouette faint
body = plt.Circle((0, 0), 0.9, color='white', fill=False, lw=3, alpha=0.2)
ax.add_patch(body)

# Brain/skull region
brain_center = [0.0, 0.62]
ax.scatter(brain_center[0], brain_center[1], c='indigo', s=800, alpha=0.4)

# === Ark Overlay on Skull ===
# Golden box (cranium)
ark_box = Rectangle((-0.25, 0.55), 0.5, 0.25, color='gold', alpha=0.6, lw=4, edgecolor='white')
ax.add_patch(ark_box)

# Mercy seat (sella turcica / pituitary cradle)
mercy = Arc((0, 0.68), 0.3, 0.15, theta1=0, theta2=180, color='white', lw=5, alpha=0.8)
ax.add_patch(mercy)

# Cherubim wings (cerebral hemispheres)
for side in [-1, 1]:
    wing = plt.Polygon([[side*0.3, 0.75], [side*0.5, 0.70], [side*0.4, 0.65]], color='gold', alpha=0.7)
    ax.add_patch(wing)

# Tablets inside (pineal + pituitary)
ax.scatter([ -0.08, 0.08 ], [0.68, 0.68], c='white', s=150, alpha=0.9, edgecolor='gold', linewidth=3)

# Ark label
ax.text(0, 0.85, 'ARK OF THE COVENANT\n= HUMAN BRAIN', color='gold', fontsize=16, ha='center', fontweight='bold')

# Subtle vibration sources
vibration_sources = []
power_rings = []

def animate(frame):
    t = frame * 0.05
    
    # Clear dynamic elements
    for v in vibration_sources:
        v.remove()
    for r in power_rings:
        r.remove()
    vibration_sources.clear()
    power_rings.clear()
    
    # Breath-driven coherence
    breath = np.sin(t * 0.8)
    coherence = 0.5 + 0.5 * (breath + 1)/2
    
    # Subtle microtubule vibration (fast gamma)
    vib_freq = 30 + 20 * coherence
    vib_phase = np.sin(t * vib_freq)
    
    # Vibration lines from brain
    for i in range(20):
        angle = i * np.pi / 10 + t
        length = 0.1 + 0.2 * coherence * (vib_phase + 1)/2
        x = np.cos(angle) * length
        y = 0.62 + np.sin(angle) * length
        line = ax.plot([0, x], [0.62, y], c='cyan', lw=2, alpha=0.6 + 0.4*coherence)[0]
        vibration_sources.append(line)
    
    # When coherence high → massive power release
    if coherence > 0.8:
        # Radiant plasma rings expanding
        for r in np.linspace(0.2, 1.2, 8):
            alpha = coherence - r * 0.5
            if alpha > 0:
                ring = plt.Circle((0, 0.62), r, color='white', fill=False, lw=5, alpha=alpha)
                ax.add_patch(ring)
                power_rings.append(ring)
        
        # Crown explosion
        ax.scatter(0, 0.65, c='white', s=2000 * coherence, alpha=coherence, marker='*')
        
        title_text = "ARK ACTIVATED — SUBTLE VIBRATION UNLEASHES DIVINE POWER"
        color = 'white'
    else:
        title_text = f"Subtle Coherent Vibration Building...\nCoherence: {coherence*100:.0f}%"
        color = 'gold'
    
    ax.set_title(title_text, color=color, fontsize=20, pad=100)

anim = FuncAnimation(fig, animate, interval=60, repeat=True)

# Boundary
boundary = plt.Circle((0, 0), 1, color='gold', fill=False, ls='--', lw=5)
ax.add_patch(boundary)

plt.show()

print("🕍⚡ Ark Mode activated")
print("Ark of the Covenant overlaid on brain")
print("Subtle microtubule/breath vibration → when coherent → massive radiant power release")
print("The Ark wasn't a box — it was the awakened human")
