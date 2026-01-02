# experiments/human_body_proton_gradient_mb.py
# Proton Gradient + Methylene Blue Delivery Simulation
# MB enhances ETC → steeper gradient → more ATP

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Wedge

fig, ax = plt.subplots(figsize=(16, 24))
ax.set_facecolor('black')
ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1.4, 1.0)
ax.axis('off')

# Human body regions
regions = {
    'mouth': [0.0, 0.55],
    'stomach': [0.0, -0.15],
    'blood': [0.0, 0.0],
    'cells': [[0.0, 0.62], [0.0, 0.0], [0.0, -0.35], [-0.3, 0.1], [0.3, -0.1]],
}

# Mitochondria in cells
mito_centers = []
for cell in regions['cells']:
    for _ in range(4):
        offset = np.random.uniform(-0.08, 0.08, 2)
        mito_centers.append(np.array(cell) + offset)

# Proton gradient visualization (inner membrane)
gradients = []
atp_bursts = []

# Methylene blue particles (start in mouth)
mb_particles = [np.array([0.0 + random.uniform(-0.05, 0.05), 0.55 + random.uniform(-0.05, 0.05)]) for _ in range(50)]
mb_active = []  # Reduced form in mitochondria

def animate(frame):
    t = frame * 0.03
    
    # Clear dynamic elements
    for g in gradients + atp_bursts:
        if g in ax.patches:
            g.remove()
    gradients.clear()
    atp_bursts.clear()
    
    # Move MB through body (oral → stomach → blood → cells)
    stage = min(frame // 100, 3)
    for i, p in enumerate(mb_particles):
        if stage == 0:  # Mouth/throat
            p[1] = 0.55 - (frame % 100) * 0.005
        elif stage == 1:  # Stomach
            p[0] += np.sin(t + i) * 0.002
            p[1] = -0.15 + np.cos(t + i) * 0.05
        elif stage == 2:  # Bloodstream
            p += np.random.normal(0, 0.01, 2)
            p[1] -= 0.003
        else:  # Enter cells/mitochondria
            target = random.choice(mito_centers)
            p += (target - p) * 0.05
            if np.linalg.norm(p - target) < 0.05:
                mb_active.append(target.copy())
    
    # Draw MB (blue = oxidized, colorless = reduced)
    ax.scatter([p[0] for p in mb_particles], [p[1] for p in mb_particles], 
               c='deepskyblue', s=40, alpha=0.8, edgecolor='cyan')
    ax.scatter([p[0] for p in mb_active], [p[1] for p in mb_active], 
               c='lightgray', s=30, alpha=0.6)  # Reduced form
    
    # Proton gradient — stronger with more active MB
    mb_boost = len(mb_active) / 50.0
    base_intensity = 0.5 + 0.5 * mb_boost
    
    for center in mito_centers:
        # Outer membrane
        outer = plt.Circle(center, 0.08, color='gray', fill=False, lw=2, alpha=0.4)
        ax.add_patch(outer)
        
        # Inner membrane gradient (steeper with MB)
        for layer in range(5):
            alpha = base_intensity - layer * 0.1
            color = 'red' if layer < 3 else 'blue'  # H+ accumulation inside
            ring = plt.Circle(center, 0.06 - layer*0.01, color=color, fill=False, lw=3, alpha=alpha)
            ax.add_patch(ring)
            gradients.append(ring)
        
        # ATP synthase rotation + burst
        angle = t * (10 + 20 * mb_boost) % 360
        rotor = Wedge(center, 0.05, angle, angle + 180, width=0.02, color='gold', alpha=0.8)
        ax.add_patch(rotor)
        
        if random.random() < 0.1 + mb_boost:
            for _ in range(5):
                spark = np.random.uniform(-0.05, 0.05, 2) + center
                b = ax.scatter(spark[0], spark[1], c='yellow', s=30, marker='*', alpha=0.9)
                atp_bursts.append(b)

    # Title
    stages = ["Oral Intake", "Stomach Absorption", "Bloodstream Distribution", "Cellular Uptake — Enhanced ETC"]
    ax.set_title(f"Methylene Blue Delivery & Proton Gradient Boost\nStage: {stages[min(stage,3)]}\nMB reduces → ETC acceleration → steeper gradient → more ATP", 
                 color='white', fontsize=18, pad=80)

anim = FuncAnimation(fig, animate, interval=80, repeat=True)

# Body faint
body = plt.Circle((0, 0), 0.9, color='deepskyblue', fill=False, lw=4, alpha=0.3)
ax.add_patch(body)
boundary = plt.Circle((0, 0), 1, color='cyan', fill=False, ls='--', lw=5)
ax.add_patch(boundary)

plt.show()

print("💊⚡ Methylene Blue + Proton Gradient Simulation activated")
print("Watch blue dye travel: mouth → stomach → blood → mitochondria")
print("MB reduces (colorless) → boosts electron transport → steeper proton gradient → more ATP bursts")
print("Real biohacking visualization — MB as mitochondrial enhancer")
