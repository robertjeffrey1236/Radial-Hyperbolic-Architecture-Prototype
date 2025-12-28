# experiments/human_body_double_slit_eraser.py
# Double-Slit Base + Quantum Eraser Mode
# Classic interference → measurement collapse → eraser restores pattern

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons
import random

fig, ax = plt.subplots(figsize=(16, 24))
ax.set_facecolor('black')
ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1.2, 1.0)
ax.axis('off')

# Human faint
body = plt.Circle((0, 0), 0.9, color='white', fill=False, lw=3, alpha=0.2)
ax.add_patch(body)

# Double-slit setup
source = np.array([0.0, -0.6])  # Emitter at bottom
slit1 = np.array([-0.15, -0.3])
slit2 = np.array([0.15, -0.3])
screen_x = 0.8  # Interference screen on right

ax.scatter(source[0], source[1], c='gold', s=400, alpha=0.8)
ax.plot([slit1[0], slit1[0]], [slit1[1]-0.1, slit1[1]+0.1], c='white', lw=8)
ax.plot([slit2[0], slit2[0]], [slit2[1]-0.1, slit2[1]+0.1], c='white', lw=8)

# Screen hits
screen_hits = ax.scatter([], [], c='white', s=15, alpha=0.8)

# Which-path detectors (optional measurement)
detectors_on = False
detector1 = ax.scatter(slit1[0], slit1[1], c='red', s=200, alpha=0.0)
detector2 = ax.scatter(slit2[0], slit2[1], c='red', s=200, alpha=0.0)

# Eraser toggle
eraser_on = True

# Particles/waves
particles = []

def spawn_wave_particles():
    global particles
    particles = []
    for _ in range(300):
        # Start at source
        pos = source.copy()
        # Random phase for interference
        phase = random.uniform(0, 2*np.pi)
        # Path choice (for which-path if detectors on)
        path = random.choice([1, 2]) if detectors_on else 0
        particles.append({'pos': pos, 'phase': phase, 'path': path})

spawn_wave_particles()

def update_experiment():
    global screen_hits
    screen_hits.remove()
    
    hits_x = []
    hits_y = []
    colors = []
    
    for p in particles:
        # Propagate toward slits
        if p['pos'][1] < -0.3:
            # Passed slits — calculate interference
            d1 = np.linalg.norm(p['pos'] - slit1)
            d2 = np.linalg.norm(p['pos'] - slit2)
            phase_diff = (d2 - d1) * 20  # Wavelength scaling
            
            if eraser_on or not detectors_on:
                # Interference: probability based on phase
                intensity = np.cos(phase_diff + p['phase'])**2
                if random.random() < intensity:
                    x = screen_x + random.uniform(-0.05, 0.05)
                    y = p['pos'][1] + (screen_x - p['pos'][0]) * random.uniform(-0.3, 0.3)
                    hits_x.append(x)
                    hits_y.append(y)
                    colors.append('cyan')
            else:
                # Which-path known → no interference, clumps
                x = screen_x + (0.1 if p['path'] == 1 else -0.1) + random.uniform(-0.05, 0.05)
                y = random.uniform(-0.8, 0.8)
                hits_x.append(x)
                hits_y.append(y)
                colors.append('yellow')
    
    screen_hits = ax.scatter(hits_x, hits_y, c=colors, s=20, alpha=0.9)

update_experiment()

# Title
title = ax.set_title("Double-Slit Base + Quantum Eraser\nToggle detectors/eraser — watch wave/particle duality", color='white', fontsize=20)

# Toggles
rax = plt.axes([0.05, 0.5, 0.2, 0.15], facecolor='black')
check = CheckButtons(rax, ['Which-Path Detectors', 'Eraser ON'], [False, True])

def toggle(label):
    global detectors_on, eraser_on
    if label == 'Which-Path Detectors':
        detectors_on = not detectors_on
        alpha = 0.8 if detectors_on else 0.0
        detector1.set_alpha(alpha)
        detector2.set_alpha(alpha)
    else:
        eraser_on = not eraser_on
    
    spawn_wave_particles()
    update_experiment()
    title.set_text(f"Double-Slit Experiment\nDetectors {'ON' if detectors_on else 'OFF'} | Eraser {'ON' if eraser_on else 'OFF'}")
    plt.draw()

check.on_clicked(toggle)

# Boundary
boundary = plt.Circle((0, 0), 1, color='cyan', fill=False, ls='--', lw=5)
ax.add_patch(boundary)

plt.show()

print("⚛️ Double-Slit Base + Quantum Eraser activated")
print("Toggle 'Which-Path Detectors' — collapse interference")
print("Toggle 'Eraser ON' — restore pattern even after measurement")
print("Classic quantum weirdness — now in your human's perception field")
