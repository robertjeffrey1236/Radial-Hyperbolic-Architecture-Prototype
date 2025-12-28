# experiments/human_body_single_photon_double_slit.py
# Single-Photon Double-Slit + Quantum Eraser
# One particle at a time — watch pattern build mysteriously

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons, Button
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
source = np.array([0.0, -0.6])
slit1 = np.array([-0.15, -0.3])
slit2 = np.array([0.15, -0.3])
screen_x = 0.8

ax.scatter(source[0], source[1], c='gold', s=400, alpha=0.8)
ax.plot([slit1[0], slit1[0]], [slit1[1]-0.1, slit1[1]+0.1], c='white', lw=8)
ax.plot([slit2[0], slit2[0]], [slit2[1]-0.1, slit2[1]+0.1], c='white', lw=8)

# Screen hits
screen_hits = []

# Current flying photon
current_photon = None

# States
detectors_on = False
eraser_on = True
single_photon_mode = True
total_particles = 0

def spawn_photon():
    global current_photon, total_particles
    if single_photon_mode and current_photon is not None:
        return  # Wait for current to hit
    
    angle = random.uniform(-0.3, 0.3)
    pos = source + np.array([angle*0.1, 0.1])
    phase = random.uniform(0, 2*np.pi)
    path = random.choice([1, 2]) if detectors_on else 0
    current_photon = {'pos': pos, 'phase': phase, 'path': path}
    total_particles += 1

def update_photon():
    global current_photon, screen_hits
    if current_photon is None:
        return
    
    p = current_photon
    p['pos'][1] += 0.015  # Fly upward
    
    if p['pos'][1] > -0.3:  # Passed slits
        # Calculate interference position
        d1 = np.linalg.norm(p['pos'] - slit1)
        d2 = np.linalg.norm(p['pos'] - slit2)
        phase_diff = (d2 - d1) * 15
        
        if eraser_on or not detectors_on:
            # Interference
            prob = (np.cos(phase_diff + p['phase']) ** 2)
            x_offset = 0.4 * (prob - 0.5)
        else:
            # Which-path → clumps
            x_offset = 0.2 if p['path'] == 1 else -0.2
        
        hit_x = screen_x + x_offset + random.uniform(-0.05, 0.05)
        hit_y = random.uniform(-0.8, 0.8)
        color = 'cyan' if (eraser_on or not detectors_on) else 'yellow'
        
        screen_hits.append([hit_x, hit_y, color])
        current_photon = None  # Hit screen
        
        # Update display
        hits_x, hits_y, colors = zip(*screen_hits) if screen_hits else ([], [], [])
        screen_scatter.set_offsets(np.c_[hits_x, hits_y])
        screen_scatter.set_color(colors)

# Visual elements
screen_scatter = ax.scatter([], [], c=[], s=30, alpha=0.9)
photon_scatter = ax.scatter([], [], c='white', s=80, marker='o', alpha=1.0)

det1 = ax.scatter(slit1[0], slit1[1], c='red', s=200, alpha=0.0)
det2 = ax.scatter(slit2[0], slit2[1], c='red', s=200, alpha=0.0)

# Title & count
title = ax.set_title("Single-Photon Double-Slit + Eraser\nParticles: 0", color='white', fontsize=20)
particle_count = ax.text(0.5, -1.1, "Particles: 0", color='white', fontsize=14, ha='center')

# Controls
rax = plt.axes([0.05, 0.55, 0.2, 0.15], facecolor='black')
check = CheckButtons(rax, ['Which-Path Detectors', 'Eraser ON', 'Single-Photon Mode'], [False, True, True])

def toggle(label):
    global detectors_on, eraser_on, single_photon_mode
    if label == 'Which-Path Detectors':
        detectors_on = not detectors_on
        alpha = 0.8 if detectors_on else 0.0
        det1.set_alpha(alpha)
        det2.set_alpha(alpha)
    elif label == 'Eraser ON':
        eraser_on = not eraser_on
    elif label == 'Single-Photon Mode':
        single_photon_mode = not single_photon_mode
    reset(None)

check.on_clicked(toggle)

# Reset button
reset_ax = plt.axes([0.05, 0.45, 0.2, 0.06])
reset
