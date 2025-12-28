# experiments/human_body_breath_heartbeat_brainwaves.py
# Ultimate Integration: Breath → Heartbeat → Brainwaves Coherence Simulation
# All systems synchronized — the living, breathing, thinking human

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from core.geometry import golden_spiral_points, poincare_disk_project

N_POINTS = 30000
BRAIN_DENSITY = 6000
FRAMES = 400  # Full breath cycle

# Lattices
points_nd = golden_spiral_points(n_points=N_POINTS, dim=37, radius_scale=0.99)
points_2d = poincare_disk_project(points_nd[:, :2])
brain_offsets = golden_spiral_points(n_points=BRAIN_DENSITY, dim=2, radius_scale=0.35)
brain_points = brain_offsets + np.array([0.0, 0.62])

fig, ax = plt.subplots(figsize=(18, 28))
ax.set_facecolor('black')

# Faint substrate
substrate = ax.scatter(points_2d[:, 0], points_2d[:, 1], c='gray', s=1, alpha=0.05)

# Brain
brain_scatter = ax.scatter(brain_points[:, 0], brain_points[:, 1], c='indigo', s=10, alpha=0.6)

# Heart
heart = ax.scatter(0, 0, c='crimson', s=800, edgecolor='white', linewidth=4, alpha=0.8, zorder=15)

# Body silhouette (will expand/contract with breath)
breath_body = plt.Circle((0, 0), 0.9, color='white', fill=False, lw=4, alpha=0.3)
ax.add_patch(breath_body)

# Aura pulse
aura = plt.Circle((0, 0), 1.3, color='white', fill=False, lw=8, alpha=0.2)
ax.add_patch(aura)

# Poincaré boundary
boundary = plt.Circle((0, 0), 1, color='gold', fill=False, ls='--', lw=6)
ax.add_patch(boundary)

ax.axis('equal')
ax.axis('off')
title = ax.set_title("Living Human: Breath • Heartbeat • Brainwaves in Coherence\nInhale... Exhale... Be.", color='white', fontsize=22, pad=100)

def animate(frame):
    t = frame / FRAMES * 2 * np.pi  # Full cycle
    
    # === Breath Cycle (slow 6-second breath) ===
    breath_phase = np.sin(t * 0.5)  # Slow rhythm
    breath_scale = 0.9 + 0.15 * breath_phase  # Body expands on inhale
    breath_body.set_radius(breath_scale)
    aura.set_radius(1.3 + 0.3 * breath_phase)
    aura.set_alpha(0.2 + 0.3 * (breath_phase + 1)/2)
    
    # === Heartbeat (faster, synced to breath) ===
    heart_rate = 1.2 + 0.4 * breath_phase  # Faster on inhale (RSA)
    heart_pulse = np.abs(np.sin(t * heart_rate * 2)) ** 0.5
    heart_size = 800 + 600 * heart_pulse
    heart_alpha = 0.8 + 0.2 * heart_pulse
    heart.set_sizes([heart_size])
    heart.set_alpha(heart_alpha)
    
    # Ripple from heart
    ripple_radius = heart_pulse * 0.8
    ripple = plt.Circle((0, 0), ripple_radius, color='crimson', fill=False, lw=4, alpha=0.5 - heart_pulse*0.3)
    if 'current_ripple' in globals():
        current_ripple.remove()
    global current_ripple
    current_ripple = ax.add_patch(ripple)
    
    # === Brainwaves (entrain to breath) ===
    if breath_phase > 0:  # Inhale → activation (beta/gamma)
        wave_freq = 6 + 4 * breath_phase
        color_map = 'plasma'
        state = "Activation ↑"
    else:  # Exhale → relaxation (alpha/theta)
        wave_freq = 3 + 2 * np.abs(breath_phase)
        color_map = 'winter'
        state = "Relaxation ↓"
    
    brain_phase = np.sin(t * wave_freq + np.linalg.norm(brain_offsets, axis=1) * 8)
    intensity = (brain_phase + 1) / 2
    colors = plt.cm.get_cmap(color_map)(intensity)
    brain_scatter.set_color(colors)
    brain_scatter.set_alpha(0.6 + 0.4 * intensity)
    
    # Coherence glow when all aligned
    coherence = np.abs(breath_phase) * heart
