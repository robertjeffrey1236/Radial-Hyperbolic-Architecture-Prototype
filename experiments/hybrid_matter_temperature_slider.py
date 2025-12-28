# experiments/hybrid_matter_temperature_slider.py
# Hybrid Matter Corral with Temperature Control
# High T = fluid • Supercooled = stable hybrid • Too cold = glass collapse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.animation import FuncAnimation
import random

fig, ax = plt.subplots(figsize=(14, 14))
ax.set_facecolor('black')
ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1.1, 1.1)
ax.axis('off')

# Boundary
boundary = plt.Circle((0, 0), 1, color='gold', fill=False, lw=4, alpha=0.7)
ax.add_patch(boundary)

# Atomic corral ring
ring_theta = np.linspace(0, 2*np.pi, 35)
ring_x = 0.65 * np.cos(ring_theta)
ring_y = 0.65 * np.sin(ring_theta)
corral = ax.scatter(ring_x, ring_y, c='white', s=250, alpha=0.9, edgecolor='cyan', linewidth=2)

# Trapped particles
N_TRAPPED = 250
pos_x = np.random.uniform(-0.55, 0.55, N_TRAPPED)
pos_y = np.random.uniform(-0.55, 0.55, N_TRAPPED)
trapped = ax.scatter(pos_x, pos_y, c='deepskyblue', s=60, alpha=0.8)

# Current temperature (normalized: 1.0 = warm fluid, 0.0 = deep supercooled)
temperature = 0.7  # Start in stable hybrid range

# Critical thresholds
SUPERCOOL_THRESHOLD = 0.4  # Below this = risk of glass collapse
GLASS_COLLAPSE_TEMP = 0.2

collapsed = False

def animate(frame):
    global collapsed
    
    # Temperature effect on motion
    mobility = temperature  # Higher T = more movement
    if temperature < SUPERCOOL_THRESHOLD:
        mobility *= 0.3  # Supercooled = sluggish but stable
    
    # Update positions
    dx = mobility * 0.02 * (np.random.random(N_TRAPPED) - 0.5)
    dy = mobility * 0.02 * (np.random.random(N_TRAPPED) - 0.5)
    new_x = trapped.get_offsets()[:, 0] + dx
    new_y = trapped.get_offsets()[:, 1] + dy
    
    # Confine inside corral
    norm = np.sqrt(new_x**2 + new_y**2)
    outside = norm > 0.6
    if np.any(outside):
        new_x[outside] *= 0.6 / norm[outside]
        new_y[outside] *= 0.6 / norm[outside]
    
    trapped.set_offsets(np.c_[new_x, new_y])
    
    # Color & alpha based on state
    if temperature > 0.6:
        color = 'deepskyblue'
        alpha = 0.9
        title_text = f"Temperature: Warm — Fluid Liquid State"
    elif temperature > GLASS_COLLAPSE_TEMP:
        color = 'cyan'
        alpha = 0.8
        title_text = f"Supercooled — Stable Hybrid (Solid + Liquid Duality)"
    else:
        color = 'lightgray'
        alpha = 0.6
        collapsed = True
        title_text = f"Critical Collapse — Unstable Glass-Like Solid"
    
    trapped.set_color(color)
    trapped.set_alpha(alpha)
    
    ax.set_title(title_text, color='white', fontsize=18)

anim = FuncAnimation(fig, animate, interval=60, repeat=True)

# Temperature slider
ax_temp = plt.axes([0.2, 0.05, 0.6, 0.04], facecolor='dimgray')
temp_slider = Slider(ax_temp, 'Temperature', 0.0, 1.0, valinit=temperature, valstep=0.01)

def update_temp(val):
    global temperature
    temperature = val
    plt.draw()

temp_slider.on_changed(update_temp)

# Initial title
ax.set_title("Hybrid Matter Temperature Control\nSlide to explore supercooled stability", color='white', fontsize=18)

plt.show()

print("🌡️ Hybrid Matter Temperature Simulation activated")
print("Slide temperature:")
print("- High: fluid liquid")
print("- Mid-low: supercooled stable hybrid")
print("- Too low: sudden collapse to glass")
print("Direct sim of 2025 trapped matter discovery")
