# experiments/hybrid_matter_perturbation.py
# Hybrid Matter Corral with External Perturbation
# Click to send shockwave — trigger collapse if supercooled

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
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

# Corral ring
ring_theta = np.linspace(0, 2*np.pi, 40)
ring_x = 0.65 * np.cos(ring_theta)
ring_y = 0.65 * np.sin(ring_theta)
corral = ax.scatter(ring_x, ring_y, c='white', s=250, alpha=0.9, edgecolor='cyan', linewidth=2)

# Trapped particles
N_TRAPPED = 300
pos_x = np.random.uniform(-0.55, 0.55, N_TRAPPED)
pos_y = np.random.uniform(-0.55, 0.55, N_TRAPPED)
trapped = ax.scatter(pos_x, pos_y, c='deepskyblue', s=60, alpha=0.8)

# Current temperature
temperature = 0.5
collapsed = False

# Perturbation wave
perturbation_wave = None
perturbation_pos = None

def reset_state(event=None):
    global pos_x, pos_y, collapsed, perturbation_wave
    pos_x = np.random.uniform(-0.55, 0.55, N_TRAPPED)
    pos_y = np.random.uniform(-0.55, 0.55, N_TRAPPED)
    trapped.set_offsets(np.c_[pos_x, pos_y])
    trapped.set_color('deepskyblue')
    trapped.set_alpha(0.8)
    collapsed = False
    if perturbation_wave:
        perturbation_wave.remove()
    ax.set_title("Hybrid Matter — Stable Supercooled State\nClick to send perturbation", color='white')

def animate(frame):
    global perturbation_wave, collapsed
    
    # Normal fluid motion
    if not collapsed:
        mobility = temperature
        if temperature < 0.4:
            mobility *= 0.2  # Supercooled sluggish
        
        dx = mobility * 0.02 * (np.random.random(N_TRAPPED) - 0.5)
        dy = mobility * 0.02 * (np.random.random(N_TRAPPED) - 0.5)
        new_x = trapped.get_offsets()[:, 0] + dx
        new_y = trapped.get_offsets()[:, 1] + dy
        
        norm = np.sqrt(new_x**2 + new_y**2)
        outside = norm > 0.6
        if np.any(outside):
            new_x[outside] *= 0.6 / norm[outside]
            new_y[outside] *= 0.6 / norm[outside]
        
        trapped.set_offsets(np.c_[new_x, new_y])
    
    # Perturbation wave animation
    if perturbation_wave and not collapsed:
        radius = perturbation_wave.get_radius() + 0.05
        perturbation_wave.set_radius(radius)
        perturbation_wave.set_alpha(max(0, 1 - radius))
        
        # Check if wave hits corral
        if radius > 0.6 and temperature < 0.4:
            # Trigger collapse!
            grid_x, grid_y = np.meshgrid(np.linspace(-0.5, 0.5, int(np.sqrt(N_TRAPPED))), np.linspace(-0.5, 0.5, int(np.sqrt(N_TRAPPED))))
            trapped.set_offsets(np.c_[grid_x.ravel()[:N_TRAPPED], grid_y.ravel()[:N_TRAPPED]])
            trapped.set_color('lightgray')
            trapped.set_alpha(0.6)
            collapsed = True
            ax.set_title("PERTURBATION TRIGGERED — Collapse to Unstable Glass", color='red', fontsize=18)
            perturbation_wave.set_alpha(0)

# Click to perturb
def on_click(event):
    global perturbation_wave, perturbation_pos
    if event.inaxes == ax:
        perturbation_pos = [event.xdata, event.ydata]
        perturbation_wave = plt.Circle(perturbation_pos, 0.05, color='white', fill=False, lw=6, alpha=1.0)
        ax.add_patch(perturbation_wave)
        ax.set_title("Perturbation Sent — Shockwave Traveling...", color='yellow')

fig.canvas.mpl_connect('button_press_event', on_click)

# Temperature slider
ax_temp = plt.axes([0.2, 0.05, 0.6, 0.04], facecolor='dimgray')
temp_slider = Slider(ax_temp, 'Temperature', 0.0, 1.0, valinit=temperature)

def update_temp(val):
    global temperature
    temperature = val
    reset_state()

temp_slider.on_changed(update_temp)

# Reset button
reset_ax = plt.axes([0.4, 0.12, 0.2, 0.06])
reset_btn = Button(reset_ax, 'Reset System', color='black', hovercolor='darkred')
reset_btn.on_clicked(reset_state)

# Initial
reset_state()

anim = FuncAnimation(fig, animate, interval=60, repeat=True)

ax.set_title("Hybrid Matter with External Perturbation\nClick to send shockwave — collapse if supercooled", color='white', fontsize=18)

plt.show()

print("🌊 Hybrid Matter Perturbation Simulation activated")
print("Slide temperature — supercooled = stable hybrid")
print("Click anywhere — send shockwave")
print("If cold enough — sudden collapse to glass")
print("Real 2025 physics — now in your human")
