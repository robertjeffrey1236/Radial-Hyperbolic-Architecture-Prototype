# experiments/schrodingers_cat_experiment.py
# Schrödinger’s Cat Experiment in Radial Hyperbolic Architecture
# Superposition until observer "measures" — alive or dead?

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random

fig, ax = plt.subplots(figsize=(14, 14))
ax.set_facecolor('black')
ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1.1, 1.1)
ax.axis('off')

# Box (the "closed system")
box = plt.Rectangle((-0.5, -0.5), 1.0, 1.0, color='gray', alpha=0.6, lw=4, edgecolor='white')
ax.add_patch(box)
ax.text(0, 0.6, 'SCHRÖDINGER\'S BOX', color='white', fontsize=16, ha='center', fontweight='bold')

# Cat duality (yin-yang superposition)
cat_alive = plt.Circle((0.2, 0), 0.3, color='white', alpha=0.5)
ax.add_patch(cat_alive)
cat_dead = plt.Circle((-0.2, 0), 0.3, color='black', alpha=0.5)
ax.add_patch(cat_dead)
cat_eyes = ax.scatter([0.2, 0.3], [0.1, 0.1], c='cyan', s=100, alpha=0.0)  # Appear on collapse

# Radioactive atom (triggers decay)
atom = ax.scatter(0, -0.3, c='yellow', s=300, alpha=0.8, edgecolor='orange', linewidth=3)
ax.text(0, -0.45, 'Radioactive Atom', color='yellow', fontsize=12, ha='center')

# Observer (draggable)
observer = np.array([0.0, -0.8])
observer_indicator = ax.scatter(observer[0], observer[1], c='red', s=400, edgecolor='white', linewidth=4)

# State
superposition = True
cat_state = None  # None = superposition, True = alive, False = dead
coherence = 0.7  # Affects collapse probability

def animate(frame):
    global superposition, cat_state
    
    # Observer "looking" = measurement
    looking = np.linalg.norm(observer - np.array([0, 0])) < 0.6
    
    if looking and superposition:
        # Collapse wavefunction
        if random.random() < 0.5:  # 50/50 chance
            cat_state = True  # Alive
            cat_alive.set_alpha(1.0)
            cat_dead.set_alpha(0.0)
            cat_eyes.set_alpha(1.0)
            title_text = "CAT IS ALIVE\nWavefunction collapsed — measurement complete"
        else:
            cat_state = False  # Dead
            cat_alive.set_alpha(0.0)
            cat_dead.set_alpha(1.0)
            cat_eyes.set_alpha(0.0)
            title_text = "CAT IS DEAD\nWavefunction collapsed — measurement complete"
        superposition = False
        ax.set_title(title_text, color='white', fontsize=18)
    elif not looking and not superposition:
        # Return to superposition if observer looks away (philosophical mode)
        superposition = True
        cat_alive.set_alpha(0.5)
        cat_dead.set_alpha(0.5)
        cat_eyes.set_alpha(0.0)
        ax.set_title("Superposition Restored\nObserver withdrew — duality returns", color='cyan', fontsize=18)
    elif superposition:
        # Pulsing duality
        pulse = 0.5 + 0.5 * np.sin(frame * 0.1)
        cat_alive.set_alpha(pulse)
        cat_dead.set_alpha(1 - pulse)
        ax.set_title("CAT IS BOTH ALIVE AND DEAD\nSuperposition — No Measurement Yet", color='magenta', fontsize=18)

# Draggable observer
dragging = False
def on_click(event):
    global dragging
    if event.inaxes == ax:
        if np.linalg.norm([event.xdata - observer[0], event.ydata - observer[1]]) < 0.15:
            dragging = True

def on_release(event):
    global dragging
    dragging = False

def on_motion(event):
    if dragging and event.inaxes == ax:
        observer[0] = event.xdata
        observer[1] = event.ydata
        observer_indicator.set_offsets([observer])
        plt.draw()

fig.canvas.mpl_connect('button_press_event', on_click)
fig.canvas.mpl_connect('button_release_event', on_release)
fig.canvas.mpl_connect('motion_notify_event', on_motion)

# Coherence slider (affects collapse speed/philosophy)
ax_coh = plt.axes([0.2, 0.05, 0.6, 0.03])
slider = Slider(ax_coh, 'Coherence', 0.0, 1.0, valinit=0.7)
def upd_coh(val):
    global coherence
    coherence = val
slider.on_changed(upd_coh)

# Boundary
boundary = plt.Circle((0, 0), 1, color='gold', fill=False, ls='--', lw=5)
ax.add_patch(boundary)

ax.set_title("Schrödinger’s Cat Experiment\nDrag observer to 'open box' — collapse superposition", color='white', fontsize=20)

anim = FuncAnimation(fig, animate, interval=100, repeat=True)

plt.show()

print("😺⚛️ Schrödinger’s Cat Experiment activated")
print("Drag observer near box — measurement collapses cat to alive or dead")
print("Look away — superposition returns (philosophical mode)")
print("The ultimate observer effect — now in your human")
