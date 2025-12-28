# experiments/human_body_touch_observer.py
# Wholesome Human with Touch as Distributed Observer
# Skin feels vibration/texture — focus hand for detailed tactile perception

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from core.geometry import golden_spiral_points, poincare_disk_project

N_POINTS = 25000
DIM = 37

# Lattice (tactile "texture" of reality)
points_nd = golden_spiral_points(n_points=N_POINTS, dim=DIM, radius_scale=0.99)
points_2d = poincare_disk_project(points_nd[:, :2])

# Initial hand focus position (right hand near torso)
hand_pos = np.array([0.25, 0.0])

fig, ax = plt.subplots(figsize=(16, 20))
ax.set_facecolor('black')
plt.subplots_adjust(bottom=0.15)

# Background lattice (unfelt)
unfelt = ax.scatter(points_2d[:, 0], points_2d[:, 1], c='gray', s=1, alpha=0.08)

# Current touch feedback
touch_scatter = None
vibration_circles = []

def update_touch(hand, sensitivity=0.4, whole_body=True):
    global touch_scatter, vibration_circles
    
    # Clear previous
    if touch_scatter: touch_scatter.remove()
    for circ in vibration_circles: circ.remove()
    vibration_circles.clear()
    
    # Distance from hand focus
    dist_hand = np.linalg.norm(points_2d - hand, axis=1)
    in_touch = dist_hand < sensitivity
    
    # Tactile intensity — closer = stronger vibration
    intensity = np.exp(-dist_hand * 4 / sensitivity)
    
    global touch_scatter
    touch_scatter = ax.scatter(points_2d[in_touch, 0], points_2d[in_touch, 1],
                               c='orange', s=20 * intensity[in_touch], alpha=0.9, zorder=10)
    
    # Vibration ripples from hand
    for r in np.linspace(0.03, sensitivity, 6):
        ripple = plt.Circle(hand, r, color='gold', fill=False, lw=2, alpha=0.4 - r/sensitivity*0.3)
        ax.add_patch(ripple)
        vibration_circles.append(ripple)
    
    # Whole-body subtle touch awareness
    if whole_body:
        skin_glow = plt.Circle((0, 0), 0.9, color='peachpuff', fill=False, lw=8, alpha=0.2)
        ax.add_patch(skin_glow)
        vibration_circles.append(skin_glow)

# Initial
update_touch(hand_pos)

# Human silhouette (neutral)
theta = np.linspace(0, 2*np.pi, 200)
skin_x = 0.45 * np.cos(theta)
skin_y = 1.1 * np.sin(theta) - 0.1
ax.plot(skin_x, skin_y, c='peachpuff', lw=4, alpha=0.5)

# Face faint
ax.scatter([-0.12, 0.12], [0.72, 0.72], c='deepskyblue', s=200, alpha=0.4)
ax.scatter(0, 0.75, c='violet', s=150, alpha=0.4)

# Hand focus indicator
hand_indicator = ax.scatter(hand_pos[0], hand_pos[1], c='gold', s=400, edgecolor='orange', linewidth=4, alpha=0.9, zorder=12)

# Internal faint
ax.scatter(0, 0, c='crimson', s=500, alpha=0.3)

# Poincaré boundary
circle = plt.Circle((0, 0), 1, color='orange', fill=False, ls='--', lw=5)
ax.add_patch(circle)

ax.axis('equal')
ax.axis('off')
plt.title("Human with Touch as Distributed Observer\nDrag hand focus to feel texture • Skin senses subtle vibration everywhere", color='white', fontsize=20)

# Draggable hand focus
dragging = False
def on_click(event):
    global dragging
    if event.inaxes != ax: return
    dist_hand = np.linalg.norm([event.xdata - hand_pos[0], event.ydata - hand_pos[1]])
    if dist_hand < 0.1:
        dragging = True

def on_release(event):
    global dragging
    dragging = False

def on_motion(event):
    global hand_pos
    if dragging and event.inaxes == ax:
        hand_pos = [event.xdata, event.ydata]
        hand_indicator.set_offsets([hand_pos])
        update_touch(hand_pos, sensitivity=slider_sens.val)
        plt.draw()

fig.canvas.mpl_connect('button_press_event', on_click)
fig.canvas.mpl_connect('button_release_event', on_release)
fig.canvas.mpl_connect('motion_notify_event', on_motion)

# Touch sensitivity slider
ax_sens = plt.axes([0.2, 0.05, 0.6, 0.03])
slider_sens = Slider(ax_sens, 'Touch Sensitivity', 0.1, 0.8, valinit=0.4)
def update_sens(val):
    update_touch(hand_pos, sensitivity=val)
    plt.draw()
slider_sens.on_changed(update_sens)

plt.show()

print("✋🌿 Touch as Observer activated — drag the golden hand focus to explore tactile reality")
print("Whole skin feels subtle presence • Focused touch reveals vibration and texture")
