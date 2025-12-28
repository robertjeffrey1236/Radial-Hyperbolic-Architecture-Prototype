# -*- coding: utf-8 -*-
"""
Created on Sun Dec 28 00:09:33 2025

@author: amham
"""

"""
Interactive Radial Hyperbolic Architecture
-----------------------------------------

- Dual golden-ratio spirals in Poincaré disk
- Counter-rotating Merkaba overlay (NEW: pulsing animation)
- Draggable observer (red point)
- Slider to adjust PERCEPTION_RADIUS
- Smooth lazy-loading animation
"""

import math
import torch
import geoopt
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.animation import FuncAnimation

# ============================================================
# 1. Hyperbolic Space Setup
# ============================================================

ball = geoopt.PoincareBall(c=1.0)
PHI = (1 + math.sqrt(5)) / 2
PERCEPTION_RADIUS = 1.99

# ============================================================
# 2. Dual Golden-Ratio Spirals
# ============================================================

def golden_spiral_points(n_points=1500, direction=1.0):
    theta = torch.linspace(0, 12 * math.pi, n_points)
    r = torch.exp(theta / PHI)
    x = r * torch.cos(direction * theta)
    y = r * torch.sin(direction * theta)
    points = torch.stack([x, y], dim=1)
    norm = torch.norm(points, dim=1, keepdim=True)
    points = points / (norm + 1e-6) * 0.99
    return ball.expmap0(points)

primal_spiral = golden_spiral_points(direction=1.0)
dual_spiral = golden_spiral_points(direction=-1.0)
all_points = torch.cat([primal_spiral, dual_spiral])

# ============================================================
# 3. Lazy Observer Loading
# ============================================================

def lazy_load(points, observer, radius):
    distances = ball.dist(observer, points)
    return points[distances < radius]

# ============================================================
# 4. Merkaba Overlay (unchanged)
# ============================================================

def merkaba_points(scale=0.6, up_angle=0.0, down_angle=0.0):
    vertices = torch.tensor([
        [ 1,  1,  1],
        [ 1, -1, -1],
        [-1,  1, -1],
        [-1, -1,  1]
    ], dtype=torch.float32) / math.sqrt(3)

    def rotation_z(angle):
        return torch.tensor([
            [math.cos(angle), -math.sin(angle), 0],
            [math.sin(angle),  math.cos(angle), 0],
            [0, 0, 1]
        ])

    up = (vertices * scale) @ rotation_z(up_angle).T
    down = (-vertices * scale) @ rotation_z(down_angle).T
    points_2d = torch.cat([up, down], dim=0)[:, :2]
    norm = torch.norm(points_2d, dim=1, keepdim=True)
    points_2d = points_2d / (norm + 1e-6) * 0.9
    return ball.expmap0(points_2d)

# ============================================================
# 5. Visualization Setup
# ============================================================

plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(9, 9))
plt.subplots_adjust(bottom=0.18)
ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1.1, 1.1)
ax.set_aspect("equal")
ax.set_title("Interactive Radial Hyperbolic Architecture", fontsize=16)

ax.text(-1, 1.05,
        "Drag the red observer | Adjust PERCEPTION_RADIUS | Merkaba pulses & rotates",
        fontsize=10, color='lightgray', va='top')

disk = plt.Circle((0, 0), 1, fill=False, color="white", lw=2, alpha=0.6)
ax.add_artist(disk)

# ============================================================
# 6. Observer
# ============================================================

observer = torch.zeros(1, 2)
observer_artist = ax.scatter(
    0, 0,
    s=180, c='red', edgecolors='white', linewidths=1.2, zorder=5
)

# ============================================================
# 7. Initial Rendering
# ============================================================

visible_points = lazy_load(all_points, observer, PERCEPTION_RADIUS).cpu().numpy()
scatter_visible = ax.scatter(
    visible_points[:, 0],
    visible_points[:, 1],
    s=12, c="violet", alpha=0.75
)

merkaba = merkaba_points().cpu().numpy()
scatter_merkaba = ax.scatter(
    merkaba[:, 0],
    merkaba[:, 1],
    s=90, c="gold", marker="*", edgecolors='black'
)

# ============================================================
# 8. Slider PERCEPTION_RADIUS
# ============================================================

ax_radius = plt.axes([0.2, 0.05, 0.6, 0.03], facecolor='dimgray')
radius_slider = Slider(
    ax_radius, 'Perception radius',
    0.1, 3, valinit=PERCEPTION_RADIUS, valstep=0.001
)

def update_radius(val):
    global PERCEPTION_RADIUS
    PERCEPTION_RADIUS = radius_slider.val
    visible = lazy_load(all_points, observer, PERCEPTION_RADIUS).cpu().numpy()
    scatter_visible.set_offsets(visible)
    fig.canvas.draw_idle()

radius_slider.on_changed(update_radius)

# ============================================================
# 9. Draggable Observer
# ============================================================

dragging = {"active": False}

def on_press(event):
    if event.inaxes == ax and observer_artist.contains(event)[0]:
        dragging["active"] = True

def on_release(event):
    dragging["active"] = False

def on_motion(event):
    if dragging["active"] and event.inaxes == ax:
        observer[0] = torch.tensor([event.xdata, event.ydata])
        observer_artist.set_offsets([[event.xdata, event.ydata]])
        visible = lazy_load(all_points, observer, PERCEPTION_RADIUS).cpu().numpy()
        scatter_visible.set_offsets(visible)
        fig.canvas.draw_idle()

fig.canvas.mpl_connect("button_press_event", on_press)
fig.canvas.mpl_connect("button_release_event", on_release)
fig.canvas.mpl_connect("motion_notify_event", on_motion)

# ============================================================
# 10. Animation (rotation + pulsing)
# ============================================================

angle = 0.0
pulse = 0.0

def animate(frame):
    global angle, pulse
    angle += 0.02
    pulse += 0.05

    pulse_scale = 0.55 + 0.05 * math.sin(pulse)

    m = merkaba_points(
        scale=pulse_scale,
        up_angle=angle,
        down_angle=-angle
    ).cpu().numpy()

    scatter_merkaba.set_offsets(m)
    return scatter_merkaba,

ani = FuncAnimation(fig, animate, interval=30)
plt.show()
