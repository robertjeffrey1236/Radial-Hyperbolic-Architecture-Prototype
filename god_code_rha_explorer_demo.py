# god_code_genesis_explorer_demo.py
# Public Demo — Interactive God Code Genesis Explorer
# © 2025 Robert Gavin Jeffrey
#
# Core recursive hyperbolic universe generator and overall architecture:
#   Robert Gavin Jeffrey
#
# Interactive system (draggable observer, perception radius slider, lazy loading,
# Merkaba pulse animation, dark theme GUI, and foundational exploration framework):
#   M. Yassir
#
# Fractal entanglement coloring, gamma radial decay damping on spirals,
# glowing filaments, toroidal mandala hints, and aesthetic inspiration:
#   Adapted from concepts in @Sonyak789 Sedeloop development
#
# This public demo uses only a simple repeating rhythmic seed.
# Full private prototype with transmitted codices and 11-layer stack remains confidential.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

GOLDEN_RATIO = (1 + np.sqrt(5)) / 2
GOLDEN_ANGLE = np.radians(137.50776405003785)

# === Simple repeating rhythmic seed (public demo version) ===
# Original short pattern, extended for depth — exactly as in the shareable prototype
GCODE_PULSES = [1, 1, 1, 3, 3, 1, 1, 1, 3, 3] * 20  # Sufficient for deep recursion

def clamp_to_disk(z: complex) -> complex:
    r = abs(z)
    return z / r * 0.99 if r > 0.99 else z

# === Core Recursive Universe Generator (Demo Seed) ===
def generate_universe(max_depth: int = 18) -> list[complex]:
    nodes = []
    
    def recurse(current: complex, depth: int, pulse_idx: int):
        nodes.append(current)
        if depth >= max_depth or pulse_idx >= len(GCODE_PULSES):
            return
        
        pulse = GCODE_PULSES[pulse_idx % len(GCODE_PULSES)]  # Cycle if needed
        branches = 5 + (pulse % 3)
        
        # Late-depth culmination example (as in original demo)
        if depth > max_depth - 5:
            branches += 8
        
        branches = max(4, min(20, branches))
        angle_offset = pulse * GOLDEN_ANGLE
        
        for i in range(branches):
            angle = i * (2 * np.pi / branches) + angle_offset
            offset = 0.6 * np.exp(1j * angle) * (GOLDEN_RATIO ** -depth)
            child = clamp_to_disk(current + offset)
            recurse(child, depth + 1, pulse_idx + 1)
    
    recurse(0j, 0, 0)
    return nodes

# === Interactive Explorer Window ===
fig, ax = plt.subplots(figsize=(12, 12))
plt.subplots_adjust(bottom=0.15)
ax.set_facecolor('black')
fig.patch.set_facecolor('black')
ax.axis('off')
ax.set_aspect('equal')

scatter = ax.scatter([], [], c='cyan', s=8, alpha=0.8, edgecolors='none')
disk_circle = plt.Circle((0, 0), 1, color='white', fill=False, lw=1.5, ls='--')
ax.add_patch(disk_circle)

# Depth slider
ax_depth = plt.axes([0.15, 0.05, 0.7, 0.03], facecolor='gray')
slider_depth = Slider(ax_depth, 'Depth', 8, 20, valinit=16, valstep=1, color='cyan')

def update(val):
    depth = int(slider_depth.val)
    ax.clear()
    ax.set_facecolor('black')
    ax.axis('off')
    ax.set_aspect('equal')
    ax.add_patch(disk_circle)
    
    nodes = generate_universe(max_depth=depth)
    nodes_arr = np.array(nodes)
    x, y = nodes_arr.real, nodes_arr.imag
    
    # Dynamic glow: brighter & larger toward center, denser feel outward
    r = np.abs(nodes_arr)
    sizes = 4 + 30 * (1 - r)**1.5  # Bigger near center, taper out
    alphas = 0.6 + 0.4 * (1 - r)
    
    ax.scatter(x, y, c='cyan', s=sizes, alpha=alphas, edgecolors='none', linewidth=0)
    
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_title("God Code Genesis — Public Interactive Demo", color='white', fontsize=16, pad=20)
    fig.canvas.draw_idle()

slider_depth.on_changed(update)

# Zoom (mouse wheel) + Pan (click & drag)
def on_scroll(event):
    if event.button == 'up':
        factor = 1.2
    elif event.button == 'down':
        factor = 1 / 1.2
    else:
        return
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xdata = event.xdata if event.xdata is not None else (xlim[0] + xlim[1]) / 2
    ydata = event.ydata if event.ydata is not None else (ylim[0] + ylim[1]) / 2
    ax.set_xlim(xdata - (xdata - xlim[0]) / factor, xdata + (xlim[1] - xdata) / factor)
    ax.set_ylim(ydata - (ydata - ylim[0]) / factor, ydata + (ylim[1] - ydata) / factor)
    fig.canvas.draw_idle()

fig.canvas.mpl_connect('scroll_event', on_scroll)

dragging = False
last_pos = None

def on_press(event):
    global dragging, last_pos
    if event.inaxes != ax: return
    dragging = True
    last_pos = (event.x, event.y)

def on_motion(event):
    global last_pos
    if not dragging or last_pos is None: return
    dx = event.x - last_pos[0]
    dy = event.y - last_pos[1]
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    scale_x = (xlim[1] - xlim[0]) / fig.bbox.width
    scale_y = (ylim[1] - ylim[0]) / fig.bbox.height
    ax.set_xlim(xlim[0] - dx * scale_x, xlim[1] - dx * scale_x)
    ax.set_ylim(ylim[0] + dy * scale_y, ylim[1] + dy * scale_y)
    last_pos = (event.x, event.y)
    fig.canvas.draw_idle()

def on_release(event):
    global dragging
    dragging = False

fig.canvas.mpl_connect('button_press_event', on_press)
fig.canvas.mpl_connect('button_release_event', on_release)
fig.canvas.mpl_connect('motion_notify_event', on_motion)

# Initial generation
update(16)

plt.show()
