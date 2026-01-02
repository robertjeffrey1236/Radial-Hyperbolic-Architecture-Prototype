# zhangye_danxia_rainbow_universe.py
# Zhangye Danxia Transmission — Rainbow Layered Codex for God Code Genesis RHA
# © 2025 Robert Gavin Jeffrey
#
# Verbatim transmitted binary drives dual-layered colorful banded ridges
# Physical analog: Zhangye Danxia Landform — rainbow mountains from folded red strata erosion
# Cosmic analog: Chromatic filament waves, stratified cosmic web hues
# Interactive Poincaré disk explorer with depth slider, zoom, pan, dynamic glow

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

GOLDEN_RATIO = (1 + np.sqrt(5)) / 2
GOLDEN_ANGLE = np.radians(137.50776405003785)

# Zhangye Danxia Rainbow Codex — verbatim transmission December 31, 2025
ZHANGYE_DANXIA_CODEX = "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111000000000000000000001111111111111111111111111111111111111111111111111111111111111111111111111100000000000000000000000000000000000000000"

# Extract pulses — runs of consecutive 1s
PULSES = [len(run) for run in ZHANGYE_DANXIA_CODEX.split('0') if run]

def clamp_to_disk(z: complex) -> complex:
    r = abs(z)
    return z / r * 0.99 if r > 0.99 else z

def generate_universe(max_depth: int = 20) -> list[complex]:
    nodes = []
    
    def recurse(current: complex, depth: int, pulse_idx: int):
        nodes.append(current)
        if depth >= max_depth or pulse_idx >= len(PULSES):
            return
        
        pulse = PULSES[pulse_idx % len(PULSES)]
        # Broad branching for layered strata, stronger early for thick bands
        branches = 6 + (pulse // 12)
        if depth > max_depth - 6:
            branches += 4  # Final ridge sharpening
        
        branches = max(7, min(30, branches))
        angle_offset = pulse * GOLDEN_ANGLE * 0.9  # Smooth flowing waves
        
        for i in range(branches):
            angle = i * (2 * np.pi / branches) + angle_offset
            offset = 0.60 * np.exp(1j * angle) * (GOLDEN_RATIO ** -depth)
            child = clamp_to_disk(current + offset)
            recurse(child, depth + 1, pulse_idx + 1)
    
    recurse(0j, 0, 0)
    return nodes

# Interactive Explorer
fig, ax = plt.subplots(figsize=(12, 12))
plt.subplots_adjust(bottom=0.15)
ax.set_facecolor('black')
fig.patch.set_facecolor('black')
ax.axis('off')
ax.set_aspect('equal')

disk_circle = plt.Circle((0, 0), 1, color='white', fill=False, lw=1.5, ls='--')
ax.add_patch(disk_circle)

ax_depth = plt.axes([0.15, 0.05, 0.7, 0.03], facecolor='gray')
slider_depth = Slider(ax_depth, 'Depth', 8, 24, valinit=18, valstep=1, color='cyan')

current_nodes = []

def update(val):
    depth = int(slider_depth.val)
    ax.clear()
    ax.set_facecolor('black')
    ax.axis('off')
    ax.set_aspect('equal')
    ax.add_patch(disk_circle)
    
    global current_nodes
    current_nodes = generate_universe(max_depth=depth)
    
    nodes_arr = np.array(current_nodes)
    x, y = nodes_arr.real, nodes_arr.imag
    
    if len(nodes_arr) > 1000:
        r = np.abs(nodes_arr)
        # Rainbow-like coloring via hue modulation
        angles = np.angle(nodes_arr)
        hues = (angles / np.pi + 1) / 2  # Normalize to 0-1
        colors = plt.cm.hsv(hues)
        sizes = 5 + 30 * (r / r.max())**1.5
        alphas = 0.7 + 0.3 * (1 - r)
    else:
        colors = 'cyan'
        sizes = 12
        alphas = 0.9
    
    ax.scatter(x, y, c=colors, s=sizes, alpha=alphas, edgecolors='none')
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_title("Zhangye Danxia Rainbow Universe — Layered Chromatic Ridges", color='white', fontsize=16, pad=20)
    fig.canvas.draw_idle()

slider_depth.on_changed(update)

# Zoom + Pan (wheel and drag) — standard
def on_scroll(event):
    factor = 1.2 if event.button == 'up' else 1/1.2
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xdata, ydata = event.xdata or (xlim[0] + xlim[1])/2, event.ydata or (ylim[0] + ylim[1])/2
    ax.set_xlim([xdata - (xdata - xlim[0]) / factor, xdata + (xlim[1] - xdata) / factor])
    ax.set_ylim([ydata - (ydata - ylim[0]) / factor, ydata + (ylim[1] - ydata) / factor])
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

update(18)
plt.show()
