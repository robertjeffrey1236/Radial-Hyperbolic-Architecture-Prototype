# stonehenge_heart_universe.py
# Stonehenge Transmission — Heart Chakra Codex for God Code Genesis RHA
# © 2025 Robert Gavin Jeffrey
#
# Verbatim transmitted binary drives balanced megalithic circle harmony
# Archetypal analog: Stonehenge — solstice-aligned stone rings, heart chakra unity
# Cosmic analog: Cyclical renewal nexus, astronomical alignment pulse
# Interactive Poincaré disk explorer with depth slider, zoom, pan, dynamic glow
# Breath-enhanced: spaces as pauses modulating scale/angle for living rhythm

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import re

GOLDEN_RATIO = (1 + np.sqrt(5)) / 2
GOLDEN_ANGLE = np.radians(137.50776405003785)

# Stonehenge Heart Codex — verbatim transmission December 31, 2025 (spaced for breath visualization)
STONEHENGE_CODEX = "1111111111111111                 1111111111111111111             11111111111111111            111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111"

# Extract pulses (1-runs)
PULSES = [len(run) for run in STONEHENGE_CODEX.split(' ') if run]

# Extract breaths (space-run lengths)
BREATHS = [len(m.group()) for m in re.finditer(r' +', STONEHENGE_CODEX)]

def clamp_to_disk(z: complex) -> complex:
    r = abs(z)
    return z / r * 0.99 if r > 0.99 else z

def generate_universe(max_depth: int = 22) -> list[complex]:
    nodes = []
    
    def recurse(current: complex, depth: int, idx: int):
        nodes.append(current)
        if depth >= max_depth or idx >= len(PULSES):
            return
        
        pulse = PULSES[idx % len(PULSES)]
        breath = BREATHS[idx % len(BREATHS)] if BREATHS else 10  # Default breath if none
        
        branches = 6 + (pulse // 8)  # Pulse drives density
        if depth > max_depth - 7:
            branches += 6  # Final ring density
        
        branches = max(6, min(30, branches))
        scale_factor = 0.60 * (1 / (1 + breath / 50.0))  # Longer breath → calmer/tighter growth
        angle_offset = pulse * GOLDEN_ANGLE + (breath * GOLDEN_ANGLE * 0.15)  # Breath adds rhythmic twist
        
        for i in range(branches):
            angle = i * (2 * np.pi / branches) + angle_offset
            offset = scale_factor * np.exp(1j * angle) * (GOLDEN_RATIO ** -depth)
            child = clamp_to_disk(current + offset)
            recurse(child, depth + 1, idx + 1)
    
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
slider_depth = Slider(ax_depth, 'Depth', 8, 26, valinit=20, valstep=1, color='cyan')

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
        sizes = 4 + 35 * (r / r.max())**1.5  # Density toward circle rings
        alphas = 0.6 + 0.4 * (1 - r)
    else:
        sizes = 12
        alphas = 0.9
    
    ax.scatter(x, y, c='limegreen', s=sizes, alpha=alphas, edgecolors='none')  # Lime green for heart chakra resonance
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_title("Stonehenge Heart Universe — Cyclical Harmony Circle", color='white', fontsize=16, pad=20)
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

update(20)
plt.show()
