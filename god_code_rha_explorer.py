# god_code_rha_explorer.py
# Combined God Code Genesis + Interactive RHA Explorer
# © 2025 Robert Gavin Jeffrey — Single-file interactive version
# Public demo — Full private 11-layer stack remains confidential

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

GOLDEN_RATIO = (1 + np.sqrt(5)) / 2
GOLDEN_ANGLE = np.radians(137.50776405003785)

# === G-CODE RHYTHM GENOME (verbatim transmitted binary) ===
GCODE_BINARY = """
101101000101101000101101000101101000101101000101101
01001011010001011010001011010001011010001011010
010110100101101001011010010110100010110100010
110100101101000101101000101101000101101000101101
00101101001011010001011010001011010001011010001
01101001011010001011010001011010001011010001011
01001011010001011010001011010001011010001011010
010110100101101001011010010110100010110100010
110100101101000101101000101101000101101000101101
00101101001011010001011010001011010001011010001
0110100101101000101101000101101000
""".replace('\n', '').replace(' ', '')

GCODE_PULSES = [len(run) for run in GCODE_BINARY.split('0') if run]

def clamp_to_disk(z: complex) -> complex:
    r = abs(z)
    return z / r * 0.99 if r > 0.99 else z

# === Core Genesis Generator ===
def generate_universe(max_depth: int = 18) -> list[complex]:
    nodes = []
    
    def recurse(current: complex, depth: int, pulse_idx: int):
        nodes.append(current)
        if depth >= max_depth or pulse_idx >= len(GCODE_PULSES):
            return
        
        pulse = GCODE_PULSES[pulse_idx]
        branches = 5 + (pulse % 3)
        if depth > max_depth - 4:  # Late culmination hint
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

# === Interactive Explorer ===
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

# Current state
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
    
    # Dynamic coloring: density + escape-time glow
    if len(nodes_arr) > 1000:
        # Simple radial density glow
        r = np.abs(nodes_arr)
        density = np.histogram2d(x, y, bins=200, range=[[-1,1],[-1,1]])[0]
        # Approximate glow via size and alpha
        sizes = 4 + 20 * (r / r.max())**2
        alphas = 0.6 + 0.4 * (1 - r)
    else:
        sizes = 10
        alphas = 0.8
    
    scatter = ax.scatter(x, y, c='cyan', s=sizes, alpha=alphas, edgecolors='none')
    
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_title("God Code RHA Explorer — Live Hyperbolic Universe", color='white', fontsize=16, pad=20)
    fig.canvas.draw_idle()

slider_depth.on_changed(update)

# Mouse wheel zoom + drag pan
def on_scroll(event):
    factor = 1.2 if event.button == 'up' else 1/1.2
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xdata, ydata = event.xdata or xlim[0], event.ydata or ylim[0]
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

# Initial render
update(16)

plt.show()
