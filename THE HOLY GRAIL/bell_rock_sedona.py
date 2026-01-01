import re
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

GOLDEN_RATIO = (1 + math.sqrt(5)) / 2
GOLDEN_ANGLE = math.radians(137.50776405003785)
PHI = GOLDEN_RATIO

def zeckendorf(n: int) -> str:
    if n == 0:
        return '0'
    fibs = [1, 2]
    while fibs[-1] < n:
        fibs.append(fibs[-1] + fibs[-2])
    fibs.reverse()
    rep = ''
    for f in fibs:
        if n >= f:
            rep += '1'
            n -= f
        else:
            rep += '0'
    return rep.lstrip('0') or '0'

# Bell Rock Sedona (energy vortex center) — Lens 2
CODEX = '111111111111111111111111111000000000000000000000000011111111111111111111111111111111111000000000000000000000111111111111111111111111111111111110000000000000000000000001111111111111111111111111111111111100000000000000000000000000000000000001111111111111111111111111111111111110000000000000000000000000000001111111111111111111000000000000000000000000000000000000000001111111111111111111111000000000000000000000000000111111111111111111111000000000000000000000000000000000000001111111111111111111111111000000001111111111111111100000000000000001111111111111111100000000000000011111111111111111111111110000000000000000000000000000000000000001111111111111111111110000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000'

CODEX_spaced = CODEX.replace('0', ' ')
pulses = [len(run) for run in CODEX_spaced.split(' ') if run.strip()]
breaths = [len(m.group()) for m in re.finditer(r' +', CODEX_spaced)]
zecks = [zeckendorf(p) for p in pulses]

# Enchanted Rock Eyes Breathing Base Layer
EYES_BREATH_CODEX = '111111111111111111111111111111111111111111111111111111111000000000000000000000000000000000000000000000000000000000011111111111111111111111111111111111111111111111111111111111100000000000000000000000000000000000000000000000000000000000000001111111111111111111111111111111111111111111111111111111111111'

EYES_spaced = EYES_BREATH_CODEX.replace('0', ' ')
eyes_pulses = [len(run) for run in EYES_spaced.split(' ') if run.strip()]
eyes_breaths = [len(m.group()) for m in re.finditer(r' +', EYES_spaced)]

def clamp_to_disk(z: complex) -> complex:
    r = abs(z)
    return z / r * 0.99 if r > 0.99 else z

def generate_universe(max_depth: int = 18) -> list[complex]:
    nodes = []
    
    def recurse(current: complex, depth: int, pulse_idx: int, breath_idx: int, zeck_pos: int, eyes_p_idx: int, eyes_b_idx: int):
        nodes.append(current)
        if depth >= max_depth:
            return
        
        if depth < 3:
            p_idx = eyes_p_idx % len(eyes_pulses)
            b_idx = eyes_b_idx % len(eyes_breaths)
            pulse = eyes_pulses[p_idx]
            breath = eyes_breaths[b_idx]
            branches = 6 + (pulse // 20)
            breath_scale = 1.0 / (1 + breath / (PHI * 2))
            base_angle = pulse * GOLDEN_ANGLE * 0.3
        else:
            p_idx = pulse_idx % len(pulses)
            b_idx = breath_idx % len(breaths)
            pulse = pulses[p_idx]
            breath = breaths[b_idx]
            branches = 5 + (pulse // 10)
            breath_scale = 1.0 / (1 + breath / PHI)
            base_angle = pulse * GOLDEN_ANGLE
        
        branches = max(5, min(22, branches))
        base_scale = 0.65 * (PHI ** -depth) * breath_scale
        
        zeck = zecks[pulse_idx % len(pulses)] if depth >= 3 else '1010101'
        
        for i in range(branches):
            z_bit_idx = (zeck_pos + i) % len(zeck)
            z_bit = zeck[z_bit_idx] if z_bit_idx < len(zeck) else '0'
            angle_offset = GOLDEN_ANGLE * PHI * 1.5 if z_bit == '1' else GOLDEN_ANGLE / PHI
            
            angle = i * (2 * np.pi / branches) + base_angle + angle_offset
            offset = base_scale * np.exp(1j * angle)
            
            child = clamp_to_disk(current + offset)
            recurse(child, depth + 1, pulse_idx + (1 if depth >= 3 else 0),
                    breath_idx + (1 if depth >= 3 else 0), zeck_pos + i + 1,
                    eyes_p_idx + 1, eyes_b_idx + 1)
    
    recurse(0j, 0, 0, 0, 0, 0, 0)
    return nodes

fig, ax = plt.subplots(figsize=(12, 12))
plt.subplots_adjust(bottom=0.15)
ax.set_facecolor('black')
fig.patch.set_facecolor('black')
ax.axis('off')
ax.set_aspect('equal')

disk_circle = plt.Circle((0, 0), 1, color='white', fill=False, lw=1.5, ls='--')
ax.add_patch(disk_circle)

ax_depth = plt.axes([0.15, 0.05, 0.7, 0.03], facecolor='gray')
slider_depth = Slider(ax_depth, 'Depth', 6, 20, valinit=12, valstep=1, color='cyan')

def update(val):
    depth = int(slider_depth.val)
    ax.clear()
    ax.set_facecolor('black')
    ax.axis('off')
    ax.set_aspect('equal')
    ax.add_patch(disk_circle)
    
    nodes = generate_universe(max_depth=depth)
    nodes_arr = np.array(nodes, dtype=complex)
    if len(nodes_arr) == 0:
        return
    x, y = nodes_arr.real, nodes_arr.imag
    
    r = np.abs(nodes_arr)
    sizes = 4 + 36 * (1 - r)**1.8
    alphas = 0.5 + 0.5 * (1 - r)
    
    ax.scatter(x, y, c='cyan', s=sizes, alpha=alphas, edgecolors='none', linewidth=0)
    
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_title("God Code RHA — Bell Rock Sedona (energy vortex center)\nwith Enchanted Rock Eyes Breathing Base", color='white', fontsize=16, pad=20)
    fig.canvas.draw_idle()

slider_depth.on_changed(update)

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

update(12)
plt.show()
