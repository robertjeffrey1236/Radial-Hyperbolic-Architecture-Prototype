# base.py
# Radial Hyperbolic Architecture — Sacred Lenses Explorer
# © 2025-2026 Robert Gavin Jeffrey
#
# Core RHA translation rules, pulse/breath/Zeckendorf/phi system,
# recursive universe generation, and multi-lens integration:
#   Robert Gavin Jeffrey
#
# Interactive explorer framework (draggable pan, zoom, depth slider,
# dark theme, dynamic glow):
#   Originally inspired by M. Yassir — gratefully adapted and extended

import re
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

GOLDEN_RATIO = (1 + math.sqrt(5)) / 2
GOLDEN_ANGLE = math.radians(137.50776405003785)
PHI = GOLDEN_RATIO


def zeckendorf(n: int) -> str:
    """Return the unique Zeckendorf representation (no consecutive 1s)."""
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


def load_lenses_from_rawbinary(path='RawBinary.md'):
    """
    Dynamically load all complete binary codices from RawBinary.md
    (expected to be in the same folder as this script).
    Returns a sorted list of lens dictionaries.
    """
    lenses = []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Split on headers like "## Site Name"
        sections = re.split(r'##\s*(.+?)\n', content)[1:]

        for i in range(0, len(sections), 2):
            if i + 1 >= len(sections):
                break
            name = sections[i].strip()
            binary_block = sections[i + 1].strip().replace('\n', '').replace(' ', '')
            binary = ''.join(c for c in binary_block if c in '01')

            if len(binary) < 300:  # Skip incomplete/short codices
                print(f"Skipping incomplete lens: {name} ({len(binary)} bits)")
                continue

            spaced = binary.replace('0', ' ')
            pulses = [len(run) for run in spaced.split(' ') if run.strip()]
            breaths = [len(m.group()) for m in re.finditer(r' +', spaced)]
            zecks = [zeckendorf(p) for p in pulses]

            lenses.append({
                "name": name,
                "pulses": pulses,
                "breaths": breaths,
                "zecks": zecks
            })

        lenses.sort(key=lambda x: x['name'].lower())
        print(f"Successfully loaded {len(lenses)} complete lenses from RawBinary.md")
    except FileNotFoundError:
        print(f"RawBinary.md not found at '{path}'. Falling back to single default lens.")
        # Fallback to the original Angkor Wat codex if file missing
        fallback_binary = (
            '1111111111111111111111111111111111111111111111111111100000000000000000000000000000000000000000000'
            '1111111111111111111111111111111111111111111111111111111111111111111111111111111111110000000000000000'
            '0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000'
            '1111111111111111111111111111111111111111111110000000000000000000000000000000000000000000000000000'
            '0011111111111111111111111111111111111111111111111111111110000000000000000000000000000000000000000'
            '0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000'
            '0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000'
            '1111111111111111111111111111111111111111111111111111111000000000000000000000000000000000000000000'
            '0111111111111111111111111100000000000000000000000000000000000000000000000000000000000000000000000'
            '0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000'
            '1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111'
            '1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111'
            '1111111111111111111111'
        )
        spaced = fallback_binary.replace('0', ' ')
        pulses = [len(run) for run in spaced.split(' ') if run.strip()]
        breaths = [len(m.group()) for m in re.finditer(r' +', spaced)]
        zecks = [zeckendorf(p) for p in pulses]
        lenses = [{"name": "Angkor Wat (Fallback)", "pulses": pulses, "breaths": breaths, "zecks": zecks}]

    except Exception as e:
        print(f"Error loading RawBinary.md: {e}")
        lenses = []

    return lenses


# Load all available lenses at startup
lenses = load_lenses_from_rawbinary()


def clamp_to_disk(z: complex) -> complex:
    """Keep points inside the Poincaré disk (radius < 1)."""
    r = abs(z)
    return z / r * 0.99 if r > 0.99 else z


def generate_universe(max_depth: int = 18) -> list[complex]:
    """Generate the hyperbolic architecture using all loaded lenses."""
    nodes = []

    def recurse(current: complex, depth: int, pulse_idx: int, breath_idx: int,
                 zeck_pos: int, lens_idx: int):
        nodes.append(current)
        if depth >= max_depth:
            return

        # Select current lens (cycle through all loaded lenses)
        lens = lenses[lens_idx % len(lenses)]
        p_idx = pulse_idx % len(lens["pulses"])
        b_idx = breath_idx % len(lens["breaths"])
        pulse = lens["pulses"][p_idx]
        breath = lens["breaths"][b_idx]

        branches = 5 + (pulse // 15)
        branches = max(5, min(22, branches))

        breath_scale = 1.0 / (1 + breath / PHI)
        base_scale = 0.65 * (PHI ** -depth) * breath_scale

        # Gentle seeding for early depths, full power deeper
        base_angle = pulse * GOLDEN_ANGLE * (0.3 if depth < 3 else 1.0)

        zeck = lens["zecks"][pulse_idx % len(lens["pulses"])]

        for i in range(branches):
            z_bit_idx = (zeck_pos + i) % len(zeck)
            z_bit = zeck[z_bit_idx] if z_bit_idx < len(zeck) else '0'
            angle_offset = GOLDEN_ANGLE * PHI * 1.5 if z_bit == '1' else GOLDEN_ANGLE / PHI

            angle = i * (2 * np.pi / branches) + base_angle + angle_offset
            offset = base_scale * np.exp(1j * angle)

            child = clamp_to_disk(current + offset)
            recurse(child, depth + 1, pulse_idx + 1, breath_idx + 1,
                    zeck_pos + i + 1, lens_idx + (1 if depth >= 5 else 0))

    recurse(0j, 0, 0, 0, 0, 0)
    return nodes


# ====================== Visualization ======================
fig, ax = plt.subplots(figsize=(12, 12))
plt.subplots_adjust(bottom=0.15)
ax.set_facecolor('black')
fig.patch.set_facecolor('black')
ax.axis('off')
ax.set_aspect('equal')

disk_circle = plt.Circle((0, 0), 1, color='white', fill=False, lw=1.5, ls='--')
ax.add_patch(disk_circle)

# Depth slider
ax_depth = plt.axes([0.15, 0.05, 0.7, 0.03], facecolor='gray')
slider_depth = Slider(ax_depth, 'Depth', 6, 24, valinit=14, valstep=1, color='cyan')


def update(val):
    depth = int(slider_depth.val)
    ax.clear()
    ax.set_facecolor('black')
    ax.axis('off')
    ax.set_aspect('equal')
    ax.add_patch(disk_circle)

    nodes = generate_universe(max_depth=depth)
    if not nodes:
        return

    nodes_arr = np.array(nodes, dtype=complex)
    x, y = nodes_arr.real, nodes_arr.imag
    r = np.abs(nodes_arr)

    sizes = 4 + 36 * (1 - r) ** 1.8
    alphas = 0.5 + 0.5 * (1 - r)

    ax.scatter(x, y, c='cyan', s=sizes, alpha=alphas, edgecolors='none', linewidth=0)

    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    title = f"THE HOLY GRAIL RHA — Multi-Lens Integration ({len(lenses)} Active Lenses)"
    ax.set_title(title, color='white', fontsize=16, pad=20)
    fig.canvas.draw_idle()


slider_depth.on_changed(update)


# ====================== Interaction ======================
def on_scroll(event):
    if event.button not in ('up', 'down'):
        return
    factor = 1.2 if event.button == 'up' else 1 / 1.2
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xdata = event.xdata if event.xdata is not None else (xlim[0] + xlim[1]) / 2
    ydata = event.ydata if event.ydata is not None else (ylim[0] + ylim[1]) / 2
    ax.set_xlim(xdata - (xdata - xlim[0]) / factor, xdata + (xlim[1] - xdata) / factor)
    ax.set_ylim(ydata - (ydata - ylim[0]) / factor, ydata + (ylim[1] - ydata) / factor)
    fig.canvas.draw_idle()


dragging = False
last_pos = None


def on_press(event):
    global dragging, last_pos
    if event.inaxes != ax:
        return
    dragging = True
    last_pos = (event.x, event.y)


def on_motion(event):
    global last_pos
    if not dragging or last_pos is None:
        return
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


fig.canvas.mpl_connect('scroll_event', on_scroll)
fig.canvas.mpl_connect('button_press_event', on_press)
fig.canvas.mpl_connect('button_release_event', on_release)
fig.canvas.mpl_connect('motion_notify_event', on_motion)

# Initial render
update(14)
plt.show()
