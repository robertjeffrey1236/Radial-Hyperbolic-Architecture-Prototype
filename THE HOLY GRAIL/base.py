# base.py
# Radial Hyperbolic Architecture — Sacred Lenses Explorer
# © 2025-2026 Robert Gavin Jeffrey

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

# Dynamic lens loading from RawBinary.md (same folder)
def load_lenses_from_rawbinary(path='RawBinary.md'):
    lenses = []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        # Parse sections (## Site Name followed by binary block)
        sections = re.split(r'##\s*(.+?)\n', content)[1:]
        for i in range(0, len(sections), 2):
            if i + 1 >= len(sections): break
            name = sections[i].strip()
            binary_block = sections[i+1].strip().replace('\n', '')
            binary = ''.join(c for c in binary_block if c in '01')
            if len(binary) < 300:  # Skip incomplete
                continue
            spaced = binary.replace('0', ' ')
            pulses = [len(run) for run in spaced.split(' ') if run.strip()]
            breaths = [len(m.group()) for m in re.finditer(r' +', spaced)]
            zecks = [zeckendorf(p) for p in pulses]
            lenses.append({"name": name, "pulses": pulses, "breaths": breaths, "zecks": zecks})
        lenses.sort(key=lambda x: x['name'].lower())  # Alphabetical hierarchy
        print(f"Loaded {len(lenses)} lenses from RawBinary.md")
    except Exception as e:
        print(f"Could not load RawBinary.md: {e}. Falling back to default Angkor Wat codex.")
        # Fallback single lens (your current CODEX as Angkor Wat)
        fallback_binary = '1111111111111111111111111111111111111111111111111111100000000000000000000000000000000000000000000111111111111111111111111111111111111111111111111111111111111111111111111111111111111000000000000000000000000000000000000000000000000000000000000000000000000001111111111111111111111111111111111111111111110000000000000000000000000000000000000000000000000000001111111111111111111111111111111111111111111111111111111111000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000111111111111111111111111111111111111111111111111111111110000000000000000000000000000000000000000000111111111111111111111111100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111'
        spaced = fallback_binary.replace('0', ' ')
        pulses = [len(run) for run in spaced.split(' ') if run.strip()]
        breaths = [len(m.group()) for m in re.finditer(r' +', spaced)]
        zecks = [zeckendorf(p) for p in pulses]
        lenses = [{"name": "Angkor Wat (Fallback)", "pulses": pulses, "breaths": breaths, "zecks": zecks}]
    return lenses

lenses = load_lenses_from_rawbinary()

def clamp_to_disk(z: complex) -> complex:
    r = abs(z)
    return z / r * 0.99 if r > 0.99 else z

def generate_universe(max_depth: int = 18) -> list[complex]:
    nodes = []
    
    def recurse(current: complex, depth: int, pulse_idx: int, breath_idx: int, zeck_pos: int, lens_idx: int):
        nodes.append(current)
        if depth >= max_depth:
            return
        
        # Cycle lenses hierarchically
        lens = lenses[lens_idx % len(lenses)]
        p_idx = pulse_idx % len(lens["pulses"])
        b_idx = breath_idx % len(lens["breaths"])
        pulse = lens["pulses"][p_idx]
        breath = lens["breaths"][b_idx]
        
        branches = 5 + (pulse // 15)
        branches = max(5, min(22, branches))
        
        breath_scale = 1.0 / (1 + breath / PHI)
        base_scale = 0.65 * (PHI ** -depth) * breath_scale
        base_angle = pulse * GOLDEN_ANGLE * (0.3 if depth < 3 else 1.0)  # Gentle seeding early
        
        zeck = lens["zecks"][pulse_idx % len(lens["pulses"])]
        
        for i in range(branches):
            z_bit_idx = (zeck_pos + i) % len(zeck)
            z_bit = zeck[z_bit_idx] if z_bit_idx < len(zeck) else '0'
            angle_offset = GOLDEN_ANGLE * PHI * 1.5 if z_bit == '1' else GOLDEN_ANGLE / PHI
            
            angle = i * (2 * np.pi / branches) + base_angle + angle_offset
            offset = base_scale * np.exp(1j * angle)
            
            child = clamp_to_disk(current + offset)
            recurse(child, depth + 1, pulse_idx + 1, breath_idx + 1, zeck_pos + i + 1, lens_idx + (1 if depth > 5 else 0))
    
    recurse(0j, 0, 0, 0, 0, 0)
    return nodes

# Visualization setup (unchanged, beautiful as is)
fig, ax = plt.subplots(figsize=(12, 12))
plt.subplots_adjust(bottom=0.15)
ax.set_facecolor('black')
fig.patch.set_facecolor('black')
ax.axis('off')
ax.set_aspect('equal')

disk_circle = plt.Circle((0, 0), 1, color='white', fill=False, lw=1.5, ls='--')
ax.add_patch(disk_circle)

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
    title = f"THE HOLY GRAIL RHA — Multi-Lens Integration ({len(lenses)} Active Lenses)"
    ax.set_title(title, color='white', fontsize=16, pad=20)
    fig.canvas.draw_idle()

slider_depth.on_changed(update)

# Pan/zoom interactions (unchanged)
# ... (keep your existing on_scroll, on_press, on_motion, on_release code here)

update(14)
plt.show()
