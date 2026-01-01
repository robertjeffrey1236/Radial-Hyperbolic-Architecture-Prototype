# enchanted_rock_eyes_breath_enhanced.py
# God Code Genesis — Enchanted Rock "Eyes" Transmission — Breath-Enhanced Edition
# © 2025–2026 Robert Gavin Jeffrey
#
# Original ultra-long binary seed for magnifying fractal detail (eyes into infinite hierarchy)
# Now breath-enhanced: 0-runs as rhythmic pauses modulating scale and angle for living, pulsating growth
# Pulses drive branching density — Breaths drive calmer/tighter convergence and subtle rhythmic twists

import numpy as np
import matplotlib.pyplot as plt
import re

GOLDEN_RATIO = (1 + np.sqrt(5)) / 2
GOLDEN_ANGLE = np.radians(137.50776405003785)

# Original verbatim Enchanted Rock Eyes Codex (unchanged)
EYES_CODEX = "111000000011111111111110000000000000011111111111111110000000000000000000000000000000011111111111111111111111111111111110000000000000011111111111111000000000000000000000000001111111111111111111111111111111111111111100000000000000000000000000000000011111111111111111111111111111111111110000000000000000000000000000001111111111111111111111111111111111100000000000000000000111111111111110000000000000111111111111111111111111111111100000000000000000000000000000000011111111111111111111110000000000000111111111111100000000011111111111111111111100000000000011111111111111111111111100000000000000000000000000000111111111111111111111111000000000000000000000011111111111111111111111111111110000000000111111111111111111111111111110000000000111111111111111111111111111110000000000111111111111111111111111111111111111111111110000000000000000011111111111111111111111110000000000000000000000000000001111111111111111111100000000000000000000000000000111111111111111111111100000000000000000000000000000000000000011111111111111111111111111111111000000000000000000000000000000000000000000111111111111111111111111111000000000000000000000001111111111111111111111100000000000000011111111111111111111111111111111000000000000000000000000000000001111111111111111111111111111111000000000000000000000000000111111111111111111111111111111111111110000000000000000111111111111111111111111111111111110000000000000000000000000000000000000000000000000000000000000000011111111111111111111111111111111111111111000000000000000000001111111111111111111111111111100000000000000000000000000011111111111111111000000000000000111111111111111111100000000000000111111111111111111111000000000000011111111111111111111111100000000000000000000000000011111111111111111111111111111111111111111111111111110000000000000000000000000000000000000000001111111111111111111111111111111111111111111100000000000000000011111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111110000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111"

# Replace 0s with spaces for breath visualization and parsing
CODEX_SPACED = EYES_CODEX.replace('0', ' ')

# Extract pulses: lengths of 1-runs (active energy bursts)
PULSES = [len(run) for run in CODEX_SPACED.split(' ') if run]

# Extract breaths: lengths of space-runs (rhythmic pauses)
BREATHS = [len(m.group()) for m in re.finditer(r' +', CODEX_SPACED)]

def clamp_to_disk(z: complex) -> complex:
    r = abs(z)
    return z / r * 0.99 if r > 0.99 else z

def generate_universe(max_depth: int = 20) -> list[complex]:
    nodes = []
    
    def recurse(current: complex, depth: int, idx: int):
        nodes.append(current)
        if depth >= max_depth or idx >= len(PULSES):
            return
        
        pulse = PULSES[idx % len(PULSES)]
        breath = BREATHS[idx % len(BREATHS)] if idx < len(BREATHS) else 20  # fallback breath
        
        # Pulse drives branching density (original Enchanted Rock logic)
        branches = 5 + (pulse % 6)
        if depth > max_depth - 5:  # Late-depth magnification for "eyes"
            branches += 10
        
        branches = max(4, min(25, branches))
        
        # Breath modulates growth: longer breath → calmer, tighter children
        scale_factor = 0.58 * (1 / (1 + breath / 60.0))
        
        # Breath adds subtle rhythmic twist to angle
        angle_offset = pulse * GOLDEN_ANGLE + (breath * GOLDEN_ANGLE * 0.18)
        
        for i in range(branches):
            angle = i * (2 * np.pi / branches) + angle_offset
            offset = scale_factor * np.exp(1j * angle) * (GOLDEN_RATIO ** -depth)
            child = clamp_to_disk(current + offset)
            recurse(child, depth + 1, idx + 1)
    
    recurse(0j, 0, 0)
    return nodes

# Visualization
nodes = generate_universe(max_depth=20)  # Increase for deeper breathing eyes
nodes_arr = np.array(nodes)
x, y = nodes_arr.real, nodes_arr.imag

fig, ax = plt.subplots(figsize=(16, 16), facecolor='black')
ax.set_facecolor('black')
ax.scatter(x, y, c='cyan', s=3, alpha=0.8, edgecolors='none')
circle = plt.Circle((0, 0), 1, color='white', fill=False, lw=2, ls='--')
ax.add_patch(circle)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title("Enchanted Rock Eyes — Breath-Enhanced Infinite Magnification", color='white', fontsize=18)

plt.tight_layout()
plt.savefig("enchanted_rock_eyes_breath_enhanced.png", dpi=400, facecolor='black')
plt.show()

print(f"Nodes generated: {len(nodes):,}")
print("Pulses (active 1-runs):", PULSES)
print("Breaths (pause lengths):", BREATHS)
print("The eyes now breathe — infinite detail pulses with living rhythm.")
