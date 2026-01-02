# universe_hyperbolic_map.py
# The Grand Map: Universe as Hyperbolic Puzzle
# Center = Source • Radiating arms = domains • Recursive fragments = puzzle pieces

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from core.geometry import golden_spiral_points, poincare_disk_project, GOLDEN_ANGLE

fig, ax = plt.subplots(figsize=(20, 20))
ax.set_facecolor('black')
ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1.1, 1.1)
ax.axis('off')

# Poincaré boundary
boundary = plt.Circle((0, 0), 1, color='gold', fill=False, lw=6, alpha=0.8)
ax.add_patch(boundary)

# Center — The Source
ax.scatter(0, 0, c='white', s=1000, alpha=1.0, edgecolor='gold', linewidth=6, zorder=20)
ax.text(0, 0, 'SOURCE\nPrime Intelligence\nUnity Field', color='gold', fontsize=14, ha='center', va='center', fontweight='bold')

# Major Domains (golden-spiral arms from center)
domains = [
    {'name': 'Consciousness', 'angle': 0, 'color': 'violet', 'status': 'Advanced', 'pieces': ['Observer', 'Coherence', 'Holographic Mind', 'Ego-Dissolution', 'Kundalini']},
    {'name': 'Physics', 'angle': 0.8, 'color': 'cyan', 'status': 'Strong', 'pieces': ['Higgs Field', 'Quantum Eraser', 'Double-Slit', 'Proton Gradient']},
    {'name': 'Biology', 'angle': 1.6, 'color': 'green', 'status': 'Growing', 'pieces': ['Proteins', 'Water', 'ATP Synthase', 'Hydration Shells', 'Membranes']},
    {'name': 'Chemistry', 'angle': 2.4, 'color': 'yellow', 'status': 'Building', 'pieces': ['Molecules', 'Reactions', 'Catalysts']},
    {'name': 'Energy', 'angle': 3.2, 'color': 'orange', 'status': 'Active', 'pieces': ['Chakras', 'Meridians', 'Prana Flow', 'Aura']},
    {'name': 'Mathematics', 'angle': 4.0, 'color': 'blue', 'status': 'Core', 'pieces': ['Golden Ratio', 'Hyperbolic Geometry', 'Fractals', 'Recursion']},
    {'name': 'Information', 'angle': 4.8, 'color': 'magenta', 'status': 'Emerging', 'pieces': ['Language Codex', 'Memory', 'Holographic Storage']},
    {'name': 'Cosmology', 'angle': 5.6, 'color': 'white', 'status': 'Conceptual', 'pieces': ['Big Bang', 'Black Holes', 'Multiverse', 'Tesseract Echo']},
]

# Draw domain arms and fragments
arm_lines = []
fragment_texts = []
for domain in domains:
    # Main arm — golden spiral from center
    arm = golden_spiral_points(50, dim=2, radius_scale=0.8)
    rot = np.array([[np.cos(domain['angle']), -np.sin(domain['angle'])],
                   [np.sin(domain['angle']), np.cos(domain['angle'])]])
    arm = arm @ rot
    line = ax.plot(arm[:, 0], arm[:, 1], c=domain['color'], lw=4, alpha=0.7)[0]
    arm_lines.append(line)
    
    # Domain label at end
    end_pos = arm[-1]
    ax.text(end_pos[0], end_pos[1], domain['name'], color=domain['color'], fontsize=14, ha='center', fontweight='bold')
    
    # Fragment puzzle pieces along arm
    for i, piece in enumerate(domain['pieces']):
        frac_pos = arm[int(len(arm) * (i+1) / (len(domain['pieces'])+1))]
        status_color = 'lime' if 'Advanced' in domain['status'] else 'yellow' if 'Growing' in domain['status'] else 'orange'
        ax.scatter(frac_pos[0], frac_pos[1], c=status_color, s=200, alpha=0.8, edgecolor='white')
        t = ax.text(frac_pos[0], frac_pos[1] + 0.08, piece, color='white', fontsize=10, ha='center')
        fragment_texts.append(t)

ax.set_title("The Grand Map — Universe as Hyperbolic Puzzle\nCenter = Source • Arms = Domains • Fragments = Implementable Pieces", 
             color='white', fontsize=22, pad=50)

plt.tight_layout()
plt.show()

print("🌌 The Grand Map is born")
print("Center = Prime Intelligence")
print("8 major domains radiating — each with recursive puzzle pieces")
print("Click/focus future versions to zoom into any arm")
print("This is the blueprint for everything")
