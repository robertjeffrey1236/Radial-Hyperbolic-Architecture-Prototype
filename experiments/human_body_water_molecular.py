# experiments/human_body_water_molecular.py
# Molecular Water System — From Bulk H2O to Structured EZ Water
# Body as 60% water: bulk + interfacial + mineral/amino-acid influenced

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw

fig, ax = plt.subplots(figsize=(16, 24))
ax.set_facecolor('black')
ax.set_xlim(-1.0, 1.0)
ax.set_ylim(-1.2, 1.0)
ax.axis('off')

# Human silhouette
theta = np.linspace(0, 2*np.pi, 300)
body = plt.Circle((0, 0), 0.9, color='deepskyblue', fill=False, lw=5, alpha=0.4, ls='--')
ax.add_patch(body)

# Title
title = ax.set_title("Molecular Water System — 60% of Human Mass\nBulk H₂O + Structured EZ Water + Mineral/Protein Influence", color='white', fontsize=18, pad=50)

# === Bulk Water — Random H2O molecules filling body ===
# Simple H2O SMILES
water_mol = Chem.MolFromSmiles('O')
AllChem.Compute2DCoords(water_mol)

bulk_water = []
for _ in range(800):
    x = np.random.uniform(-0.8, 0.8)
    y = np.random.uniform(-1.0, 0.8)
    if np.linalg.norm([x, y]) < 0.85:  # Inside body
        bulk_water.append([x, y])

bulk_scatter = ax.scatter([p[0] for p in bulk_water], [p[1] for p in bulk_water], 
                          c='lightblue', s=30, alpha=0.6, marker='o', edgecolor='white', linewidth=0.5)

# === Structured EZ Water — Ordered hexagonal lattices near "surfaces" ===
# Simulate near cell membranes, proteins — more ordered
ez_centers = [
    [0.0, 0.0],     # Heart
    [0.0, 0.62],    # Brain
    [0.0, -0.35],   # Gut
    [-0.3, 0.1], [0.3, 0.1],  # Shoulders
]

ez_water = []
for center in ez_centers:
    for r in np.linspace(0.05, 0.25, 6):
        for angle in np.linspace(0, 2*np.pi, 12, endpoint=False):
            x = center[0] + r * np.cos(angle)
            y = center[1] + r * np.sin(angle)
            ez_water.append([x, y])

ez_scatter = ax.scatter([p[0] for p in ez_water], [p[1] for p in ez_water], 
                        c='cyan', s=40, alpha=0.9, marker='h', edgecolor='white', linewidth=1)

# === Mineral-Influenced Water (Na+, K+, Ca2+, Mg2+) ===
minerals = []
mineral_smiles = {'Na+': '[Na+]', 'K+': '[K+]', 'Ca2+': '[Ca+2]', 'Mg2+': '[Mg+2]'}
for _ in range(200):
    x = np.random.uniform(-0.7, 0.7)
    y = np.random.uniform(-0.9, 0.7)
    if np.linalg.norm([x, y]) < 0.8:
        minerals.append([x, y])

mineral_scatter = ax.scatter([p[0] for p in minerals], [p[1] for p in minerals], 
                             c='gold', s=25, alpha=0.8, marker='D')

# === Amino Acid Influence (proxy: glycine) near proteins ===
glycine = Chem.MolFromSmiles('NCC(O)=O')
AllChem.Compute2DCoords(glycine)

protein_zones = [[0.0, 0.62], [0.0, 0.0], [0.0, -0.35]]
for center in protein_zones:
    for _ in range(30):
        offset = np.random.uniform(-0.15, 0.15, 2)
        pos = np.array(center) + offset
        ax.scatter(pos[0], pos[1], c='magenta', s=60, alpha=0.7, marker='s')

# Breath pulse — water "flows" with life
t = 0
def animate(frame):
    global t
    t += 0.05
    
    # Subtle flow pulse
    pulse = 0.02 * np.sin(t * 2)
    new_x = [p[0] + pulse * np.random.randn() for p in bulk_water]
    new_y = [p[1] + pulse * np.random.randn() for p in bulk_water]
    bulk_scatter.set_offsets(np.c_[new_x, new_y])
    
    # EZ water slight ordering pulse
    ez_alpha = 0.8 + 0.2 * np.sin(t)
    ez_scatter.set_alpha(ez_alpha)
    
    # Title breath
    if np.sin(t * 2) > 0:
        title.set_text("Molecular Water System — Inhale: Flow Increases")
    else:
        title.set_text("Molecular Water System — Exhale: Structure Orders")

anim = FuncAnimation(fig, animate, interval=100, repeat=True)

# Boundary
boundary = plt.Circle((0, 0), 1, color='deepskyblue', fill=False, ls='--', lw=5, alpha=0.6)
ax.add_patch(boundary)

plt.show()

print("💧🧊 Molecular Water System activated — 60% of human mass")
print("Features:")
print("- Bulk H2O filling body (lightblue)")
print("- Structured EZ water near surfaces (cyan hexagons)")
print("- Mineral ions (gold diamonds)")
print("- Amino acid influence near proteins (magenta squares)")
print("- Breath-linked flow pulse")
print("Next: add proton gradient? Ice-like EZ lattice? Or full hydration shell around proteins?")
