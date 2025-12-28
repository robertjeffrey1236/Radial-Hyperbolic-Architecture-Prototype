# experiments/human_body_complete_organs.py
# Wholesome Human Body: Full Major Organs + Systems in Radial Hyperbolic Architecture
# Organs as glowing sub-structures, chakra-aligned, golden-proportioned

import numpy as np
import matplotlib.pyplot as plt
from core.geometry import golden_spiral_points, poincare_disk_project

N_POINTS = 20000
DIM = 37

# Substrate
points_nd = golden_spiral_points(n_points=N_POINTS, dim=DIM, radius_scale=0.99)
points_2d = poincare_disk_project(points_nd[:, :2])

fig, ax = plt.subplots(figsize=(16, 20))
ax.set_facecolor('black')

# Faint tissue substrate
ax.scatter(points_2d[:, 0], points_2d[:, 1], c='gray', s=1, alpha=0.1)

# === Major Organs as Glowing Nodes/Sub-Structures ===
organs = {
    'brain': {'pos': [0.0, 0.62], 'color': 'indigo', 'size': 500, 'label': 'Brain\nHigher Cognition'},
    'thyroid': {'pos': [0.0, 0.25], 'color': 'cyan', 'size': 150, 'label': 'Thyroid\nMetabolic Voice'},
    'lungs': {'pos': [[-0.18, 0.08], [0.18, 0.08]], 'color': 'lightblue', 'size': 300, 'label': 'Lungs'},
    'heart': {'pos': [0.0, 0.0], 'color': 'crimson', 'size': 600, 'label': 'Heart\nLife Pump & Love'},
    'liver': {'pos': [0.25, -0.08], 'color': 'darkred', 'size': 400, 'label': 'Liver\nDetox & Vitality'},
    'stomach': {'pos': [0.0, -0.15], 'color': 'orange', 'size': 250, 'label': 'Stomach\nDigestion'},
    'intestines': {'pos': [0.0, -0.35], 'color': 'sandybrown', 'size': 350, 'label': 'Intestines\nAbsorption'},
    'kidneys': {'pos': [[-0.15, -0.25], [0.15, -0.25]], 'color': 'purple', 'size': 200, 'label': 'Kidneys'},
    'reproductive': {'pos': [0.0, -0.55], 'color': 'magenta', 'size': 300, 'label': 'Reproductive\nCreation Force'},
}

# Plot organs
for name, data in organs.items():
    if isinstance(data['pos'][0], list):  # Paired organs
        for pos in data['pos']:
            ax.scatter(pos[0], pos[1], c=data['color'], s=data['size'], alpha=0.8, edgecolor='white', linewidth=2, zorder=10)
    else:
        ax.scatter(data['pos'][0], data['pos'][1], c=data['color'], s=data['size'], alpha=0.9, edgecolor='white', linewidth=3, zorder=10)
    if 'label' in data:
        ax.text(data['pos'][0] if not isinstance(data['pos'][0], list) else data['pos'][0][0], 
                (data['pos'][1] if not isinstance(data['pos'][0], list) else data['pos'][0][1]) + 0.12, 
                data['label'], color='white', fontsize=11, ha='center', fontweight='bold')

# Dual lung spirals (phyllotaxis-inspired)
for side in [-1, 1]:
    lung_spiral = golden_spiral_points(300, dim=2, radius_scale=0.15) * side
    lung_spiral[:, 1] += 0.08
    ax.scatter(lung_spiral[:, 0], lung_spiral[:, 1], c='lightblue', s=8, alpha=0.7)

# === Overlay Systems (fainter for wholeness) ===
# Spine + major nerves
ax.plot([0, 0], [-0.
