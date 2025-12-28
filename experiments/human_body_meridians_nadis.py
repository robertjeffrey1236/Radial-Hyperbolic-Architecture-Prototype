# experiments/human_body_meridians_nadis.py
# Wholesome Human with Meridians & Nadis (Subtle Energy Channels)
# Major pranic pathways — Ida, Pingala, Sushumna + 12 TCM meridians stylized in golden spirals

import numpy as np
import matplotlib.pyplot as plt
from core.geometry import golden_spiral_points, poincare_disk_project, GOLDEN_ANGLE

N_POINTS = 35000
DIM = 37

# Substrate
points_nd = golden_spiral_points(n_points=N_POINTS, dim=DIM, radius_scale=0.99)
points_2d = poincare_disk_project(points_nd[:, :2])

fig, ax = plt.subplots(figsize=(18, 30))
ax.set_facecolor('black')

# Faint substrate (pranic field)
ax.scatter(points_2d[:, 0], points_2d[:, 1], c='white', s=0.5, alpha=0.02)

# Chakra positions (energy nodes)
chakras = {
    'crown': [0.0, 0.65], 'third_eye': [0.0, 0.45], 'throat': [0.0, 0.20],
    'heart': [0.0, 0.0], 'solar_plexus': [0.0, -0.15], 'sacral': [0.0, -0.40], 'root': [0.0, -0.60]
}
for name, pos in chakras.items():
    ax.scatter(pos[0], pos[1], c='gold', s=400, alpha=0.7, edgecolor='white', linewidth=2, zorder=10)

# === Central Nadis: Sushumna, Ida, Pingala ===
# Sushumna — central white column
ax.plot([0, 0], [-0.7, 0.7], c='white', lw=8, alpha=0.8)

# Ida (left, lunar, feminine) — cool cyan spiral
ida_path = golden_spiral_points(100, dim=2, radius_scale=0.3)
ida_path[:, 0] -= 0.15  # Shift left
ida_path[:, 1] = np.linspace(-0.6, 0.6, 100)
ax.plot(ida_path[:, 0], ida_path[:, 1], c='cyan', lw=4, alpha=0.7)

# Pingala (right, solar, masculine) — warm gold spiral
pingala_path = ida_path.copy()
pingala_path[:, 0] *= -1  # Mirror right
pingala_path[:, 0] += 0.3
ax.plot(pingala_path[:, 0], pingala_path[:, 1], c='gold', lw
