# experiments/human_body_acupuncture_energy_points.py
# Wholesome Human with Acupuncture Points & All Known Energy Points
# 361+ classical points + major marma + extra points along meridians/nadis

import numpy as np
import matplotlib.pyplot as plt
from core.geometry import golden_spiral_points, poincare_disk_project

N_POINTS = 40000
DIM = 37

# Substrate
points_nd = golden_spiral_points(n_points=N_POINTS, dim=DIM, radius_scale=0.99)
points_2d = poincare_disk_project(points_nd[:, :2])

fig, ax = plt.subplots(figsize=(20, 32))
ax.set_facecolor('black')

# Faint pranic field
ax.scatter(points_2d[:, 0], points_2d[:, 1], c='white', s=0.5, alpha=0.02)

# === Meridians/Nadis Base (from previous) ===
# Central Sushumna
ax.plot([0, 0], [-0.8, 0.8], c='white', lw=6, alpha=0.6)

# Ida & Pingala
ida_x = -0.15 + 0.1 * np.sin(np.linspace(0, 10*np.pi, 200))
ida_y = np.linspace(-0.7, 0.7, 200)
ax.plot(ida_x, ida_y, c='cyan', lw=3, alpha=0.6)

pingala_x = -ida_x
ax.plot(pingala_x, ida_y, c='gold', lw=3, alpha=0.6)

# Simplified major meridians
meridians = [
    ([-0.4, -0.1], [-0.2, 0.1]), ( [0.4, -0.1], [0.2, 0.1]),   # Lung
    ([-0.3, 0.05], [-0.5, -0.2]), ([0.3, 0.05], [0.5, -0.2]),  # Pericardium
    ([0, 0.6], [0.1, -0.8]), ([0, 0.6], [-0.1, -0.8]),         # Stomach
    ([-0.2, -0.9], [-0.1, -0.1]), ([0.2, -0.9], [0.1, -0.1]),  # Kidney
]

for start, end in meridians:
    t = np.linspace(0, 1, 80)
    x = start[0] + (end[0] - start[0]) * t + 0.08 * np.sin(t * np.pi * 5)
    y = start[1] + (end[1] - start[1]) * t
    ax.plot(x, y, c='lightblue', lw=2, alpha=0.5)

# === Acupuncture & Energy Points ===
# Major classical points (selected key ones for clarity + beauty)
key_points = [
    # Head & Crown
    ('GV20 Baihui', [0.0, 0.70], 'violet'),
    ('GV16 Fengfu', [0.0, 0.55], 'indigo'),
    ('Yintang', [0.0, 0.68], 'deepskyblue'),
    # Face
    ('LI20 Yingxiang', [-0.08, 0.62], 'cyan'), ('LI20 Yingxiang', [0.08, 0.62], 'cyan'),
    ('ST2 Sibai', [-0.1, 0.58], 'white'), ('ST2 Sibai', [0.1, 0.58], 'white'),
    # Neck/Throat
    ('CV22 Tiantu', [0.0, 0.25], 'cyan'),
    # Chest/Heart
    ('CV17 Tanzhong', [0.0, 0.05], 'green'),
    ('LU1 Zhongfu', [-0.2, 0.1], 'lightblue'), ('LU1 Zhongfu', [0.2, 0.1], 'lightblue'),
    # Arms/Hands (PC6 Neiguan, LI4 Hegu)
    ('PC6 Neiguan', [-0.35, -0.05], 'crimson'), ('PC6 Neiguan', [0.35, -0.05], 'crimson'),
    ('LI4 Hegu', [-0.5, -0.15], 'white'), ('LI4 Hegu', [0.5, -0.15], 'white'),
    # Abdomen
    ('CV12 Zhongwan', [0.0, -0.1], 'yellow'),
    ('CV6 Qihai', [0.0, -0.3], 'orange'),
    # Lower
