# experiments/human_body_neutral_face.py
# Wholesome Human with Detailed Neutral Face
# Golden-ratio eyes, nose, mouth/teeth, ears — sacred and balanced

import numpy as np
import matplotlib.pyplot as plt
from core.geometry import golden_spiral_points, poincare_disk_project

N_POINTS = 35000
DIM = 37

# Substrate
points_nd = golden_spiral_points(n_points=N_POINTS, dim=DIM, radius_scale=0.99)
points_2d = poincare_disk_project(points_nd[:, :2])

fig, ax = plt.subplots(figsize=(18, 28))
ax.set_facecolor('black')

# Faint substrate
ax.scatter(points_2d[:, 0], points_2d[:, 1], c='gray', s=0.5, alpha=0.03)

# Head base (from previous)
head_center = np.array([0.0, 0.68])
ax.scatter(head_center[0], head_center[1], c='white', s=600, alpha=0.5, edgecolor='gold', linewidth=3)

# === Face Features (Golden-Ratio Proportioned) ===
# Eyes — windows to the soul, third-eye alignment
eye_left = [-0.12, 0.72]
eye_right = [0.12, 0.72]
ax.scatter(eye_left, eye_right, c='deepskyblue', s=300, alpha=0.9, edgecolor='cyan', linewidth=3, zorder=10)
# Irises + pupils
ax.scatter(eye_left, c='indigo', s=150, alpha=1.0)
ax.scatter(eye_right, c='indigo', s=150, alpha=1.0)
ax.scatter(eye_left, c='black', s=60, alpha=1.0)
ax.scatter(eye_right, c='black', s=60, alpha=1.0)
# Third-eye glow
ax.scatter(0, 0.75, c='violet', s=200, alpha=0.8, edgecolor='white', linewidth=2)

# Nose — central bridge
nose_points = [[-0.03, 0.70], [0.03, 0.70], [0, 0.62]]
ax.plot([p[0] for p in nose_points], [p[1] for p in nose_points], c='gold', lw=4, alpha=0.8)

# Mouth — expression/smile, crystalline teeth hint
mouth_curve = np.linspace(-0.1, 0.1, 50)
mouth_y = 0.58 - 0.03 * np.sin((mouth_curve + 0.1) * np.pi / 0.2)
ax.plot(mouth_curve, mouth_y, c='crimson', lw=4, alpha=0.9)
# Teeth subtle crystals
for x in np.linspace(-0.08, 0.08, 8):
    ax.plot([x, x], [0.58, 0.55], c='white', lw=2, alpha=0.7)

# Ears — receptive spirals
for side in [-1, 1]:
    ear_center = [side * 0.2, 0.68]
    ear_spiral = golden_spiral_points(50, dim=2, radius_scale=0.06) * side
    ear_spiral[:, 1] += 0.68
    ear_spiral[:, 0] += side * 0.2
    ax.plot(ear_spiral[:, 0], ear_spiral[:, 1], c='gold', lw=3, alpha=0.8)

# Facial golden-ratio guides (subtle)
ax.plot([-0.2, 0.2], [0.78, 0.78], c='gold', lw=1, alpha=0.4, ls='--')  # Brow line
ax.plot([-0.2, 0.2], [0.55, 0.55], c='gold', lw=1, alpha=0.4, ls='--')  # Mouth base

# Internal layers faint
ax.scatter(0, 0, c='crimson', s=600, alpha=0.3)
ax.plot([0, 0], [-0.8, 0.8], c='cyan', lw=3, alpha=0.2)

# Poincaré boundary
circle = plt.Circle((0, 0), 1, color='gold', fill=False, ls='--', lw=6, alpha=0.9)
ax.add_patch(circle)

ax.axis('equal')
ax.axis('off')
plt.title("Wholesome Human with Detailed Neutral Face\nGolden-Ratio Eyes, Nose, Mouth/Te
