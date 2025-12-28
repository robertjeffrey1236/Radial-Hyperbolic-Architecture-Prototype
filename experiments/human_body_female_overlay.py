# experiments/human_body_female_overlay.py
# Wholesome Human with Anatomical Female Overlay
# Feminine expression: curvier build, fuller chest/hips, womb center, graceful strength

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

# Faint internal substrate
ax.scatter(points_2d[:, 0], points_2d[:, 1], c='gray', s=0.5, alpha=0.03)

# === Base Skin Envelope (neutral translucent) ===
theta = np.linspace(0, 2*np.pi, 300)
base_radius_x = 0.4 + 0.15 * np.abs(np.sin(theta * 3))
base_radius_y = 1.1 - 0.3 * np.abs(np.cos(theta))
base_x = base_radius_x * np.cos(theta) * np.abs(np.cos(theta))
base_y = base_radius_y * np.sin(theta)
ax.fill(base_x, base_y, c='peachpuff', alpha=0.12, zorder=4)

# === Female Overlay: Curvier, Graceful Contour ===
female_radius_x = 0.45 + 0.25 * np.abs(np.sin(theta * 2))  # Wider hips
female_radius_y = 1.15 - 0.2 * np.abs(np.cos(theta))
female_x = female_radius_x * np.cos(theta)
female_y = female_radius_y * np.sin(theta)

# Feminine silhouette fill and glow
ax.fill(female_x, female_y, c='palevioletred', alpha=0.22, zorder=6)
ax.plot(female_x, female_y, c='hotpink', lw=5, alpha=0.8, zorder=7)

# Fuller chest/breast region
ax.scatter([ -0.15, 0.15 ], [0.15, 0.15], c='pink', s=450, alpha=0.7, edgecolor='magenta', linewidth=2, zorder=8)

# Womb/ovaries — sacred creative center (sacral)
ax.scatter(0, -0.35, c='deeppink', s=500, alpha=0.8, edgecolor='gold', linewidth=4)
ax.scatter([ -0.1, 0.1 ], [-0.4, -0.4], c='magenta', s=250, alpha=0.7, zorder=9)
ax.text(0, -0.25, 'Womb\nLife Creation', color='gold', fontsize=13, ha='center', fontweight='bold')

# Softer, graceful muscle glow (arms, legs, core)
graceful_muscles = [
    ([-0.3, 0.15], 'arm left'), ([0.3, 0.15], 'arm right'),
    ([-0.25, -0.6], 'thigh left'), ([0.25, -0.6], 'thigh right'),
]
for pos, label in graceful_muscles:
    ax.scatter(pos[0], pos[1], c='palevioletred', s=350, alpha=0.6)

# Softer facial contour (gentler jaw)
ax.plot([-0.1, -0.06, 0.06, 0.1], [0.68, 0.71, 0.71, 0.68], c='gold', lw=3, alpha=0.8)

# === Internal Layers Visible Through Overlay ===
ax.scatter(0, 0.62, c='indigo', s=500, alpha=0.4)      # Brain
ax.scatter(0, 0, c='crimson', s=600, alpha=0.4)        # Heart
ax.scatter(0, -0.35, c='orange', s=400, alpha=0.3)     # Gut (now aligned with womb)
ax.plot([0, 0], [-0.8, 0.8], c='cyan', lw=4, alpha=0.3)  # Spine

# Poincaré boundary — warm feminine glow
circle = plt.Circle((0, 0), 1, color='hotpink', fill=False, ls='--', lw=6, alpha=0.9)
ax.add_patch(circle)

ax.axis('equal')
ax.axis('off')
plt.title("Wholesome Human with Anatomical Female Overlay\nCurvier Grace | Fuller Form | Womb Creation Center | Divine Feminine Archetype", 
          color='white', fontsize=22, pad=90)
plt.tight_layout()
plt.savefig("human_body_female_overlay.png", dpi=600, facecolor='black', bbox_inches='tight')
plt.show()

print("♀️🌸 Anatomical Female Overlay complete — grace, nurturing, and creative power expressed")
print("Features: Wider hips, fuller chest, womb/ovaries emphasis, warm rose-gold tones")
print("Saved: human_body_female_overlay.png — the feminine archetype blooms from the universal form")
