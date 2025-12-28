# experiments/human_body_chakras_aura.py
# Wholesome Human with Chakras & Aura/Energetic Field
# Seven spiraling energy centers + multi-layered toroidal aura

import numpy as np
import matplotlib.pyplot as plt
from core.geometry import golden_spiral_points, poincare_disk_project, GOLDEN_ANGLE

N_POINTS = 30000
DIM = 37

# Substrate
points_nd = golden_spiral_points(n_points=N_POINTS, dim=DIM, radius_scale=0.99)
points_2d = poincare_disk_project(points_nd[:, :2])

fig, ax = plt.subplots(figsize=(18, 28))
ax.set_facecolor('black')

# Very faint substrate (etheric field)
ax.scatter(points_2d[:, 0], points_2d[:, 1], c='white', s=0.5, alpha=0.02)

# Chakra positions & classic colors
chakras = [
    ('Root',     [0.0, -0.60], 'red',      0.18),
    ('Sacral',   [0.0, -0.40], 'orange',   0.20),
    ('Solar Plexus', [0.0, -0.15], 'yellow', 0.22),
    ('Heart',    [0.0,  0.00], 'green',    0.25),
    ('Throat',   [0.0,  0.20], 'cyan',     0.20),
    ('Third Eye',[0.0,  0.45], 'indigo',   0.18),
    ('Crown',    [0.0,  0.65], 'violet',   0.22),
]

# Draw each chakra with dual counter-rotating golden spirals
for name, pos, color, scale in chakras:
    pos = np.array(pos)
    
    # Inner glow core
    ax.scatter(pos[0], pos[1], c=color, s=600, alpha=0.8, edgecolor='white', linewidth=3, zorder=10)
    ax.scatter(pos[0], pos[1], c='white', s=200, alpha=0.9, zorder=11)
    
    # Counter-rotating spirals (inflow/outflow)
    for direction in [1, -1]:
        spiral = golden_spiral_points(80, dim=2, radius_scale=scale)
        spiral[:, 0] *= direction  # Mirror for counter-rotation
        spiral = spiral @ np.array([[np.cos(GOLDEN_ANGLE*direction), -np.sin(GOLDEN_ANGLE*direction)],
                                   [np.sin(GOLDEN_ANGLE*direction), np.cos(GOLDEN_ANGLE*direction)]])  # Rotate
        spiral += pos
        ax.plot(spiral[:, 0], spiral[:, 1], c=color, lw=2, alpha=0.6)
    
    # Chakra name
    ax.text(pos[0], pos[1] + 0.15, name, color='white', fontsize=12, ha='center', fontweight='bold', alpha=0.9)

# === Aura / Energetic Field ===
# Multi-layered toroidal aura
for i, (radius, color, alpha) in enumerate([
    (1.1, 'white', 0.3),      # Inner etheric
    (1.3, 'gold', 0.2),       # Emotional body
    (1.5, 'rainbow', 0.15),   # Mental rainbow sheen
    (1.8, 'violet', 0.1),     # Spiritual outer
]):
    if color == 'rainbow':
        rainbow_theta = np.linspace(0, 2*np.pi, 200)
        for offset in np.linspace(0, 2*np.pi, 12, endpoint=False):
            aura_x = radius * np.cos(rainbow_theta + offset)
            aura_y = radius * np.sin(rainbow_theta + offset)
            ax.plot(aura_x, aura_y, c=plt.cm.hsv(offset/(2*np.pi)), lw=3, alpha=alpha)
    else:
        aura = plt.Circle((0, 0), radius, color=color, fill=False, lw=5, alpha=alpha)
        ax.add_patch(aura)

# Central column of light (sushumna)
ax.plot([0, 0], [-0.8, 0.8], c='white', lw=6, alpha=0.4)

# Physical body faint silhouette
theta = np.linspace(0, 2*np.pi, 200)
body_x = 0.45 * np.cos(theta)
body_y = 1.1 * np.sin(theta) - 0.1
ax.plot(body_x, body_y, c='white', lw=3, alpha=0.2)

# Poincaré boundary (interface with cosmic field)
circle = plt.Circle((0, 0), 1, color='white', fill=False, ls='--', lw=6, alpha=0.5)
ax.add_patch(circle)

ax.axis('equal')
ax.axis('off')
plt.title("Wholesome Human with Activated Chakras & Full Aura\nSpiraling Energy Centers | Multi-Layered Toroidal Field | Pranic Flow", 
          color='white', fontsize=22, pad=100)
plt.tight_layout()
plt.savefig("human_body_chakras_aura.png", dpi=600, facecolor='black', bbox_inches='tight')
plt.show()

print("🌈✨ Chakras & Aura fully activated — seven spiraling vortices with counter-rotating golden energy")
print("Multi-layered aura: etheric, emotional, mental, spiritual — the human as radiant being")
print("Saved: human_body_chakras_aura.png — the energetic body is now complete and luminous")
