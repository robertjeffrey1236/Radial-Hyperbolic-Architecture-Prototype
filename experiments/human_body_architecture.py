# experiments/human_body_architecture.py
# Basic Human Body Mapped into Radial Hyperbolic Architecture
# Foundation for bio-inspired extensions (nervous system, chakras, meridians, etc.)

import numpy as np
import matplotlib.pyplot as plt
from core.geometry import golden_spiral_points, poincare_disk_project, build_hyperbolic_graph

N_POINTS = 8000
DIM = 37
SCALE_BODY = 0.8

# Generate base lattice - represents the "field" or substrate
points_nd = golden_spiral_points(n_points=N_POINTS, dim=DIM, radius_scale=0.99)
points_2d = poincare_disk_project(points_nd[:, :2])

# Build connectivity
neighbors = build_hyperbolic_graph(points_2d, k_neighbors=8)

# === Define Key Body Landmarks (in normalized Poincaré disk coords) ===
# These are hand-placed but follow golden-ratio proportions where possible
body_landmarks = {
    'root': np.array([0.0, -0.6]),           # Base of spine / root chakra
    'sacral': np.array([0.0, -0.4]),
    'solar_plexus': np.array([0.0, -0.15]),
    'heart': np.array([0.0, 0.05]),           # Central observer / core
    'throat': np.array([0.0, 0.25]),
    'third_eye': np.array([0.0, 0.45]),
    'crown': np.array([0.0, 0.65]),           # Top of head
    'left_shoulder': np.array([-0.3, 0.1]),
    'right_shoulder': np.array([0.3, 0.1]),
    'left_hand': np.array([-0.55, -0.1]),
    'right_hand': np.array([0.55, -0.1]),
    'left_foot': np.array([-0.2, -0.85]),
    'right_foot': np.array([0.2, -0.85]),
}

# Colors for chakra/body regions
chakra_colors = {
    'root': 'red',
    'sacral': 'orange',
    'solar_plexus': 'yellow',
    'heart': 'green',
    'throat': 'cyan',
    'third_eye': 'blue',
    'crown': 'violet',
}

fig, ax = plt.subplots(figsize=(12, 14))
ax.set_facecolor('black')

# Plot faint background lattice
ax.scatter(points_2d[:, 0], points_2d[:, 1], c='gray', s=3, alpha=0.3)

# Plot body connections (spine + limbs)
spine_points = np.array([body_landmarks[k] for k in ['root', 'sacral', 'solar_plexus', 'heart', 'throat', 'third_eye', 'crown']])
ax.plot(spine_points[:, 0], spine_points[:, 1], c='white', lw=4, alpha=0.8, label='Spinal Column')

# Arms
ax.plot([body_landmarks['left_shoulder'][0], body_landmarks['heart'][0], body_landmarks['right_shoulder'][0]],
        [body_landmarks['left_shoulder'][1], body_landmarks['heart'][1], body_landmarks['right_shoulder'][1]],
        c='magenta', lw=3, alpha=0.7)
ax.plot([body_landmarks['left_shoulder'][0], body_landmarks['left_hand'][0]],
        [body_landmarks['left_shoulder'][1], body_landmarks['left_hand'][1]], c='magenta', lw=2)
ax.plot([body_landmarks['right_shoulder'][0], body_landmarks['right_hand'][0]],
        [body_landmarks['right_shoulder'][1], body_landmarks['right_hand'][1]], c='magenta', lw=2)

# Legs
ax.plot([body_landmarks['root'][0], body_landmarks['left_foot'][0]],
        [body_landmarks['root'][1], body_landmarks['left_foot'][1]], c='magenta', lw=2)
ax.plot([body_landmarks['root'][0], body_landmarks['right_foot'][0]],
        [body_landmarks['root'][1], body_landmarks['right_foot'][1]], c='magenta', lw=2)

# Plot major nodes (chakras/organs)
for name, pos in body_landmarks.items():
    color = chakra_colors.get(name, 'white')
    size = 150 if 'heart' in name or 'crown' in name or 'third_eye' in name else 100
    ax.scatter(pos[0], pos[1], c=color, s=size, edgecolors='white', linewidth=2, alpha=0.9, zorder=5)
    ax.text(pos[0], pos[1] + 0.08, name.replace('_', ' ').title(), color='white', fontsize=10, ha='center')

# Poincaré disk boundary
circle = plt.Circle((0, 0), 1, color='cyan', fill=False, ls='--', lw=2, alpha=0.7)
ax.add_patch(circle)

ax.axis('equal')
ax.axis('off')
plt.title("Human Body Architecture in Hyperbolic Space\nGolden-Ratio Proportioned | Chakra-Aligned | Fractal Substrate", color='white', fontsize=16)
plt.tight_layout()
plt.savefig("human_body_hyperbolic.png", dpi=300, facecolor='black')
plt.show()

print("🧘 Human Body Architecture visualized in your Radial Hyperbolic Framework")
print("Saved as 'human_body_hyperbolic.png' — foundation for bio-extensions (nerves, meridians, consciousness mapping)")
