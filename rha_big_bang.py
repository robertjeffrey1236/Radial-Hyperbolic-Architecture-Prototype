import numpy as np
import matplotlib.pyplot as plt

phi = (1 + np.sqrt(5)) / 2

# Radial golden-ratio "time" layers: singularity → expansion → CMB-like outer shell
max_epochs = 12  # "Time" steps post-Bang
points = []
for epoch in range(max_epochs + 1):
    hyper_dist = phi ** epoch / 2  # Scaled for visual expansion
    r_poincare = 0.99 * np.tanh(hyper_dist)
    num = max(10, int(30 * (phi ** epoch)))  # More structure as universe "cools"
    theta = np.linspace(0, 2*np.pi, num) + epoch * 0.5
    x = r_poincare * np.cos(theta)
    y = r_poincare * np.sin(theta)
    points.extend(np.column_stack((x, y)))

points = np.array(points)

# Plot
fig, ax = plt.subplots(figsize=(10,10), facecolor='black')
ax.set_aspect('equal')
ax.axis('off')

# Disk (infinite future)
circle = plt.Circle((0,0), 1, color='white', fill=False, lw=2, alpha=0.7)
ax.add_patch(circle)

# Color by "epoch": Hot core (red/orange) → plasma → galaxies → CMB glow (blue outer)
core = points[np.linalg.norm(points, axis=1) < 0.2]
early = points[(np.linalg.norm(points, axis=1) >= 0.2) & (np.linalg.norm(points, axis=1) < 0.5)]
mid = points[(np.linalg.norm(points, axis=1) >= 0.5) & (np.linalg.norm(points, axis=1) < 0.8)]
outer = points[np.linalg.norm(points, axis=1) >= 0.8]

ax.scatter(core[:,0], core[:,1], c='red', s=80, alpha=0.95, edgecolors='orange')
ax.scatter(early[:,0], early[:,1], c='yellow', s=50, alpha=0.9)
ax.scatter(mid[:,0], mid[:,1], c='cyan', s=30, alpha=0.8)
ax.scatter(outer[:,0], outer[:,1], c='blue', s=20, alpha=0.7, edgecolors='white', linewidth=0.3)

# Singularity flash
ax.scatter(0, 0, c='white', s=300, marker='*', alpha=0.9)

plt.title('Big Bang Toy Model\nRadial Hyperbolic Expansion from Singularity via Golden-Ratio Recursion', 
          color='white', fontsize=14, pad=30)
plt.tight_layout()
plt.savefig('rha_big_bang.png', dpi=300, facecolor='black')
plt.show()
