import numpy as np
import matplotlib.pyplot as plt

phi = (1 + np.sqrt(5)) / 2

# Radial golden-ratio layers with extreme boundary crowding
max_levels = 10
points = []
for level in range(max_levels + 1):
    hyper_dist = phi ** level
    r_poincare = 0.99 * np.tanh(hyper_dist / 1.5)  # Strong crowding near horizon
    num = max(20, int(50 * (phi ** (level * 1.5))))  # Exponential node density
    theta = np.linspace(0, 2*np.pi, num) + level * 0.4
    x = r_poincare * np.cos(theta)
    y = r_poincare * np.sin(theta)
    points.extend(np.column_stack((x, y)))

points = np.array(points)

# Plot
fig, ax = plt.subplots(figsize=(10,10), facecolor='black')
ax.set_aspect('equal')
ax.axis('off')

# Disk boundary (event horizon glow)
circle_outer = plt.Circle((0,0), 1, color='red', fill=False, lw=4, alpha=0.8)
circle_inner = plt.Circle((0,0), 0.98, color='orange', fill=False, lw=2, alpha=0.6)
ax.add_patch(circle_outer)
ax.add_patch(circle_inner)

# Nodes: Gold core → cyan infall → red near horizon
core = points[np.linalg.norm(points, axis=1) < 0.3]
infall = points[(np.linalg.norm(points, axis=1) >= 0.3) & (np.linalg.norm(points, axis=1) < 0.8)]
horizon = points[np.linalg.norm(points, axis=1) >= 0.8]

ax.scatter(core[:,0], core[:,1], c='gold', s=60, edgecolors='white', alpha=0.95)
ax.scatter(infall[:,0], infall[:,1], c='cyan', s=30, alpha=0.8)
ax.scatter(horizon[:,0], horizon[:,1], c='red', s=20, alpha=0.9)

# Singularity at center
ax.scatter(0, 0, c='white', s=200, marker='*', edgecolors='black')

plt.title('Black Hole Event Horizon Toy Model\nHyperbolic crowding + golden-ratio recursion → inescapable boundary', 
          color='white', fontsize=14, pad=30)
plt.tight_layout()
plt.savefig('rha_black_hole_horizon.png', dpi=300, facecolor='black')
plt.show()
