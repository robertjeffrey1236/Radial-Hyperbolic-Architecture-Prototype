import numpy as np
import matplotlib.pyplot as plt

phi = (1 + np.sqrt(5)) / 2

# Central "mass" influence: denser recursion near center
max_levels = 8
points = []
for level in range(max_levels + 1):
    r_hyper = 0.99 * np.tanh((phi ** level) / 3)  # Slower growth for warp effect
    num = max(12, int(20 * (phi ** level)))  # Denser outward but pulled in
    theta = np.linspace(0, 2*np.pi, num) + level * 0.2
    x = r_hyper * np.cos(theta)
    y = r_hyper * np.sin(theta)
    points.extend(np.column_stack((x, y)))

points = np.array(points)

# Geodesics "bent" by gravity (radial pull + curved paths)
fig, ax = plt.subplots(figsize=(10,10), facecolor='black')
ax.set_aspect('equal')
ax.axis('off')

# Disk
circle = plt.Circle((0,0), 1, color='white', fill=False, lw=2, alpha=0.7)
ax.add_patch(circle)

# Warped grid lines (radial + circular, converging inward)
for angle in np.linspace(0, 2*np.pi, 24):
    x = np.linspace(0, 0.99 * np.cos(angle), 50)
    y = np.linspace(0, 0.99 * np.sin(angle), 50)
    ax.plot(x, y, color='cyan', lw=1, alpha=0.4)

for r in np.linspace(0.1, 0.99, 10):
    theta = np.linspace(0, 2*np.pi, 100)
    warp = 1 - 0.5 * np.exp(-5 * (1 - r))  # Stronger warp near center
    x = r * warp * np.cos(theta)
    y = r * warp * np.sin(theta)
    ax.plot(x, y, color='magenta', lw=1, alpha=0.5)

# Nodes (matter in spacetime)
ax.scatter(points[:,0], points[:,1], c='gold', s=30, edgecolors='white', alpha=0.8)

# "Orbit" paths (closed geodesics curving around central mass)
for offset in [0.4, 0.6, 0.8]:
    theta = np.linspace(0, 2*np.pi, 200)
    r = offset + 0.05 * np.sin(8 * theta)  # Perturbed for gravity feel
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    ax.plot(x, y, color='white', lw=1.5, alpha=0.7, linestyle='--')

# Central mass
ax.scatter(0, 0, c='red', s=200, marker='*', edgecolors='orange', linewidth=1)

plt.title('Spacetime & Gravity as One: Emergent Curvature in Hyperbolic Space\nCentral mass warps radial golden-ratio geodesics', color='white', fontsize=14, pad=30)
plt.tight_layout()
plt.savefig('rha_spacetime_gravity.png', dpi=300, facecolor='black')
plt.show()
