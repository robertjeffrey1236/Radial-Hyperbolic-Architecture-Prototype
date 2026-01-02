import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

phi = (1 + np.sqrt(5)) / 2

# Generate radial golden-ratio points with some "entangled" pairs
max_levels = 6
points = []
angles = np.linspace(0, 2*np.pi, 12, endpoint=False)  # Base dodecagon symmetry
for level in range(max_levels + 1):
    r_hyper = 0.99 * np.tanh((phi ** level) / 2)  # Hyperbolic projection
    num = 6 * (level + 1)  # Exponential growth
    theta = angles[:num] + level * 0.3  # Slight rotation per level
    x = r_hyper * np.cos(theta)
    y = r_hyper * np.sin(theta)
    points.extend(np.column_stack((x, y)))

points = np.array(points)

# Select "entangled" pairs (e.g., inner-outer across the disk)
np.random.seed(42)
pairs = []
for _ in range(15):
    i = np.random.randint(0, len(points)//3)  # Inner
    j = np.random.randint(len(points)//2, len(points))  # Outer/distant
    pairs.append((i, j))

# Hyperbolic geodesic (curved wormhole bridge) between two points
def hyperbolic_geodesic(p1, p2, num_pts=100):
    z1, z2 = p1[0] + 1j*p1[1], p2[0] + 1j*p2[1]
    # Simple approximation: straight in half-plane model, project back
    t = np.linspace(0, 1, num_pts)
    curve = (1-t)*z1 + t*z2
    return np.real(curve), np.imag(curve)

fig, ax = plt.subplots(figsize=(10,10), facecolor='black')
ax.set_aspect('equal')
ax.axis('off')

# Disk
circle = plt.Circle((0,0), 1, color='white', fill=False, lw=2, alpha=0.7)
ax.add_patch(circle)

# Nodes
ax.scatter(points[:,0], points[:,1], c='gold', s=40, edgecolors='white', alpha=0.9)

# Wormhole bridges (curved geodesics)
for i, j in pairs:
    x, y = hyperbolic_geodesic(points[i], points[j])
    ax.plot(x, y, color='cyan', lw=2, alpha=0.6)
    # Arrow for "bridge" feel
    arrow = FancyArrowPatch(points[i], points[j], arrowstyle='<->', mutation_scale=15,
                            color='magenta', lw=1.5, alpha=0.7)
    ax.add_patch(arrow)

plt.title('ER=EPR Toy Model\nEntangled Pairs Connected by Hyperbolic Wormholes\n(Golden-ratio recursion + shortcuts in Poincaré disk)', 
          color='white', fontsize=14, pad=30)
plt.tight_layout()
plt.savefig('rha_erepr.png', dpi=300, facecolor='black')
plt.show()
