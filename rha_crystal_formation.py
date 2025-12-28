import numpy as np
import matplotlib.pyplot as plt

phi = (1 + np.sqrt(5)) / 2

# Recursive golden-angle branching for quasi-crystal growth (snowflake/quasicrystal inspired)
def grow_quasicrystal(levels=8, base_scale=0.99):
    points = [np.array([0.0, 0.0])]
    angles = np.array([0.0])
    
    for level in range(1, levels + 1):
        new_points = []
        scale = (phi ** -level) * base_scale  # Golden-ratio contraction
        golden_angle = np.pi * (phi - 1) * 2  # ~137.5° optimal divergence
        
        for i, p in enumerate(points):
            for branch in range(5):  # 5-fold symmetry for quasicrystal feel
                theta = angles[i] + branch * (2 * np.pi / 5) + level * golden_angle
                offset = scale * np.array([np.cos(theta), np.sin(theta)])
                new_p = p + offset
                # Hyperbolic projection to keep bounded
                r = np.linalg.norm(new_p)
                if r < 1.0:
                    new_p = new_p / (1 + r**2) * 0.99  # Approx stereographic to disk
                new_points.append(new_p)
                angles = np.append(angles, theta)
        
        points.extend(new_points)
    
    return np.array(points)

points = grow_quasicrystal(levels=7)

# Plot
fig, ax = plt.subplots(figsize=(10,10), facecolor='black')
ax.set_aspect('equal')
ax.axis('off')

# Disk boundary
circle = plt.Circle((0,0), 1, color='white', fill=False, lw=2, alpha=0.7)
ax.add_patch(circle)

# Crystal points: ice-blue with golden glow
ax.scatter(points[:,0], points[:,1], c='cyan', s=20, edgecolors='gold', linewidth=0.4, alpha=0.9)

# Connections for lattice feel (nearest neighbors)
for i in range(len(points)):
    for j in range(i+1, len(points)):
        if np.linalg.norm(points[i] - points[j]) < 0.15:
            ax.plot(points[[i,j],0], points[[i,j],1], color='lightblue', alpha=0.3, lw=0.8)

plt.title('Crystal Formation Toy Model\nGolden-Ratio Recursive Branching → Quasicrystal in Hyperbolic Disk', 
          color='white', fontsize=14, pad=30)
plt.tight_layout()
plt.savefig('rha_crystal_formation.png', dpi=300, facecolor='black')
plt.show()
