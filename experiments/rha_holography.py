import numpy as np
import matplotlib.pyplot as plt

phi = (1 + np.sqrt(5)) / 2

# Toy holographic model: Golden-ratio radial shells in Poincaré disk
# Outer shells dominate → emergent boundary encoding bulk

max_gen = 7
shell_nodes = []
radii = np.cumsum(phi ** np.arange(max_gen))  # Phi-scaled distances
poincare_r = 0.99 * np.tanh(radii / 2)  # Hyperbolic projection

colors = ['#ff9999', '#ffcc99', '#ffff99', '#ccff99', '#99ff99', '#99ccff', '#cc99ff']

fig, ax = plt.subplots(figsize=(10,10), facecolor='black')
ax.set_aspect('equal')
ax.axis('off')

circle = plt.Circle((0,0), 1, color='white', fill=False, lw=2, alpha=0.8)
ax.add_patch(circle)

total_nodes = 0
boundary_fraction = 0

for gen in range(max_gen):
    r = poincare_r[gen] if gen < len(poincare_r) else 0.99
    nodes = int(20 * (phi ** (2 * gen)))  # Exponential growth ~ phi^{2gen} for hyperbolic area
    shell_nodes.append(nodes)
    total_nodes += nodes
    
    angles = np.linspace(0, 2*np.pi, max(6, nodes//10), endpoint=False)
    x = r * np.cos(angles)
    y = r * np.sin(angles)
    ax.scatter(x, y, c=colors[gen], s=30, edgecolors='white', alpha=0.9)
    
    if gen == max_gen - 1:
        boundary_fraction = nodes / total_nodes

ax.text(0, -0.1, f'Boundary (outer shell) encodes ~{boundary_fraction:.3f} of total\n(≈ 1/φ = 0.618 in limit)', 
        color='white', ha='center', fontsize=14)

plt.title('Toy Holographic Principle in Radial Hyperbolic Space\nBulk hierarchy encoded on boundary via negative curvature + Φ recursion', 
          color='white', fontsize=14, pad=30)
plt.tight_layout()
plt.savefig('rha_holography.png', dpi=300, facecolor='black')
plt.show()

print(f"Boundary fraction: {boundary_fraction:.3f} ≈ 1/φ")
