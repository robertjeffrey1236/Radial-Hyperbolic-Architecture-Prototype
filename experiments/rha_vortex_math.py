import numpy as np
import matplotlib.pyplot as plt

phi = (1 + np.sqrt(5)) / 2

# Vortex math 3-6-9 pattern on golden-ratio radial layers
max_levels = 9
points = []
labels = []
colors = []

for level in range(1, max_levels + 1):
    r_hyper = 0.99 * np.tanh((phi ** level) / 3)  # Hyperbolic projection
    num = 24  # Points per layer for full cycle
    theta = np.linspace(0, 2*np.pi, num, endpoint=False) + level * 0.2
    for i in range(num):
        x = r_hyper * np.cos(theta[i])
        y = r_hyper * np.sin(theta[i])
        points.append([x, y])
        
        # Vortex math number: doubling mod 9 +1 (classic 1-2-4-8-7-5 pattern, highlights 3-6-9)
        num_val = (2 ** (level * num + i)) % 9
        if num_val == 0: num_val = 9
        labels.append(str(num_val))
        
        # Color 3-6-9 special
        if num_val in [3,6,9]:
            colors.append('magenta')
        else:
            colors.append('cyan')

points = np.array(points)

# Plot
fig, ax = plt.subplots(figsize=(10,10), facecolor='black')
ax.set_aspect('equal')
ax.axis('off')

# Disk
circle = plt.Circle((0,0), 1, color='white', fill=False, lw=2, alpha=0.7)
ax.add_patch(circle)

# Points with labels
ax.scatter(points[:,0], points[:,1], c=colors, s=50, edgecolors='white', alpha=0.9)
for i, label in enumerate(labels):
    ax.text(points[i,0], points[i,1], label, color='white', ha='center', va='center', fontsize=8)

# Toroidal flow hint (curved arrows)
for angle in [0, np.pi]:
    ax.arrow(0, 0, 0.5 * np.cos(angle), 0.5 * np.sin(angle), head_width=0.05, color='gold', alpha=0.7)

plt.title('Vortex Math 3-6-9 Toy Model\nGolden-Ratio Layers + Modular Doubling in Hyperbolic Toroidal Flow', 
          color='white', fontsize=14, pad=30)
plt.tight_layout()
plt.savefig('rha_vortex_math.png', dpi=300, facecolor='black')
plt.show()
