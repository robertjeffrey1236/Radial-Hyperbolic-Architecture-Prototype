import numpy as np
import matplotlib.pyplot as plt

# Golden ratio
phi = (1 + np.sqrt(5)) / 2

# Simulate generations with phi-scaled radial steps
generations = np.arange(0, 7)  # 0 to 6 for visualization
euclidean_radial = phi ** generations
hyperbolic_cumulative = np.cumsum(euclidean_radial)

# Poincaré disk projection (tanh maps hyperbolic distance to disk radius)
disk_radius = 0.99
poincare_r = disk_radius * np.tanh(hyperbolic_cumulative / 2)

# Create representative points for each generation (ring of nodes)
points_2d = []
shell_labels = []
colors = ['#c88cc7', '#d0bdce', '#a04515', '#0fd2b2', '#6d87a7', '#91322b', '#f93c55']  # From your post

for gen, r in enumerate(poincare_r):
    if r < 1e-3:
        points_2d.append([0, 0])
        continue
    num_points = max(6, int(20 * r))  # More points outward for visual density
    angles = np.linspace(0, 2*np.pi, num_points, endpoint=False)
    x = r * np.cos(angles)
    y = r * np.sin(angles)
    points_2d.extend(np.column_stack((x, y)))
    shell_labels.extend([gen] * num_points)

points_2d = np.array(points_2d)

# Plot
fig, ax = plt.subplots(figsize=(10,10), facecolor='black')
ax.set_aspect('equal')
ax.axis('off')

# Disk boundary
circle = plt.Circle((0,0), 1, color='white', fill=False, lw=2, alpha=0.8)
ax.add_patch(circle)

# Generation rings (color-coded, fading outward)
for gen in range(7):
    mask = np.array(shell_labels) == gen
    if np.any(mask):
        alpha = 0.95 if gen < 3 else 0.4  # Highlight first three
        size = 40 if gen < 3 else 10
        ax.scatter(points_2d[mask,0], points_2d[mask,1], c=colors[gen], s=size, edgecolors='white', linewidth=0.3, alpha=alpha)

# Annotations
ax.text(0, 0.05, 'Generation 0 (Core)', color=colors[0], ha='center', va='bottom', fontsize=12)
ax.text(0, 0.4, 'Gen 1', color=colors[1], ha='center', fontsize=14)
ax.text(0, 0.7, 'Gen 2', color=colors[2], ha='center', fontsize=14)
ax.text(0, 0.9, 'Gen 3+ (crowded boundary)', color='gray', ha='center', fontsize=10)

plt.title('Why Three Generations?\nHyperbolic + Golden-Ratio Radial Growth → Natural Tripling Before Saturation', 
          color='white', fontsize=16, pad=30)
plt.tight_layout()
plt.savefig('rha_three_generations.png', dpi=300, facecolor='black')
plt.show()
