import numpy as np
import matplotlib.pyplot as plt

phi = (1 + np.sqrt(5)) / 2

# Planet names and approximate real semi-major axes (AU) scaled with golden-ratio for toy model
planets = ['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune']
base_au = 0.1  # Starting scale
au_scaled = [base_au * (phi ** i) for i in range(len(planets))]

# Hyperbolic projection to Poincaré disk
r_poincare = 0.99 * np.tanh(np.array(au_scaled) * 2)  # Adjust for crowding

# Plot
fig, ax = plt.subplots(figsize=(10,10), facecolor='black')
ax.set_aspect('equal')
ax.axis('off')

# Disk boundary
circle = plt.Circle((0,0), 1, color='white', fill=False, lw=2, alpha=0.7)
ax.add_patch(circle)

# Sun
ax.scatter(0, 0, c='yellow', s=400, edgecolors='orange', linewidth=2, alpha=0.95, label='Sun')

# Orbits (concentric rings)
for r in r_poincare:
    orbit = plt.Circle((0,0), r, color='cyan', fill=False, lw=1, alpha=0.5)
    ax.add_patch(orbit)

# Planets (colored dots on orbits)
colors = ['gray', 'orange', 'blue', 'red', 'brown', 'gold', 'lightblue', 'darkblue']
sizes = [20, 40, 45, 30, 120, 100, 70, 65]
for i, (r, name, col, sz) in enumerate(zip(r_poincare, planets, colors, sizes)):
    angle = np.pi / 4  # Arbitrary positions for visual
    x = r * np.cos(angle + i * 0.3)
    y = r * np.sin(angle + i * 0.3)
    ax.scatter(x, y, c=col, s=sz, edgecolors='white', alpha=0.9)
    ax.text(x, y + 0.05, name, color='white', ha='center', fontsize=10)

plt.title('Sun-Centric Planetary System Toy Model\nGolden-Ratio Scaled Orbits in Hyperbolic (Poincaré) Space', 
          color='white', fontsize=14, pad=30)
plt.tight_layout()
plt.savefig('rha_planetary_system.png', dpi=300, facecolor='black')
plt.show()
