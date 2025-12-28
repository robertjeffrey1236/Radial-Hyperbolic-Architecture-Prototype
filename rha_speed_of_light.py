import numpy as np
import matplotlib.pyplot as plt

phi = (1 + np.sqrt(5)) / 2
c = 1.0  # Normalized speed of light

# Golden-ratio radial levels in hyperbolic space
levels = np.arange(0, 12)
radial_hyper = np.cumsum(phi ** levels)
poincare_r = 0.99 * np.tanh(radial_hyper / 2)  # Approaches boundary asymptotically

# Effective speed increases toward c at boundary
effective_speed = c * poincare_r

# Plot
fig, ax = plt.subplots(figsize=(10,10), facecolor='black')
ax.set_aspect('equal')
ax.axis('off')

# Poincaré disk boundary
circle = plt.Circle((0,0), 1, color='white', fill=False, lw=2, alpha=0.8)
ax.add_patch(circle)

# Radial shells: Brightness & density represent rising effective speed
for i, r in enumerate(poincare_r):
    if r < 0.01: continue
    num_points = int(30 + 200 * effective_speed[i])  # Denser near boundary
    angles = np.linspace(0, 2*np.pi, num_points)
    x = r * np.cos(angles)
    y = r * np.sin(angles)
    intensity = 0.3 + 0.7 * effective_speed[i]  # Brighter = faster
    ax.scatter(x, y, c='cyan', s=15, alpha=intensity, edgecolors='white', linewidth=0.2)

# Labels
ax.text(0, 0.3, 'Deep Hierarchy:\nLow effective speed', color='gray', ha='center', fontsize=12)
ax.text(0, -0.9, f'Boundary:\nApproaches c = {c}\n(Universal limit)', color='cyan', ha='center', fontsize=14)

plt.title('Toy Model: Speed of Light as Hyperbolic Boundary Limit\nGolden-ratio recursion enforces finite, universal c', 
          color='white', fontsize=14, pad=30)
plt.tight_layout()
plt.savefig('rha_speed_of_light.png', dpi=300, facecolor='black')
plt.show()
