import numpy as np
import matplotlib.pyplot as plt

phi = (1 + np.sqrt(5)) / 2

# Simulate photonic quantum wire: Spiral path with lossless ray propagation
def photonic_spiral_ray(num_rays=20, levels=10):
    rays = []
    for ray in range(num_rays):
        theta = np.linspace(0, levels * 2 * np.pi, 500)
        r = np.exp(theta / phi) / np.exp(levels * 2 * np.pi / phi) * 0.99  # Normalized to disk
        x = r * np.cos(theta + ray * 2 * np.pi / num_rays)
        y = r * np.sin(theta + ray * 2 * np.pi / num_rays)
        rays.append((x, y))
    return rays

rays = photonic_spiral_ray()

# Hyperbolic projection (Poincaré disk)
fig, ax = plt.subplots(figsize=(10,10), facecolor='black')
ax.set_aspect('equal')
ax.axis('off')

circle = plt.Circle((0,0), 1, color='white', fill=False, lw=2, alpha=0.7)
ax.add_patch(circle)

# Rays: Lossless flow (no scattering)
for x, y in rays:
    ax.plot(x, y, color='cyan', lw=1.5, alpha=0.8)

# Atomic chain hint (central line)
ax.plot([0, 0.5], [0, 0], color='magenta', lw=3, alpha=0.9)

plt.title('Quantum Wire Photonic Toy Model\nLossless Spiral Flow in Hyperbolic Space via Golden-Ratio Paths', color='white', fontsize=14, pad=30)
plt.tight_layout()
plt.savefig('rha_quantum_wire.png', dpi=300, facecolor='black')
plt.show()
