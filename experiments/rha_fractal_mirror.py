import numpy as np
import matplotlib.pyplot as plt

phi = (1 + np.sqrt(5)) / 2

# Core "higher power" simulation: Deep golden-ratio spiral
def generate_core_spiral(depth=10):
    theta = np.linspace(0, depth * 2 * np.pi, 1000)
    r = np.exp(theta / phi) / np.exp(depth * 2 * np.pi / phi) * 0.99
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y

# Distort to fragments: Apply mirrors/transforms (e.g., rotation, scaling, noise)
def distort_fragment(x, y, distort_type='rotate', param=0.5):
    if distort_type == 'rotate':
        theta_offset = param * 2 * np.pi
        x_new = x * np.cos(theta_offset) - y * np.sin(theta_offset)
        y_new = x * np.sin(theta_offset) + y * np.cos(theta_offset)
    elif distort_type == 'scale':
        x_new, y_new = param * x, param * y
    elif distort_type == 'noise':
        x_new = x + param * np.random.normal(size=len(x))
        y_new = y + param * np.random.normal(size=len(y))
    else:
        x_new, y_new = x, y
    r_new = np.sqrt(x_new**2 + y_new**2)
    mask = r_new < 0.99  # Keep bounded
    return x_new[mask], y_new[mask]

# Simulate universe: Core + fragments
core_x, core_y = generate_core_spiral(depth=8)  # High-power core
fragments = []
for distort in [('rotate', 0.3), ('scale', 0.8), ('noise', 0.05)]:
    frag_x, frag_y = distort_fragment(core_x, core_y, *distort)
    fragments.append((frag_x, frag_y))

# Plot
fig, ax = plt.subplots(figsize=(10,10), facecolor='black')
ax.set_aspect('equal')
ax.axis('off')

circle = plt.Circle((0,0), 1, color='white', fill=False, lw=2, alpha=0.7)
ax.add_patch(circle)

# Core (bright central)
ax.plot(core_x, core_y, color='gold', lw=2, alpha=0.9, label='Higher Power Core')

# Fragments (distorted mirrors)
colors = ['cyan', 'magenta', 'green']
for i, (fx, fy) in enumerate(fragments):
    ax.plot(fx, fy, color=colors[i], lw=1.5, alpha=0.7, label=f'Fragment {i+1}')

plt.title('Fractal Mirror Toy Model\nCore Simulation + Distorted Fragments → Efficient Universe Scaling', color='white', fontsize=14, pad=30)
plt.tight_layout()
plt.savefig('rha_fractal_mirror.png', dpi=300, facecolor='black')
plt.show()
