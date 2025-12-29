# experiments/fractal_entanglement_visualizer.py
# A dedicated visualizer for the Radial-Hyperbolic-Architecture (RHA) prototype
# Integrates fractal entanglement mapping inspired by your friend's sedeloop visuals:
#   - Mandelbrot/Julia-like escape-time coloring for multi-scale quantum correlations
#   - Counter-rotating golden ratio (Φ) spirals with gamma radian decay damping
#   - Glowing filaments based on percolation thresholds
#   - Toroidal mandala overlays at high-percolation hubs
#   - Merkaba pulse animation
#   - Just Intonation harmonic hints via node sizing/color saturation

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
from math import phi as golden_ratio  # ≈1.618

# Hyperbolic geometry helpers (Poincaré disk projection)
def hyperbolic_to_poincare(z):
    """Map hyperbolic point to Poincaré disk (unit disk)."""
    r = np.abs(z)
    if r >= 1:
        return z / r * 0.99  # Clamp to inside disk
    return 2 * z / (1 + r**2 + 1e-8)

# Dual golden spirals (Krystal inward / Fibonacci outward) with damping
def golden_spiral_points(num_points=500, direction=1, gamma_decay=0.05, max_depth=10):
    """Generate points along a logarithmic golden spiral with exponential damping."""
    theta = np.linspace(0, max_depth * 2 * np.pi, num_points)
    r = np.exp(gamma_decay * theta) * (golden_ratio ** (direction * theta / (2 * np.pi)))
    points = r * np.exp(1j * theta)
    return [hyperbolic_to_poincare(p) for p in points]

# Simple Mandelbrot escape time for fractal coloring
def mandelbrot_escape(c, max_iter=50):
    z = 0j
    for i in range(max_iter):
        z = z*z + c
        if abs(z) > 2:
            return i
    return max_iter

# Generate RHA lattice nodes (recursive fractal branching in hyperbolic space)
def generate_rha_nodes(center=0j, depth=6, branching=5, scale_factor=0.6, nodes=None):
    if nodes is None:
        nodes = []
    if depth == 0:
        return
    nodes.append(center)
    angle_step = 2 * np.pi / branching
    for i in range(branching):
        angle = i * angle_step + (i % 2) * (np.pi / branching)  # Alternate offset for density
        child = center + scale_factor * np.exp(1j * angle) * golden_ratio**(-depth)
        # Hyperbolic adjustment
        child = hyperbolic_to_poincare(child)
        generate_rha_nodes(child, depth-1, branching, scale_factor, nodes)
    return nodes

# Percolation-like glow intensity
def percolation_glow(depth, threshold=0.7):
    prob = 1 - np.exp(-depth / 3)
    return prob if np.random.rand() < threshold else prob * 0.3

# Setup figure
fig, ax = plt.subplots(figsize=(12, 12), facecolor='black')
ax.set_aspect('equal')
ax.axis('off')
ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1.1, 1.1)

# Background cosmic web glow
ax.add_patch(Circle((0,0), 1, color='#0a001f', alpha=0.8))

# Generate nodes
nodes_complex = generate_rha_nodes(depth=7, branching=6)
nodes_x = [p.real for p in nodes_complex]
nodes_y = [p.imag for p in nodes_complex]
depths = [abs(p) for p in nodes_complex]  # Approximate depth proxy

# Fractal coloring: use node position as c in Mandelbrot
escape_times = [mandelbrot_escape(p + 0.1j * d, max_iter=60) for p, d in zip(nodes_complex, depths)]
norm_escape = np.array(escape_times) / max(escape_times)

# Glow based on percolation + escape time
glows = [percolation_glow(d) * (1 + 0.5 * e) for d, e in zip(depths, norm_escape)]

# Scatter for nodes
scatter = ax.scatter(nodes_x, nodes_y, c=norm_escape, cmap='inferno', s=20 * np.array(glows)**2,
                     edgecolors='white', linewidths=0.5, alpha=0.9)

# Dual spirals
spiral_in = golden_spiral_points(direction=-1, gamma_decay=0.03)  # Inward Krystal
spiral_out = golden_spiral_points(direction=1, gamma_decay=0.02)  # Outward Fibonacci
ax.plot([p.real for p in spiral_in], [p.imag for p in spiral_in], color='#ffaa00', lw=1.5, alpha=0.7)
ax.plot([p.real for p in spiral_out], [p.imag for p in spiral_out], color='#00ffff', lw=1.5, alpha=0.7)

# Toroidal mandalas at high-percolation hubs
high_hubs = [n for n, g in zip(nodes_complex, glows) if g > 0.9]
for hub in high_hubs[:10]:  # Limit for performance
    torus = Circle((hub.real, hub.imag), 0.05 + 0.03 * glows[0], color='#ff00ff', alpha=0.3, fill=False, lw=2)
    ax.add_patch(torus)
    inner = Circle((hub.real, hub.imag), 0.03, color='#ffff00', alpha=0.2, fill=False, lw=1)
    ax.add_patch(inner)

# Merkaba central pulse
merkaba_pulse = ax.add_patch(Circle((0,0), 0.1, color='#ffffff', alpha=0.3))

# Animation: Merkaba pulse + subtle node breathing
def animate(frame):
    pulse_radius = 0.08 + 0.05 * np.sin(frame * 0.1)
    merkaba_pulse.set_radius(pulse_radius)
    merkaba_pulse.set_alpha(0.3 + 0.2 * np.abs(np.sin(frame * 0.05)))
    
    # Subtle breathing on high-glow nodes
    sizes = 20 * np.array(glows)**2 * (1 + 0.1 * np.sin(frame * 0.08 + depths))
    scatter.set_sizes(sizes)
    return scatter, merkaba_pulse

ani = FuncAnimation(fig, animate, frames=200, interval=50, blit=False, repeat=True)

plt.tight_layout()
plt.show()
