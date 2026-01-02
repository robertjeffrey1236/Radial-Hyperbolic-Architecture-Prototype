# experiments/fractal_entanglement_visualizer_js_enhanced.py
# Upgraded version of the fractal entanglement visualizer
# Now incorporates Jensen-Shannon divergence directly into the base node generation:
# - Each node carries a simple Just Intonation-based probability distribution
# - Recursion only expands children if JS divergence to parent is low enough (high similarity)
# - This creates natural resonant clusters, rotation-invariant percolation, and efficiency gains
# - Glow intensity now modulated by average JS similarity to nearby nodes (information-theoretic resonance)
# Fully standalone - run directly!

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle

golden_ratio = (1 + np.sqrt(5)) / 2

# Hyperbolic to Poincaré disk
def hyperbolic_to_poincare(z):
    r = np.abs(z)
    if r >= 1:
        return z / r * 0.99
    return 2 * z / (1 + r**2 + 1e-8)

# Dual golden spirals with gamma decay
def golden_spiral_points(num_points=500, direction=1, gamma_decay=0.05, max_depth=10):
    theta = np.linspace(0, max_depth * 2 * np.pi, num_points)
    r = np.exp(gamma_decay * theta) * (golden_ratio ** (direction * theta / (2 * np.pi)))
    points = r * np.exp(1j * theta)
    return [hyperbolic_to_poincare(p) for p in points]

# Mandelbrot escape time for fractal coloring
def mandelbrot_escape(c, max_iter=50):
    z = 0j
    for i in range(max_iter):
        z = z*z + c
        if abs(z) > 2:
            return i
    return max_iter

# Entropy and Jensen-Shannon divergence (core upgrade)
def entropy(x):
    x = x[x > 0]
    return -np.sum(x * np.log(x))

def jensen_shannon_divergence(p, q, eps=1e-12):
    p = np.asarray(p) + eps
    q = np.asarray(q) + eps
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    return 0.5 * (entropy(p) + entropy(q)) - entropy(m)

# Just Intonation base ratios
base_ratios = np.array([1, 3/2, 5/3, 7/4, 9/5, 11/6])

# Node probability distribution (harmonic "signature")
def node_distribution(depth, angles):
    weights = np.exp(-0.3 * depth) * np.array([np.cos(a * np.pi / 6)**2 + 0.5 for a in angles])
    dist = base_ratios * weights
    return dist / dist.sum()

# Enhanced RHA lattice generation with JS gating
def generate_rha_nodes(center=0j, depth=7, branching=6, scale_factor=0.6, js_threshold=0.35, parent_dist=None, nodes=None):
    if nodes is None:
        nodes = []
    current_angles = np.linspace(0, 2*np.pi, branching, endpoint=False)
    current_dist = node_distribution(depth, current_angles)
    
    # JS gate: only expand if similar enough to parent (or root)
    if parent_dist is not None:
        js_div = jensen_shannon_divergence(current_dist, parent_dist)
        if js_div > js_threshold:
            return nodes  # Prune dissonant branches
    
    nodes.append((center, current_dist, depth))
    
    if depth == 0:
        return nodes
    
    angle_step = 2 * np.pi / branching
    for i in range(branching):
        angle = i * angle_step + (i % 2) * (np.pi / branching)
        child_offset = scale_factor * np.exp(1j * angle) * golden_ratio**(-depth)
        child = hyperbolic_to_poincare(center + child_offset)
        generate_rha_nodes(child, depth-1, branching, scale_factor, js_threshold, current_dist, nodes)
    return nodes

# Generate the mesh
nodes_data = generate_rha_nodes(depth=7, branching=6, scale_factor=0.6, js_threshold=0.35)

print(f"JS-Enhanced RHA Mesh Generated: {len(nodes_data)} nodes")
print("This uses total Jensen-inspired gating for efficient, resonant, rotation-invariant hierarchy.")

# Extract data for visualization
nodes_complex = [n[0] for n in nodes_data]
dists = [n[1] for n in nodes_data]
depths = [n[2] for n in nodes_data]

nodes_x = [p.real for p in nodes_complex]
nodes_y = [p.imag for p in nodes_complex]

# Fractal coloring
escape_times = [mandelbrot_escape(p + 0.1j * d, max_iter=60) for p, d in zip(nodes_complex, depths)]
norm_escape = np.array(escape_times) / max(max(escape_times, default=[1]))

# Glow from JS similarity (average similarity to sample of other nodes = resonance)
js_similarities = []
sample_size = min(50, len(dists))
for dist in dists:
    others = np.random.choice(len(dists), sample_size, replace=False)
    avg_js = np.mean([jensen_shannon_divergence(dist, dists[i]) for i in others])
    js_similarities.append(1 - avg_js)  # high similarity = high glow

glows = np.array(js_similarities) * (1 + norm_escape)

# Setup plot
fig, ax = plt.subplots(figsize=(12, 12), facecolor='black')
ax.set_aspect('equal')
ax.axis('off')
ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1.1, 1.1)

ax.add_patch(Circle((0,0), 1, color='#0a001f', alpha=0.8))

# Nodes
scatter = ax.scatter(nodes_x, nodes_y, c=norm_escape, cmap='plasma', s=30 * glows**2,
                     edgecolors='white', linewidths=0.5, alpha=0.9)

# Dual spirals
spiral_in = golden_spiral_points(direction=-1, gamma_decay=0.03)
spiral_out = golden_spiral_points(direction=1, gamma_decay=0.02)
ax.plot([p.real for p in spiral_in], [p.imag for p in spiral_in], color='#ffaa00', lw=1.5, alpha=0.7)
ax.plot([p.real for p in spiral_out], [p.imag for p in spiral_out], color='#00ffff', lw=1.5, alpha=0.7)

# Toroidal mandalas at high-resonance hubs
high_hubs_idx = np.argsort(glows)[-15:]
for idx in high_hubs_idx:
    hub = nodes_complex[idx]
    radius = 0.04 + 0.04 * glows[idx]
    ax.add_patch(Circle((hub.real, hub.imag), radius, color='#ff00ff', alpha=0.3, fill=False, lw=2))
    ax.add_patch(Circle((hub.real, hub.imag), radius*0.6, color='#ffff00', alpha=0.2, fill=False, lw=1))

# Merkaba pulse
merkaba_pulse = ax.add_patch(Circle((0,0), 0.1, color='#ffffff', alpha=0.3))

# Animation
def animate(frame):
    pulse_radius = 0.08 + 0.05 * np.sin(frame * 0.1)
    merkaba_pulse.set_radius(pulse_radius)
    merkaba_pulse.set_alpha(0.3 + 0.2 * np.abs(np.sin(frame * 0.05)))
    
    sizes = 30 * glows**2 * (1 + 0.1 * np.sin(frame * 0.08 + depths))
    scatter.set_sizes(sizes)
    return scatter, merkaba_pulse

ani = FuncAnimation(fig, animate, frames=200, interval=50, blit=False, repeat=True)

plt.tight_layout()
plt.show()
