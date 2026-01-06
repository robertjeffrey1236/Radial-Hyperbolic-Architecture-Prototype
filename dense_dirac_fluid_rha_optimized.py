import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial import KDTree  # For fast neighbor search

# Larger honeycomb lattice
def honeycomb_lattice(n_rows=40, n_cols=40):
    points = []
    for row in range(n_rows):
        for col in range(n_cols):
            x = col * 1.5
            y = row * np.sqrt(3) + (col % 2) * (np.sqrt(3)/2)
            points.append([x, y])
    return np.array(points)

lattice = honeycomb_lattice(40, 40)
lattice += np.random.normal(0, 0.05, lattice.shape)

# High-density particles (scaled for gaming PC)
n_particles = 5000  # Balanced for performance
particles = lattice[np.random.choice(len(lattice), n_particles, replace=True)]
particles += np.random.normal(0, 0.18, particles.shape)

vel = np.random.normal(0, 0.08, particles.shape)

center = np.mean(lattice, axis=0)
PHI = (1 + np.sqrt(5)) / 2
pulse_strength = 5.0

# Figure
fig, ax = plt.subplots(figsize=(12, 12))
ax.set_facecolor('black')
ax.set_aspect('equal')
ax.set_xlim(lattice[:,0].min()-4, lattice[:,0].max()+4)
ax.set_ylim(lattice[:,1].min()-4, lattice[:,1].max()+4)
ax.set_title('High-Density Dirac Fluid in RHA Honeycomb Toroid (KDTree Optimized)', color='white', fontsize=16)

# Subtle lattice
ax.plot(lattice[:,0], lattice[:,1], 'o', c='gray', alpha=0.08, markersize=0.8)

# Particles
sc = ax.scatter(particles[:,0], particles[:,1], c='cyan', s=10, alpha=0.85, edgecolor='none')

# Central light
central = ax.scatter([center[0]], [center[1]], c='white', s=600, edgecolor='yellow', linewidth=3)

plt.legend([central], ['Central Light (Dirac Point)'], loc='upper right', facecolor='black', labelcolor='white', fontsize=12)

# Pre-compute toroidal ranges
x_min, x_max = lattice[:,0].min(), lattice[:,0].max()
y_min, y_max = lattice[:,1].min(), lattice[:,1].max()
x_range = x_max - x_min
y_range = y_max - y_min

def update(frame):
    global vel, particles
    
    to_center = center - particles
    dist = np.linalg.norm(to_center, axis=1, keepdims=True)
    
    # Phi-modulated pulse
    force_magnitude = pulse_strength * np.sin(frame / 15 + dist.squeeze() / PHI)
    force = force_magnitude[:, np.newaxis] * to_center / (dist + 0.6)
    vel += force * 0.01
    
    # Fast KDTree neighbor alignment
    tree = KDTree(particles)
    neighbors = tree.query_ball_point(particles, r=6.0)
    avg_vel = np.zeros_like(vel)
    for i in range(n_particles):
        neighs = [j for j in neighbors[i] if j != i]
        if neighs:
            avg_vel[i] = np.mean(vel[neighs], axis=0)
    vel += 0.15 * avg_vel
    
    vel *= 0.95
    particles += vel
    
    # Toroidal wrap
    particles[:,0] = np.mod(particles[:,0] - x_min, x_range) + x_min
    particles[:,1] = np.mod(particles[:,1] - y_min, y_range) + y_min
    
    sc.set_offsets(particles)
    
    # Charge brightness
    colors = np.clip(1 / (dist.squeeze() / 8 + 0.4), 0.5, 1)
    sc.set_array(colors)
    
    # Central pulse
    pulse_size = 600 + 200 * np.sin(frame / 15)
    central.set_sizes([pulse_size])
    
    return sc, central

# High-quality animation
ani = FuncAnimation(fig, update, frames=800, interval=25, blit=True)

try:
    ani.save('dense_dirac_fluid_rha_optimized.gif', writer='pillow', fps=40, dpi=80)
    print("High-quality GIF saved successfully!")
except Exception as e:
    print(f"Save failed: {e}. Displaying live animation.")
    plt.show()

plt.show()
