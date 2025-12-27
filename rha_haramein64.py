import numpy as np
import matplotlib.pyplot as plt

# Cuboctahedron (vector equilibrium) - 12 vertices
def cuboctahedron_vertices(scale=1.0):
    coords = []
    for x in [-1, 1]:
        for y in [-1, 1]:
            coords.extend([[x, y, 0], [x, 0, y], [0, x, y]])
    return np.array(coords) * scale

# Regular tetrahedron vertices
def tetrahedron_vertices(scale=1.0, offset=np.zeros(3)):
    v = np.array([[1,1,1], [1,-1,-1], [-1,1,-1], [-1,-1,1]]) 
    v = v / np.linalg.norm(v[0]) * scale
    return v + offset

# Generate approximate 64-grid with fractal golden-ratio layers
def generate_haramein64_grid(max_levels=4, base_scale=0.3):
    points = []
    phi = (1 + np.sqrt(5)) / 2
    
    # Central cuboctahedron
    points.extend(cuboctahedron_vertices(base_scale))
    
    # Recursive tetrahedral layers (up/down alternating)
    for level in range(1, max_levels + 1):
        scale = base_scale * (phi ** level)
        offset_up = np.array([0, 0, scale * 0.8])
        offset_down = np.array([0, 0, -scale * 0.8])
        
        points.extend(tetrahedron_vertices(scale, offset_up))
        points.extend(tetrahedron_vertices(scale, offset_down) * np.array([1, 1, -1]))  # Inverted down
        
    return np.array(points)

# Hyperbolic (Poincaré) projection to disk
def poincare_projection(points_3d, radius=0.99):
    projected = []
    for p in points_3d:
        r = np.linalg.norm(p)
        if r > 1e-6:
            factor = radius * np.tanh(r / 2) / r   # Hyperbolic tanh projection
        else:
            factor = 0
        projected.append(p * factor)
    return np.array(projected)[:, :2]  # xy plane

# Main visualization
def plot_haramein64():
    points_3d = generate_haramein64_grid(max_levels=4)
    points_2d = poincare_projection(points_3d)
    
    fig, ax = plt.subplots(figsize=(10,10), facecolor='black')
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Disk boundary
    circle = plt.Circle((0,0), 1, color='white', fill=False, lw=2, alpha=0.8)
    ax.add_patch(circle)
    
    # Points: golden, larger in center
    sizes = 30 + 100 * np.exp(-np.linspace(0, 4, len(points_2d)))
    ax.scatter(points_2d[:,0], points_2d[:,1], c='gold', s=sizes, edgecolors='white', linewidth=0.5, alpha=0.95)
    
    # Light connections for lattice feel
    for i in range(len(points_2d)):
        for j in range(i+1, len(points_2d)):
            if np.linalg.norm(points_2d[i] - points_2d[j]) < 0.3:
                ax.plot(points_2d[[i,j],0], points_2d[[i,j],1], color='cyan', alpha=0.2, lw=0.5)
    
    plt.title('Radial-Hyperbolic Haramein 64 Tetrahedron Grid Exploration\n(Golden-ratio recursion + Poincaré projection)', color='white', fontsize=14)
    plt.tight_layout()
    plt.savefig('rha_haramein64.png', dpi=300, facecolor='black')
    plt.show()

# Run it
if __name__ == "__main__":
    plot_haramein64()
