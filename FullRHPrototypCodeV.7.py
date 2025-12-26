import torch
import math
import matplotlib.pyplot as plt
import networkx as nx
import random
import numpy as np
from scipy.spatial import ConvexHull
from torch import cdist
from pyscf import gto, scf

# Device management
device = torch.device('cpu')  # Or 'cuda' if available
torch.set_default_device(device)

# Custom hyperbolic functions for Poincare ball (c=1.0, curvature -1)
c = 1.0
sqrt_c = c ** 0.5

def mobius_add(x, y, c=1.0):
    inner = torch.sum(x * y, dim=-1, keepdim=True)
    x2 = torch.sum(x ** 2, dim=-1, keepdim=True)
    y2 = torch.sum(y ** 2, dim=-1, keepdim=True)
    num = (1 + 2 * c * inner + c * y2) * x + (1 - c * x2) * y
    denom = 1 + 2 * c * inner + c ** 2 * x2 * y2
    return num / denom.clamp_min(1e-15)

def expmap0(v, c=1.0):
    norm_v = torch.norm(v, p=2, dim=-1, keepdim=True).clamp_min(1e-15)
    return torch.tanh(sqrt_c * norm_v) * (v / (sqrt_c * norm_v))

def dist(x, y, c=1.0):
    mob = mobius_add(-x, y, c)
    norm = torch.norm(mob, p=2, dim=-1)
    arg = (sqrt_c * norm).clamp(max=1 - 1e-7)
    return (2 / sqrt_c) * torch.atanh(arg)

def dist0(x, c=1.0):
    norm = torch.norm(x, p=2, dim=-1)
    arg = (sqrt_c * norm).clamp(max=1 - 1e-7)
    return (2 / sqrt_c) * torch.atanh(arg)

# Golden ratio
PHI = (1 + math.sqrt(5)) / 2

def golden_spiral_points(n_points=500, direction=1.0):
    """Generate points along a logarithmic golden spiral (direction: 1 or -1 for counter-rotation)"""
    theta = torch.linspace(0, 12 * math.pi, n_points)
    r = torch.exp(theta / PHI)
    x = r * torch.cos(direction * theta)
    y = r * torch.sin(direction * theta)
    points = torch.stack([x, y], dim=1)
    norm = torch.norm(points, dim=1, keepdim=True)
    points = points / (norm + 1e-6) * 0.99 # Keep strictly inside unit disk
    return expmap0(points, c) # Map to hyperbolic space

# Combine Dual Spirals
primal_points = golden_spiral_points(direction=1.0)
# Expansive
dual_points = golden_spiral_points(direction=-1.0)
# Contractive
all_points = torch.cat([primal_points, dual_points])

def recursive_branch(points, depth=1, scale_factor=0.3):
    if depth == 0: return points
    offsets = torch.tensor([[scale_factor, 0.0], [-scale_factor/2, scale_factor * PHI / 2], [-scale_factor/2, -scale_factor * PHI / 2]])
    branched = [points]
    for offset in offsets:
        offset_exp = expmap0(offset.unsqueeze(0), c)
        new_branch = mobius_add(offset_exp, points, c)
        branched.append(new_branch)
    # Adaptive scaling: divide by PHI for next level to prevent crowding
    return recursive_branch(torch.cat(branched), depth-1, scale_factor / PHI)

def lazy_load(points, observer=torch.zeros(1, 2), max_dist=6.0):
    """Resolve only points within hyperbolic distance of observer"""
    dists = dist(observer, points, c)
    return points[dists < max_dist]

# Φ Bond Percolation Extension with optimized graph building
def build_and_percolate(points, edge_threshold=0.5):
    G = nx.Graph()
    num_points = len(points)
    # Add nodes with positions
    for i in range(num_points):
        G.add_node(i, pos=points[i])

    # Optimized edge addition using cdist (Euclidean proxy, fast)
    dist_matrix = cdist(points.cpu(), points.cpu())
    mask = (dist_matrix < edge_threshold) & (dist_matrix > 0)
    rows, cols = np.nonzero(mask)
    edges = [(i, j) for i, j in zip(rows, cols) if i < j] # Undirected
    G.add_edges_from(edges)

    # Percolate: keep edges with prob p = phi_conj ** r_hyper, tuned for slower decay
    phi_conj = (math.sqrt(5) - 1) / 2 # ~0.618
    percolated_G = G.copy()
    for u, v in list(G.edges()):
        # r_hyper: hyperbolic dist of edge's mean point from origin
        mean_point = (points[u] + points[v]) / 2
        r_hyper = dist0(mean_point, c)
        p = 1.5 * (phi_conj ** (r_hyper / 2)) # Tuned: slower decay, multiply constant
        if random.random() > p:
            percolated_G.remove_edge(u, v)
    return percolated_G

# Zeckendorf Φ-Binary Encoding (primary recommendation for addresses/packets)
def fib_sequence(n=50):
    fib = [1, 2]
    for i in range(2, n):
        fib.append(fib[i-1] + fib[i-2])
    return fib

FIB = fib_sequence()

def zeckendorf_encode(n):
    """Encode integer n as Zeckendorf bit string (no adjacent 1s)"""
    if n == 0: return '0'
    bits = []
    for f in reversed(FIB):
        if f <= n:
            bits.append('1')
            n -= f
        else:
            if bits: bits.append('0') # Skip leading zeros
    return ''.join(bits)

def zeckendorf_decode(bits):
    """Decode back to integer"""
    n = 0
    for i, b in enumerate(reversed(bits)):
        if b == '1':
            n += FIB[i]
    return n

# Run-Length Encoding (RLE) for spiral paths or bit sequences
def rle_encode(data):
    """RLE on sequence (e.g., list of 0/1 bits)"""
    if not data: return []
    encoded = []
    count = 1
    for i in range(1, len(data)):
        if data[i] == data[i-1]:
            count += 1
        else:
            encoded.append((data[i-1], count))
            count = 1
    encoded.append((data[-1], count))
    return encoded

# Trigonometric Compression (Polar coords)
def to_polar(points):
    r = torch.norm(points, dim=1)
    theta = torch.atan2(points[:, 1], points[:, 0])
    return torch.stack([r, theta], dim=1)

def from_polar(polar):
    r, theta = polar[:, 0], polar[:, 1]
    return torch.stack([r * torch.cos(theta), r * torch.sin(theta)], dim=1)

# Base-φ Number System (experimental)
def to_base_phi(n, digits=20):
    bits = []
    for _ in range(digits):
        n *= PHI
        bit = int(n)
        bits.append(bit)
        n -= bit
    return bits

# Example Usage with Multi-Point and φ-Encoding Incorporation
# Optionally apply recursion (shared lattice)
all_points = recursive_branch(all_points, depth=1) # Reduced depth for efficiency
# Multiple observers (private manifolds; add more as needed)
observer1 = torch.zeros(1, 2) # Central observer
observer2 = expmap0(torch.tensor([[0.3, 0.2]]), c) # Offset observer
observers = [observer1, observer2] # Extendable list for distributed nodes

# Lazy load visible points for each observer and union for shared substrate
visible_sets = [lazy_load(all_points, obs, max_dist=7.0) for obs in observers]
all_visible = torch.unique(torch.cat(visible_sets, dim=0), dim=0) # Union of resolved regions

# Build and percolate graph on combined visible points (emergent network)
G_percolated = build_and_percolate(all_visible)

# Assign Zeckendorf addresses to nodes (for hierarchical addressing)
for node in G_percolated.nodes:
    G_percolated.nodes[node]['address'] = zeckendorf_encode(node)

# Visualization: Color by connected components (run locally to see plot)
components = list(nx.connected_components(G_percolated))
colors = plt.cm.rainbow(np.linspace(0, 1, len(components)))
plt.figure(figsize=(8, 8))
for idx, comp in enumerate(components):
    comp_pos = {n: G_percolated.nodes[n]['pos'].cpu().numpy() for n in comp}
    nx.draw_networkx_nodes(G_percolated.subgraph(comp), comp_pos, node_color=[colors[idx]], node_size=20, alpha=0.7)
    nx.draw_networkx_edges(G_percolated.subgraph(comp), comp_pos, edge_color='gray', alpha=0.3)

# Plot observers
for obs in observers:
    obs_np = obs.cpu().numpy()
    plt.scatter(obs_np[:, 0], obs_np[:, 1], c='red', s=100, marker='*', label='Observer')

# Overlap detection and portal visualization (vesica piscis approx via overlapping circles)
threshold = 5.0 # Hyperbolic distance threshold for overlap/ resonance
for i in range(len(observers)):
    for j in range(i + 1, len(observers)):
        d = dist(observers[i], observers[j], c)
        if d < threshold:
            # Log portal (for shared computation)
            print(f"Portal open between observer {i} and {j}, dist: {d.item()}")
            # Simulate message packet: encode e.g., num_nodes as Zeckendorf and "send"
            sample_packet = len(G_percolated.nodes) # Example data: total nodes
            encoded_packet = zeckendorf_encode(sample_packet)
            print(f"Sent packet from {i} to {j}: {encoded_packet} (decodes to {zeckendorf_decode(encoded_packet)}")

            # Visualize approx vesica piscis (Euclidean circles for lens shape)
            o1 = observers[i].cpu().numpy()[0]
            o2 = observers[j].cpu().numpy()[0]
            center_dist = np.linalg.norm(o1 - o2)
            r = center_dist * 1.2 # Adjust for visible overlap
            circle1 = plt.Circle(o1, r, color='blue', fill=False, linestyle='--', alpha=0.5)
            circle2 = plt.Circle(o2, r, color='blue', fill=False, linestyle='--', alpha=0.5)
            plt.gca().add_artist(circle1)
            plt.gca().add_artist(circle2)

circle = plt.Circle((0, 0), 1, fill=False, color='black', linewidth=2)
plt.gca().add_artist(circle)
plt.axis('equal')
plt.title("RHA with Multi-Point Observers, Φ Percolation, Portals, and Encoding")
plt.legend()
plt.show() # Or plt.savefig('rha_multi_point_encoded.png') for export

# Measure example stats
print(f"Number of components: {len(components)}")
largest = max(components, key=len)
print(f"Largest cluster size: {len(largest)}")

# Fractal dimension proxy
if len(largest) > 10:
    largest_pos = np.array([G_percolated.nodes[n]['pos'].cpu().numpy() for n in largest])
    try:
        hull = ConvexHull(largest_pos)
        coverage_radius = np.max(np.linalg.norm(largest_pos, axis=1))
        print(f"Approximate coverage radius: {coverage_radius}")
    except:
        print("ConvexHull failed (points may be degenerate)")

# Demo Φ-Encoding Features
# Sample addresses
sample_nodes = list(G_percolated.nodes)[:5]
print("Sample node addresses (Zeckendorf):")
for node in sample_nodes:
    print(f"Node {node}: {G_percolated.nodes[node]['address']}")

# RLE demo on a bit sequence (e.g., from a node's address)
if sample_nodes:
    bits = list(G_percolated.nodes[sample_nodes[0]]['address']) # As list of chars '0','1'
    rle = rle_encode(bits)
    print(f"RLE on first address {bits}: {rle}")

# Polar compression demo
polar = to_polar(all_visible[:10]) # First 10 points
reconstructed = from_polar(polar)
print(f"Polar compression demo (original shape {all_visible.shape}, polar {polar.shape})") # Storage savings: polar uses same floats, but arithmetic seq in theta for spirals enables further compression

# Base-φ demo
sample_r = 0.5 # Example radius
base_phi = to_base_phi(sample_r, digits=10)
print(f"Base-φ for {sample_r}: {base_phi}")

def load_molecule(mol_formula='C6H6', basis='3-21g'):  # Changed basis
    if mol_formula == 'C6H6':
        atom = '''
C  0.0000  1.4027  0.0000
H  0.0000  2.4902  0.0000
C -1.2148  0.7014  0.0000
H -2.1566  1.2451  0.0000
C -1.2148 -0.7014  0.0000
H -2.1566 -1.2451  0.0000
C  0.0000 -1.4027  0.0000
H  0.0000 -2.4902  0.0000
C  1.2148 -0.7014  0.0000
H  2.1566 -1.2451  0.0000
C  1.2148  0.7014  0.0000
H  2.1566  1.2451  0.0000
        '''  # Standard benzene coordinates
    else:
        atom = mol_formula
    mol = gto.M(atom=atom, basis=basis)
    mf = scf.RHF(mol)
    mf.max_cycle = 200  # Increase iterations
    mf.diis_space = 15  # Larger DIIS for stability
    mf.kernel()
    if not mf.converged:
        print("SCF not converged; try better basis or coordinates")
    coords = torch.tensor(mol.atom_coords()[:, :2])
    norm = torch.norm(coords, dim=1, keepdim=True)
    points = coords / (norm + 1e-6) * 0.99
    return expmap0(points, c), mf.e_tot

# Run the fixed load_molecule to check
atomic_points, atomic_energy = load_molecule()
print(atomic_energy)</parameter>
</xai:function_call>