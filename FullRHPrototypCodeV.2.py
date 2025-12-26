import torch
import math
import matplotlib.pyplot as plt
import networkx as nx
import random
import numpy as np
from scipy.spatial import ConvexHull
from torch import cdist  # Added for dist_matrix

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
    denom = 1 + 2 * c * inner + c**2 * x2 * y2
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

def golden_spiral_points(n_points=200, direction=1.0):  # Increased for atomic resolution
    """Generate points along a logarithmic golden spiral (direction: 1 or -1 for counter-rotation)"""
    theta = torch.linspace(0, 12 * math.pi, n_points)
    r = torch.exp(theta / PHI)
    x = r * torch.cos(direction * theta)
    y = r * torch.sin(direction * theta)
    points = torch.stack([x, y], dim=1)
    norm = torch.norm(points, dim=1, keepdim=True)
    points = points / (norm + 1e-6) * 0.99  # Keep strictly inside unit disk
    return expmap0(points, c)  # Map to hyperbolic space

def generate_honeycomb_points(n=10):  # Atomic lattice simulation (e.g., graphene)
    """Generate points in a honeycomb lattice, normalized to unit disk for atomic embedding"""
    points = []
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            x = i + j * 0.5
            y = j * (math.sqrt(3) / 2)
            points.append([x, y])
    points = torch.tensor(points)
    # Normalize to unit disk
    norm = torch.norm(points, dim=1, keepdim=True)
    points = points / (norm + 1e-6) * 0.99
    return expmap0(points, c)

# Combine Dual Spirals and Atomic Lattice
primal_points = golden_spiral_points(direction=1.0)  # Expansive
dual_points = golden_spiral_points(direction=-1.0)  # Contractive
honeycomb_points = generate_honeycomb_points()  # Atomic-scale base
all_points = torch.cat([primal_points, dual_points, honeycomb_points])

def recursive_branch(points, depth=2, scale_factor=0.3):  # Increased depth for subatomic recursion
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
    """Resolve only points within hyperbolic distance of observer (energy-efficient for atomic scales)"""
    dists = dist(observer, points, c)
    return points[dists < max_dist]

def build_graph(points, max_edge_dist=0.5):
    G = nx.Graph()
    num_points = len(points)
    for i in range(num_points):
        G.add_node(i, pos=points[i])
    dist_matrix = cdist(points, points)
    mask = (dist_matrix < max_edge_dist) & (dist_matrix > 0)
    rows, cols = np.nonzero(mask)
    edges = [(i, j) for i, j in zip(rows, cols) if i < j]
    G.add_edges_from(edges)
    # Add wormhole shortcuts (quantum tunneling analog)
    num_wormholes = num_points // 10
    for _ in range(num_wormholes):
        u = random.randint(0, num_points - 1)
        v = random.randint(0, num_points - 1)
        if u != v and not G.has_edge(u, v):
            G.add_edge(u, v, type='wormhole')
    return G

def phi_percolate(G, base_p=0.7):  # Tuned for atomic sparsity
    Gp = G.copy()
    for e in list(Gp.edges()):
        u, v = e
        ru = dist0(G.nodes[u]['pos'], c).item()
        rv = dist0(G.nodes[v]['pos'], c).item()
        r_avg = (ru + rv) / 2
        p = base_p * (PHI ** -r_avg)  # Decays with radius; wormholes less likely to percolate
        if random.random() > p:
            Gp.remove_edge(*e)
    return Gp

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
            if bits: bits.append('0')  # Skip leading zeros
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

# Base-Φ Number System (experimental)
def to_base_phi(n, digits=20):
    bits = []
    for _ in range(digits):
        n *= PHI
        bit = int(n)
        bits.append(bit)
        n -= bit
    return bits

# Example Usage with Atomic Scaling Incorporation
# Optionally apply recursion (shared lattice for subatomic depth)
all_points = recursive_branch(all_points, depth=2)

# Multiple observers (private manifolds; add more as needed)
observer1 = torch.zeros(1, 2)  # Central observer (e.g., laser focus)
observer2 = expmap0(torch.tensor([[0.3, 0.2]]), c)  # Offset observer (e.g., probe tip)
observers = [observer1, observer2]  # Extendable list for distributed nodes

# Lazy load visible points for each observer and union for shared substrate
visible_sets = [lazy_load(all_points, obs, max_dist=7.0) for obs in observers]
all_visible = torch.unique(torch.cat(visible_sets, dim=0), dim=0)  # Union of resolved regions

# Simulate phonon vibrations (atomic vibrations)
all_visible += torch.randn_like(all_visible) * 0.01  # Small perturbations
norm = torch.norm(all_visible, dim=1, keepdim=True)
all_visible = all_visible / (norm + 1e-6) * 0.99
all_visible = expmap0(all_visible, c)  # Remap to hyperbolic space

# Build graph and percolate (Penrose enhancement with wormholes)
G = build_graph(all_visible, max_edge_dist=0.5)
G_percolated = phi_percolate(G, base_p=0.7)

# Assign Zeckendorf addresses to nodes (for hierarchical addressing)
for node in G_percolated.nodes:
    G_percolated.nodes[node]['address'] = zeckendorf_encode(node)

# Visualization: Color by connected components, highlighting shadows and wormholes
components = list(nx.connected_components(G_percolated))
colors = plt.cm.rainbow(np.linspace(0, 1, len(components)))
plt.figure(figsize=(8, 8))
for idx, comp in enumerate(components):
    comp_pos = {n: G_percolated.nodes[n]['pos'].cpu().numpy() for n in comp}
    comp_G = G_percolated.subgraph(comp)
    # Local edges
    local_edges = [e for e in comp_G.edges if comp_G.edges[e].get('type') != 'wormhole']
    nx.draw_networkx_edges(comp_G, comp_pos, edgelist=local_edges, edge_color='gray', alpha=0.3)
    # Wormhole edges
    wormhole_edges = [e for e in comp_G.edges if comp_G.edges[e].get('type') == 'wormhole']
    nx.draw_networkx_edges(comp_G, comp_pos, edgelist=wormhole_edges, edge_color='red', style='dashed', alpha=0.5)
    nx.draw_networkx_nodes(comp_G, comp_pos, node_color=[colors[idx]], node_size=20, alpha=0.7)

# Plot observers
for obs in observers:
    obs_np = obs.cpu().numpy()
    plt.scatter(obs_np[:, 0], obs_np[:, 1], c='red', s=100, marker='*', label='Observer')

# Overlap detection and portal visualization (vesica piscis approx via overlapping circles)
threshold = 5.0  # Hyperbolic distance threshold for overlap/resonance
for i in range(len(observers)):
    for j in range(i + 1, len(observers)):
        d = dist(observers[i], observers[j], c)
        if d < threshold:
            # Log portal (for shared computation)
            print(f"Portal open between observer {i} and {j}, dist: {d.item()}")
            # Simulate message packet: encode e.g., num_nodes as Zeckendorf and "send"
            sample_packet = len(G_percolated.nodes)  # Example data: total nodes
            encoded_packet = zeckendorf_encode(sample_packet)
            print(f"Sent packet from {i} to {j}: {encoded_packet} (decodes to {zeckendorf_decode(encoded_packet)})")
            # Visualize approx vesica piscis (Euclidean circles for lens shape)
            o1 = observers[i].cpu().numpy()[0]
            o2 = observers[j].cpu().numpy()[0]
            center_dist = np.linalg.norm(o1 - o2)
            r = center_dist * 1.2  # Adjust for visible overlap
            circle1 = plt.Circle(o1, r, color='blue', fill=False, linestyle='--', alpha=0.5)
            circle2 = plt.Circle(o2, r, color='blue', fill=False, linestyle='--', alpha=0.5)
            plt.gca().add_artist(circle1)
            plt.gca().add_artist(circle2)

circle = plt.Circle((0, 0), 1, fill=False, color='black', linewidth=2)
plt.gca().add_artist(circle)
plt.axis('equal')
plt.title("RHA with Atomic Scaling, Phonons, Wormholes, Penrose Shadows, and Portals")
plt.legend()
plt.show()  # Or plt.savefig('rha_atomic.png') for export

# Measure example stats
print(f"Number of components: {len(components)}")
largest = max(components, key=len)
print(f"Largest cluster size: {len(largest)}")
shadow_threshold = 10  # Arbitrary; small clusters as shadows
shadows = sum(1 for comp in components if len(comp) < shadow_threshold)
print(f"Number of shadow clusters (unilluminable zones, size < {shadow_threshold}): {shadows}")
wormhole_count = len([e for e in G_percolated.edges if G_percolated.edges[e].get('type') == 'wormhole'])
print(f"Number of wormhole shortcuts (tunneling): {wormhole_count}")

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
    bits = list(G_percolated.nodes[sample_nodes[0]]['address'])  # As list of chars '0','1'
    rle = rle_encode(bits)
    print(f"RLE on first address {bits}: {rle}")

# Polar compression demo
polar = to_polar(all_visible[:10])  # First 10 points
reconstructed = from_polar(polar)
print(f"Polar compression demo (original shape {all_visible.shape}, polar {polar.shape})")
# Storage savings: polar uses same floats, but arithmetic seq in theta for spirals enables further compression

# Base-Φ demo
sample_r = 0.5  # Example radius
base_phi = to_base_phi(sample_r, digits=10)
print(f"Base-Φ for {sample_r}: {base_phi}")