import torch
import math
import matplotlib.pyplot as plt
import networkx as nx
import random
import numpy as np
from scipy.spatial import ConvexHull
from torch import cdist

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

# Emerald Tablet Integration: Parse and Map (Step 1)
emerald_aphorisms = [
    "Tis true without lying, certain & most true.",
    "That wch is below is like that wch is above & that wch is above is like that wch is below to do ye miracles of one only thing.",
    "And as all things have been & arose from one by ye mediation of one: so all things have their birth from this one thing by adaptation.",
    "The Sun is its father, the moon its mother,",
    "The wind hath carried it in its belly, the earth its nourse.",
    "The father of all perfection in ye whole world is here.",
    "Its force or power is entire if it be converted into earth.",
    "Seperate thou ye earth from ye fire, ye subtile from the gross sweetly wth great indoustry.",
    "It ascends from ye earth to ye heaven & again it desends to ye earth & receives ye power of things above & below.",
    "By this means you shall have ye glory of ye whole world & thereby all obscurity shall fly from you.",
    "Its force is above all force. ffor it overcomes every subtile thing & penetrates every solid thing.",
    "So was ye world created.",
    "From this are & do come admirable adaptaions whereof ye means (Or process) is here in this.",
    "Hence I am called Hermes Trismegist, having the three parts of ye philosophy of ye whole world.",
    "That wch I have said of ye operation of ye Sun is accomplished & ended."
]
num_aphorisms = len(emerald_aphorisms)  # 15
recursion_depth = num_aphorisms // 3  # 5, from "three parts"
branching_factor = 3  # From "three parts"
scale_factor = 1 / PHI  # Adaptation efficiency

def golden_spiral_points(n_points=500, direction=1.0):
    """Generate points along a logarithmic golden spiral (direction: 1 or -1 for counter-rotation)"""
    theta = torch.linspace(0, 12 * math.pi, n_points)
    r = torch.exp(theta / PHI)
    x = r * torch.cos(direction * theta)
    y = r * torch.sin(direction * theta)
    points = torch.stack([x, y], dim=1)
    norm = torch.norm(points, dim=1, keepdim=True)
    points = points / (norm + 1e-6) * 0.99  # Keep strictly inside unit disk
    return expmap0(points, c)  # Map to hyperbolic space

# Integrate into Geometry: Perturb with aphorism hashes (Step 2)
primal_points = golden_spiral_points(direction=1.0)  # Expansive ("Sun")
dual_points = golden_spiral_points(direction=-1.0)  # Contractive ("Moon")
aphorism_hashes = [abs(hash(a)) % 100 for a in emerald_aphorisms]  # Simplified hashes
for i, h in enumerate(aphorism_hashes):
    offset = torch.tensor([[h / 1000.0, 0.0]])  # Scale to small vector
    primal_points = mobius_add(expmap0(offset), primal_points, c)  # Perturb expansive
    dual_points = mobius_add(expmap0(-offset), dual_points, c)  # Perturb contractive (duality)

all_points = torch.cat([primal_points, dual_points])

# Enhance Recursion with Transformation (Step 3)
def recursive_branch(points, depth=recursion_depth, scale_factor=scale_factor):
    if depth == 0:
        return points
    # Offsets with polarity for ascent/descent (transformation principles)
    offsets = torch.tensor([[scale_factor, 0.0], [-scale_factor/2, scale_factor * PHI / 2], [-scale_factor/2, -scale_factor * PHI / 2]])
    branched = [points]
    for offset in offsets:
        offset_exp = expmap0(offset.unsqueeze(0), c)
        new_branch = mobius_add(offset_exp, points, c)
        branched.append(new_branch)
    # Adaptive scaling: divide by PHI for next level to prevent crowding
    return recursive_branch(torch.cat(branched), depth-1, scale_factor / PHI)

# Lazy loading with Tablet-derived max_dist (Step 3)
def lazy_load(points, observer=torch.zeros(1, 2), max_dist=num_aphorisms / 2.0):  # ~7.5 for power reception
    """Resolve only points within hyperbolic distance of observer"""
    dists = dist(observer, points, c)
    return points[dists < max_dist]

# Φ Bond Percolation Extension (original)
def build_and_percolate(points, edge_threshold=0.5):
    G = nx.Graph()
    num_points = len(points)
    for i in range(num_points):
        G.add_node(i, pos=points[i])
    dist_matrix = cdist(points.cpu(), points.cpu())
    mask = (dist_matrix < edge_threshold) & (dist_matrix > 0)
    rows, cols = np.nonzero(mask)
    edges = [(i, j) for i, j in zip(rows, cols) if i < j]
    G.add_edges_from(edges)
    phi_conj = (math.sqrt(5) - 1) / 2
    percolated_G = G.copy()
    for u, v in list(G.edges()):
        mean_point = (points[u] + points[v]) / 2
        r_hyper = dist0(mean_point, c)
        p = 1.5 * (phi_conj ** (r_hyper / 2))
        if random.random() > p:
            percolated_G.remove_edge(u, v)
    return percolated_G

# Zeckendorf Φ-Binary Encoding (original, used for compression metrics)
def fib_sequence(n=50):
    fib = [1, 2]
    for i in range(2, n):
        fib.append(fib[i-1] + fib[i-2])
    return fib

FIB = fib_sequence()

def zeckendorf_encode(n):
    if n == 0: return '0'
    bits = []
    for f in reversed(FIB):
        if f <= n:
            bits.append('1')
            n -= f
        else:
            if bits: bits.append('0')
    return ''.join(bits)

def zeckendorf_decode(bits):
    n = 0
    for i, b in enumerate(reversed(bits)):
        if b == '1':
            n += FIB[i]
    return n

# Run-Length Encoding (RLE) (original)
def rle_encode(data):
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

# Trigonometric Compression (original)
def to_polar(points):
    r = torch.norm(points, dim=1)
    theta = torch.atan2(points[:, 1], points[:, 0])
    return torch.stack([r, theta], dim=1)

def from_polar(polar):
    r, theta = polar[:, 0], polar[:, 1]
    return torch.stack([r * torch.cos(theta), r * torch.sin(theta)], dim=1)

# Base-φ Number System (original)
def to_base_phi(n, digits=20):
    bits = []
    for _ in range(digits):
        n *= PHI
        bit = int(n)
        bits.append(bit)
        n -= bit
    return bits

# Phononic Substrate for Emergent Dynamics (Step 4)
def simulate_phonons(G, start_node=0, freq=PHI, steps=50, damping=0.01):
    """Model Tablet 'force' as vibrations; check for amplification near 55 meV analog"""
    energy = {n: 0.0 for n in G.nodes}
    energy[start_node] = 1.0  # Seed from unity principle
    history = []
    for t in range(steps):
        new_energy = energy.copy()
        for n in G.nodes:
            if energy[n] > 0:
                for neigh in G.neighbors(n):
                    phase = math.sin(t * freq)  # Φ-harmonic (transformation cycles)
                    transfer = energy[n] * phase * (1 - damping)  # Low loss penetration
                    new_energy[neigh] += transfer
        energy = new_energy
        total_amp = sum(abs(e) for e in energy.values())
        history.append(total_amp)
    return history  # For emergence check (e.g., peaks)

# Example Usage (original, with integrations)
all_points = recursive_branch(all_points, depth=recursion_depth)  # Apply enhanced recursion
observer1 = torch.zeros(1, 2)
observer2 = expmap0(torch.tensor([[0.3, 0.2]]), c)
observers = [observer1, observer2]
visible_sets = [lazy_load(all_points, obs) for obs in observers]  # Enhanced lazy load
all_visible = torch.unique(torch.cat(visible_sets, dim=0), dim=0)
G_percolated = build_and_percolate(all_visible)

# Assign addresses (original)
for node in G_percolated.nodes:
    G_percolated.nodes[node]['address'] = zeckendorf_encode(node)

# Simulate phonons on largest component (Step 4)
components = list(nx.connected_components(G_percolated))
largest = max(components, key=len)
sub_G = G_percolated.subgraph(largest)
wave_history = simulate_phonons(sub_G)
print(f"Phonon amplitudes: {wave_history}")  # Check for emergence/amplification

# Visualization (original, run locally)
# ... (your plt code here, e.g., draw nodes/edges, observers, portals)

# Feasibility Metrics (Step 5)
print(f"Number of components: {len(components)}")
print(f"Largest cluster size: {len(largest)}")
if len(largest) > 10:
    largest_pos = np.array([G_percolated.nodes[n]['pos'].cpu().numpy() for n in largest])
    try:
        hull = ConvexHull(largest_pos)
        coverage_radius = np.max(np.linalg.norm(largest_pos, axis=1))
        print(f"Approximate coverage radius: {coverage_radius}")
    except:
        print("ConvexHull failed")
print(f"States generated: {len(all_visible)}")
print(f"Compression bits for 10^10: {len(zeckendorf_encode(10**10))}")

# Demo encodings (original)
# ... (your sample nodes, RLE, polar, base-phi demos)