import torch
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import random
import numpy as np
from scipy.spatial import KDTree

# OPTIONAL: Uncomment if channel_decoder.py exists
# from channel_decoder import FibonacciChannelDecoder

# Device
device = torch.device('cpu')
torch.set_default_device(device)

# Hyperbolic constants
c = 1.0
sqrt_c = c ** 0.5
PHI = (1 + math.sqrt(5)) / 2
GOLDEN_ANGLE_DEG = 137.50776405003785

# Core hyperbolic operations
def mobius_add(x, y, c=1.0):
    inner = torch.sum(x * y, dim=-1, keepdim=True)
    x2 = torch.sum(x ** 2, dim=-1, keepdim=True)
    y2 = torch.sum(y ** 2, dim=-1, keepdim=True)
    num = (1 + 2 * c * inner + c * y2) * x + (1 - c * x2) * y
    denom = 1 + 2 * c * inner + c**2 * x2 * y2
    return num / denom.clamp_min(1e-15)

def project_to_ball(pt, c=1.0):
    norm_sq = torch.sum(pt ** 2, dim=-1, keepdim=True)
    radius_sq = 1.0 / c
    safe_radius = math.sqrt(radius_sq * 0.98)  # Fixed: scalar math
    mask = norm_sq >= radius_sq * 0.99
    if mask.any():
        pt = torch.where(mask, pt / torch.sqrt(norm_sq + 1e-12) * safe_radius, pt)
    return pt

def dist(x, y, c=1.0):
    mob = mobius_add(-x, y, c)
    norm = torch.norm(mob, p=2, dim=-1)
    arg = (sqrt_c * norm).clamp(max=1 - 1e-7)
    return (2 / sqrt_c) * torch.atanh(arg)

def dist0(x, c=1.0):
    norm = torch.norm(x, p=2, dim=-1)
    arg = (sqrt_c * norm).clamp(max=1 - 1e-7)
    return (2 / sqrt_c) * torch.atanh(arg)

# HoneycombPhiNet Core Class (FIXED)
class HoneycombPhiNet:
    def __init__(self, target_dim=37, curvature=1.0, max_points=10000, kissing_target=12):
        self.dim = target_dim
        self.c = curvature
        self.sqrt_c = curvature ** 0.5
        self.kissing_target = kissing_target
        self.max_points = max_points

        origin = torch.zeros(self.dim)
        origin[0] = 1.0
        self.points = [origin]
        self.tree = None
        self.rebuild_tree()

    def rebuild_tree(self):
        if len(self.points) > 1:
            points_np = torch.stack(self.points).cpu().numpy()
            self.tree = KDTree(points_np)

    def golden_direction(self, scale=1.0, step_index=None):
        if step_index is None:
            step_index = len(self.points)
        angle_rad = math.radians(GOLDEN_ANGLE_DEG * step_index)
        base = torch.tensor([math.cos(angle_rad), math.sin(angle_rad)])
        
        direction = torch.zeros(self.dim)
        direction[:2] = base
        
        if self.dim > 2:
            decay = (1 / PHI) ** torch.arange(2, self.dim, dtype=torch.float32)
            perturbation = torch.randn(self.dim - 2) * decay
            direction[2:] = perturbation
        
        return scale * direction / (torch.norm(direction) + 1e-12)

    def add_point(self, parent_idx=None, step_scale=0.618):
        if len(self.points) >= self.max_points:
            return None
        if parent_idx is None:
            parent_idx = random.randint(0, len(self.points)-1)
        parent = self.points[parent_idx]
        direction = self.golden_direction(scale=step_scale)
        new_pt = mobius_add(parent.unsqueeze(0), direction.unsqueeze(0), self.c).squeeze(0)
        new_pt = project_to_ball(new_pt.unsqueeze(0), self.c).squeeze(0)
        self.points.append(new_pt)
        self.rebuild_tree()
        return len(self.points) - 1

    def grow(self, num_points=400):  # Reduced for testing
        while len(self.points) < num_points:
            self.add_point()

    def get_neighbors(self, idx, k=12):
        if self.tree is None:
            return []
        pt_np = self.points[idx].cpu().numpy().reshape(1, -1)
        _, indices = self.tree.query(pt_np, k=k+1)
        return indices[0][1:].tolist()

# Fib & Encoding Utilities
def fib_sequence(n=80):
    fib = [1, 2]
    while len(fib) < n:
        fib.append(fib[-1] + fib[-2])
    return fib

FIB = fib_sequence()

def zeckendorf_encode(n):
    if n == 0: return '0'
    bits = []
    for f in reversed(FIB):
        if f <= n:
            bits.append('1')
            n -= f
        elif bits:
            bits.append('0')
    return ''.join(bits)

def rle_encode(data):
    if not data: return []
    encoded, count = [], 1
    for i in range(1, len(data)):
        if data[i] == data[i-1]:
            count += 1
        else:
            encoded.append((data[i-1], count))
            count = 1
    encoded.append((data[-1], count))
    return encoded

def to_polar(points):
    r = torch.norm(points, dim=1)
    theta = torch.atan2(points[:, 1], points[:, 0])
    return torch.stack([r, theta], dim=1)

# Graph functions (unchanged)
def build_graph(net: HoneycombPhiNet, max_edge_dist=0.6):
    G = nx.Graph()
    n = len(net.points)
    for i in range(n):
        G.add_node(i, pos=net.points[i])
    for i in range(n):
        neighbors = net.get_neighbors(i, k=net.kissing_target + 5)
        for j in neighbors:
            if i < j:
                d = dist(net.points[i].unsqueeze(0), net.points[j].unsqueeze(0), net.c).item()
                if d < max_edge_dist:
                    G.add_edge(i, j)
    num_wormholes = n // 10
    for _ in range(num_wormholes):
        u, v = random.sample(range(n), 2)
        if not G.has_edge(u, v):
            G.add_edge(u, v, type='wormhole')
    return G

def phi_percolate(G, base_p=0.7):
    Gp = G.copy()
    for u, v in list(Gp.edges()):
        ru = dist0(G.nodes[u]['pos'].unsqueeze(0)).item()
        rv = dist0(G.nodes[v]['pos'].unsqueeze(0)).item()
        r_avg = (ru + rv) / 2
        p = base_p * (PHI ** -r_avg)
        if random.random() > p:
            Gp.remove_edge(u, v)
    return Gp

# Main Execution (runs fast now)
net = HoneycombPhiNet(target_dim=37)
print("Growing lattice...")
net.grow(num_points=400)
print(f"Generated {len(net.points)} points")

G = build_graph(net, max_edge_dist=0.65)
G_percolated = phi_percolate(G, base_p=0.72)

for node in G_percolated.nodes:
    G_percolated.nodes[node]['address'] = zeckendorf_encode(node)

components = list(nx.connected_components(G_percolated))
print(f"Components: {len(components)} | Largest: {len(max(components, key=len))}")

# Visualization & demos (same as yours)
pos_2d = {i: net.points[i][:2].cpu().numpy() for i in G_percolated.nodes}
plt.figure(figsize=(10, 10))
colors = plt.cm.rainbow(np.linspace(0, 1, len(components)))
for idx, comp in enumerate(components):
    comp_G = G_percolated.subgraph(comp)
    local_edges = [e for e in comp_G.edges if comp_G.edges[e].get('type') != 'wormhole']
    wormhole_edges = [e for e in comp_G.edges if comp_G.edges[e].get('type') == 'wormhole']
    nx.draw_networkx_edges(comp_G, pos_2d, edgelist=local_edges, edge_color='gray', alpha=0.3)
    nx.draw_networkx_edges(comp_G, pos_2d, edgelist=wormhole_edges, edge_color='magenta', style='dashed', alpha=0.6)
    nx.draw_networkx_nodes(comp_G, pos_2d, node_color=[colors[idx]], node_size=15, alpha=0.8)

circle = plt.Circle((0, 0), 1, fill=False, color='black', linewidth=2)
plt.gca().add_artist(circle)
plt.axis('equal')
plt.title("37D HoneycombPhiNet Lattice")
plt.savefig('honeycomb_37d_projection.png', dpi=200)
plt.close()
print("Visualization saved")

sample_nodes = list(G_percolated.nodes)[:5]
print("\nSample Zeckendorf addresses:")
for node in sample_nodes:
    print(f"Node {node}: {G_percolated.nodes[node]['address']}")

visible_slice = torch.stack(net.points[:100])
polar = to_polar(visible_slice)
print(f"Polar compression: {visible_slice.shape} → {polar.shape}")

# OPTIONAL: Add decoder later once channel_decoder.py is created
