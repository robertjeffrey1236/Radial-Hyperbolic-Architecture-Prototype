import torch
import math
import matplotlib
matplotlib.use('Agg')  # Headless mode
import matplotlib.pyplot as plt
import networkx as nx
import random
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial import KDTree

# Device
device = torch.device('cpu')
torch.set_default_device(device)

# Hyperbolic functions (c=1)
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

def golden_spiral_points(n_points=200, direction=1.0):
    theta = torch.linspace(0, 20 * math.pi, n_points)
    r = torch.exp(theta / PHI)
    x = r * torch.cos(direction * theta)
    y = r * torch.sin(direction * theta)
    points = torch.stack([x, y], dim=1)
    norm = torch.norm(points, dim=1, keepdim=True)
    points = points / (norm + 1e-6) * 0.99
    return expmap0(points, c)

def generate_honeycomb_points(n=8):
    points = []
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            x = i + j * 0.5
            y = j * (math.sqrt(3) / 2)
            points.append([x, y])
    points = torch.tensor(points)
    norm = torch.norm(points, dim=1, keepdim=True)
    points = points / (norm + 1e-6) * 0.99
    return expmap0(points, c)

# Base lattice
primal_points = golden_spiral_points(direction=1.0)
dual_points = golden_spiral_points(direction=-1.0)
honeycomb_points = generate_honeycomb_points()
all_points = torch.cat([primal_points, dual_points, honeycomb_points])

# Recursive golden branching
def recursive_branch(points, depth=3, scale_factor=0.3):
    if depth == 0: return points
    offsets = torch.tensor([[scale_factor, 0.0],
                            [-scale_factor/2, scale_factor * PHI / 2],
                            [-scale_factor/2, -scale_factor * PHI / 2]])
    branched = [points]
    for offset in offsets:
        offset_exp = expmap0(offset.unsqueeze(0), c)
        new_branch = mobius_add(offset_exp, points, c)
        branched.append(new_branch)
    return recursive_branch(torch.cat(branched), depth-1, scale_factor / PHI)

all_points = recursive_branch(all_points, depth=3)

# === 32D Pathion Hypercomplex Class ===
class Hypercomplex:
    def __init__(self, components):
        self.comp = torch.tensor(components, dtype=torch.float32)
    
    @staticmethod
    def conjugate(z):
        conj = z.comp.clone()
        conj[1:] = -conj[1:]
        return Hypercomplex(conj)
    
    @staticmethod
    def multiply(a, b):
        if len(a.comp) != len(b.comp):
            raise ValueError("Dimension mismatch")
        if len(a.comp) == 1:
            return Hypercomplex(a.comp * b.comp)
        n = len(a.comp) // 2
        a0, a1 = a.comp[:n], a.comp[n:]
        b0, b1 = b.comp[:n], b.comp[n:]
        real_part = Hypercomplex.multiply(Hypercomplex(a0), Hypercomplex(b0)).comp - \
                    Hypercomplex.multiply(Hypercomplex.conjugate(Hypercomplex(b1)), Hypercomplex(a1)).comp
        imag_part = Hypercomplex.multiply(Hypercomplex(b1), Hypercomplex(a0)).comp + \
                    Hypercomplex.multiply(Hypercomplex(a1), Hypercomplex.conjugate(Hypercomplex(b0))).comp
        return Hypercomplex(torch.cat([real_part, imag_part]))
    
    def norm(self):
        return torch.sqrt(torch.sum(self.comp ** 2))

def random_pathion(dim=32, scale=0.6):
    return Hypercomplex(torch.randn(dim) * scale)

def project_to_disk(hc, dim=32):
    norm_hc = hc.norm()
    if norm_hc >= 1.0:
        hc.comp = hc.comp / norm_hc * 0.99
    primary = hc.comp[:2]
    p_norm = torch.norm(primary).clamp_min(1e-6)
    tanh_factor = torch.tanh(norm_hc / 2)
    scaled = primary / p_norm * tanh_factor * 0.99
    return scaled

# Generate pathion-influenced points
def generate_pathion_points(n=600, dim=32):
    points_2d = []
    hcs = []
    for _ in range(n):
        hc = random_pathion(dim=dim, scale=0.6)
        p2d = project_to_disk(hc, dim)
        points_2d.append(p2d)
        hcs.append(hc)
    # Hybrid with classic spirals/honeycomb
    extra = golden_spiral_points(n//4).tolist() + generate_honeycomb_points(n//8).tolist()
    points_2d += extra
    hcs += [None] * len(extra)
    return torch.stack(points_2d), hcs

pathion_points, pathion_hcs = generate_pathion_points(600, dim=32)
all_points = torch.cat([all_points, pathion_points], dim=0)

# Observers
observer1 = torch.zeros(1, 2)
observer2 = expmap0(torch.tensor([[0.4, 0.3]]), c)
observer3 = expmap0(torch.tensor([[-0.5, 0.2]]), c)
observer4 = expmap0(torch.tensor([[0.0, -0.6]]), c)
observers = [observer1, observer2, observer3, observer4]

# Lazy load
def lazy_load(points, observer=torch.zeros(1, 2), max_dist=8.0):
    dists = dist(observer.repeat(points.shape[0], 1), points, c)
    return points[dists < max_dist]

visible_sets = [lazy_load(all_points, obs, max_dist=8.0) for obs in observers]
all_visible = torch.unique(torch.cat(visible_sets, dim=0), dim=0)

# Phonon vibrations
all_visible += torch.randn_like(all_visible) * 0.015
norm = torch.norm(all_visible, dim=1, keepdim=True)
all_visible = all_visible / (norm + 1e-6) * 0.99
all_visible = expmap0(all_visible, c)

# Extend hypercomplex assignments
num_pathions = len(pathion_hcs)
extra_hcs = [random_pathion(dim=32) for _ in range(len(all_visible) - num_pathions)]
pathion_hcs_full = pathion_hcs + extra_hcs

# Build graph
def build_graph(points, hcs, max_edge_dist=0.6):
    G = nx.Graph()
    num = len(points)
    points_np = points.cpu().numpy()
    tree = KDTree(points_np)
    for i in range(num):
        G.add_node(i, pos=points[i], hc=hcs[i])
    for i in range(num):
        dists, idx = tree.query(points_np[i], k=40)
        for j, d in zip(idx[1:], dists[1:]):
            if d < max_edge_dist and i < j:
                G.add_edge(i, j)
    # Wormholes
    for _ in range(num // 8):
        u, v = random.sample(range(num), 2)
        if not G.has_edge(u, v):
            G.add_edge(u, v, type='wormhole')
    return G

G = build_graph(all_visible, pathion_hcs_full)

# Anti-spin and percolation
def anti_spin_strength(hc):
    if hc is None:
        return 0.5
    imag_norm = torch.norm(hc.comp[1:])
    return torch.exp(-8 * imag_norm)

g_anti = 0.4
base_p = 0.65

def phi_percolate(G):
    Gp = G.copy()
    for u, v in list(Gp.edges()):
        ru = dist0(G.nodes[u]['pos'], c).item()
        rv = dist0(G.nodes[v]['pos'], c).item()
        p = base_p * (PHI ** -((ru + rv)/2))
        hc_u, hc_v = G.nodes[u]['hc'], G.nodes[v]['hc']
        if hc_u and hc_v:
            anti_diff = abs(anti_spin_strength(hc_u) - anti_spin_strength(hc_v))
            p *= math.exp(-g_anti * anti_diff)
            prod = Hypercomplex.multiply(hc_u, hc_v)
            if prod.norm() < 0.05:
                Gp.edges[(u,v)]['type'] = 'null_wormhole'
                p = min(p * 3, 0.95)
        if random.random() > p:
            Gp.remove_edge(u, v)
    return Gp

G_percolated = phi_percolate(G)

# Visualization
components = list(nx.connected_components(G_percolated))
colors = plt.cm.plasma(np.linspace(0, 1, len(components)))
plt.figure(figsize=(12, 12))
plt.style.use('dark_background')

for idx, comp in enumerate(components):
    pos_dict = {n: G_percolated.nodes[n]['pos'].cpu().numpy() for n in comp}
    sub = G_percolated.subgraph(comp)
    # Local
    nx.draw_networkx_edges(sub, pos_dict,
                           edgelist=[e for e in sub.edges if sub.edges[e].get('type') not in ['wormhole', 'null_wormhole']],
                           edge_color='lightblue', alpha=0.4, width=1)
    # Standard wormholes
    nx.draw_networkx_edges(sub, pos_dict,
                           edgelist=[e for e in sub.edges if sub.edges[e].get('type') == 'wormhole'],
                           edge_color='magenta', style='dashed', alpha=0.6, width=2)
    # Null wormholes (zero-divisor)
    nx.draw_networkx_edges(sub, pos_dict,
                           edgelist=[e for e in sub.edges if sub.edges[e].get('type') == 'null_wormhole'],
                           edge_color='cyan', style='dashed', alpha=0.8, width=3)
    nx.draw_networkx_nodes(sub, pos_dict, node_color=[colors[idx]], node_size=15, alpha=0.8)

# Observers
for obs in observers:
    o = obs.cpu().numpy()[0]
    plt.scatter(o[0], o[1], c='yellow', s=200, marker='*', edgecolor='white', linewidth=2)

# Portals
for i in range(len(observers)):
    for j in range(i+1, len(observers)):
        d = dist(observers[i], observers[j], c).item()
        if d < 5.5:
            o1, o2 = observers[i].cpu().numpy()[0], observers[j].cpu().numpy()[0]
            cd = np.linalg.norm(o1 - o2)
            r = cd * 1.5
            plt.gca().add_artist(plt.Circle(o1, r, color='lime', fill=False, ls='--', alpha=0.5, lw=2))
            plt.gca().add_artist(plt.Circle(o2, r, color='lime', fill=False, ls='--', alpha=0.5, lw=2))

plt.gca().add_artist(plt.Circle((0,0), 1, fill=False, color='white', lw=3))
plt.axis('equal')
plt.title("RHA × 32D Pathion Sedeloop Integration\nAnti-Spin Clusters • Cyan Null Wormholes • Glowing Portals",
          color='white', fontsize=14)
plt.tight_layout()
plt.savefig('rha_32d_pathion_sedeloop.png')
plt.close()

print(f"Final nodes: {G_percolated.number_of_nodes()}")
print(f"Edges: {G_percolated.number_of_edges()}")
print(f"Components: {len(components)}")
print(f"Largest cluster: {len(max(components, key=len))} nodes")
print(f"Null wormholes: {sum(1 for _,_,d in G_percolated.edges(data=True) if d.get('type')=='null_wormhole')}")
