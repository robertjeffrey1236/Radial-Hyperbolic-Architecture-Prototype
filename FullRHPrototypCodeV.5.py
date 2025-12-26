import torch
import math
import matplotlib.pyplot as plt
import networkx as nx
import random
import numpy as np
from scipy.spatial import ConvexHull
from torch import cdist
import cmath  # For Berry phase complex math
import mido  # For MIDI output
from pyscf import gto, scf  # For molecular loading (hessian optional for full phonons)

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

# Emerald Tablet Integration: Parse and Map (ultra-compressible seed rules)
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
recursion_depth = num_aphorisms // 3  # 5
branching_factor = 3  # From "three parts"
scale_factor = 1 / PHI  # Adaptation efficiency

def golden_spiral_points(n_points=500, direction=1.0):
    theta = torch.linspace(0, 12 * math.pi, n_points)
    r = torch.exp(theta / PHI)
    x = r * torch.cos(direction * theta)
    y = r * torch.sin(direction * theta)
    points = torch.stack([x, y], dim=1)
    norm = torch.norm(points, dim=1, keepdim=True)
    points = points / (norm + 1e-6) * 0.99
    return expmap0(points, c)

# Perturb with Emerald hashes for compressible seed
primal_points = golden_spiral_points(direction=1.0)
dual_points = golden_spiral_points(direction=-1.0)
aphorism_hashes = [abs(hash(a)) % 100 for a in emerald_aphorisms]
for i, h in enumerate(aphorism_hashes):
    offset = torch.tensor([[h / 1000.0, 0.0]])
    primal_points = mobius_add(expmap0(offset), primal_points, c)
    dual_points = mobius_add(expmap0(-offset), dual_points, c)

all_points = torch.cat([primal_points, dual_points])

# Recursion with Emerald transformation
def recursive_branch(points, depth=recursion_depth, scale_factor=scale_factor):
    if depth == 0:
        return points
    offsets = torch.tensor([[scale_factor, 0.0], [-scale_factor/2, scale_factor * PHI / 2], [-scale_factor/2, -scale_factor * PHI / 2]])
    branched = [points]
    for offset in offsets:
        offset_exp = expmap0(offset.unsqueeze(0), c)
        new_branch = mobius_add(offset_exp, points, c)
        branched.append(new_branch)
    return recursive_branch(torch.cat(branched), depth-1, scale_factor / PHI)

all_points = recursive_branch(all_points)

# Lazy loading
def lazy_load(points, observer=torch.zeros(1, 2), max_dist=num_aphorisms / 2.0):
    dists = dist(observer, points, c)
    return points[dists < max_dist]

# Percolation
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

# Zeckendorf, RLE, Polar, Base-Φ (as before)
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

def to_polar(points):
    r = torch.norm(points, dim=1)
    theta = torch.atan2(points[:, 1], points[:, 0])
    return torch.stack([r, theta], dim=1)

def from_polar(polar):
    r, theta = polar[:, 0], polar[:, 1]
    return torch.stack([r * torch.cos(theta), r * torch.sin(theta)], dim=1)

def to_base_phi(n, digits=20):
    bits = []
    for _ in range(digits):
        n *= PHI
        bit = int(n)
        bits.append(bit)
        n -= bit
    return bits

# Berry Phase
def berry_phase(path_points):
    phase = 0j
    for i in range(len(path_points) - 1):
        z1 = complex(path_points[i][0], path_points[i][1])
        z2 = complex(path_points[i+1][0], path_points[i+1][1])
        dz = (z2 - z1) / (1 - z1.conjugate() * z2)
        phase += cmath.log(1j * dz / abs(dz))
    return cmath.phase(phase) % (2 * math.pi)

# Telegeodynamics (Tesla resonance with longitudinal modes, criticality, music)
def simulate_telegeodynamics(G, start_node=0, freq=PHI, steps=50, damping=0.01):
    energy = {n: 0.0 for n in G.nodes}
    phase_dict = {n: 0.0 for n in G.nodes}
    energy[start_node] = 1.0
    history = []
    for t in range(steps):
        new_energy = energy.copy()
        new_phase = phase_dict.copy()
        for n in G.nodes:
            if energy[n] > 0:
                for neigh in G.neighbors(n):
                    mode_sign = 1 if n % 2 == 0 else -1
                    osc_phase = math.sin(t * freq) * mode_sign
                    path = [G.nodes[n]['pos'].cpu().numpy(), G.nodes[neigh]['pos'].cpu().numpy()]
                    berry_shift = berry_phase(path)
                    transfer = energy[n] * (osc_phase + math.sin(berry_shift)) * (1 - damping)
                    new_energy[neigh] += transfer
                    new_phase[neigh] += berry_shift
        energy = new_energy
        phase_dict = new_phase
        total_amp = sum(abs(e) for e in energy.values())
        history.append(total_amp)
        if total_amp > 1e6:
            print(f"Telegeodynamic criticality at step {t}: amp {total_amp}")
            break
    return history, phase_dict

# JI Music Mapping
def map_to_ji(amp, phase, base_freq=440):
    ratio = 1 + (abs(amp) % 1) * (PHI - 1) + math.sin(phase) * 0.1
    intervals = [1, 3/2, 5/4, 8/5, 9/8]
    ji_ratio = intervals[int(amp + phase) % len(intervals)]
    return int(base_freq * ji_ratio * ratio)

# Generate MIDI
def generate_midi(history, phases, filename='rha_telegeo.mid'):
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)
    for t, amp in enumerate(history):
        note = map_to_ji(amp, list(phases.values())[t % len(phases)])
        track.append(mido.Message('note_on', note=note, velocity=64, time=0))
        track.append(mido.Message('note_off', note=note, velocity=64, time=120))
    mid.save(filename)
    print(f"Saved MIDI: {filename}")

# Atomic-Scale Layer
def load_molecule(mol_formula='C6H6', basis='sto-3g'):
    if mol_formula == 'C6H6':
        atom = '''
        C 0 0 0; C 1.42 0 0; C 0.71 1.23 0; C 2.13 1.23 0; C 0.71 2.46 0; C 2.13 2.46 0;
        H 0 2.49 0; H 2.16 1.25 0; H 2.16 -1.25 0; H 0 -2.49 0; H -2.16 -1.25 0; H -2.16 1.25 0
        '''
    else:
        atom = mol_formula
    mol = gto.M(atom=atom, basis=basis)
    mf = scf.RHF(mol)
    mf.kernel()
    if not mf.converged:
        print("SCF not converged")
    coords = torch.tensor(mol.atom_coords()[:, :2])
    norm = torch.norm(coords, dim=1, keepdim=True)
    points = coords / (norm + 1e-6) * 0.99
    return expmap0(points, c), mf.e_tot

atomic_points, atomic_energy = load_molecule()
all_points = torch.cat([all_points, atomic_points])

# Probe Simulation for Atomic "Game"
edge_threshold = 0.5  # Global for percolation
def simulate_probe_observation(probe_pos=torch.tensor([[0.0, 0.0]]), field_strength=0.5):
    visible = lazy_load(all_points, expmap0(probe_pos), max_dist=7.0)
    global edge_threshold
    edge_threshold += field_strength * 0.1
    G_percolated = build_and_percolate(visible)
    wave_history, phase_dict = simulate_telegeodynamics(G_percolated)
    generate_midi(wave_history, phase_dict, 'atomic_resonance.mid')
    return G_percolated

# Example Usage
observer1 = torch.zeros(1, 2)
observer2 = expmap0(torch.tensor([[0.3, 0.2]]), c)
observers = [observer1, observer2]
visible_sets = [lazy_load(all_points, obs) for obs in observers]
all_visible = torch.unique(torch.cat(visible_sets, dim=0), dim=0)
G_percolated = build_and_percolate(all_visible)
for node in G_percolated.nodes:
    G_percolated.nodes[node]['address'] = zeckendorf_encode(node)

components = list(nx.connected_components(G_percolated))
colors = plt.cm.rainbow(np.linspace(0, 1, len(components)))
plt.figure(figsize=(8, 8))
for idx, comp in enumerate(components):
    comp_pos = {n: G_percolated.nodes[n]['pos'].cpu().numpy() for n in comp}
    nx.draw_networkx_nodes(G_percolated.subgraph(comp), comp_pos, node_color=[colors[idx]], node_size=20, alpha=0.7)
    nx.draw_networkx_edges(G_percolated.subgraph(comp), comp_pos, edge_color='gray', alpha=0.3)

for obs in observers:
    obs_np = obs.cpu().numpy()
    plt.scatter(obs_np[:, 0], obs_np[:, 1], c='red', s=100, marker='*', label='Observer')

threshold = 5.0
for i in range(len(observers)):
    for j in range(i + 1, len(observers)):
        d = dist(observers[i], observers[j], c)
        if d < threshold:
            print(f"Portal open between observer {i} and {j}, dist: {d.item()}")
            sample_packet = len(G_percolated.nodes)
            encoded_packet = zeckendorf_encode(sample_packet)
            print(f"Sent packet from {i} to {j}: {encoded_packet} (decodes to {zeckendorf_decode(encoded_packet)}")

            o1 = observers[i].cpu().numpy()[0]
            o2 = observers[j].cpu().numpy()[0]
            center_dist = np.linalg.norm(o1 - o2)
            r = center_dist * 1.2
            circle1 = plt.Circle(o1, r, color='blue', fill=False, linestyle='--', alpha=0.5)
            circle2 = plt.Circle(o2, r, color='blue', fill=False, linestyle='--', alpha=0.5)
            plt.gca().add_artist(circle1)
            plt.gca().add_artist(circle2)

circle = plt.Circle((0, 0), 1, fill=False, color='black', linewidth=2)
plt.gca().add_artist(circle)
plt.axis('equal')
plt.title("RHA with Multi-Point Observers, Φ Percolation, Portals, and Encoding")
plt.legend()
# plt.show()  # Uncomment to display

largest = max(components, key=len)
sub_G = G_percolated.subgraph(largest)
wave_history, phase_dict = simulate_telegeodynamics(sub_G)
generate_midi(wave_history, phase_dict)

probe = torch.tensor([[0.3, 0.2]])
game_G = simulate_probe_observation(probe)
print(f"Atomic game lattice: {len(game_G.nodes)} nodes, energy {atomic_energy}")

# Stats and Demos
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

sample_nodes = list(G_percolated.nodes)[:5]
print("Sample node addresses (Zeckendorf):")
for node in sample_nodes:
    print(f"Node {node}: {G_percolated.nodes[node]['address']}")

if sample_nodes:
    bits = list(G_percolated.nodes[sample_nodes[0]]['address'])
    rle = rle_encode(bits)
    print(f"RLE on first address {bits}: {rle}")

polar = to_polar(all_visible[:10])
reconstructed = from_polar(polar)
print(f"Polar compression demo (original shape {all_visible.shape}, polar {polar.shape})")

sample_r = 0.5
base_phi = to_base_phi(sample_r, digits=10)
print(f"Base-φ for {sample_r}: {base_phi}")