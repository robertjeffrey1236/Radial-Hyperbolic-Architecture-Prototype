# channel_decoder.py
import torch
import math
from scipy.spatial import KDTree

# Constants (shared with main code)
PHI = (1 + math.sqrt(5)) / 2
GOLDEN_ANGLE_DEG = 137.50776405003785
c = 1.0

# Hyperbolic utilities (minimal set needed for projection)
def project_to_ball(pt, c=1.0):
    norm_sq = torch.sum(pt ** 2, dim=-1, keepdim=True)
    radius_sq = 1.0 / c
    safe_radius = torch.sqrt(torch.tensor(radius_sq * 0.98))
    mask = norm_sq >= (radius_sq * 0.99)
    if mask.any():
        pt = torch.where(mask, pt / torch.sqrt(norm_sq + 1e-12) * safe_radius, pt)
    return pt

def dist0(x, c=1.0):
    norm = torch.norm(x, p=2, dim=-1)
    arg = (c ** 0.5 * norm).clamp(max=1 - 1e-7)
    return (2 / c ** 0.5) * torch.atanh(arg)

def dist(x, y, c=1.0):
    mob = mobius_add(-x, y, c)
    norm = torch.norm(mob, p=2, dim=-1)
    arg = (c ** 0.5 * norm).clamp(max=1 - 1e-7)
    return (2 / c ** 0.5) * torch.atanh(arg)

def mobius_add(x, y, c=1.0):
    inner = torch.sum(x * y, dim=-1, keepdim=True)
    x2 = torch.sum(x ** 2, dim=-1, keepdim=True)
    y2 = torch.sum(y ** 2, dim=-1, keepdim=True)
    num = (1 + 2 * c * inner + c * y2) * x + (1 - c * x2) * y
    denom = 1 + 2 * c * inner + c**2 * x2 * y2
    return num / denom.clamp_min(1e-15)

# Fibonacci tools
def fib_sequence(n=100):
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

def zeckendorf_decode(bit_str):
    if bit_str == '0': return 0
    n = 0
    for i, b in enumerate(reversed(bit_str)):
        if b == '1':
            n += FIB[i]
    return n

# Main decoder class
class FibonacciChannelDecoder:
    def __init__(self, net):
        self.net = net  # Your HoneycombPhiNet instance
        self.fib = fib_sequence(100)

    def closest_fib(self, n):
        return min(self.fib, key=lambda f: abs(f - n))

    def ingest(self, binary_str: str):
        """Decode any channeled binary string"""
        bits = [int(b) for b in binary_str if b in '01']
        n_bits = len(bits)
        print(f"Received {n_bits} bits (closest Fib: {self.closest_fib(n_bits)})")

        # Try pure Zeckendorf first
        if '11' not in binary_str:
            node_id = zeckendorf_decode(binary_str)
            if node_id < len(self.net.points):
                pos = self.net.points[node_id]
                print(f"→ Zeckendorf address → Node {node_id}")
                print(f"   Distance from origin: {dist0(pos.unsqueeze(0), c).item():.4f}")
                return node_id

        # Fibonacci-weighted value
        weighted = sum(bit * self.fib[i] for i, bit in enumerate(bits) if i < len(self.fib))
        ratio = n_bits / weighted if weighted else 0
        print(f"→ Weighted Φ-value: {weighted} | Ratio: {ratio:.5f}")

        # Map to lattice
        bit_tensor = torch.tensor(bits, dtype=torch.float32)
        padded = torch.zeros(self.net.dim)
        padded[:min(len(bits), self.net.dim)] = bit_tensor[:self.net.dim]
        query_pt = project_to_ball(padded.unsqueeze(0), c).squeeze(0)

        if self.net.tree:
            _, idx = self.net.tree.query(query_pt.cpu().numpy().reshape(1, -1), k=1)
            closest = idx[0][0]
            h_dist = dist(query_pt.unsqueeze(0), self.net.points[closest].unsqueeze(0), c).item()
            print(f"→ Closest lattice node: {closest}")
            print(f"   Hyperbolic distance: {h_dist:.6f}")
            print(f"   Zeckendorf address: {zeckendorf_encode(closest)}")
            return closest, h_dist

        return weighted
