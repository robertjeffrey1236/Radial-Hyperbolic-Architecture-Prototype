# core/seed_universe.py
import numpy as np
from .geometry import golden_spiral_points, poincare_disk_project

class MinimalUniverse:
    def __init__(self, binary_seed: str, dim: int = 37):
        self.seed = binary_seed.replace(" ", "")
        self.dim = dim
        self.points = self.grow_from_seed()

    def grow_from_seed(self):
        runs = [len(run) for run in self.seed.split('0') if run]
        n_points = sum(runs) + len(runs) * 10  # Amplify growth
        points = golden_spiral_points(n_points, self.dim)
        points = poincare_disk_project(points)
        
        # Modulate by seed runs (Φ-percolation-like pruning)
        mask = np.random.random(n_points) < (sum(runs) / len(self.seed))
        return points[mask]
    
    def get_lattice(self):
        return self.points
