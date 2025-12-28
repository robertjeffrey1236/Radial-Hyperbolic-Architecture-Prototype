# experiments/lindblad_noise.py
import numpy as np
import matplotlib.pyplot as plt
from qutip import *
from core.geometry import golden_spiral_points

# Simple 2-level systems on lattice nodes
N = 50
points = golden_spiral_points(N, dim=2)

# Example: collective dephasing Lindblad simulation
gamma = 0.1
H = tensor([sigmax() for _ in range(3)])  # Toy Hamiltonian
c_ops = [np.sqrt(gamma) * tensor([sigmaz() for _ in range(3)])]

rho0 = ket("000").proj()  # Initial ground state
tlist = np.linspace(0, 50, 200)
result = mesolve(H, rho0, tlist, c_ops)

# Plot expectation values
plt.plot(tlist, expect(sigmax() * sigmay() * sigmaz(), result.states))
plt.title("Decoherence in Hyperbolic Lattice Toy Model")
plt.xlabel("Time")
plt.ylabel("<σx σy σz>")
plt.show()
