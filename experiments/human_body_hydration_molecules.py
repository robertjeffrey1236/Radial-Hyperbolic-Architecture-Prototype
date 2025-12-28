# experiments/human_body_hydration_molecules.py
# Hydration Shells + Creatine & Amino Acids
# Water structured around biomolecules — realistic cellular environment

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from rdkit import Chem
from rdkit.Chem import AllChem

fig, ax = plt.subplots(figsize=(18, 28))
ax.set_facecolor('black')
ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1.4, 1.0)
ax.axis('off')

# Human silhouette
theta = np.linspace(0, 2*np.pi, 400)
body = plt.Circle((0, 0), 0.9, color='deepskyblue', fill=False, lw=4, alpha=0.3)
ax.add_patch(body)

# Title
title = ax.set_title("Hydration Shells + Key Molecules\nCreatine • Amino Acids • Structured Water Layers", color='white', fontsize=20)

# === Key Biomolecules ===
molecules = {
    'creatine': 'CN(CC(=O)O)C(=N)N',           # Creatine (muscle/brain energy)
    'glycine': 'C(C(=O)O)N',                   # Simplest amino acid
    'glutamate': 'C(CC(=O)O)C(C(=O)O)N',        # Charged, excitatory
    'lysine': 'C(CCN)CC(C(=O)O)N',             # Positive charge
}

# Generate 3D → 2D coords
mols_2d = {}
for name, smiles in molecules.items():
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.Compute2DCoords(mol)
    mols_2d[name] = mol

# Place molecules in high-relevant regions
placements = [
    ('creatine', [0.0, 0.0], 0.15),    # Heart/muscle
    ('creatine', [0.0, 0.62], 0.12),  # Brain
    ('glycine', [0.0, -0.35], 0.1),    # Gut
    ('glutamate', [-0.3, 0.1], 0.1),   # Left side
    ('lysine', [0.3, -0.2], 0.1),      # Right side
]

# Draw molecules + hydration shells
hydration_layers = []
for name, center, scale in placements:
    mol = mols_2d[name]
    
    # Draw molecule (simplified as cluster)
    conf = mol.GetConformer()
    coords = np.array([conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())])[:, :2] * scale
    coords += center
    ax.scatter(coords[:, 0], coords[:, 1], c='yellow', s=80, alpha=0.9, edgecolor='gold', zorder=10)
    
    # Hydration shells — 2–3 ordered water layers
    for layer in range(1, 4):
        radius = 0.08 * layer * scale
        shell_theta = np.linspace(0, 2*np.pi, 30)
        shell_x = center[0] + radius * np.cos(shell_theta)
        shell_y = center[1] + radius * np.sin(shell_theta)
        shell = ax.scatter(shell_x, shell_y, c='cyan', s=20, alpha=0.8 - layer*0.2)
        hydration_layers.append(shell)

# === Bulk + Structured Water Background ===
# Bulk H2O
bulk_x = np.random.uniform(-0.8, 0.8, 1200)
bulk_y = np.random.uniform(-1.1, 0.8, 1200)
mask = bulk_x**2 + bulk_y**2 < 0.81
bulk = ax.scatter(bulk_x[mask], bulk_y[mask], c='lightblue', s=15, alpha=0.5)

# Mineral ions
ions_x = np.random.uniform(-0.7, 0.7, 200)
ions_y = np.random.uniform(-1.0, 0.7, 200)
ion_mask = ions_x**2 + ions_y**2 < 0.7
ions = ax.scatter(ions_x[ion_mask], ions_y[ion_mask], c='gold', s=20, alpha=0.7, marker='D')

# Breath pulse — hydration ordering
t = 0
def animate(frame):
    global t
    t += 0.05
    pulse = np.sin(t * 1.5) * 0.1
    
    # Hydration shells pulse with breath
    for i, layer in enumerate(hydration_layers):
        alpha = 0.6 + 0.3 * (np.sin(t + i*0.5) + 1)/2
        layer.set_alpha(alpha)
    
    # Bulk water subtle flow
    bulk.set_offsets(bulk.get_offsets() + pulse * np.random.randn(len(bulk.get_offsets()), 2) * 0.01)
    
    phase = "Inhale — Flow" if np.sin(t * 1.5) > 0 else "Exhale — Order"
    title.set_text(f"Hydration Shells + Molecules\n{phase} — Water Structures Around Life")

anim = FuncAnimation(fig, animate, interval=80, repeat=True)

# Boundary
boundary = plt.Circle((0, 0), 1, color='cyan', fill=False, ls='--', lw=5)
ax.add_patch(boundary)

plt.show()

print("💧🧬 Hydration Shells + Molecules activated")
print("Features:")
print("- Creatine in heart/brain")
print("- Glycine, glutamate, lysine placed anatomically")
print("- 2–3 hydration shells around each (ordered water)")
print("- Bulk water + mineral ions")
print("- Breath-linked pulsing/ordering")
print("The human is now chemically hydrated — water as living matrix")
