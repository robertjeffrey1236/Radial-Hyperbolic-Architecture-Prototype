import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem

# Example molecule SMILES (change for different compounds)
smiles = 'Cn1cnc2c1c(=O)n(c(=O)n2C)C'  # Caffeine

# Generate molecule and 2D coords
mol = Chem.MolFromSmiles(smiles)
AllChem.Compute2DCoords(mol)
coords = mol.GetConformer().GetPositions()[:, :2]  # 2D x,y

# Normalize coords to fit in unit disk
coords -= coords.mean(0)
coords /= np.max(np.abs(coords)) * 1.2  # Scale to ~0.8 radius

# Hyperbolic Poincaré projection (simple tanh map for distortion)
r = np.linalg.norm(coords, axis=1)
poincare_r = 0.99 * np.tanh(r / 1.5)  # Distort to hyperbolic
theta = np.arctan2(coords[:,1], coords[:,0])
x_hyper = poincare_r * np.cos(theta)
y_hyper = poincare_r * np.sin(theta)

# Plot in Poincaré disk
fig, ax = plt.subplots(figsize=(10,10), facecolor='black')
ax.set_aspect('equal')
ax.axis('off')

circle = plt.Circle((0,0), 1, color='white', fill=False, lw=2, alpha=0.7)
ax.add_patch(circle)

# Atoms as gold points
ax.scatter(x_hyper, y_hyper, c='gold', s=100, edgecolors='white', alpha=0.9)

# Bonds as cyan lines
for bond in mol.GetBonds():
    i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
    ax.plot([x_hyper[i], x_hyper[j]], [y_hyper[i], y_hyper[j]], color='cyan', lw=2, alpha=0.7)

plt.title('Chemistry Compounds in Hyperbolic Space\nMolecule Atoms Projected into Poincaré Disk', color='white', fontsize=14, pad=30)
plt.tight_layout()
plt.savefig('rha_chemistry_compounds.png', dpi=300, facecolor='black')
plt.show()
