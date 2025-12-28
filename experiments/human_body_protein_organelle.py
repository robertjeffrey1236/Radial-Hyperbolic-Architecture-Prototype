# experiments/human_body_protein_organelle.py
# Protein & Organelle Level — Real Molecular Structures via RDKit
# Mitochondria (ATP synthase), Hemoglobin, DNA, Ribosomes embedded in human

import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
import io
from PIL import Image
import random

# Simple molecules to embed (real SMILES)
molecules = {
    'ATP_synthase_proxy': 'CC(C)(COP(=O)(O)OP(=O)(O)OC1C(C(C(O1)N2C=NC3=C2N=CN=C3N)O)O)C(CO)O',  # Simplified ATP
    'hemoglobin_proxy': 'CC1=C(C2=CC3=C(C(=C([N-]3)C=C4C(=C(C(=C5C(=C(C(=C6[N-]5)C=C1[N-]2)C=C)C)C=C)C)C=C)C=C)C=C.[Fe+2]',  # Heme group
    'dna_fragment': 'c1cc2c(nc1)n(c=n2)C3C(C(C(O3)CO)O)O',  # Guanine base as proxy
}

# Generate 3D conformers
mols = {}
for name, smiles in molecules.items():
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    AllChem.UFFOptimizeMolecule(mol)
    mols[name] = mol

# Main human view
fig, ax = plt.subplots(figsize=(16, 24))
ax.set_facecolor('black')

# Substrate faint
theta = np.linspace(0, 2*np.pi, 1000)
ax.plot(0.9 * np.cos(theta), 0.9 * np.sin(theta), c='white', lw=3, alpha=0.3)

# Organs with molecular zoom hints
organ_centers = {
    'brain': [0.0, 0.62],
    'heart': [0.0, 0.0],
    'liver': [0.25, -0.08],
    'gut': [0.0, -0.35],
    'cells': [[-0.3, 0.2], [0.3, -0.2], [0.0, -0.6]],  # Generic cells
}

for name, pos in organ_centers.items():
    if isinstance(pos[0], list):
        for p in pos:
            ax.scatter(p[0], p[1], c='cyan', s=200, alpha=0.6)
    else:
        ax.scatter(pos[0], pos[1], c='magenta', s=400, alpha=0.7)

# Embed molecular views in organelles
def draw_mol_in_place(mol, center, scale=0.15):
    drawer = rdMolDraw2D.MolDraw2DCairo(300, 300)
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    bio = io.BytesIO(drawer.GetDrawingText())
    img = Image.open(bio)
    img = img.resize((int(300*
