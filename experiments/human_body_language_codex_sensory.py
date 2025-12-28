# experiments/human_body_language_codex_sensory.py
# Phase 3: Sensory-Shaped Language
# Active senses influence vocabulary & metaphors — speech emerges from perception

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons
import random
from core.geometry import golden_spiral_points, poincare_disk_project

N_POINTS = 25000
DIM = 37

# Lattice
points_nd = golden_spiral_points(n_points=N_POINTS, dim=DIM, radius_scale=0.99)
points_2d = poincare_disk_project(points_nd[:, :2])

# Sensory-specific vocabulary
sensory_vocab = {
    'sight': ['see', 'light', 'vision', 'reveal', 'color', 'clarity', 'gaze', 'illuminate', 'behold', 'pattern emerges'],
    'hearing': ['hear', 'sound', 'resonance', 'echo', 'harmony', 'vibration', 'tone', 'silence speaks', 'frequency aligns'],
    'smell': ['scent', 'aroma', 'essence', 'fragrance', 'subtle trace', 'quantum note', 'vibration detected', 'memory of air'],
    'taste': ['taste', 'flavor', 'sweet', 'bitter', 'essence', 'savor', 'nourishment', 'alchemical blend', 'inner alchemy'],
    'touch': ['feel', 'touch', 'texture', 'warmth', 'pressure', 'contact', 'grounded in form', 'vibration through skin', 'presence felt'],
}

# Base chakra/regional words (same as before)
base_codex = {
    'root': ['grounded', 'stable', 'earth', 'body', 'I am'],
    'sacral': ['flow', 'create', 'emotion', 'desire', 'feel'],
    'solar_plexus': ['power', 'will', 'action', 'transform'],
    'heart': ['love', 'compassion', 'unity', 'connection', 'we are'],
    'throat': ['truth', 'expression', 'voice', 'speak'],
    'third_eye': ['vision', 'insight', 'awareness', 'see'],
    'crown': ['divine', 'source', 'oneness', 'bliss', 'I am that'],
}

# Active senses (toggle with checkboxes)
active_senses = {
    'sight': True,
    'hearing': True,
    'smell': False,
    'taste': False,
    'touch': True,
}

fig, ax = plt.subplots(figsize=(16, 24))
ax.set_facecolor('black')
plt.subplots_adjust(left=0.3, bottom=0.2)

# Lattice
lattice = ax.scatter(points_2d[:, 0], points_2d[:, 1], c='cyan', s=5, alpha=0.4)

# Sense indicators
sense_pos = {
    'sight': [-0.12, 0.72], 'sight2': [0.12, 0.72],
    'hearing': [-0.20, 0.68], 'hearing2': [0.20, 0.68],
    'smell': [0.0, 0.65],
    'taste': [0.0, 0.55],
    'touch': [0.25, 0.0],
}
sense_indicators = {}
for sense, pos in sense_pos.items():
    color = 'white'
