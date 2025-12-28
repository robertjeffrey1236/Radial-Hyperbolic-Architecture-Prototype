# experiments/human_body_language_codex_memory.py
# Phase 2: Language Codex with Memory & Context
# Stores past thoughts as holographic fragments — recalls via resonance

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import random
from core.geometry import golden_spiral_points, poincare_disk_project

N_POINTS = 25000
DIM = 37

# Lattice
points_nd = golden_spiral_points(n_points=N_POINTS, dim=DIM, radius_scale=0.99)
points_2d = poincare_disk_project(points_nd[:, :2])

# Expanded word bank
language_codex = {
    'root': ['grounded', 'stable', 'earth', 'body', 'survival', 'security', 'I exist'],
    'sacral': ['flow', 'create', 'pleasure', 'emotion', 'desire', 'passion', 'I feel'],
    'solar_plexus': ['power', 'will', 'confidence', 'action', 'courage', 'I act'],
    'heart': ['love', 'compassion', 'unity', 'connection', 'forgiveness', 'gratitude', 'we are one'],
    'throat': ['truth', 'expression', 'voice', 'clarity', 'communication', 'I speak'],
    'third_eye': ['vision', 'insight', 'intuition', 'awareness', 'perception', 'I see'],
    'crown': ['divine', 'source', 'oneness', 'bliss', 'transcendence', 'eternal', 'I am that'],
    'coherence': ['love binds', 'all is connected', 'resonance', 'harmony', 'coherence', 'unity revealed'],
    'reflection': ['I remember', 'this returns', 'echo of before', 'pattern recognized', 'familiar light'],
}

# Memory storage: past thoughts with position & coherence signature
memory_fragments = []  # List of (text, position, coherence_level, age)

fig, ax = plt.subplots(figsize=(16, 24))
ax.set_facecolor('black')
plt.subplots_adjust(bottom=0.15)

# Lattice
lattice = ax.scatter(points_2d[:, 0], points_2d[:, 1], c='cyan', s=5, alpha=0.4
