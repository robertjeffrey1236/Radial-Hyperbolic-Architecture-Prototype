# experiments/human_body_language_codex_memory_poetry.py
# Phase 5: Memory-Infused Poetry
# Past poems stored as golden echoes — recalled and woven into new verses

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons, Slider
import random
from core.geometry import golden_spiral_points, poincare_disk_project

N_POINTS = 25000
DIM = 37

# Lattice
points_nd = golden_spiral_points(n_points=N_POINTS, dim=DIM, radius_scale=0.99)
points_2d = poincare_disk_project(points_nd[:, :2])

# Vocabulary
sensory_vocab = {
    'sight': ['light', 'vision', 'color', 'reveal', 'gaze', 'illuminate', 'behold', 'dawn', 'clarity'],
    'hearing': ['sound', 'resonance', 'echo', 'harmony', 'vibration', 'tone', 'silence', 'whisper', 'song'],
    'smell': ['scent', 'aroma', 'essence', 'fragrance', 'breath', 'trace', 'bloom', 'memory'],
    'taste': ['flavor', 'sweet', 'bitter', 'savor', 'nectar', 'essence', 'nourish', 'truth'],
    'touch': ['feel', 'touch', 'texture', 'warmth', 'pressure', 'embrace', 'ground', 'pulse', 'caress'],
}

base_codex = {
    'root': ['earth', 'ground', 'stable', 'body', 'deep', 'foundation'],
    'sacral': ['flow', 'create', 'emotion', 'river', 'desire', 'dance'],
    'solar_plexus': ['fire', 'power', 'will', 'sun', 'action', 'radiance'],
    'heart': ['love', 'compassion', 'unity', 'rose', 'connection', 'embrace'],
    'throat': ['truth', 'voice', 'expression', 'sky', 'word', 'breath'],
    'third_eye': ['vision', 'insight', 'awareness', 'moon', 'dream', 'see'],
    'crown': ['divine', 'source', 'oneness', 'bliss', 'eternal', 'infinite'],
}

# Active senses & state
active_senses = {k: True for k in sensory_vocab}
current_region = 'heart'
coherence = 0.7

# Memory: past poems stored with position & coherence
poem_memory = []  # List of (poem_lines, position, coherence)

fig, ax = plt.subplots(figsize=(16, 28))
ax.set_facecolor('black')
plt.subplots_adjust(left=0.3, bottom=0.2)

lattice = ax.scatter(points_2d[:, 0], points_2d[:, 1], c='cyan', s=5, alpha=0.4)

# Memory echoes
memory_scatters = []

# Current poem display
poem_text = ax.text(0, -1.4, "The poet awakens...\nMemory stirs...", color='white', fontsize=15, ha='center', va='top', linespacing=1.8)

def compose_memory_poetry():
    sense
