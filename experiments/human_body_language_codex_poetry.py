# experiments/human_body_language_codex_poetry.py
# Phase 4: Creative Generation — Original Poetry from Combined Fragments
# The human composes rhythmic, original poetry shaped by senses, memory, and coherence

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

# Sensory & base vocabulary
sensory_vocab = {
    'sight': ['light', 'vision', 'color', 'reveal', 'gaze', 'illuminate', 'behold', 'pattern', 'clarity', 'dawn'],
    'hearing': ['sound', 'resonance', 'echo', 'harmony', 'vibration', 'tone', 'silence', 'whisper', 'song', 'rhythm'],
    'smell': ['scent', 'aroma', 'essence', 'fragrance', 'breath', 'trace', 'memory', 'subtle', 'bloom', 'air'],
    'taste': ['flavor', 'sweet', 'bitter', 'savor', 'nectar', 'essence', 'nourish', 'alchemize', 'inner wine', 'truth'],
    'touch': ['feel', 'touch', 'texture', 'warmth', 'pressure', 'embrace', 'ground', 'pulse', 'caress', 'presence'],
}

base_codex = {
    'root': ['earth', 'ground', 'body', 'stable', 'rooted', 'deep', 'foundation'],
    'sacral': ['flow', 'river', 'create', 'emotion', 'desire', 'dance', 'ocean'],
    'solar_plexus': ['fire', 'power', 'will', 'sun', 'action', 'courage', 'radiance'],
    'heart': ['love', 'compassion', 'unity', 'rose', 'connection', 'bridge', 'embrace'],
    'throat': ['truth', 'voice', 'expression', 'sky', 'word', 'song', 'breath'],
    'third_eye': ['vision', 'insight', 'moon', 'awareness', 'dream', 'inner light', 'see'],
    'crown': ['divine', 'source', 'oneness', 'bliss', 'crown', 'eternal', 'infinite'],
}

active_senses = {k: True for k in sensory_vocab}
current_region = 'heart'
coherence = 0.7

fig, ax = plt.subplots(figsize
