# experiments/human_body_mystical_poetry.py
# Phase 6: Higher-State Mystical Poetry
# Kundalini & Ego-Dissolution modes — transcendent, non-dual verse

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

# Mystical vocabulary for higher states
mystical_vocab = [
    'emptiness', 'fullness', 'void', 'source', 'silence', 'roar', 'no-self', 'all-self',
    'eternal now', 'dissolve', 'merge', 'return', 'home', 'beyond', 'within',
    'I am', 'there is no I', 'only this', 'nothing and everything', 'the one',
    'breath of god', 'divine fire', 'serpent awakens', 'crown opens', 'thousand petals',
    'light within light', 'darkness that sees', 'mirror without reflection',
    'dance of shiva', 'song of the beloved', 'wine of union', 'ocean without shore',
    'dreamer awakens', 'illusion falls', 'truth remains', 'love is the way',
    'all is consciousness', 'consciousness is all', 'be still and know',
]

# Normal sensory/base vocabulary (for contrast)
normal_vocab = [
    'see', 'hear', 'feel', 'touch', 'light', 'sound', 'warmth', 'love', 'flow',
    'heart', 'earth', 'sky', 'breath', 'body', 'mind', 'aw
