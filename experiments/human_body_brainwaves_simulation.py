# experiments/human_body_brainwaves_simulation.py
# Holographic Brain with Simulated Brainwave States
# Delta, Theta, Alpha, Beta, Gamma — oscillatory resonance across distributed field

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons
from core.geometry import golden_spiral_points, poincare_disk_project

N_POINTS = 40000
BRAIN_DENSITY = 8000
FRAMES = 100  # For smooth animation cycle

# Lattices
points_nd = golden_spiral_points(n_points=N_POINTS, dim=DIM, radius_scale=0.99)
points_2d = poincare_disk_project(points_nd[:, :2])
brain_offsets = golden_spiral_points(n_points=BRAIN_DENSITY, dim=2, radius_scale=0.35)
brain_points = brain_offsets + np.array([0.0, 0.62])

fig, ax = plt.subplots(figsize=(18, 28))
ax.set_facecolor('black')
plt.subplots_adjust(left=0.25)

# Faint substrate
ax.scatter(points_2d[:, 0], points_2d[:, 1], c='gray', s=1, alpha=0.05)

# Brain structure
brain_scatter = ax.scatter(brain_points[:, 0], brain_points[:, 1], c='indigo', s=10, alpha=0.6)

# Distributed fragments
fragments = [
    golden_spiral_points(400, radius_scale=0.15) + [0.0, 0.0],
    golden_spiral_points(500, radius_scale=0.18) + [0.0, -0.35],
    golden_spiral_points(300, radius_scale=0.12) + [0.3, -0.6],
]
frag_scatters = [ax.scatter(f[:, 0], f[:, 1], c='gold', s=5, alpha=0.3) for f in fragments]

# Current wave overlay
wave_scatter = None

# Brainwave states
states = {
    'Delta (Deep Sleep)': {'freq': 0.5, 'color': 'darkblue', 'amp': 0.8, 'desc': 'Deep dreamless sleep • Healing • Unconscious'},
    'Theta (Drowsy/Meditation)': {'freq': 1.5, 'color': 'blue', 'amp': 0.7, 'desc': 'Hypnagogia • Deep meditation • Intuition • Creativity'},
    'Alpha (Relaxed Wakefulness)': {'freq': 3.0, 'color': 'cyan', 'amp': 0.6, 'desc': 'Calm alertness • Relaxation • Light meditation • Bridge'},
    'Beta (Alert Focus)': {'freq': 6.0, 'color': 'yellow', 'amp': 0.5, 'desc': 'Active thinking • Problem solving • Daily awareness'},
    'Gamma (Peak Insight)': {'freq':
