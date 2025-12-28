# experiments/ludicrous_speed_sim.py
# Toy Sim: Ludicrous Speed - Breaking the Light Barrier in Hyperbolic Space
# Inspired by Spaceballs: "They've gone to plaid!"

import numpy as np
import matplotlib.pyplot as plt
from core.geometry import golden_spiral_points, poincare_disk_project, build_hyperbolic_graph

N_POINTS = 10000
DIM = 37
SPEED_LEVELS = ['sub_light', 'light_speed', 'ludicrous_speed']  # Three stages

# Generate base lattice (the "stars" in space)
points_nd = golden_spiral_points(n_points=N_POINTS, dim=DIM, radius_scale=0.99)
points_2d_base = poincare_disk_project(points_nd[:, :2])

# Build connections
neighbors = build_hyperbolic_graph(points_2d_base, k_neighbors=6)

def plot_speed_level(level: str, points_2d: np.ndarray, save_path: str):
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_facecolor('black')
    
    if level == 'sub_light':
        # Normal: Static lattice
        ax.scatter(points_2d[:, 0], points_2d[:, 1], c='cyan', s=8, alpha=0.8)
        title = "Sub-Light Speed - Normal Space (Static Lattice)"
        color_edges = 'white'
        
    elif level == 'light_speed':
        # Relativistic: Bunch forward, add radial streaks
        ax.scatter(points_2d[:, 0], points_2d[:, 1], c='yellow', s=12, alpha=0.9)
        # Streaking lines from center
        for i in range(0, N_POINTS, 50):
            ax.plot([0, points_2d[i, 0]], [0, points_2d[i, 1]], c='white', lw=1, alpha=0.6)
        title = "Light Speed - Relativistic Aberration (Stars Streaking Forward)"
        color_edges = 'lime'
        
    else:  # ludicrous_speed
        # FTL Break: Exponential warp + rainbow plaid
        colors = plt.cm.hsv(np.linspace(0, 1, len(neighbors)))  # Rainbow trails
        for i, nbrs in enumerate(neighbors):
            for j in nbrs:
                if j > i:
                    ax.plot(*zip(points_2d[i], points_2d[j]), c=colors[i], lw=1.5, alpha=0.7)
        ax.scatter(points_2d[:, 0], points_2d[:, 1], c='magenta', s=15, alpha=1.0)
        # Central explosion glow
        ax.scatter(0, 0, c='white', s=200, alpha=0.5, marker='*')
        title = "LUDICROUS SPEED - They've Gone to Plaid! (Hyperbolic Breakthrough)"
    
    # Draw edges (subdued for lower speeds)
    if level != 'ludicrous_speed':
        for i, nbrs in enumerate(neighbors):
            for j in nbrs:
                if j > i:
                    ax.plot(*zip(points_2d[i], points_2d[j]), c=color_edges, lw=0.5, alpha=0.4)
    
    # Poincaré disk boundary
    circle = plt.Circle((0, 0), 1, color='red', fill=False, ls='--', lw=3, alpha=0.8)
    ax.add_patch(circle)
    
    ax.axis('equal')
    ax.axis('off')
    plt.title(title, color='white', fontsize=14)
    plt.savefig(save_path, dpi=300, facecolor='black', bbox_inches='tight')
    plt.show()

# Simulate progression
print("🚀 Engaging Ludicrous Speed Sequence...")

# 1. Sub-light: Slight distortion toward center
points_sub = points_2d_base * 0.9
plot_speed_level('sub_light', points_sub, "sub_light_normal.png")

# 2. Light speed: Boost toward center (aberration)
boost_factor = 3.0
points_light = points_2d_base / (1 + boost_factor * np.linalg.norm(points_2d_base, axis=1)[:, None])
plot_speed_level('light_speed', points_light, "light_speed_streaks.png")

# 3. Ludicrous speed: Break into full hyperbolic warp (exponential expansion)
points_ludicrous = points_2d_base * (1 + 5 * np.linalg.norm(points_2d_base, axis=1)[:, None]**2)
points_ludicrous /= np.max(np.linalg.norm(points_ludicrous, axis=1)) * 0.95  # Keep in disk
plot_speed_level('ludicrous_speed', points_ludicrous, "ludicrous_speed_plaid.png")

print("They've gone to plaid! Images saved: sub_light_normal.png | light_speed_streaks.png | ludicrous_speed_plaid.png")
print("When you see the plaid one... you've broken through. 🌌💨")
