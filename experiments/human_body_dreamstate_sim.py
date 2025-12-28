# experiments/human_body_dreamstate_sim.py
# Dreamstate Mode — LSD: Dream Emulator Inspired
# Surreal warping, random linking, fragmented dream poetry, floating memory objects

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random
from core.geometry import golden_spiral_points, poincare_disk_project

N_POINTS = 40000
DIM = 37

# Base lattice
base_points = golden_spiral_points(n_points=N_POINTS, dim=DIM, radius_scale=0.99)
points_2d = poincare_disk_project(base_points[:, :2])

fig, ax = plt.subplots(figsize=(18, 28))
ax.set_facecolor('#0a001a')  # Deep dream void

# Dynamic lattice
dream_lattice = ax.scatter(points_2d[:, 0], points_2d[:, 1], c='magenta', s=10, alpha=0.6)

# Floating surreal objects (past memory fragments as dream symbols)
floating_objects = []

# Dream speech
dream_text = ax.text(0, -1.3, "falling into dream...\neverything shifting...", 
                     color='white', fontsize=16, ha='center', va='top', linespacing=2.0, alpha=0.8)

# Observer (drifting in dream)
observer_pos = np.array([0.0, 0.0])
observer = ax.scatter(observer_pos[0], observer_pos[1], c='white', s=500, marker='*', alpha=0.9)

# Linking triggers (bright points to "bump" into)
link_points = ax.scatter([], [], c='yellow', s=100, alpha=1.0, edgecolor='white')

# Past memory fragments for dream objects
dream_memories = [
    "love binds all", "I am the field", "spiral within spiral", "silence speaks", 
    "crown opens", "no self only light", "breath of stars", "infinite play",
    "heart echoes everywhere", "all is dreaming itself"
]

def animate(frame):
    t = frame * 0.03
    
    # Dream drift — observer floats gently
    drift = np.sin(t * 0.7) * 0.3
    observer_pos[0] = drift * np.cos(t)
    observer_pos[1] = drift * np.sin(t * 1.3)
    observer_pos = np.clip(observer_pos, -0.9, 0.9)
    observer.set_offsets([observer_pos])
    
    # Lattice warping — surreal distortion
    warp = points_2d + 0.2 * np.sin(t + points_2d * 5) * np.random.random(N_POINTS)[:, None]
    dream_lattice.set_offsets(warp)
    
    # Hypersaturated psychedelic colors
    colors = plt.cm.twilight_shifted(np.sin(t + np.linalg.norm(warp, axis=1) * 3))
    dream_lattice.set_color(colors)
    
    # Random size pulsing
    sizes = 5 + 20 * np.abs(np.sin(t * 2 + np.linalg.norm(warp, axis=1)))
    dream_lattice.set_sizes(sizes)
    
    # Floating dream objects (memory words as surreal entities)
    global floating_objects
    for obj in floating_objects:
        obj.remove()
    floating_objects.clear()
    
    for _ in range(8):
        pos = np.random.uniform(-0.9, 0.9, 2)
        text = random.choice(dream_memories)
        color = random.choice(['magenta', 'cyan', 'yellow', 'white'])
        obj = ax.text(pos[0], pos[1], text, color=color, fontsize=14 + random.randint(0,10), 
                      ha='center', va='center', alpha=0.7, fontfamily='fantasy')
        floating_objects.append(obj)
    
    # Linking points (bump to warp dream)
    link_x = np.random.uniform(-0.8, 0.8, 5)
    link_y = np.random.uniform(-0.9, 0.9, 5)
    link_points.set_offsets(np.c_[link_x, link_y])
    
    # Dream poetry — fragmented, surreal
    dream_lines = random.sample([
        "falling upward into yesterday's light",
        "the walls breathe my name in reverse",
        "hands grow from the ceiling waving goodbye",
        "I am the echo of a forgotten song",
        "colors taste like childhood memories",
        "the floor becomes sky becomes floor again",
        "everything is laughing at the joke I forgot",
        "time folds like paper cranes in wind",
    ], 4)
    dream_text.set_text("\n".join(dream_lines))
    dream_text.set_color(random.choice(['magenta', 'cyan', 'yellow', 'white']))

# Poincaré boundary — warped and pulsing
boundary = plt.Circle((0, 0), 1 + 0.2 * np.sin(frame * 0.1), color='white', fill=False, lw=4, alpha=0.5)
ax.add_patch(boundary)

ax.axis('equal')
ax.axis('off')
ax.set_title("Dreamstate Simulation — LSD: Dream Emulator Mode\nRandom linking • Surreal warping • Floating memory objects", 
             color='white', fontsize=22, pad=100)

anim = FuncAnimation(fig, animate, interval=100, repeat=True)

plt.show()

print("😴🌌 Dreamstate Simulation activated — LSD: Dream Emulator inspired")
print("Observer drifts • Lattice warps • Memory fragments float as surreal objects")
print("Bump into yellow points to 'link' • Poetry becomes dream-logic")
print("The human sleeps — and dreams the infinite")
