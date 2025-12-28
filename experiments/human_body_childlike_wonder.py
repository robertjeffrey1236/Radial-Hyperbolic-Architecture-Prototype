# experiments/human_body_childlike_wonder.py
# Childlike Wonder Mode — Innocence, Play, Awe
# The human rediscovers existence with fresh eyes and open heart

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from core.geometry import golden_spiral_points, poincare_disk_project

N_POINTS = 30000
DIM = 37

# Substrate
points_nd = golden_spiral_points(n_points=N_POINTS, dim=DIM, radius_scale=0.99)
points_2d = poincare_disk_project(points_nd[:, :2])

fig, ax = plt.subplots(figsize=(18, 28))
ax.set_facecolor('#0a0a1a')  # Soft deep night sky

# Twinkling wonder stars (the lattice as magic)
wonder_stars = ax.scatter(points_2d[:, 0], points_2d[:, 1], 
                          c='white', s=8, alpha=0.6, edgecolor='lavender', linewidth=0.5)

# Gentle body glow — childlike aura
body_aura = plt.Circle((0, 0), 0.9, color='peachpuff', fill=False, lw=6, alpha=0.4)
ax.add_patch(body_aura)

# Heart of wonder
heart = ax.scatter(0, 0, c='pink', s=800, alpha=0.8, edgecolor='gold', linewidth=4)

# Eyes wide open
eyes = ax.scatter([-0.12, 0.12], [0.72, 0.72], c='lightblue', s=400, alpha=0.9, edgecolor='white', linewidth=3)

# Smile
smile = np.linspace(-0.15, 0.15, 50)
smile_y = 0.55 + 0.08 * np.sin((smile + 0.15) * np.pi / 0.3)
ax.plot(smile, smile_y, c='pink', lw=5, alpha=0.9)

# Wonder sparkles (floating magic)
sparkles = []

def animate(frame):
    t = frame * 0.05
    
    # Breathing wonder
    breath = np.sin(t * 0.6) * 0.1
    body_aura.set_radius(0.9 + breath)
    body_aura.set_alpha(0.4 + 0.2 * (breath + 0.1))
    
    # Heart pulse of joy
    joy = np.abs(np.sin(t * 1.2))
    heart_size = 800 + 400 * joy
    heart.set_sizes([heart_size])
    
    # Twinkling stars
    alphas = 0.4 + 0.6 * np.random.random(N_POINTS)
    sizes = 6 + 10 * np.random.random(N_POINTS)
    wonder_stars.set_alpha(alphas)
    wonder_stars.set_sizes(sizes)
    
    # Floating sparkles
    global sparkles
    for s in sparkles:
        s.remove()
    sparkles.clear()
    
    for _ in range(15):
        x = np.random.uniform(-0.8, 0.8)
        y = np.random.uniform(-0.9, 0.9)
        color = random.choice(['pink', 'lightblue', 'lavender', 'gold', 'white'])
        s = ax.scatter(x, y, c=color, s=100 + 200 * np.random.random(), alpha=0.8, marker='*')
        sparkles.append(s)
    
    # Title of pure wonder
    messages = [
        "Wow... everything sparkles!",
        "Look! The stars are dancing!",
        "I can feel the magic breathing...",
        "Everything is alive and saying hello!",
        "Love makes the whole world shine!",
        "What if we're all made of wonder?",
        "Heehee... this feels like flying inside!",
    ]
    title.set_text(random.choice(messages))

title = ax.set_title("Childlike Wonder Mode\nPure innocence • Awe • Playful discovery", color='white', fontsize=22, pad=100)

# Poincaré boundary — soft rainbow glow
boundary = plt.Circle((0, 0), 1, color='lavender', fill=False, ls='--', lw=6, alpha=0.6)
ax.add_patch(boundary)

ax.axis('equal')
ax.axis('off')

anim = FuncAnimation(fig, animate, interval=100, repeat=True)

plt.show()

print("🌟👶 Childlike Wonder Mode activated")
print("The human returns to innocence — wide-eyed, playful, full of awe")
print("Everything twinkles • Heart dances • Magic floats everywhere")
print("This is the original state — before knowing, only wondering")
