# experiments/human_body_quantum_eraser.py
# Quantum Eraser Mode — Delayed-Choice Quantum Eraser Simulation
# Observer "measurement" retroactively shapes interference pattern

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons
import random

fig, ax = plt.subplots(figsize=(16, 24))
ax.set_facecolor('black')
ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1.2, 1.0)
ax.axis('off')

# Human faint
body = plt.Circle((0, 0), 0.9, color='white', fill=False, lw=3, alpha=0.2)
ax.add_patch(body)

# "Screen" for interference pattern (right side)
screen = ax.scatter([], [], c='white', s=10, alpha=0.8)

# Entangled photon source (third-eye)
source = np.array([0.0, 0.45])
ax.scatter(source[0], source[1], c='gold', s=400, alpha=0.8, edgecolor='white')

# Observer "detector" (draggable)
observer = np.array([0.0, 0.0])
detector = ax.scatter(observer[0], observer[1], c='cyan', s=500, marker='x', linewidth=4, alpha=0.9)

# Eraser toggle state
eraser_on = True  # Start with eraser (interference visible)

# Photon pairs
pairs = []

def spawn_pairs():
    global pairs
    pairs = []
    for _ in range(80):
        angle = random.uniform(0, 2*np.pi)
        path1 = source + 0.4 * np.array([np.cos(angle), np.sin(angle)])
        path2 = source + 0.4 * np.array([np.cos(angle + np.pi), np.sin(angle + np.pi)])
        which_path = random.choice([0, 1])  # Random path info
        pairs.append({'p1': path1, 'p2': path2, 'which': which_path})

spawn_pairs()

def update_pattern():
    global screen
    screen.remove()
    
    hits = []
    for pair in pairs:
        # "Signal" photon hits screen
        hits.append(pair['p1'])
        
        # Idler photon "measured" by observer/detector
        measured = np.linalg.norm(pair['p2'] - observer) < 0.3
        
        # Eraser on = path info erased → interference
        # Eraser off = path info known → no interference
        if eraser_on or not measured:
            # Interference: pattern builds on screen (right side)
            x = 0.6 + 0.4 * np.sin(pair['p1'][0] * 10 + pair['p1'][1] * 5)
            y = pair['p1'][1]
            hits.append([x, y])
    
    hits = np.array(hits)
    colors = ['white' if eraser_on else 'gray' for _ in hits]
    alphas = [0.9 if eraser_on else 0.4 for _ in hits]
    global screen
    screen = ax.scatter(hits[:, 0], hits[:, 1], c=colors, s=15, alpha=alphas)

update_pattern()

# Title
title = ax.set_title("Quantum Eraser Mode\nToggle eraser — watch interference appear/disappear retroactively", color='white', fontsize=20)

# Eraser toggle
rax = plt.axes([0.05, 0.5, 0.2, 0.1], facecolor='black')
check = CheckButtons(rax, ['Eraser ON (Interference)'], [True])

def toggle_eraser(label):
    global eraser_on
    eraser_on = not eraser_on
    update_pattern()
    title.set_text(f"Quantum Eraser Mode — Eraser {'ON' if eraser_on else 'OFF'}\nMeasurement links information → pattern {'appears' if eraser_on else 'vanishes'}")
    plt.draw()

check.on_clicked(toggle_eraser)

# Draggable detector (observer)
dragging = False
def on_click(event):
    global dragging
    if event.inaxes == ax:
        if np.linalg.norm([event.xdata - observer[0], event.ydata - observer[1]]) < 0.15:
            dragging = True

def on_release(event):
    global dragging
    dragging = False

def on_motion(event):
    if dragging and event.inaxes == ax:
        observer[0] = event.xdata
        observer[1] = event.ydata
        detector.set_offsets([observer])
        update_pattern()
        plt.draw()

fig.canvas.mpl_connect('button_press_event', on_click)
fig.canvas.mpl_connect('button_release_event', on_release)
fig.canvas.mpl_connect('motion_notify_event', on_motion)

# Boundary
boundary = plt.Circle((0, 0), 1, color='cyan', fill=False, ls='--', lw=5)
ax.add_patch(boundary)

plt.show()

print("⚛️ Quantum Eraser Mode activated")
print("Toggle 'Eraser ON/OFF' — watch interference pattern appear/disappear")
print("Drag detector (observer) — measurement links information retroactively")
print("Your focus literally shapes quantum reality — just like the experiment")
