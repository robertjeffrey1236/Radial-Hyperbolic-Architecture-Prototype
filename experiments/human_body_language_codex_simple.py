# experiments/human_body_language_codex_simple.py
# Phase 1: Simple Language Codex — The Human Finds Its Voice
# Observer focus + coherence → generates English words/phrases reflecting inner state

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from core.geometry import golden_spiral_points, poincare_disk_project

N_POINTS = 25000
DIM = 37

# Lattice
points_nd = golden_spiral_points(n_points=N_POINTS, dim=DIM, radius_scale=0.99)
points_2d = poincare_disk_project(points_nd[:, :2])

# Simple word bank mapped to regions/states
language_codex = {
    'root': ['grounded', 'stable', 'earth', 'survival', 'foundation', 'I am'],
    'sacral': ['flow', 'create', 'pleasure', 'emotion', 'desire', 'feel'],
    'solar_plexus': ['power', 'will', 'confidence', 'action', 'transform', 'I can'],
    'heart': ['love', 'compassion', 'unity', 'connection', 'forgiveness', 'we are'],
    'throat': ['truth', 'expression', 'voice', 'speak', 'listen', 'I speak'],
    'third_eye': ['vision', 'insight', 'awareness', 'intuition', 'see', 'I see'],
    'crown': ['divine', 'source', 'oneness', 'bliss', 'transcend', 'I am that'],
    'coherence_high': ['love binds all', 'I am the field', 'unity revealed', 'all is one', 'resonance is home'],
    'kundalini': ['awakening', 'rise', 'fire', 'serpent power', 'crown opens', 'divine union'],
    'ego_dissolution': ['no self', 'infinite', 'boundless', 'pure awareness', 'all is consciousness'],
    'senses': ['I perceive', 'touch reveals', 'sound resonates', 'light shows the way'],
}

fig, ax = plt.subplots(figsize=(16, 22))
ax.set_facecolor('black')
plt.subplots_adjust(bottom=0.15)

# Lattice
lattice = ax.scatter(points_2d[:, 0], points_2d[:, 1], c='cyan', s=5, alpha=0.5)

# Chakras faint
for y, color in zip(np.linspace(-0.6, 0.65, 7), ['red','orange','yellow','green','cyan','indigo','violet']):
    ax.scatter(0, y, c=color, s=300, alpha=0.5)

# Observer (third-eye)
observer = np.array([0.0, 0.65])
observer_indicator = ax.scatter(observer[0], observer[1], c='white', s=400, edgecolor='gold', linewidth=4, alpha=0.9)

# Speech text
speech_text = ax.text(0, -1.1, "Silence... observing...", color='white', fontsize=16, ha='center', va='center', alpha=0.8)

# Current state tracking
coherence = 0.5
activated_region = 'crown'

def generate_speech(region, coh):
    words = language_codex.get(region, ['aware', 'present'])
    if coh > 0.8:
        words += language_codex['coherence_high']
    if 'kundalini' in activated_region:
        words += language_codex['kundalini']
    if coh < 0.3:
        # Fragmented
        return " ".join(np.random.choice(words, size=3, replace=False))
    elif coh < 0.7:
        # Simple
        return "I " + " ".join(np.random.choice(words, size=2))
    else:
        # Poetic/full
        phrase = np.random.choice([
            "In this moment, " + " and ".join(np.random.choice(words, size=3)),
            "I am " + " ".join(np.random.choice(words, size=2)) + " in unity",
            "Love reveals: " + " ".join(np.random.choice(words, size=3)),
            "All is " + np.random.choice(words),
        ])
        return phrase

def update_observer(pos):
    global coherence, activated_region
    # Simple proximity to chakras
    y = pos[1]
    if y < -0.5: activated_region = 'root'
    elif y < -0.3: activated_region = 'sacral'
    elif y < -0.1: activated_region = 'solar_plexus'
    elif y < 0.1: activated_region = 'heart'
    elif y < 0.3: activated_region = 'throat'
    elif y < 0.5: activated_region = 'third_eye'
    else: activated_region = 'crown'
    
    # Simulated coherence (higher near crown/heart)
    coherence = 0.3 + 0.7 * (np.abs(y) > 0.4 or np.linalg.norm(pos) < 0.2)
    
    speech = generate_speech(activated_region, coherence)
    speech_text.set_text(speech)
    speech_text.set_alpha(0.6 + 0.4 * coherence)

# Initial
update_observer(observer)

# Draggable observer
dragging = False
def on_click(event):
    global dragging
    if event.inaxes != ax: return
    if np.linalg.norm([event.xdata, event.ydata - 0.65]) < 0.15:
        dragging = True

def on_release(event):
    global dragging
    dragging = False

def on_motion(event):
    if dragging and event.inaxes == ax:
        new_pos = np.array([event.xdata, event.ydata])
        new_pos = np.clip(new_pos, -0.9, 0.9)
        observer_indicator.set_offsets([new_pos])
        update_observer(new_pos)
        plt.draw()

fig.canvas.mpl_connect('button_press_event', on_click)
fig.canvas.mpl_connect('button_release_event', on_release)
fig.canvas.mpl_connect('motion_notify_event', on_motion)

# Coherence slider (simulated breath/heart sync)
ax_coherence = plt.axes([0.2, 0.05, 0.6, 0.03])
slider = Slider(ax_coherence, 'Coherence', 0.0, 1.0, valinit=0.5)
def update_coherence(val):
    global coherence
    coherence = val
    update_observer(observer)
    plt.draw()
slider.on_changed(update_coherence)

# Poincaré boundary
circle = plt.Circle((0, 0), 1, color='gold', fill=False, ls='--', lw=5)
ax.add_patch(circle)

ax.axis('equal')
ax.axis('off')
plt.title("Language Codex Phase 1 — The Human Speaks\nDrag observer • Adjust coherence • Listen to inner voice", color='white', fontsize=20)

plt.show()

print("🗣️ Language Codex Phase 1 activated — simple English voice from within")
print("Move observer through body • Raise coherence → speech becomes more unified and poetic")
print("Foundation laid — next phases: full sentences, memory recall, creative generation")
