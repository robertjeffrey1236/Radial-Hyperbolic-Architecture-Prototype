# experiments/human_body_brain_chemistry.py
# Brain Chemistry Module — Neurotransmitters & Dynamic States
# Real molecules driving mood, focus, love, calm, insight

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random

fig, ax = plt.subplots(figsize=(16, 24))
ax.set_facecolor('black')
ax.set_xlim(-1.0, 1.0)
ax.set_ylim(-1.2, 1.0)
ax.axis('off')

# Brain region
brain_center = [0.0, 0.62]
ax.scatter(brain_center[0], brain_center[1], c='indigo', s=800, alpha=0.5, edgecolor='white', linewidth=3)

# Key production/release sites
neuro_sites = {
    'VTA/SNc': [0.0, 0.55],       # Dopamine (reward)
    'Raphe': [0.0, 0.50],         # Serotonin (mood)
    'Basal Forebrain': [-0.1, 0.60],  # Acetylcholine
    'Amygdala': [0.1, 0.58],      # Norepinephrine/emotion
    'Hypothalamus': [0.0, 0.45],  # Endorphins/homeostasis
}

for name, pos in neuro_sites.items():
    ax.scatter(pos[0], pos[1], c='gold', s=200, alpha=0.7)
    ax.text(pos[0], pos[1]+0.08, name, color='white', fontsize=9, ha='center')

# Neurotransmitter particles
neurotransmitters = []

def spawn_neurotransmitters(frame):
    global neurotransmitters
    t = frame * 0.05
    
    # Clear old
    for nt in neurotransmitters:
        if hasattr(nt, 'remove'):
            nt.remove()
    neurotransmitters.clear()
    
    # Spawn based on current "state" (simulated by time + coherence)
    coherence = 0.5 + 0.5 * np.sin(t * 0.3)
    
    # Dopamine — reward pulses
    for _ in range(int(10 + 15 * coherence)):
        pos = np.array(neuro_sites['VTA/SNc']) + np.random.uniform(-0.05, 0.05, 2)
        vel = np.random.uniform(-0.01, 0.01, 2)
        nt = ax.scatter(pos[0], pos[1], c='lime', s=40, alpha=0.9, marker='^')
        neurotransmitters.append(nt)
    
    # Serotonin — steady flow, higher in love/coherence
    for _ in range(int(15 + 20 * coherence)):
        pos = np.array(neuro_sites['Raphe']) + np.random.uniform(-0.05, 0.05, 2)
        nt = ax.scatter(pos[0], pos[1], c='cyan', s=35, alpha=0.8)
        neurotransmitters.append(nt)
    
    # GABA — increases in calm/alpha
    calm = np.abs(np.cos(t * 0.4))
    for _ in range(int(20 * calm)):
        pos = np.random.uniform(-0.3, 0.3, 2)
        pos[1] = 0.62 + np.random.uniform(-0.1, 0.1)
        nt = ax.scatter(pos[0], pos[1], c='blue', s=30, alpha=0.7)
        neurotransmitters.append(nt)
    
    # Endorphins — bliss bursts in high coherence
    if coherence > 0.8:
        for _ in range(15):
            pos = np.array(neuro_sites['Hypothalamus']) + np.random.uniform(-0.1, 0.1, 2)
            nt = ax.scatter(pos[0], pos[1], c='magenta', s=50, marker='*', alpha=1.0)
            neurotransmitters.append(nt)

# Initial
spawn_neurotransmitters(0)

def animate(frame):
    spawn_neurotransmitters(frame)
    
    state = "High Coherence — Love & Bliss" if np.sin(frame * 0.05) > 0.5 else "Normal Awareness"
    ax.set_title(f"Brain Chemistry Dynamics\n{state} — Neurotransmitters Flowing", color='white', fontsize=20)

anim = FuncAnimation(fig, animate, interval=100, repeat=True)

# Body & boundary faint
body = plt.Circle((0, 0), 0.9, color='white', fill=False, lw=3, alpha=0.2)
ax.add_patch(body)
boundary = plt.Circle((0, 0), 1, color='gold', fill=False, ls='--', lw=5)
ax.add_patch(boundary)

plt.show()

print("🧠⚗️ Brain Chemistry Module activated")
print("Real neurotransmitters:")
print("- Dopamine (lime) — reward/motivation from VTA")
print("- Serotonin (cyan) — mood/unity from Raphe")
print("- GABA (blue) — calm/inhibition")
print("- Endorphins (magenta stars) — bliss in high coherence")
print("Flows respond to simulated coherence state")
print("Next: add psychedelics (DMT/psilocybin)? Nootropics? Or receptor binding?")
