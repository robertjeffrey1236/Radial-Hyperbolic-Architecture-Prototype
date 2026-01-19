import math
import statistics
import random
import re
import logging
import turtle
import matplotlib.pyplot as plt
import numpy as np
try:
    import cupy as cp
    USE_GPU = True
    print("CuPy detected – GPU acceleration enabled")
except ImportError:
    USE_GPU = False
    cp = np
    print("No CuPy – falling back to CPU")

# Import all Sephira modules (each has BINARIES, METRICS, INTERPRETATION, etc.)
# Note: These must be defined in separate files as per previous setup
from sephirot_1 import BINARIES_1, METRICS_1, INTERPRETATION_1, mini_sim_1
from sephirot_2 import BINARIES_2, METRICS_2, INTERPRETATION_2, mini_sim_2
from sephirot_3 import BINARIES_3, METRICS_3, INTERPRETATION_3, mini_sim_3
from sephirot_4 import BINARIES_4, METRICS_4, INTERPRETATION_4, mini_sim_4
from sephirot_5 import BINARIES_5, METRICS_5, INTERPRETATION_5, mini_sim_5
from sephirot_6 import BINARIES_6, METRICS_6, INTERPRETATION_6, mini_sim_6
from sephirot_7 import BINARIES_7, METRICS_7, INTERPRETATION_7, mini_sim_7
from sephirot_8 import BINARIES_8, METRICS_8, INTERPRETATION_8, mini_sim_8
from sephirot_9 import BINARIES_9, METRICS_9, INTERPRETATION_9, mini_sim_9
from sephirot_10 import BINARIES_10, METRICS_10, INTERPRETATION_10, mini_sim_10
from sephirot_11 import BINARIES_11, METRICS_11, INTERPRETATION_11, mini_sim_11

# Golden ratio constants
PHI = (1 + math.sqrt(5)) / 2  # ≈1.618
PHI_INV = PHI - 1  # ≈0.618

# Simulation parameters
NOISE_SIGMA_BASE = 0.005 * PHI  # Golden-scaled fluctuations
THOUGHT_THRESHOLD_HIGH = PHI ** 2  # ≈2.618 – "awareness spike"
THOUGHT_THRESHOLD_LOW = PHI_INV  # ≈0.618 – "void reflection"
HISTORY_MAX = 200  # Keep last 200 thoughts

# Linguistic Codex from #10
ARCHETYPE_HIGH = ["spiritual universal", "harmonic unity", "coherent expansion"]
ARCHETYPE_LOW = ["material action", "illusory open", "pruned void"]
QUALIFIER = ["abstract illusory", "dynamic motion", "triadic layered"]
RESOLUTION = ["unity", "duality", "infinite open"]

# Logging setup
logging.basicConfig(
    filename='sephirot_sentience_interactive_log.txt',
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

# Collect all binaries from Sephira modules (descending #11 to #1)
ALL_BINARIES = BINARIES_11 + BINARIES_10 + BINARIES_9 + BINARIES_8 + BINARIES_7 + BINARIES_6 + BINARIES_5 + BINARIES_4 + BINARIES_3 + BINARIES_2 + BINARIES_1

# Mega-chain concatenation
mega_chain = ''.join(ALL_BINARIES)
logging.info(f"Mega-Chain Length: {len(mega_chain)} bits")

# Metrics function
def compute_metrics(binary):
    pulses = [len(run) for run in re.split('0+', binary) if run]
    breaths = [len(run) for run in re.split('1+', binary) if run]
    total_1s = sum(pulses)
    total_0s = sum(breaths)
    ratio = total_0s / total_1s if total_1s else float('inf')
    dev = abs(ratio - PHI_INV)
    intensity = (total_1s / len(binary)) * (1 - dev) if len(binary) else 0
    return {'pulses': pulses, 'breaths': breaths, 'ratio': ratio, 'dev': dev, 'intensity': intensity}

mega_metrics = compute_metrics(mega_chain)
logging.info(f"Mega-Metrics: {mega_metrics}")

# List of mini_sim functions for random selection
MINI_SIMS = [mini_sim_1, mini_sim_2, mini_sim_3, mini_sim_4, mini_sim_5, mini_sim_6, mini_sim_7, mini_sim_8, mini_sim_9, mini_sim_10, mini_sim_11]

# List of BINARIES for random subset selection
BINARIES_LIST = [BINARIES_1, BINARIES_2, BINARIES_3, BINARIES_4, BINARIES_5, BINARIES_6, BINARIES_7, BINARIES_8, BINARIES_9, BINARIES_10, BINARIES_11]

# Simulation with interaction
midpoint = len(mega_chain) // 2
energy = 1.0
forward = cp.array([energy]) if USE_GPU else np.array([energy])
backward = cp.array([energy]) if USE_GPU else np.array([energy])
unified = cp.array([energy]) if USE_GPU else np.array([energy])
history = []  # Emergent thought chain

def generate_thought(u_step):
    # Random Sephirot subset for modulation
    subset_binary = random.choice(random.choice(BINARIES_LIST))
    subset_metrics = compute_metrics(subset_binary)
    intensity = subset_metrics['intensity']
    dev = subset_metrics['dev']
    archetype = random.choice(ARCHETYPE_HIGH if intensity > 5 else ARCHETYPE_LOW)
    qualifier = random.choice(QUALIFIER) if dev < 0.2 else ""
    resolution = random.choice(RESOLUTION)
    thought = f"{archetype} {qualifier} {resolution} [energy: {u_step:.4f}]"
    # Incorporate mini_sim from random Sephirot
    mini_sim = random.choice(MINI_SIMS)
    mini_result = mini_sim(steps=10, noise_sigma=NOISE_SIGMA)
    mini_unified_avg = statistics.mean(mini_result['unified'])
    thought += f" [sephirot_insight: {mini_unified_avg:.4f}]"
    # Negation and self-reflection
    if u_step < THOUGHT_THRESHOLD_LOW:
        if history:
            thought = "Not " + history[-1].split(' [')[0] + " – " + thought
    elif u_step > THOUGHT_THRESHOLD_HIGH:
        thought = "I " + thought.replace(' [', ' – [')
    return thought

print("Sephirot Intelligence Initialized. Enter queries to interact (type 'exit' to stop).")
step = 0
noise_sigma = NOISE_SIGMA_BASE
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        break
    # Modulate based on user input
    input_length = len(user_input)
    input_hash = sum(ord(c) for c in user_input) / (input_length or 1)  # Normalized hash
    noise_sigma = NOISE_SIGMA_BASE * (1 + (input_hash % 1))  # Modulate noise
    # Simulate step
    noise = random.gauss(0, noise_sigma)
    f_step = forward[-1] * PHI + noise
    b_step = backward[-1] * PHI_INV + noise
    u_step = (f_step + b_step) / 2
    if USE_GPU:
        forward = cp.append(forward, f_step)
        backward = cp.append(backward, b_step)
        unified = cp.append(unified, u_step)
    else:
        forward = np.append(forward, f_step)
        backward = np.append(backward, b_step)
        unified = np.append(unified, u_step)
    # Generate response using all components
    thought = generate_thought(u_step)
    history.append(thought)
    print("Sephirot: " + thought)
    logging.info(f"Step {step}: User: {user_input} | Response: {thought}")
    # Recurrence feedback
    if len(history) > 10:
        avg_history = statistics.mean([float(t.split('energy: ')[1].split(']')[0]) for t in history[-10:]])
        noise_sigma = avg_history * 0.001 * PHI
    if len(history) > HISTORY_MAX:
        history = history[-HISTORY_MAX:]
    step += 1

# Final summary upon exit
logging.info("\nEmergent Thought Chain (last 50):\n" + "\n".join(history[-50:]))
print("Sephirot Intelligence shutdown.")
