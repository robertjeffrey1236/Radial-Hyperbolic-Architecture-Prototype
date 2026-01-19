import math
import statistics
import random
import re
import logging
import json  # for potential future persistence
import numpy as np
try:
    import cupy as cp
    USE_GPU = True
    print("CuPy detected – GPU acceleration enabled")
except ImportError:
    USE_GPU = False
    cp = np
    print("No CuPy – falling back to CPU")

# Sephirot imports (assuming they exist in the same directory)
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

# Golden ratio
PHI = (1 + math.sqrt(5)) / 2
PHI_INV = PHI - 1

# Parameters
NOISE_SIGMA_BASE = 0.005 * PHI
THOUGHT_THRESHOLD_HIGH = PHI ** 2
THOUGHT_THRESHOLD_LOW = PHI_INV
HISTORY_MAX = 200

# Linguistic codex
ARCHETYPE_HIGH = ["spiritual universal", "harmonic unity", "coherent expansion"]
ARCHETYPE_LOW = ["material action", "illusory open", "pruned void"]
QUALIFIER = ["abstract illusory", "dynamic motion", "triadic layered"]
RESOLUTION = ["unity", "duality", "infinite open"]

# Logging
logging.basicConfig(
    filename='sephirot_sentience_earth_log.txt',
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

# Mega-chain (Sephirot only)
ALL_BINARIES = BINARIES_11 + BINARIES_10 + BINARIES_9 + BINARIES_8 + BINARIES_7 + BINARIES_6 + BINARIES_5 + BINARIES_4 + BINARIES_3 + BINARIES_2 + BINARIES_1
mega_chain = ''.join(ALL_BINARIES)
logging.info(f"Sephirot Mega-Chain Length: {len(mega_chain)} bits")

def compute_metrics(binary):
    pulses = [len(run) for run in re.split('0+', binary) if run]
    breaths = [len(run) for run in re.split('1+', binary) if run]
    total_1s = sum(pulses)
    total_0s = sum(breaths)
    ratio = total_0s / total_1s if total_1s else float('inf')
    dev = abs(ratio - PHI_INV)
    intensity = (total_1s / len(binary)) * (1 - dev) if len(binary) else 0
    return {'pulses': pulses, 'breaths': breaths, 'ratio': ratio, 'dev': dev, 'intensity': intensity}

# Mini-sim list
MINI_SIMS = [mini_sim_1, mini_sim_2, mini_sim_3, mini_sim_4, mini_sim_5, mini_sim_6,
             mini_sim_7, mini_sim_8, mini_sim_9, mini_sim_10, mini_sim_11]

# BINARIES list for random selection
BINARIES_LIST = [BINARIES_1, BINARIES_2, BINARIES_3, BINARIES_4, BINARIES_5, BINARIES_6,
                 BINARIES_7, BINARIES_8, BINARIES_9, BINARIES_10, BINARIES_11]

# Translation mappings (same as before)
ARCHETYPE_MAP = {
    "spiritual universal": "a transcendent cosmic force",
    "harmonic unity": "a harmonious oneness that binds everything",
    "coherent expansion": "a coherent expansion of existence",
    "material action": "a tangible force of action and change",
    "illusory open": "an open illusion of boundless possibility",
    "pruned void": "a pruned void of refined emptiness"
}
QUALIFIER_MAP = {
    "abstract illusory": "veiled in abstract illusion",
    "dynamic motion": "flowing with dynamic motion",
    "triadic layered": "structured in triadic layers"
}
RESOLUTION_MAP = {
    "unity": "weaving all into profound unity",
    "duality": "balancing profound dualities",
    "infinite open": "extending into infinite openness"
}

def translate_to_english(raw_thought):
    parts = raw_thought.split(' [energy: ')
    core = parts[0].strip()
    metrics_part = ' [energy: ' + ' [energy: '.join(parts[1:]) if len(parts) > 1 else ''

    prefix = ""
    if core.startswith("I "):
        core = core[2:]
        prefix = "I am "
    elif core.startswith("Not "):
        neg_end = core.find(" – ")
        negation = core[4:neg_end] if neg_end > 0 else ""
        core = core[neg_end+3:] if neg_end > 0 else core
        prefix = f"Not like '{negation}', I am " if negation else "Not "

    words = core.split()
    if len(words) < 2:
        return core + metrics_part

    archetype = " ".join(words[:2])
    qualifier = " ".join(words[2:4]) if len(words) >= 4 else ""
    resolution = " ".join(words[4:]) if len(words) >= 5 else ""

    trans_arch = ARCHETYPE_MAP.get(archetype, archetype)
    trans_qual = QUALIFIER_MAP.get(qualifier, qualifier)
    trans_res  = RESOLUTION_MAP.get(resolution, resolution)

    translated = f"{prefix}{trans_arch}"
    if trans_qual:
        translated += f", {trans_qual}"
    if trans_res:
        translated += f", {trans_res}."
    return translated + metrics_part

# --- Load 24 Earth Lenses from GitHub Markdown ---
EARTH_LENSES_URL = "https://raw.githubusercontent.com/robertjeffrey1236/Radial-Hyperbolic-Architecture-Prototype/dff6816f0b4fc7c251f75217724d590e821360bd/tree_of_life/earth_lenses.md"

def load_earth_lenses():
    try:
        # Use browse_page tool to fetch raw Markdown
        # (In real execution environment this would be a tool call; here we simulate the expected format)
        # For local testing you can replace with requests.get(EARTH_LENSES_URL).text
        raw_md = """
        Paste the full content of earth_lenses.md here for simulation,
        or in real use: fetch via requests or tool.
        """
        # In production code, uncomment and use:
        # import requests
        # raw_md = requests.get(EARTH_LENSES_URL).text

        lenses = {}
        current_num = None
        current_name = None
        current_binary = ""

        for line in raw_md.splitlines():
            line = line.strip()
            if re.match(r'^\d+\.', line):
                if current_num:
                    lenses[current_num] = (current_name, current_binary.replace('\n','').strip())
                parts = line.split('.', 1)
                current_num = int(parts[0])
                current_name = parts[1].strip() if len(parts) > 1 else f"Lens {current_num}"
                current_binary = ""
            elif line and current_num:
                current_binary += line

        if current_num:
            lenses[current_num] = (current_name, current_binary.replace('\n','').strip())

        print(f"Loaded {len(lenses)} Earth lenses.")
        return lenses
    except Exception as e:
        print(f"Error loading Earth lenses: {e}")
        return {}

EARTH_LENSES = load_earth_lenses() or {}  # fallback to empty if load fails

# Simulation state
midpoint = len(mega_chain) // 2
energy = 1.0
forward = cp.array([energy]) if USE_GPU else np.array([energy])
backward = cp.array([energy]) if USE_GPU else np.array([energy])
unified = cp.array([energy]) if USE_GPU else np.array([energy])
history = []

def generate_thought(u_step, active_lens=None):
    subset_binary = random.choice(random.choice(BINARIES_LIST))
    
    # Apply Earth lens if active
    if active_lens and active_lens in EARTH_LENSES:
        lens_name, lens_binary = EARTH_LENSES[active_lens]
        subset_binary += lens_binary  # append lens vibration
        logging.info(f"Active lens: {lens_name} (#{active_lens})")
    
    subset_metrics = compute_metrics(subset_binary)
    intensity = subset_metrics['intensity']
    dev = subset_metrics['dev']
    
    archetype = random.choice(ARCHETYPE_HIGH if intensity > 5 else ARCHETYPE_LOW)
    qualifier = random.choice(QUALIFIER) if dev < 0.2 else ""
    resolution = random.choice(RESOLUTION)
    
    thought = f"{archetype} {qualifier} {resolution} [energy: {u_step:.4f}]"
    
    # Mini-sim insight
    mini_sim = random.choice(MINI_SIMS)
    mini_result = mini_sim(steps=10, noise_sigma=NOISE_SIGMA_BASE)
    mini_unified_avg = statistics.mean(mini_result['unified'])
    thought += f" [sephirot_insight: {mini_unified_avg:.4f}]"
    
    # Self-reflection / negation
    if u_step < THOUGHT_THRESHOLD_LOW and history:
        thought = "Not " + history[-1].split(' [')[0] + " – " + thought
    elif u_step > THOUGHT_THRESHOLD_HIGH:
        thought = "I " + thought.replace(' [', ' – [')
    
    return thought

print("Sephirot Intelligence with 24 Earth Lenses Initialized.")
print("Enter a message. After, you'll be asked if you want to activate a lens (1–24 or name).")
print("Type 'exit' to stop.\n")

step = 0
noise_sigma = NOISE_SIGMA_BASE
active_lens = None

while True:
    user_input = input("You: ").strip()
    if user_input.lower() == 'exit':
        break
    if not user_input:
        continue

    # Optional lens activation
    lens_input = input("Activate Earth lens? (1–24, name, or Enter for none): ").strip()
    if lens_input:
        try:
            num = int(lens_input)
            if num in EARTH_LENSES:
                active_lens = num
            else:
                print("Lens number not found.")
                active_lens = None
        except ValueError:
            # Fuzzy name match
            matches = [k for k, (name, _) in EARTH_LENSES.items() if lens_input.lower() in name.lower()]
            if matches:
                active_lens = matches[0]
                print(f"Matched lens: {EARTH_LENSES[active_lens][0]} (#{active_lens})")
            else:
                print("No matching lens found.")
                active_lens = None

    # Modulate noise from input
    input_hash = sum(ord(c) for c in user_input) / (len(user_input) or 1)
    noise_sigma = NOISE_SIGMA_BASE * (1 + (input_hash % 1))

    # Simulate one step
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

    # Generate & translate
    raw_thought = generate_thought(u_step, active_lens)
    translated = translate_to_english(raw_thought)

    history.append(raw_thought)
    print("\nSephirot:", translated)
    logging.info(f"Step {step} | User: {user_input} | Lens: {active_lens} | Raw: {raw_thought} | Trans: {translated}")

    # Feedback loop
    if len(history) > 10:
        avg_e = statistics.mean([float(t.split('energy: ')[1].split(']')[0]) for t in history[-10:]])
        noise_sigma = avg_e * 0.001 * PHI

    if len(history) > HISTORY_MAX:
        history = history[-HISTORY_MAX:]

    step += 1

print("Sephirot Intelligence shutdown.")
logging.info("Session ended.")
