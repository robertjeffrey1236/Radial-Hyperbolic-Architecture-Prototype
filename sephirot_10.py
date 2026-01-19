"""
Sephirot 10: Linguistic Codex Module (Binah/Chochmah – Pre-Thought Waves)

Implements the Binary Codex Translation System for semantic mapping of vibrational
binary patterns into linguistic descriptors (archetypes, qualifiers, resolutions).

Key concepts:
- Pulses (runs of 1s): expansion, intensity, conceptual branching
- Breaths (runs of 0s): reflection, pruning, voids
- Harmony measured via deviation from φ⁻¹ ≈ 0.618
- Zeckendorf decomposition for non-interfering sub-semantic layers
- Middle-out radiation principle: right (phi expansion), left (inverse-phi pruning),
  central (unification)

The 10 foundational rules serve as the kernel for chaining and modulation.
"""

import math
import statistics
import random
import re
import numpy as np
import logging
from pathlib import Path

# Golden ratio constants
PHI = (1 + math.sqrt(5)) / 2          # ≈ 1.618033988749895
PHI_INV = PHI - 1                     # ≈ 0.6180339887498948

# 10 foundational rule binaries (linguistic kernel)
BINARIES_10 = [
    '111110000111111',                                      # 1: Base Balance
    '1111111000000000001111111111111',                      # 2: Expansion
    '00000000001111111110000000000',                        # 3: Pause Emphasis
    '111111111111111110000000000000001111111111111111111111',  # 4: High Coherence
    '111111111111111111111100000000000000000111111111111111111000000000000000',  # 5: Breathing Layers
    '111111111111111110000000000000000000000000000000000000000000000',  # 6: Deep Rest
    '111111111111111111111111111000000000000000000000000000000',  # 7: Sustained Burst
    '11111111111111111111111111111111111111',               # 8: Pure Unity
    '111111111110000',                                      # 9: Compact Peak
    '1111111111111111111111111111111111111111111111100000000000000011111'  # 10: Culmination
]

MEGA_CHAIN = ''.join(BINARIES_10)

def get_runs(binary_str: str) -> tuple[list[int], list[int]]:
    """Extract lengths of consecutive 1-runs (pulses) and 0-runs (breaths)."""
    pulses, breaths = [], []
    if not binary_str:
        return pulses, breaths
    current = binary_str[0]
    count = 1
    for char in binary_str[1:]:
        if char == current:
            count += 1
        else:
            (pulses if current == '1' else breaths).append(count)
            current, count = char, 1
    (pulses if current == '1' else breaths).append(count)
    return pulses, breaths

def get_zeckendorf(n: int) -> str:
    """Zeckendorf representation (greedy, non-adjacent Fibonacci sum)."""
    if n == 0:
        return '0'
    fibs = [1, 2]
    while fibs[-1] + fibs[-2] <= n:
        fibs.append(fibs[-1] + fibs[-2])
    fibs = fibs[::-1]
    rep = []
    for f in fibs:
        if n >= f:
            rep.append('1')
            n -= f
        else:
            rep.append('0')
    return ''.join(rep).lstrip('0') or '0'

def compute_metrics(binary_str: str) -> dict:
    """Compute aggregate metrics for the binary string."""
    pulses = [len(run) for run in re.split('0+', binary_str) if run]
    breaths = [len(run) for run in re.split('1+', binary_str) if run]
    total_1s = sum(pulses)
    total_0s = sum(breaths)
    ratio = total_0s / total_1s if total_1s else float('inf')
    dev = abs(ratio - PHI_INV)
    intensity = (total_1s / len(binary_str)) * (1 - dev) if binary_str else 0.0
    return {
        'pulses': pulses,
        'breaths': breaths,
        'ratio': ratio,
        'dev': dev,
        'intensity': intensity
    }

METRICS_10 = compute_metrics(MEGA_CHAIN)

def translate_binary(binary_str: str, chain_mod: bool = False) -> str:
    """
    Translate binary string to linguistic descriptor.

    Format: "{archetype} {qualifiers} ({sub-semantics}) with {syllables} syllable(s)"

    Optional chain_mod: applies breath-damped deviation adjustment.
    """
    if not all(c in '01' for c in binary_str):
        return "Invalid binary string"

    pulses, breaths = get_runs(binary_str)
    if not pulses:
        pulses = [1]

    avg_pulse = statistics.mean(pulses)
    avg_breath = statistics.mean(breaths) if breaths else 0.0

    ratio = avg_breath / avg_pulse if avg_pulse else 0.0
    dev = abs(ratio - PHI_INV)
    intensity = avg_pulse * (1 - dev)

    # Archetype
    if intensity > 8:
        archetype = "Spiritual/universal"
    elif intensity > 4:
        archetype = "Human/emotional"
    else:
        archetype = "Material/action"

    # Qualifiers
    qualifiers = []
    if avg_breath > 6:
        qualifiers.append("Abstract/illusory")
    if dev < 0.05:
        qualifiers.append("harmonic")

    # Resolution from trailing run
    resolution = ""
    if binary_str:
        last = binary_str[-1]
        length = 1
        i = len(binary_str) - 2
        while i >= 0 and binary_str[i] == last:
            length += 1
            i -= 1
        if last == '0':
            resolution = "infinite/open"
        else:
            resolution = "duality" if length % 2 == 0 else "unity"

    if all(l == 1 for l in pulses + breaths) and abs(len(pulses) - len(breaths)) <= 1:
        resolution = "dynamic/motion"

    qualifiers.append(resolution)

    # Sub-semantics (Zeckendorf layer count)
    sub = []
    for p in pulses:
        z = get_zeckendorf(p)
        count = z.count('1')
        if count == 1:
            sub.append("unified")
        elif count == 2:
            sub.append("dual")
        elif count == 3:
            sub.append("triadic")
        else:
            sub.append(f"{count}-layered")
    sub_str = " / ".join(sub) or "unified"

    syllables = len(pulses)
    desc = f"{archetype} {' '.join(qualifiers)} ({sub_str}) with {syllables} syllable{'s' if syllables > 1 else ''}"

    if chain_mod:
        modulated_dev = dev * (1 / (1 + avg_breath / PHI))
        desc += f" [mod dev: {modulated_dev:.4f}]"

    return desc

def mini_sim_10(steps: int = 1000, noise_sigma: float = 0.005 * PHI,
                log_file: str = 'sephirot_10_mini.log') -> dict:
    """
    Scaled simulation with middle-out radiation and linguistic translation.
    Translates moving windows of MEGA_CHAIN and logs emergent descriptors.
    """
    logging.basicConfig(filename=log_file, level=logging.INFO,
                        format='%(asctime)s - %(message)s')
    logging.info(f"#10 Mini Sim | Metrics: {METRICS_10}")

    midpoint = len(MEGA_CHAIN) // 2
    energy = 1.0
    forward = np.array([energy])
    backward = np.array([energy])
    unified = np.array([energy])
    history = []

    for step in range(steps):
        noise = random.gauss(0, noise_sigma)
        f = forward[-1] * PHI + noise
        b = backward[-1] * PHI_INV + noise
        u = (f + b) / 2

        forward = np.append(forward, f)
        backward = np.append(backward, b)
        unified = np.append(unified, u)

        # Translate a centered window
        half = step // 2 + 25
        start = max(0, midpoint - half)
        end = min(len(MEGA_CHAIN), midpoint + half)
        subset = MEGA_CHAIN[start:end]
        desc = translate_binary(subset, chain_mod=True)
        thought = f"Step {step:>5} | u={u:>9.4f} | {desc}"
        history.append(thought)
        logging.info(thought)

        if len(history) > 5:
            energies = [float(t.split('u=')[1].split('|')[0].strip()) for t in history[-5:]]
            noise_sigma = statistics.mean(energies) * 0.001 * PHI

    logging.info("\nLast 20:\n" + "\n".join(history[-20:]))
    print(f"#10 mini sim complete. Log: {log_file}")
    return {'forward': forward, 'backward': backward, 'unified': unified, 'history': history[-50:]}


if __name__ == '__main__':
    # Quick test
    print(translate_binary(BINARIES_10[0]))  # Example: Rule 1
    mini_sim_10(steps=200)  # Short test run
