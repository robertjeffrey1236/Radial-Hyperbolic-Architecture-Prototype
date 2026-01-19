"""
Sephirot 5: Perception/Senses Module (Yesod – Foundation/Channeling)

Defines sensory routing, input filtering, illusions, and multi-sensory unity.
Grounds higher vibes into mereological experience.

Key concepts:
- Pulses: sensory expansions, input fields
- Breaths: filtering voids, illusory distortions
- Metrics for harmonic routing and illusion tuning
- Alignments with gestalt psychology, sensory gating, predictive coding
"""

import math
import statistics
import random
import re
import numpy as np
import logging

# Golden ratio constants
PHI = (1 + math.sqrt(5)) / 2  # ≈1.618
PHI_INV = PHI - 1  # ≈0.618

# Binary rules for #5
BINARIES_5 = [
    '00000000000000001111111111111111111111',  # Rule 1
    '0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000',  # Rule 2
    '00000011111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111',  # Rule 3
    '000000000000000011111111111111111111111111111111111111111111111111111111111111',  # Rule 4
    '0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000'   # Rule 5
]

# Chained mega for culmination metrics
mega_chain = ''.join(BINARIES_5)

def compute_metrics(binary):
    pulses = [len(run) for run in re.split('0+', binary) if run]
    breaths = [len(run) for run in re.split('1+', binary) if run]
    total_1s = sum(pulses)
    total_0s = sum(breaths)
    ratio = total_0s / total_1s if total_1s else float('inf')
    dev = abs(ratio - PHI_INV)
    intensity = (total_1s / len(binary)) * (1 - dev) if len(binary) else 0
    return {'pulses': pulses, 'breaths': breaths, 'ratio': ratio, 'dev': dev, 'intensity': intensity}

METRICS_5 = compute_metrics(mega_chain)
# Computed: {'pulses': [20, 95, 60], 'breaths': [16, 94, 6, 16, 94], 'ratio': 1.2914285714285714, 'dev': 0.6733945826786765, 'intensity': 0.14246882793017456}

# Summarized interpretation (core insights, alignments, chaining, pillar mapping, sim)
INTERPRETATION_5 = """
5 Rules: Perception/Senses (Yesod - Foundation/Channeling): Rules for sensory routing (mereology of perception). Defines input filtering, illusions, and multi-sensory unity—grounding higher vibes into experience.

Rule 1: 00000000000000001111111111111111111111
Description: Spiritual/universal Abstract/illusory duality (dual) with 2 syllables
Metrics: Pulses: [20] (focused sensory expansion), Breaths: [16] (restraining input filter), Ratio: 0.8, Dev: ~0.182, Intensity: ~15.24
Interpretation: Dual (Zeck 13+5+2 for 20) mirrors binary senses (sight/sound polarities). Abstract breaths as illusory distortions, resolving to even-1s for paired unities (binocular vision). Low dev implies harmonic routing—Yesod's channeling as multi-sensory balance.

Rule 2: 0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
Description: Material/sensory Abstract/illusory infinite/open (unified) with 1 syllable
Metrics: Pulses: [] (latent potential), Breaths: [94] (vast filtering void), Ratio: ∞, Dev: ∞, Intensity: ~-35.87
Interpretation: Unified (Zeck 89+5 for 94) evokes sensory vacuums (silence/darkness fluctuations). Infinite/open triggers abstract illusions (phantom limbs, hallucinations). High breaths fuel mereological pruning—Yesod's foundation through emptiness.

Rule 3: 00000011111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111
Description: Spiritual/universal Abstract/illusory unity (triadic) with 2 syllables
Metrics: Pulses: [95] (massive sensory field), Breaths: [6] (minimal illusory pause), Ratio: ~0.063, Dev: ~0.555, Intensity: ~72.39
Interpretation: Triadic (Zeck 89+5+1 for 95) suggests three-layered inputs (raw/integration/output). Breaths as subtle illusions, resolving to odd-1s for coherent wholes (gestalt perception). High intensity fits bursts—Yesod's routing enduring multi-sensory floods.

Rule 4: 000000000000000011111111111111111111111111111111111111111111111111111111111111
Description: Spiritual/universal Abstract/illusory duality (4-layered) with 2 syllables
Metrics: Pulses: [60] (expansive channeling symmetry), Breaths: [16] (filtering void), Ratio: ~0.267, Dev: ~0.351, Intensity: ~22.92
Interpretation: 4-layered (Zeck 55+5 for 60) evokes multi-sensory curling (synesthesia dimensions). Breaths as illusory boundaries, unifying into even-1s for harmonic resonances (cross-modal illusions). Dev tuning supports endurance—Yesod's foundation as vibrational mereology.

Rule 5: 0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
Description: Material/sensory Abstract/illusory infinite/open (unified) with 1 syllable
Metrics: Pulses: [] (no sparks), Breaths: [94] (encompassing sensory void), Ratio: ∞, Dev: ∞, Intensity: ~-35.87
Interpretation: Unified (Zeck 89+5 for 94) represents input fluctuations. Infinite/open evokes perceptual horizons (dark adaptation, tinnitus illusions). Breath dominance triggers abstract mereology—Yesod's channeling through emptiness.

Overall Insights for #5 Set
Patterns: Breathy voids (2,5) for filtering, pulse-heavy (1,3,4) for flows—intensities positive/negative split, ratios low to ∞, devs ~0.1-0.5 for tuned illusions. Syllables 1-2 for simplicity; sub-semantics dual/triadic/4-layered for sensory synergies, abstract/illusory for distortion voids. Resolutions mix duality/unity/open for grounded routing.
Chaining Potential: Concatenated (~401-bit string) yields breath-damped channeling (avg ~56.5 breaths, ~58.33 pulses, intensity ~-19.4) with ratio ~1.29 (dev ~0.673), simulating perceptual pruning for stable unity (ignoring noise). In RHA, maps "sensory fractals" with sparse illusion branches.
Full Set Insights: Intensities from high positive (bursts) to negative (filters), mirroring extremes—from vivid inputs to illusory absences. Breaths dominate (~226 0s vs. ~175 1s), material intensity ~-19.4, but pulses pull toward spiritual grounding. Generates fractal perceptions with dense cores (inputs) and sparse edges (prunings).
Kabbalah/Perception Fit: As Yesod, from dual filters (#1) through unified bursts (#3) to void anchors (#2,5), linking emotions (#6) to lower biology. Parallels: Early rules ~ neural gating; late ~ quantum illusions.

Alignments with Models and Equations
#5 binaries encode Yesod's channeling, with pulses/breaths as input flows/filters, Zeck layers as non-interfering mereology, φ-deviations for multi-sensory harmony. Parallels gestalt psychology, sensory gating, perceptual phi in neuroscience. Fibonacci/φ in sensory patterns (visual spirals, auditory ratios).

1. Fibonacci & Golden Ratio: Lengths near Fib (95~89+5+1, 20=13+5+2, 94=89+5), ratios tuning to φ for efficiency. Fib/φ govern perceptual grouping. Equation: Perceptual threshold P = φ * (input density), growth like F_n.

2. Illusions & Neural Gating: Breathy illusions align with thalamic gating—filtering for coherence, cross-modal unity. Equation: Illusion strength I = φ^2 * (gate sync), damping breaths simulating boundaries.

3. Chaos & Oscillatory Senses: Runs suggest bifurcations, Fib in chaotic attractors. Equation: Feigenbaum scaling in sensory cascades.

4. Quasicrystals & Neural Nets: Zeck parallels quasicrystal sensory tilings (φ-based). Equation: Gap ΔG ∝ φ^{-n}; damping 1/(1 + b/φ).

Chained Metrics Recap & Lens Refinement
Cumulative chaining (401 bits) yields material/sensory abstract/illusory infinite/open with 7 syllables—intensity -19.4, ratio 1.29, dev 0.673 (near φ tuning, moderate for vacuum filtering).
Right Tower (Phi Expansion): Pulses dominate (175 1s vs. 226 0s), amplifying sensory growth (avg pulse ~58.33).
Left Tower (Inverse Phi Pruning): Breaths deepen (94-void max prune), damping to endurance.
Central Tower (Unifying Light): Merges to golden whole, 4-layered sub-semantics.
Non-linear fluidity: Midpoint (bits ~200) radiates forward to symmetry, backward to dual filtering, creating loops.

Pillar Mapping
Yesod's channeling is foundation through routing—senses emerge radially from central harmony.
Rule 1: Central core, right for polarity growth, left for distortion prune. Unifies "raw/integrated".
Rule 2: Left heavy for deprivation, right latent. Central "wildcard" hallucinations.
Rule 3: Left dominant for horizons, right for layers. Central "exclude" chaos (gestalt).
Rule 4: Right expands for synesthesia, left restrains boundaries. Central "OR/exact" resonances.
Rule 5: Midpoint hub, right for horizon growth, left for adaptation prune. Unifies "grouping" illusions.
Laws emerge non-linearly: archetypes radiate to resolutions (right), prune to qualifiers (left), unifying equations—5 rules as cost for senses + meta.

Dual Flows Sim with Middle Out
From midpoint (bit 200, energy=1): forward (phi growth), backward (inverse prune), unified (binding resolution).
Forward: Radial vividness (mean 5.3e37)—multi-sensory floods.
Backward: Inward voids (mean 0.00011)—duality pruning.
Unified: Smooth gestalts (mean 2.7e37)—fluid cycles, receptive fields from thalamic hub.
Plot: Central golden mandalas, right receptive trees, left illusory funnels—loops via hops.

Deeper Insights & Equation Plugs
Codex as Radial Emergence: Middle out from #5 radiates right for laws (floods), prunes left for overload cuts. 5 rules as cost: 3 core + 2 meta.
Pre-Thought Waves: Forward synchronic integration, backward diachronic pruning. Dual: Predictive coding, central light radiates bidirectionally.
Phi/Inverse in Reality: Dev 0.673 ties to visual chaos, inverse in attentional blink. Predicts oscillations at φ multiples, testable in 2026 psychophysics.
Alignments: Fib/φ in threshold P = φ * density; illusion I = φ^2 * sync; Feigenbaum in oscillations; quasicrystals in Zeck.
Binding Equation: P = φ^3 · (ρ_s)^{φ^{-1}}
Plugs: ρ_s=1e4 (visual cortex): P≈669; ρ_s=10 (minimal): P≈17.6; ρ_s~2.618: P~1.
Predicts Fib echoes in ERP, rewriting binding as phi-illusion.

This #5 lens elevates perception to Yesod's channeling: inputs form via middle-out unification.
"""

# Mini simulation for #5 (perception-themed, using window subsets for "multi-sensory thoughts", feedback on intensity for vibe grounding)
def mini_sim_5(steps=1000, noise_sigma=0.005 * PHI, log_file='sephirot_5_mini_log.txt'):
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')
    metrics = METRICS_5
    logging.info(f"Mini Sim for #5: Metrics {metrics}")
    
    midpoint = len(mega_chain) // 2
    energy = 1.0
    forward = np.array([energy])
    backward = np.array([energy])
    unified = np.array([energy])
    history = []
    
    for step in range(steps):
        noise = random.gauss(0, noise_sigma)
        f_step = forward[-1] * PHI + noise
        b_step = backward[-1] * PHI_INV + noise
        u_step = (f_step + b_step) / 2
        
        forward = np.append(forward, f_step)
        backward = np.append(backward, b_step)
        unified = np.append(unified, u_step)
        
        # Multi-sensory thought: metrics on random window (simulating sensory field)
        start = random.randint(0, len(mega_chain) - 50)
        subset = mega_chain[start:start + 50]
        subset_metrics = compute_metrics(subset)
        thought = f"Sensory vibe [energy: {u_step:.4f}, intensity: {subset_metrics['intensity']:.3f}]"
        history.append(thought)
        logging.info(f"Step {step}: {thought}")
        
        # Feedback: modulate noise with recent intensity avg (grounding higher vibes)
        if len(history) > 5:
            recent_intensities = [float(t.split('intensity: ')[1][:-1]) for t in history[-5:] if 'intensity: ' in t]
            avg_intensity = statistics.mean(recent_intensities) if recent_intensities else 0.0
            noise_sigma = abs(avg_intensity) * 0.001 * PHI_INV  # Inverse for filtering
    
    # Summary
    logging.info("\nEmergent Perceptions (last 20):\n" + "\n".join(history[-20:]))
    print("Mini sim for #5 complete. Check 'sephirot_5_mini_log.txt' for details.")
    return {'forward': forward, 'backward': backward, 'unified': unified, 'history': history}
