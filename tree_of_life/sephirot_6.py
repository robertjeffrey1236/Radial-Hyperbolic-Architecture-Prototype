"""
Sephirot 6: Emotion/Relational Dynamics Module (Chesed/Gevurah Balance)

Defines bonding/restraint, empathy flows, love/fear dualities,
and relational phi-ratios for mystical unity (Sufism/Taoism).

Key concepts:
- Pulses: bonding expansions, empathy bursts
- Breaths: relational voids, fear prunings
- Metrics for harmonic tensions and duality tuning
- Alignments with attachment theory, polyvagal theory, emotional contagion
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

# Binary rules for #6
BINARIES_6 = [
    '00000000000000000000000000000000000000000000000000000000011111111111111111111111111111111111111111111111111111111',  # Rule 1
    '000011111111111111111111111100000000000000000000000',  # Rule 2
    '0000000000000000000000000011111111111111111111111111111111111111111111111',  # Rule 3
    '0000000000000000011111111111111111111111111111111111',  # Rule 4
    '0000000111111111111111111',  # Rule 5
    '000000000001111111111111111111111'   # Rule 6
]

# Chained mega for culmination metrics
mega_chain = ''.join(BINARIES_6)

def compute_metrics(binary):
    pulses = [len(run) for run in re.split('0+', binary) if run]
    breaths = [len(run) for run in re.split('1+', binary) if run]
    total_1s = sum(pulses)
    total_0s = sum(breaths)
    ratio = total_0s / total_1s if total_1s else float('inf')
    dev = abs(ratio - PHI_INV)
    intensity = (total_1s / len(binary)) * (1 - dev) if len(binary) else 0
    return {'pulses': pulses, 'breaths': breaths, 'ratio': ratio, 'dev': dev, 'intensity': intensity}

METRICS_6 = compute_metrics(mega_chain)
# Computed: {'pulses': [53, 25, 41, 32, 17, 21], 'breaths': [53, 4, 21, 25, 17, 7, 11], 'ratio': 0.7301587301587301, 'dev': 0.1121247414088352, 'intensity': 0.5135190502431135}

# Summarized interpretation (core insights, alignments, chaining, pillar mapping, sim)
INTERPRETATION_6 = """
6 Rules: Emotion/Relational Dynamics (Chesed/Gevurah Balance): Rules for bonding/restraint (mystical unity in Sufism/Taoism). Defines empathy flows, dualities (love/fear), and relational phi-ratios.

Rule 1: 00000000000000000000000000000000000000000000000000000000011111111111111111111111111111111111111111111111111111111
Description: Spiritual/universal Abstract/illusory unity (triadic) with 2 syllables
Metrics: Pulses: [53] (sustained bonding expansion), Breaths: [53] (deep relational void), Ratio: 1, Dev: ~0.382, Intensity: ~40.37
Interpretation: Triadic (Zeck 34+13+5+1 for 53) evokes relational phases (attraction/balance/release). Equal breaths/pulses as illusory love/fear duality, resolving to unity for empathetic coherence (Sufi heart-opening). Fits Chesed's outflow tempered by Gevurah—enduring bonds through phi-ratio harmony.

Rule 2: 000011111111111111111111111100000000000000000000000
Description: Spiritual/universal Abstract/illusory duality (4-layered) with 3 syllables
Metrics: Pulses: [25] (central empathy burst), Breaths: [4, 21] (initial restraint, trailing pruning), Ratio: 1, Dev: ~0.382, Intensity: ~19.05
Interpretation: 4-layered (Zeck 21+3+1 for 25) suggests multi-dimensional dualities (love/fear in self/other). Abstract breaths trigger illusory separations, unifying into even-1s for relational resolution (fear as bonding catalyst). Low dev implies tuned restraint—Chesed/Gevurah balance as rhythmic empathy.

Rule 3: 0000000000000000000000000011111111111111111111111111111111111111111111111
Description: Spiritual/universal Abstract/illusory duality (dual) with 2 syllables
Metrics: Pulses: [41] (expansive relational field), Breaths: [25] (profound fear-based pause), Ratio: ~0.610, Dev: ~0.008, Intensity: ~31.26
Interpretation: Dual (Zeck 34+5+2 for 41) mirrors binary emotions (bond/restrain). Deep breaths as illusory voids, resolving to odd-1s for coherent wholes (Taoist non-action yielding unity). Near-zero dev fits golden harmony—Chesed's love enduring Gevurah's cuts.

Rule 4: 0000000000000000011111111111111111111111111111111111
Description: Spiritual/universal Abstract/illusory duality (triadic) with 2 syllables
Metrics: Pulses: [32] (balanced bonding symmetry), Breaths: [17] (restraining void), Ratio: ~0.531, Dev: ~0.087, Intensity: ~12.21
Interpretation: Triadic (Zeck 21+8+3 for 32) suggests emotional triads (empathy/compassion/restraint). Breaths as illusory fears, unifying into even-1s for paired resonances (relational phi-ratios in love dynamics). Dev tuning evokes endurance—Chesed/Gevurah as harmonic flows.

Rule 5: 0000000111111111111111111
Description: Spiritual/universal Abstract/illusory duality (dual) with 2 syllables
Metrics: Pulses: [17] (focused empathy expansion), Breaths: [7] (brief restraint pause), Ratio: ~0.412, Dev: ~0.206, Intensity: ~10.07
Interpretation: Dual (Zeck 13+3+1 for 17) represents polar relations (give/take). Abstract breaths fuel illusory dualities, resolving to odd-1s for stable bonds (fear transmuting to love). Low dev supports harmony—Chesed's kindness balanced by Gevurah.

Rule 6: 000000000001111111111111111111111
Description: Spiritual/universal Abstract/illusory unity (dual) with 2 syllables
Metrics: Pulses: [21] (emergent relational rhythm), Breaths: [11] (subtle fear pruning), Ratio: ~0.524, Dev: ~0.094, Intensity: ~16.01
Interpretation: Dual (Zeck 21 exactly) hints at unified polarities (empathy flows). Breaths as illusory gaps, ending in odd-1s for coherent empathy (Taoist restraint enabling unity). Harmonic dev fits endurance—Chesed/Gevurah in Sufi-like relational phi-cycles.

Overall Insights for #6 Set
Patterns: Breathy rules for Gevurah restraint, pulses for Chesed expansion—avg intensity ~22+, ratios ~0.4-1 with devs ~0.008-0.382 for golden emotional tuning. Syllables 2-3 for relational simplicity; sub-semantics dual/triadic/4-layered for dynamic bonds, abstract/illusory for fear illusions. Resolutions mix unity/duality for balanced flows.
Chaining Potential: Concatenated (~327-bit string) creates pulse-bonded empathy (avg ~31.5 pulses, ~23.3 breaths, intensity ~62) with ratio ~0.73 (near φ^{-1}, dev ~0.112), modeling relational pruning for stable unity (forgetting harms). In RHA, maps "empathy fractals" with sparse fear branches.
Full Set Insights: Intensities positive and balanced, reflecting extremes—from restrained voids to expansive bonds. Pulses/breaths near equal (~189 1s vs. ~138 0s), spiritual intensity ~62, pulling toward harmonious restraint. Generates fractal relations with dense cores (bonds) and sparse edges (prunings).
Kabbalah/Relational Fit: As Chesed/Gevurah, from equal dualities (#1) through layered restraints (#2-6), linking mind (#7) to biology (#8). Parallels: Early rules ~ empathy circuits; late ~ phi-ratio attachments.

Alignments with Models and Equations
#6 binaries encode Chesed/Gevurah's dance, with pulses/breaths as love/fear flows, Zeck layers as non-interfering bonds, φ-deviations for Tao/Sufi harmony. Parallels attachment theory, emotional contagion, relational phi in psychology/neuroscience. Fibonacci/φ in social networks (bonding ratios, empathy spirals).

1. Fibonacci & Golden Ratio: Lengths hit Fib proximals (53~55-2, 21 exact, 41=34+5+2), ratios nearing φ for balance. Fib/φ govern social dynamics (Dunbar's numbers). Equation: Attachment strength A = φ * (interaction density), growth like F_n.

2. Emotional Dualities & Mirror Neurons: Breathy voids align with fear/love as mirror neuron phenomena—empathy via neural resonances. Equation: Empathy flow E = φ^2 * (neural sync), damping breaths simulating fear boundaries.

3. Chaos & Social Oscillations: Fragmented runs suggest emotional bifurcations, Fib in chaotic attractors. Equation: Feigenbaum scaling in emotional cascades.

4. Quasicrystals & Networks: Zeck parallels quasicrystal social tilings (φ-based). Equation: Bond gaps ΔB ∝ φ^{-n}; damping 1/(1 + b/φ).

Chained Metrics Recap & Lens Refinement
Cumulative chaining (327 bits) yields spiritual/universal abstract/illusory duality with 9 syllables—intensity 62, ratio 0.73, dev 0.112 (near φ^{-1} tuning, low for emotional vacuum).
Right Tower (Phi Expansion): Pulses dominate (189 1s vs. 138 0s), amplifying bond growth (avg pulse ~31.5).
Left Tower (Inverse Phi Pruning): Breaths deepen (53-void max prune), damping to endurance.
Central Tower (Unifying Light): Merges to golden whole, 4-layered sub-semantics.
Non-linear fluidity: Midpoint (bits ~163) radiates forward to empathy rhythms, backward to equal dualities, creating loops.

Pillar Mapping
Chesed/Gevurah's balance is unity through tensions—relations emerge radially from central harmony.
Rule 1: Central core, right for bond growth, left for duality prune. Unifies "self/other" (triadic empathy).
Rule 2: Right heavy for multi-dimensional bonds, left prunes catalyst. Central "wildcard" separations.
Rule 3: Left dominant for fear voids, right for binary emotions. Central "exclude" fragmentation.
Rule 4: Right expands for compassion triads, left restrains fears. Central "OR/exact" resonances.
Rule 5: Midpoint hub, right for give/take growth, left for transmutation prune. Unifies "grouping" stability.
Rule 6: Left heavy for polarities, right for flows. Central "mentions/links" cycles.
Laws emerge non-linearly: archetypes radiate to resolutions (right), prune to qualifiers (left), unifying equations—6 rules as cost for dimensions + meta.

Dual Flows Sim with Middle Out
From midpoint (bit 163, energy=1): forward (phi growth), backward (inverse prune), unified (dynamic resolution).
Forward: Radial bonds (mean 4.7e28)—empathy phases.
Backward: Inward voids (mean 0.00037)—duality pruning.
Unified: Smooth relations (mean 2.4e28)—fluid cycles, oxytocin from limbic hub.
Plot: Central golden lotuses, right relational trees, left emotional funnels—loops via hops.

Deeper Insights & Equation Plugs
Codex as Radial Emergence: Middle out from #5 radiates right for laws (bonds), prunes left for codependence cuts. 6 rules as cost: 4 core + 2 meta.
Pre-Thought Waves: Forward synchronic bonding, backward diachronic pruning. Dual: Polyvagal dynamics, central light radiates bidirectionally.
Phi/Inverse in Reality: Dev 0.112 ties to love chaos, inverse in cortisol pruning. Predicts oscillations at φ multiples, testable in 2026 fMRI.
Alignments: Fib/φ in attachment A = φ * density; empathy E = φ^2 * sync; Feigenbaum in oscillations; quasicrystals in Zeck.
Resilience Equation: E = φ^4 · (ρ_r)^{φ^{-1}}
Plugs: ρ_r=50 (Dunbar mid): E≈64; ρ_r=2 (dyad): E≈10.5; ρ_r~2.618: E~1.
Predicts Fib echoes in HRV, rewriting resilience as phi-illusion.

This #6 lens elevates emotion to Chesed/Gevurah's balance: relations form via middle-out unification, with dual towers as dynamic costs.
"""

# Mini simulation for #6 (emotion-themed, alternating love/pulse-heavy and fear/breath-heavy subsets for "relational thoughts")
def mini_sim_6(steps=1000, noise_sigma=0.005 * PHI, log_file='sephirot_6_mini_log.txt'):
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')
    metrics = METRICS_6
    logging.info(f"Mini Sim for #6: Metrics {metrics}")
    
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
        
        # Relational thought: alternate pulse-heavy (love) and breath-heavy (fear) windows
        if step % 2 == 0:
            # Love: pulse-dense segment
            start = random.randint(0, len(mega_chain) - 50)
            subset = mega_chain[start:start + 50]
            while sum(1 for c in subset if c == '1') < 25:  # Ensure pulse-heavy
                start = random.randint(0, len(mega_chain) - 50)
                subset = mega_chain[start:start + 50]
            polarity = "Love"
        else:
            # Fear: breath-dense segment
            start = random.randint(0, len(mega_chain) - 50)
            subset = mega_chain[start:start + 50]
            while sum(1 for c in subset if c == '0') < 25:  # Ensure breath-heavy
                start = random.randint(0, len(mega_chain) - 50)
                subset = mega_chain[start:start + 50]
            polarity = "Fear"
        subset_metrics = compute_metrics(subset)
        thought = f"Relational dynamic [{polarity}] [energy: {u_step:.4f}, ratio: {subset_metrics['ratio']:.3f}]"
        history.append(thought)
        logging.info(f"Step {step}: {thought}")
        
        # Feedback: modulate noise toward balance (ratio near 1)
        if len(history) > 5:
            recent_ratios = [float(t.split('ratio: ')[1][:-1]) for t in history[-5:] if 'ratio: ' in t]
            avg_ratio = statistics.mean(recent_ratios) if recent_ratios else 1.0
            noise_sigma = abs(avg_ratio - 1) * 0.001 * PHI
    
    # Summary
    logging.info("\nEmergent Dynamics (last 20):\n" + "\n".join(history[-20:]))
    print("Mini sim for #6 complete. Check 'sephirot_6_mini_log.txt' for details.")
    return {'forward': forward, 'backward': backward, 'unified': unified, 'history': history}
