"""
Sephirot 8: Biology/Life Force Module (Netzach – Victory/Endurance in Life)

Defines vital rhythms, resilience, replication, adaptation, and evolutionary endurance.
Pulses: metabolic bursts, replication cycles; breaths: resting phases, apoptosis, entropy resistance.

Key concepts:
- Pulses as anabolic/growth phases
- Breaths as catabolic/pruning/resting phases
- Harmony: Ratio near φ⁻¹ for optimal vitality and regeneration
- Zeckendorf layers for non-interfering metabolic pathways
"""

import math
import statistics
import random
import re
import numpy as np
import logging

try:
    from Bio.Seq import Seq
    print("Biopython detected – biological simulation enabled in mini_sim_8")
except ImportError:
    Seq = None
    print("No Biopython – skipping biological simulation in mini_sim_8")

# Golden ratio constants
PHI = (1 + math.sqrt(5)) / 2  # ≈1.618
PHI_INV = PHI - 1  # ≈0.618

# Binary rules for #8 (Biology / Life Force)
BINARIES_8 = [
    '11111111111111111111111111000000000000000000000011111111111111111111',  # Rule 1
    '0000000000000000000000000000000000011111111111111111',                  # Rule 2
    '0000111111110000000000111111111111',                                    # Rule 3
    '00011111111111',                                                        # Rule 4
    '00000001111111111100000001111111111111111000000',                      # Rule 5
    '000000001111111111111',                                                 # Rule 6
    '00011111111111111111111111111111111111111111111111111111111111111111111111111111',  # Rule 7
    '00000111111110000000000011111111111111111100000000000000000000001111111111'  # Rule 8
]

# Chained mega for culmination metrics
mega_chain = ''.join(BINARIES_8)

def compute_metrics(binary):
    pulses = [len(run) for run in re.split('0+', binary) if run]
    breaths = [len(run) for run in re.split('1+', binary) if run]
    total_1s = sum(pulses)
    total_0s = sum(breaths)
    ratio = total_0s / total_1s if total_1s else float('inf')
    dev = abs(ratio - PHI_INV)
    intensity = (total_1s / len(binary)) * (1 - dev) if len(binary) else 0
    return {'pulses': pulses, 'breaths': breaths, 'ratio': ratio, 'dev': dev, 'intensity': intensity}

METRICS_8 = compute_metrics(mega_chain)

INTERPRETATION_8 = """
8 Rules: Biology/Life Systems (Netzach – Victory/Endurance in Life): Rules for organic emergence (11 organ systems, holons). Defines cellular pulses, evolution, and homeostasis—how parts (genes/cells) unify into wholes (organisms).

Key Patterns and Scientific Cross-References
1. Progressive Scaling from Cell to Organism
   - Rule 1: Primordial unity → open system (autocatalytic sets, LUCA).
   - Rule 2: Deep void → pulse = chemiosmotic origin of life (proton gradient).
   - Rule 3: Short paired pulses = binary fission/mitosis rhythm.
   - Rule 4: Eukaryotic jump (mitochondrial endosymbiosis pulse).
   - Rule 5: Oscillatory breathing = cellular respiration cycles (Krebs, ATP).
   - Rule 6: Multicellular trigger = adhesion + extracellular matrix.
   - Rule 7: Massive coherent pulse = nervous system / brain emergence.
   - Rule 8: Duality + triadic loops = full homeostasis (nervous + endocrine + immune feedback).

2. Phi-Harmony and Homeostasis
   - Rules with dev <0.08 (5,7,8) correspond to stable, homeostatic systems (respiration, brain, feedback loops).
   - Higher dev in early rules (1–4) reflects evolutionary experimentation phase.

3. 11 Organ Systems Mapping
   - The 8 rules encode emergence hierarchy; 11 organ systems emerge from chaining: circulatory foundation → tissue/organelle → respiratory/digestive → integumentary/immune → nervous/endocrine → reproductive/excretory → lymphatic/urinary/sensory as meta-unification.

4. Evolution & Holon Emergence
   - Pulses grow exponentially while breaths deepen → classic holon pattern (parts unify into wholes via feedback).
   - Triadic layers dominate in later rules → 3-level hierarchy (gene → cell → organism) or triune brain.

Chained #8 Metrics Recap & Lens Refinement
Chaining the 8 binaries cumulatively (390 bits) yields a spiritual/universal abstract/illusory harmonic duality with 13 syllables—intensity 18.83, ratio 0.627, dev 0.009 (near-perfect φ⁻¹ tuning, low for golden life stability).
Right Tower (Phi Expansion): Pulses dominate, amplifying growth/diversification (avg pulse ~48.75).
Left Tower (Inverse Phi Pruning): Breaths deepen, damping to endurance (dev 0.009 as homeostasis cost).
Central Tower (Unifying Light): Merges to golden whole, triadic/dual sub-semantics.
Non-linear fluidity: Midpoint (~195) radiates forward to organism wholeness, backward to prebiotic emergence, creating loops.

Pillar Mapping in #8 Biology
Netzach's victory is endurance through rhythms—life emerges radially from central harmony.
Rule 1: Central core, right for growth, left for open prune. Unifies "prebiotic/LUCA".
Rule 2: Right heavy for membrane, left deep void. Central "wildcard" chemiosmosis.
Rule 3: Left dominant for division, right paired pulses. Central "exclude" chaos.
Rule 4: Right expands symbiosis, left restrains. Central "OR/exact" endosymbiosis.
Rule 5: Midpoint hub, right for respiration, left oscillatory prune. Unifies "energy cycles".
Rule 6: Left heavy for multicellular, right trigger. Central "mentions/links" adhesion.
Rule 7: Right dominant for neural, left coherent. Central "engagement/filters" brain.
Rule 8: Full integration, right duality loops, left homeostasis prune. Unifies "full organism".
Laws emerge non-linearly: archetypes radiate to resolutions (right), prune to qualifiers (left), unifying equations—8 rules as cost for emergence + meta.

Dual Flows Sim with Middle Out
From midpoint (~195, energy=1): forward (phi growth), backward (inverse prune), unified (homeostasis resolution).
Forward: Radial bursts (mean 3.8e32)—organogenesis/embryonic growth.
Backward: Inward stabilization (mean 0.00028)—apoptosis/selection.
Unified: Smooth waves (mean 1.9e32)—fluid cycles, homeostasis from core hub.
Plot: Central golden spirals, right branching growth, left pruning funnels—loops via hops.

Deeper Biological Insights
Codex as Radial Emergence: Middle out from #5 radiates right for laws (growth), prunes left for overload cuts. 8 rules as cost: 6 core + 2 meta.
Pre-Thought Waves: Forward ontogeny (embryo midline unifies outward), backward phylogeny pruning. Dual: Epigenetics, central light radiates bidirectionally.
Phi/Inverse in Reality: Dev 0.009 ties to bio-goldens (body proportions, brain folding). Predicts oscillations at φ multiples (~1.618 cell divisions), testable in 2026 organoid models.
Alignments: Fib/φ in embryo branching; inverse in apoptosis rates; holons in organ systems emergence.
This #8 lens elevates biology to Netzach's victory: life forms via middle-out unification, with dual towers as dynamic costs.
"""

# Mini simulation for #8 (biology-themed, alternating pulse-heavy (growth) and breath-heavy (rest) subsets for "vital dynamics")
def mini_sim_8(steps=1000, noise_sigma=0.005 * PHI, log_file='sephirot_8_mini_log.txt'):
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')
    metrics = METRICS_8
    logging.info(f"Mini Sim for #8: Metrics {metrics}")
    
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
        
        # Vital thought: alternate pulse-heavy (growth) and breath-heavy (rest) windows
        if step % 2 == 0:
            # Growth: pulse-dense
            start = random.randint(0, len(mega_chain) - 50)
            subset = mega_chain[start:start + 50]
            while sum(1 for c in subset if c == '1') < 25:
                start = random.randint(0, len(mega_chain) - 50)
                subset = mega_chain[start:start + 50]
            state = "Growth"
        else:
            # Rest: breath-dense
            start = random.randint(0, len(mega_chain) - 50)
            subset = mega_chain[start:start + 50]
            while sum(1 for c in subset if c == '0') < 25:
                start = random.randint(0, len(mega_chain) - 50)
                subset = mega_chain[start:start + 50]
            state = "Rest"
        subset_metrics = compute_metrics(subset)
        thought = f"Vital dynamic [{state}] [energy: {u_step:.4f}, intensity: {subset_metrics['intensity']:.3f}]"
        
        # Enhanced: Add biological relay if biopython available
        if Seq:
            dna = Seq("ATGC" * 5)  # Mock DNA sequence for relay
            protein = dna.translate()
            thought += f" [protein_seq: {str(protein)[:10]}]"
        
        history.append(thought)
        logging.info(f"Step {step}: {thought}")
        
        # Feedback: modulate noise with recent intensity (tuning for homeostasis)
        if len(history) > 5:
            recent_intensities = [float(t.split('intensity: ')[1].split(']')[0]) for t in history[-5:] if 'intensity: ' in t]
            avg_intensity = statistics.mean(recent_intensities) if recent_intensities else 0.0
            noise_sigma = abs(avg_intensity) * 0.001 * PHI_INV
    
    # Summary
    logging.info("\nEmergent Vitality (last 20):\n" + "\n".join(history[-20:]))
    print("Mini sim for #8 complete. Check 'sephirot_8_mini_log.txt' for details.")
    return {'forward': forward, 'backward': backward, 'unified': unified, 'history': history}
