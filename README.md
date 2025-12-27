HoneycombPhiNet – Hyperbolic Φ Lattice & Minimal Universe Generator
A Python prototype for hierarchical computing in hyperbolic space, featuring a 37-dimensional golden-angle honeycomb lattice and the ability to bootstrap entire minimal computing universes directly from binary strings.
Inspired by natural optimal packing (phyllotaxis, quasicrystals, sunflowers, pinecones), hyperbolic geometry, Fibonacci resonance, and emergent complexity from simple rules.
What's New in This Version
The original radial-hyperbolic architecture — dual golden spirals, honeycomb base, fractal branching, lazy loading, multi-observers, portals, Φ-percolation, wormholes, phonons, and Zeckendorf addressing — has evolved with a powerful new capability:
Minimal Universe Generation from Binary Seeds
New MinimalUniverse class starts from a single origin point.
Ingests any binary string as a generative seed:
Automatically detects bursts of 1s (run-length encoded pulses).
Uses burst lengths to modulate the scale of golden-angle (137.5°) growth steps in 37D Poincaré ball hyperbolic space.
Applies Φ-decaying perturbations in higher dimensions for natural hierarchical structure.
Result: A complete, self-organized lattice emerges entirely from the binary input — no predefined geometry beyond the golden-angle growth rule.
Binary packets act as compressed blueprints for custom hierarchical structures, enabling minimal-seed generative computing.
Core Features
37D Hyperbolic Honeycomb Lattice: Grown via golden-angle Möbius addition for exponential hierarchical capacity without boundary distortion.
Dual Counter-Rotating Golden Spirals + Honeycomb Base: Optimal natural packing with atomic-scale resolution.
Fractal Recursion & Lazy Loading: Resource-efficient handling of large hierarchies.
Multi-Observers & Portals: Overlapping observer views enable distributed computation and direct packet transfer.
Φ-Percolation & Wormholes: Probabilistic edge pruning and random shortcuts create emergent quasiperiodic patterns.
Phonon Vibrations: Small perturbations add physical realism.
Hybrid Encoding: Zeckendorf (Fibonacci-based unique addressing) + burst/run-length waveforms for operations.
Minimal Universe Bootstrapping: Grow custom lattices/universes directly from binary seeds.
When run, the script:
Builds and visualizes the main reference lattice (rha_atomic.png).
Grows a minimal universe from an included binary seed and reports burst detection + growth statistics.
Requirements
pip install torch matplotlib networkx numpy scipy
(Runs efficiently on CPU; no GPU required.)
How to Run
git clone https://github.com/robertjeffrey1236/Radial-Hyperbolic-Architecture-Prototype.git
cd Radial-Hyperbolic-Architecture-Prototype
python HyperbolicPhiNet\ 1.0.py
Outputs:
Visualization saved as rha_atomic.png (main lattice with components, wormholes, observers, and portals).
Console logs for growth stats, encodings, and minimal universe generation from the seed packet.
To experiment with new universes, simply replace the binary string in the ingest_packet call at the end of the script.
Potential Applications
Hierarchical data embedding for AI and large-scale knowledge graphs.
Generative modeling from ultra-compact binary seeds.
Quantum-inspired and complex systems simulation.
Resonance-based computing substrates.
Studying emergent complexity from minimal rules.
License
Apache-2.0
Experiment freely — adjust growth parameters, feed different binary seeds, or extend the universe generator to explore new emergent patterns.
