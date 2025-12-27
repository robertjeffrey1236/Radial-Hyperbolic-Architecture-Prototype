# Radial-Hyperbolic-Architecture-Prototype

Python prototype for efficient hierarchical computing in the Poincaré disk. Features dual Φ (golden ratio) spirals, fractal recursion, lazy loading, multi-point observers with portals, Φ percolation for Penrose shadows, atomic scaling with wormholes/phonons, and music substrate via Just Intonation (JI) harmonics. Simulates emergent patterns on bounded resources.

## Overview

This prototype explores a novel computing architecture inspired by hyperbolic geometry, natural patterns (golden ratio spirals, honeycombs), and physical/quantum analogies:

- **Hyperbolic embedding** in the Poincaré disk for exponentially growing hierarchical data without boundary issues.
- **Dual golden spirals** (expansive and contractive) combined with a honeycomb lattice for atomic-scale resolution.
- **Fractal recursion** and **lazy loading** for efficient resource use on bounded hardware.
- **Multi-point observers** with "portals" (overlapping views) enabling distributed/shared computation.
- **Φ-percolation** to create emergent Penrose-like quasiperiodic shadows.
- **Wormholes** as random shortcuts (quantum tunneling analog).
- **Phonon vibrations** simulated via small perturbations.
- **Zeckendorf encoding** (Fibonacci-based) for compact hierarchical addressing.
- Future extensions planned for Just Intonation music substrate.

The code generates a visual representation of the structure using NetworkX and Matplotlib.


# HyperbolicPhiNet1.0.py
This version introduces efficiency fixes:
Memory Optimization: Replaces the full distance matrix computation (torch.cdist) in graph building with SciPy's KDTree for nearest-neighbor queries, reducing complexity from O(N²) to O(N log N) and memory usage dramatically (e.g., from GBs to MBs for N=2500+ points).
Reduced Defaults: Lowers parameters like n_points=50 for spirals, n=5 for honeycomb, depth=1 for recursion to enable quick testing; scalable up for production.
Headless Plotting: Uses matplotlib.use('Agg') to save visualizations to file ('rha_atomic.png') without requiring a display, ideal for servers or non-interactive runs.
Other Tweaks: Clamps modulations (e.g., in extensions like bit_mod ≤ 2.0) to prevent amplitude explosions; removes unused imports (e.g., mido, pyscf if not called) to streamline.
The result is a more robust simulator for exploring emergent dynamics in hierarchical spaces, suitable for AI data embedding, generative art, or quantum-inspired computing. When run with reduced params, it generates ~200-800 points, builds a percolated graph, visualizes components/observers/portals, and demos encodings—completing in seconds on CPU.
Core Components and Functionality
The script is modular, starting with setup and hyperbolic ops, then point generation, graph construction, percolation, encodings, and visualization/stats. Here's an extended breakdown:
Imports and Device Setup:
Libraries: Torch for tensors/hyperbolic math; Matplotlib/NetworkX for plotting/graphs; NumPy/SciPy for numerics/spatial tools (ConvexHull, KDTree); cmath for complex phases; mido/PySCF optional for MIDI/quantum chem.
Device: Prefers CPU (or GPU if available) for offloading; prints e.g., "Using device: cpu".
Hyperbolic Geometry Functions (Poincaré Ball Model):
Curvature c=1.0 (negative for hyperbolic expansion).
mobius_add: Hyperbolic vector addition via Möbius gyrovectors, clamping for stability.
expmap0: Projects Euclidean vectors to hyperbolic space using tanh.
dist/dist0: Computes hyperbolic distances with atanh, clamping to avoid singularities.
These enable efficient hierarchical embeddings, where distance grows exponentially—key for scaling large datasets without distortion.
Point Generation and Branching:
Golden spirals (golden_spiral_points): Logarithmic spirals in dual directions (expansive/contractive), scaled by φ for natural fractal patterns; reduced to n_points=50 for testing.
Honeycomb lattice (generate_honeycomb_points): Atomic-scale grid normalized to unit disk; n=5 yields ~100 points.
Concatenates spirals + honeycomb into all_points.
recursive_branch: Fractal expansion with φ-scaled offsets (branching factor 3); depth=1 limits to ~3x points to avoid explosion.
Phonon vibrations: Adds Gaussian noise (0.01 std) for atomic realism, remaps to hyperbolic space.
Lazy Loading and Multi-Observer Setup:
lazy_load: Filters points visible to observers based on hyperbolic dist <7.0, for efficiency.
Observers: Central (origin) and offset; unions visible sets, removing duplicates with torch.unique.
Simulates distributed views, enabling "portal" overlaps for shared computation.
Graph Construction and Percolation (Key Fix Here):
build_graph: Adds nodes with positions; uses KDTree to query top-k (50) neighbors within max_edge_dist=0.5, adding edges only if close and unique. This fixes the original O(N²) bottleneck—e.g., for N=839, it builds quickly without memory overflow.
Adds random "wormholes" (10% of nodes) as quantum shortcuts, typed for visualization.
phi_percolate: Probabilistically removes edges based on φ-decay with radius (p = 0.7 * φ^{-r_avg}), creating sparse, scale-free networks mimicking natural percolation.
Encodings and Compression:
Zeckendorf (zeckendorf_encode/decode): Fibonacci-based unique binary (no adjacent 1s) for node addresses—assigns to all nodes.
RLE (rle_encode): Compresses bit sequences, demoed on sample addresses.
Polar coords (to_polar/from_polar): Trigonometric compression for points, reducing dims while preserving structure.
Base-φ (to_base_phi): Experimental encoding with φ-multiplication for compact reps of floats.
Visualization and Portal Detection:
Plots components with rainbow colors, gray local edges, red dashed wormholes.
Stars for observers; detects overlaps (dist <5.0), logs portals, "sends" Zeckendorf-encoded packets (e.g., node count).
Approximates vesica piscis portals with overlapping circles; saves to 'rha_atomic.png'.
Includes unit disk boundary for hyperbolic context.
Statistics and Demos:
Computes components, largest cluster, coverage radius via ConvexHull (handles failures gracefully).
Prints sample encodings, RLE, polar shapes, base-φ example.
Example Run Outputs (from Test Execution)
When executed with the reduced params:
Portal detection: "Portal open between observer 0 and 1, dist: 0.7211102843284607"
Packet demo: "Sent packet from 0 to 1: 10010101010001 (decodes to 839)"
Graph stats: "Number of components: 1", "Largest cluster size: 839", "Approximate coverage radius: 0.757361888885498"
Sample addresses: Node 0: '0', Node 1: '1', Node 2: '10', etc.
RLE: On '0': [('0', 1)]
Polar demo: Original shape torch.Size([839, 2]), polar torch.Size([10, 2])
Base-φ for 0.5: [0, 1, 0, 0, 1, 0, 0, 1, 0, 0]
Potential Extensions and Use Cases
Wave Dynamics/Sonification: If re-added (from original), modulate transfers with Zeckendorf bits for "riding" waves, tying to resonance/harmonics.
Quantum Integration: Load molecules (e.g., benzene) via PySCF for atomic energies, projecting coords hyperbolically.
AI Applications: Embed data subsets hierarchically for fast retrieval, as discussed earlier.
Scalability: Increase params on GPU for larger sims (e.g., N=10k+); KDTree scales well.


## Requirements

- Python 3.7+
- Required libraries:
  - torch (PyTorch)
  - matplotlib
  - networkx
  - numpy
  - scipy

Install dependencies with:

```bash
pip install torch matplotlib networkx numpy scipy
(No GPU required – runs on CPU.)
How to Run
Clone the repository:
git clone https://github.com/robertjeffrey1236/Radial-Hyperbolic-Architecture-Prototype.git
cd Radial-Hyperbolic-Architecture-Prototype
Run the prototype:
python FullRHPrototypCodeV.2.py
A Matplotlib window will open showing the visualized structure:
Nodes colored by connected components (Penrose-like shadows)
Gray edges: local connections
Red dashed edges: wormholes
Red stars: observers
Blue dashed circles: active portals between observers
(Optional) Save the figure instead of displaying:
Uncomment the last line in the script: plt.savefig('rha_visualization.png')
Console output will show portal detections and example encoded packets sent through portals.
Key Concepts Explained
Poincaré Disk: Hyperbolic space model where distance grows exponentially toward the boundary – perfect for trees/hierarchies.
Dual Φ Spirals: Two counter-rotating golden spirals provide optimal packing and self-similarity.
Lazy Loading: Only resolve geometry near active observers – simulates energy-efficient computation.
Portals: When observers' views overlap in hyperbolic distance, a "portal" opens for direct data transfer (Zeckendorf-encoded packets).
Wormholes/Phonons: Random shortcuts + vibrations for quantum-inspired dynamics.
Φ Percolation: Edges kept with probability decaying by golden ratio – creates quasiperiodic patterns reminiscent of Penrose tilings.
Future Ideas
Integrate Just Intonation harmonics (music substrate) using frequencies derived from node distances.
Real-time interaction (e.g., move observers with mouse).
Export to 3D or interactive web visualization (Plotly/Three.js).
Benchmark against traditional tree/graph structures.
License
Apache-2.0
Feel free to tweak parameters in the code (e.g., recursion depth, number of points, percolation probability) to explore different emergent patterns!

