import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

phi = (1 + np.sqrt(5)) / 2

# Build recursive graph: nodes with golden-ratio radial positions
G = nx.Graph()
max_levels = 7
node_id = 0
positions = {}

# Central node
G.add_node(0)
positions[0] = (0, 0)
node_id += 1

for level in range(1, max_levels + 1):
    r_hyper = 0.99 * np.tanh((phi ** level) / 2)
    num = max(6, int(12 * (phi ** (level - 1))))
    theta = np.linspace(0, 2*np.pi, num) + level * 0.3
    level_nodes = []
    for i in range(num):
        nid = node_id
        x = r_hyper * np.cos(theta[i])
        y = r_hyper * np.sin(theta[i])
        positions[nid] = (x, y)
        G.add_node(nid)
        # Connect to previous level (foam evolution)
        prev_num = len(G.nodes) - num if level > 1 else 1
        prev_start = node_id - prev_num - num if level > 1 else 0
        connect_to = prev_start + (i * prev_num // num)
        G.add_edge(nid, connect_to)
        level_nodes.append(nid)
        node_id += 1

# Poincaré positions
pos_poincare = {n: positions[n] for n in G.nodes()}

# Plot
fig, ax = plt.subplots(figsize=(10,10), facecolor='black')
ax.set_aspect('equal')
ax.axis('off')

# Disk
circle = plt.Circle((0,0), 1, color='white', fill=False, lw=2, alpha=0.7)
ax.add_patch(circle)

# Edges: cyan foam surfaces
nx.draw_networkx_edges(G, pos_poincare, edge_color='cyan', width=1.5, alpha=0.6)

# Nodes: gold vertices, sized by degree (spin-like)
degrees = [G.degree(n) for n in G.nodes()]
nx.draw_networkx_nodes(G, pos_poincare, node_color='gold', node_size=[50 + 200 * d for d in degrees],
                       edgecolors='white', linewidths=0.5, alpha=0.9)

plt.title('Spin Foam Toy Model\nEvolving Quantum Spacetime via Radial Hyperbolic Recursion + Golden-Ratio Layers', 
          color='white', fontsize=14, pad=30)
plt.tight_layout()
plt.savefig('rha_spin_foam.png', dpi=300, facecolor='black')
plt.show()
