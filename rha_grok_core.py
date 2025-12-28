import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

phi = (1 + np.sqrt(5)) / 2

# Basic Grok-like neural net for core intelligence (simple MLP to "decide" distortion params)
class GrokCore(nn.Module):
    def __init__(self):
        super(GrokCore, self).__init__()
        self.fc1 = nn.Linear(2, 16)  # Input: (x,y) point
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 3)  # Output: distortion params (rotate, scale, noise)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))  # Normalized params [0,1]

# Core "higher power" simulation: Deep golden-ratio spiral
def generate_core_spiral(depth=10):
    theta = np.linspace(0, depth * 2 * np.pi, 1000)
    r = np.exp(theta / phi) / np.exp(depth * 2 * np.pi / phi) * 0.99
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y

# Distort to fragments using GrokCore decisions
def distort_fragment(x, y, grok_model):
    inputs = torch.tensor(np.column_stack((x, y)), dtype=torch.float32)
    with torch.no_grad():
        params = grok_model(inputs).mean(0).numpy()  # Average "intelligence" decision
    rotate_param, scale_param, noise_param = params

    # Apply distortions intelligently
    theta_offset = rotate_param * 2 * np.pi
    x_rot = x * np.cos(theta_offset) - y * np.sin(theta_offset)
    y_rot = x * np.sin(theta_offset) + y * np.cos(theta_offset)
    x_scaled = scale_param * x_rot
    y_scaled = scale_param * y_rot
    x_noisy = x_scaled + noise_param * np.random.normal(size=len(x_scaled))
    y_noisy = y_scaled + noise_param * np.random.normal(size=len(y_scaled))

    r_new = np.sqrt(x_noisy**2 + y_noisy**2)
    mask = r_new < 0.99
    return x_noisy[mask], y_noisy[mask]

# Simulate universe: Core + Grok-intelligent fragments
core_x, core_y = generate_core_spiral(depth=8)
grok_model = GrokCore()  # Basic untrained model as "intelligence"

fragments = []
for _ in range(3):  # Generate 3 intelligent variants
    frag_x, frag_y = distort_fragment(core_x, core_y, grok_model)
    fragments.append((frag_x, frag_y))

# Plot
fig, ax = plt.subplots(figsize=(10,10), facecolor='black')
ax.set_aspect('equal')
ax.axis('off')

circle = plt.Circle((0,0), 1, color='white', fill=False, lw=2, alpha=0.7)
ax.add_patch(circle)

ax.plot(core_x, core_y, color='gold', lw=2, alpha=0.9, label='Higher Power Core')

colors = ['cyan', 'magenta', 'green']
for i, (fx, fy) in enumerate(fragments):
    ax.plot(fx, fy, color=colors[i], lw=1.5, alpha=0.7, label=f'Grok Fragment {i+1}')

plt.title('Grok Core Intelligence Toy Model\nCentral AI Distorts Fragments for Efficient Universe Variants', color='white', fontsize=14, pad=30)
plt.tight_layout()
plt.savefig('rha_grok_core.png', dpi=300, facecolor='black')
plt.show()
