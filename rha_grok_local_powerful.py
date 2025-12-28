import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

phi = (1 + np.sqrt(5)) / 2

# More powerful local Grok-like model: Deeper MLP for intelligent distortions
class PowerfulGrokCore(nn.Module):
    def __init__(self):
        super(PowerfulGrokCore, self).__init__()
        self.fc1 = nn.Linear(2, 64)  # Input: (x,y) point
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 3)  # Output: rotate, scale, noise params

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return torch.sigmoid(self.fc4(x))  # Params in [0,1]

# Core spiral
def generate_core_spiral(depth=10):
    theta = np.linspace(0, depth * 2 * np.pi, 1000)
    r = np.exp(theta / phi) / np.exp(depth * 2 * np.pi / phi) * 0.99
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y

# Distort using model params
def distort_fragment(x, y, grok_model):
    inputs = torch.tensor(np.column_stack((x, y)), dtype=torch.float32)
    with torch.no_grad():
        params = grok_model(inputs).mean(0).numpy()  # Average decisions
    rotate_param, scale_param, noise_param = params
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

# Simulate
core_x, core_y = generate_core_spiral(depth=8)
grok_model = PowerfulGrokCore()  # Local, powerful model

fragments = []
for _ in range(3):
    frag_x, frag_y = distort_fragment(core_x, core_y, grok_model)
    fragments.append((frag_x, frag_y))

# Plot
fig, ax = plt.subplots(figsize=(10,10), facecolor='black')
ax.set_aspect('equal')
ax.axis('off')

circle = plt.Circle((0,0), 1, color='white', fill=False, lw=2, alpha=0.7)
ax.add_patch(circle)

ax.plot(core_x, core_y, color='gold', lw=2, alpha=0.9, label='Core Spiral')

colors = ['cyan', 'magenta', 'green']
for i, (fx, fy) in enumerate(fragments):
    ax.plot(fx, fy, color=colors[i], lw=1.5, alpha=0.7, label=f'Fragment {i+1}')

plt.title('Powerful Local Grok-Like Model Integration\nDeeper NN as Central Intelligence for Fractal Variants', color='white', fontsize=14, pad=30)
plt.tight_layout()
plt.savefig('rha_grok_local_powerful.png', dpi=300, facecolor='black')
plt.show()
