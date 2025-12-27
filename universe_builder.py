class MinimalUniverse:
    def __init__(self, dim=37):
        self.dim = dim
        self.points = []
        origin = torch.zeros(self.dim)
        origin[0] = 1.0
        self.points.append(origin)

    def golden_direction(self, scale=1.0):
        direction = torch.zeros(self.dim)
        angle = math.radians(137.50776405003785 * len(self.points))
        direction[0] = math.cos(angle)
        direction[1] = math.sin(angle)
        if self.dim > 2:
            decay = (1 / PHI) ** torch.arange(2, self.dim)
            perturbation = torch.randn(self.dim - 2) * decay
            direction[2:] = perturbation
        return scale * direction / torch.norm(direction)

    def ingest_packet(self, binary_str: str, growth_steps_per_burst=10):
        bits = [int(b) for b in binary_str if b in '01']
        # Extract burst lengths
        bursts = []
        current = 0
        in_burst = False
        for b in bits:
            if b == 1:
                current += 1
                in_burst = True
            else:
                if in_burst:
                    bursts.append(current)
                    in_burst = False
                    current = 0
        if in_burst:
            bursts.append(current)
        print(f"Detected {len(bursts)} bursts: {bursts}")

        # Grow from each burst
        for length in bursts:
            scale = (length / 50.0) * 0.618  # Tune for density
            for _ in range(growth_steps_per_burst):
                direction = self.golden_direction(scale=scale)
                parent = self.points[-1]  # Sequential growth (or randomize parent)
                new_pt = mobius_add(parent.unsqueeze(0), direction.unsqueeze(0), c).squeeze(0)
                new_pt = project_to_ball(new_pt.unsqueeze(0), c).squeeze(0)
                self.points.append(new_pt)
        print(f"Universe grown to {len(self.points)} points")
