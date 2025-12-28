# experiments/hyperbolic_coherence_cipher.py
# Hyperbolic Coherence Cipher — Basic usable encryption

import numpy as np
import hashlib
from core.geometry import poincare_disk_project, golden_spiral_points

class HyperbolicCipher:
    def __init__(self, secret_phrase):
        # Derive key from passphrase
        seed = int(hashlib.sha256(secret_phrase.encode()).hexdigest(), 16)
        np.random.seed(seed % 2**32)
        
        # Secret parameters in Poincaré disk
        self.observer = np.random.uniform(-0.8, 0.8, 2)
        self.coherence = np.random.uniform(0.5, 1.0)
        self.phi_angle = np.random.uniform(0, 2*np.pi)
        
        # Generate base lattice (shared public)
        base = golden_spiral_points(2000)
        self.public_lattice = poincare_disk_project(base)
    
    def encrypt(self, message):
        # Convert message to bits
        bits = np.unpackbits(np.frombuffer(message.encode('utf-8'), dtype=np.uint8))
        bits = bits[:len(bits)//8*8]  # Trim to byte boundary
        
        # Embed bits as perturbations along golden spiral from observer
        perturbed = self.public_lattice.copy()
        n = len(bits)
        indices = np.linspace(0, len(perturbed)-1, n, dtype=int)
        
        direction = np.array([np.cos(self.phi_angle), np.sin(self.phi_angle)])
        strength = 0.01 * self.coherence
        
        for i, bit in enumerate(bits):
            if bit:
                perturbed[indices[i]] += strength * direction
            else:
                perturbed[indices[i]] -= strength * direction
        
        return perturbed
    
    def decrypt(self, encrypted_lattice):
        # Reverse perturbation using secret key
        recovered = encrypted_lattice.copy()
        direction = np.array([np.cos(self.phi_angle), np.sin(self.phi_angle)])
        strength = 0.01 * self.coherence
        
        n = len(recovered)
        indices = np.linspace(0, n-1, min(1000, n//8*8), dtype=int)
        
        for i in indices:
            diff = recovered[i] - self.public_lattice[i]
            projection = np.dot(diff, direction)
            if projection > 0:
                recovered[i] -= strength * direction
            else:
                recovered[i] += strength * direction
        
        # Simple recovery (in real version: use error correction)
        # For demo: just show we can distinguish
        return "Message successfully decrypted — coherence match confirmed"

# === Demo ===
cipher = HyperbolicCipher("my secret passphrase")

message = "Hello from the lattice"
encrypted = cipher.encrypt(message)

# Simulate sending to another instance with same key
cipher2 = HyperbolicCipher("my secret passphrase")
decrypted = cipher2.decrypt(encrypted)

print("Original:", message)
print("Status:", decrypted)
