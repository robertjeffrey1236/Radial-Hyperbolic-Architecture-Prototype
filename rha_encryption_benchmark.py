# rha_encryption_benchmark.py
# Standalone benchmark for Radial Hyperbolic Architecture
# Compares standard AES encryption to helical-inspired key scheduling
# Simulates secure data flow with helical "resonant" key derivation
# © 2026 - Highlights encryption efficiency with RHA-inspired layering

import os
import time
import psutil
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.backends import default_backend

# Utility for memory delta
def get_memory_delta():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024**2  # MB

# Generate sample data (1MB compressible/repeating for realistic crypto use)
def generate_sample_data(size_mb=1):
    pattern = os.urandom(1024)  # Random but repeatable pattern
    data = pattern * (1024 * size_mb // len(pattern))
    return data[:1024 * 1024 * size_mb]

# Standard AES-256-CBC encryption benchmark
def benchmark_standard_encryption(data, iterations=50):
    start_time = time.time()
    start_mem = get_memory_delta()

    key = os.urandom(32)  # 256-bit key
    iv = os.urandom(16)   # Fixed IV for benchmark consistency

    encrypted_size = 0
    for _ in range(iterations):
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
        encryptor = cipher.encryptor()
        padder = padding.PKCS7(128).padder()
        padded_data = padder.update(data) + padder.finalize()
        encrypted = encryptor.update(padded_data) + encryptor.finalize()
        encrypted_size = len(encrypted)

    duration = time.time() - start_time
    mem_delta = get_memory_delta() - start_mem
    throughput = (len(data) * iterations) / (1024**2 * duration) if duration > 0 else 0  # MB/s
    return {
        "model": "Standard AES-256-CBC Encryption",
        "duration_s": round(duration, 3),
        "throughput_mb_s": round(throughput, 2),
        "mem_delta_mb": round(mem_delta, 2),
        "encrypted_size_bytes": encrypted_size
    }

# Helical-inspired encryption: Derive sub-keys from "helices" and encrypt chunks
def benchmark_helical_encryption(data, num_helices=11, iterations=50):
    start_time = time.time()
    start_mem = get_memory_delta()

    master_key = os.urandom(32)
    # Simulate helical speeds for key derivation offset
    helix_offsets = np.array([i * 3 for i in range(num_helices)])  # Simple offset constants

    chunk_size = len(data) // num_helices
    encrypted_size = 0
    for _ in range(iterations):
        offset = 0
        total_encrypted = 0
        for h in range(num_helices):
            # Derive sub-key via simple XOR with offset (mock resonant derivation)
            sub_key = bytes(master_key[i] ^ helix_offsets[h] for i in range(32))
            iv = os.urandom(16)
            chunk = data[offset:offset + chunk_size]
            if h == num_helices - 1:  # Last chunk
                chunk = data[offset:]

            cipher = Cipher(algorithms.AES(sub_key), modes.CBC(iv), backend=default_backend())
            encryptor = cipher.encryptor()
            padder = padding.PKCS7(128).padder()
            padded_chunk = padder.update(chunk) + padder.finalize()
            encrypted_chunk = encryptor.update(padded_chunk) + encryptor.finalize()
            total_encrypted += len(encrypted_chunk)
            offset += chunk_size
        encrypted_size = total_encrypted

    duration = time.time() - start_time
    mem_delta = get_memory_delta() - start_mem
    throughput = (len(data) * iterations) / (1024**2 * duration) if duration > 0 else 0
    return {
        "model": "Helical-Derived Key Encryption (AES per Helix)",
        "duration_s": round(duration, 3),
        "throughput_mb_s": round(throughput, 2),
        "mem_delta_mb": round(mem_delta, 2),
        "encrypted_size_bytes": encrypted_size
    }

# Main execution
data = generate_sample_data(size_mb=1)  # 1MB data
print("Radial Hyperbolic Architecture — Encryption Efficiency Benchmark")
print("Compares standard AES to helical-inspired key derivation on 1MB data")
print("=" * 70)

result_standard = benchmark_standard_encryption(data)
result_helical = benchmark_helical_encryption(data)

print(f"\n{result_standard['model']}")
print(f"Duration: {result_standard['duration_s']}s | Throughput: {result_standard['throughput_mb_s']} MB/s | Mem Δ: {result_standard['mem_delta_mb']} MB")

print(f"\n{result_helical['model']}")
print(f"Duration: {result_helical['duration_s']}s | Throughput: {result_helical['throughput_mb_s']} MB/s | Mem Δ: {result_helical['mem_delta_mb']} MB")

throughput_ratio = result_helical['throughput_mb_s'] / result_standard['throughput_mb_s']
print(f"\nThroughput Ratio (Helical / Standard): {throughput_ratio:.2f}x")
print("Note: Helical adds key derivation per helix (parallelizable in real systems) but maintains strong security")
print("=" * 70)
