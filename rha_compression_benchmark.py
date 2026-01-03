# rha_compression_benchmark.py
# Standalone benchmark for Radial Hyperbolic Architecture
# Compares standard zlib compression to helical-inspired slicing
# Simulates data transfer/compression efficiency with helical "flows"
# © 2026 - Tests overhead of helical chunking on compressible data

import zlib
import numpy as np
import time
import psutil
import os

# Utility to get memory usage
def get_memory_delta():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024**2  # MB

# Generate sample compressible data (1MB repeating pattern)
def generate_sample_data(size_mb=1):
    pattern = b'ABC' * 1024  # Compressible repeating bytes
    data = pattern * (1024 * size_mb // len(pattern))
    return data[:1024 * 1024 * size_mb]  # Exact size

# Standard zlib compression benchmark
def benchmark_standard_compression(data, iterations=100):
    start_time = time.time()
    start_mem = get_memory_delta()

    compressed_size = 0
    for _ in range(iterations):
        compressed = zlib.compress(data)
        compressed_size = len(compressed)

    duration = time.time() - start_time
    mem_delta = get_memory_delta() - start_mem
    ratio = len(data) / compressed_size if compressed_size > 0 else 0
    ops_per_sec = (len(data) * iterations) / duration if duration > 0 else 0
    return {
        "model": "Standard zlib Compression",
        "duration_s": round(duration, 3),
        "ops_per_sec": f"{ops_per_sec:,.0f}",
        "mem_delta_mb": round(mem_delta, 2),
        "compression_ratio": round(ratio, 2),
        "compressed_size_bytes": compressed_size
    }

# Helical-inspired compression: Slice data into "helices" of varying lengths, compress each
def benchmark_helical_compression(data, num_helices=11, iterations=100):
    start_time = time.time()
    start_mem = get_memory_delta()

    # Precompute helix lengths (varying like code lengths)
    helix_lengths = [len(data) // num_helices + i * 50 for i in range(num_helices)]
    helix_lengths[-1] += len(data) - sum(helix_lengths)  # Adjust last for exact fit

    compressed_size = 0
    for _ in range(iterations):
        offset = 0
        total_compressed = 0
        for length in helix_lengths:
            chunk = data[offset:offset + length]
            compressed = zlib.compress(chunk)
            total_compressed += len(compressed)
            offset += length
        compressed_size = total_compressed

    duration = time.time() - start_time
    mem_delta = get_memory_delta() - start_mem
    ratio = len(data) / compressed_size if compressed_size > 0 else 0
    ops_per_sec = (len(data) * iterations) / duration if duration > 0 else 0
    return {
        "model": "Helical Slicing + zlib Compression",
        "duration_s": round(duration, 3),
        "ops_per_sec": f"{ops_per_sec:,.0f}",
        "mem_delta_mb": round(mem_delta, 2),
        "compression_ratio": round(ratio, 2),
        "compressed_size_bytes": compressed_size
    }

# Main execution
data = generate_sample_data(size_mb=1)  # 1MB sample
print("Radial Hyperbolic Architecture — Compression Efficiency Benchmark")
print("Compares standard zlib to helical-inspired slicing on 1MB data")
print("=" * 70)

result_standard = benchmark_standard_compression(data)
result_helical = benchmark_helical_compression(data)

print(f"\n{result_standard['model']}")
print(f"Duration: {result_standard['duration_s']}s | Ops/sec: {result_standard['ops_per_sec']} | Mem Δ: {result_standard['mem_delta_mb']} MB")
print(f"Compression Ratio: {result_standard['compression_ratio']}x | Compressed Size: {result_standard['compressed_size_bytes']} bytes")

print(f"\n{result_helical['model']}")
print(f"Duration: {result_helical['duration_s']}s | Ops/sec: {result_helical['ops_per_sec']} | Mem Δ: {result_helical['mem_delta_mb']} MB")
print(f"Compression Ratio: {result_helical['compression_ratio']}x | Compressed Size: {result_helical['compressed_size_bytes']} bytes")

ratio_vs_standard = result_helical['compression_ratio'] / result_standard['compression_ratio']
print(f"\nEfficiency Ratio (Helical / Standard): {ratio_vs_standard:.2f}x compression ratio")
print("Note: Helical adds slicing for parallel potential but may reduce ratio on uniform data; shines on hierarchical patterns")
print("=" * 70)
