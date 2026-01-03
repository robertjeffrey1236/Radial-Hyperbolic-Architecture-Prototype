# rha_resonance_alignment_benchmark.py
# Standalone benchmark for Radial Hyperbolic Architecture
# Measures efficiency of frequency alignments in helical torus model
# Simulates chord convergences with full-precision 24-TET and fluid damping
# © 2026 - Highlights resonant "wild magic" with low computational cost

import numpy as np
import scipy.signal as signal  # For simple peak detection in alignments
import time
import psutil
import os

# Utility for memory delta
def get_memory_delta():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024**2  # MB

# Precompute full-precision 24-TET frequencies (A3=220 Hz base)
def generate_24tet_frequencies():
    return np.array([220 * (2 ** (n / 24)) for n in range(24)])

# Benchmark naive 2D wheel frequency alignment (loop-based)
def benchmark_2d_resonance(num_codes=11, num_ticks=20000, num_freqs=24):
    start_time = time.time()
    start_mem = get_memory_delta()

    freqs = generate_24tet_frequencies()
    total_ops = 0
    alignments = 0
    for tick in range(num_ticks):
        for code_id in range(num_codes):
            code_length = 100 + code_id * 50
            # Simulate ring positions and freq assignments
            positions = np.linspace(0, 2 * np.pi, num_freqs)
            active_freqs = freqs[(np.arange(num_freqs) + tick) % code_length < code_length // 2]
            # Mock alignment detection (peaks in "chord" sum)
            if len(active_freqs) > 0:
                chord_sum = np.sum(active_freqs)
                peaks = signal.find_peaks(np.array([chord_sum]))[0]
                alignments += len(peaks)
            total_ops += num_freqs

    duration = time.time() - start_time
    mem_delta = get_memory_delta() - start_mem
    ops_per_sec = total_ops / duration if duration > 0 else 0
    return {
        "model": "2D Wheel Resonance Alignment",
        "duration_s": round(duration, 3),
        "ops_per_sec": f"{ops_per_sec:,.0f}",
        "mem_delta_mb": round(mem_delta, 2),
        "total_alignments": alignments
    }

# Benchmark optimized helical torus resonance (vectorized + damping)
def benchmark_helical_resonance(num_codes=11, num_ticks=20000, num_freqs=24, damping_const=0.01):
    start_time = time.time()
    start_mem = get_memory_delta()

    freqs = generate_24tet_frequencies()
    code_lengths = np.array([100 + i * 50 for i in range(num_codes)])
    speeds = 2 * np.pi / code_lengths  # Differential helical speeds

    total_ops = 0
    alignments = 0
    for tick in range(num_ticks):
        # Vectorized phase with damping
        phi = speeds[:, np.newaxis] * np.arange(num_freqs) + tick * 0.1
        phi *= (1 - damping_const)  # Fluid damping constant

        # Simulate active freqs per helix (vectorized mask)
        active_mask = np.mod(np.arange(num_freqs) + tick, code_lengths[:, np.newaxis]) < code_lengths[:, np.newaxis] // 2
        active_freqs = freqs * active_mask  # Masked freq array

        # Alignment detection: peaks in summed "chords" across helices
        chord_sums = np.sum(active_freqs, axis=1)
        for sum_val in chord_sums:
            peaks = signal.find
