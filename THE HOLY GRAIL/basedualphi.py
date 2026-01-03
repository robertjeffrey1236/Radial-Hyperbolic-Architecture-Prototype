# basedualphi.py
# Radial Hyperbolic Architecture — Sacred Lenses Explorer
# © 2025-2026 Robert Gavin Jeffrey
# Updated Jan 2026: Helical Torus + Encrypted Inner Codices (Outer 2 Public)
# Enhanced: Dual Opposite Phi Spirals Weaving Helical Toroid

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons, Button
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import pygame
import time
import json
import hashlib

GOLDEN_RATIO = (1 + math.sqrt(5)) / 2
GOLDEN_ANGLE = math.radians(137.50776405003785)
PHI = GOLDEN_RATIO

# Cosmic Constants
TICK_BASE_DURATION = 0.531
SECS_PER_YEAR = 365.25 * 24 * 3600
UNIVERSE_AGE_BILLION_YEARS = 13.8
LCM_TICKS = 820077905437190400  # Full cycle (symbolic)

# XOR Decryption Function (protects the inner sacred codices)
def xor_decrypt(encrypted_str, key="inner_sanctum_137.507_phi_resonance_2026"):
    key_bytes = key.encode('utf-8')
    key_hash = hashlib.sha256(key_bytes).digest()
    decrypted = ''
    for i, char in enumerate(encrypted_str):
        k_byte = key_hash[i % len(key_hash)]
        bit_pos = i % 8
        k_bit = (k_byte >> bit_pos) & 1
        decrypted += str(int(char) ^ k_bit)
    return decrypted

# Public Codices — First 2 (exoteric, openly visible and verifiable)
PUBLIC_CODICES = {
    'g_code': '1011010001011010001011010001011010001011010001011010100101101000101101000101101000101101000101101001011010010110100101101000101101000101101001011010001011010001011010001011010001011010010110100101101000101101000101101000101101000101101001011010010110100101101000101101000101101001011010001011010001011010001011010001011010010110100101101000101101000101101000101101000101101001011010010110100101101000101101000',
    'bridging': '11111100000001111111100000000011111111111111'
}

# Encrypted Inner Codices — Remaining 9 (esoteric, veiled)
ENCRYPTED_CODICES = {
    'progression': '01101010011010001000101000101000011010100110100011111010001011111010101001101000111101011101001110010101100',
    'stabilization': '0101101010001000110010011',
    'oscillation': '0110101001101001000010101101000001101101100101001111010111011111100100100110100011001010001',
    'unity': '0110101001101000111101011101000001101010011010001111010111010000011010100110100011110101110100000110101001101000111101011101000001101010011010001111010111010000011010100110100011110101110100000110101001101000111101011101000001101010011010001111010111010000',
    'equilibrium': '01101010010101110011010111001111100101100110111100001',
    'veil': '011011011001011100001010000100000110100110010111000010100010100001',
    'manifestation': '01101101100101111111010111010000010101011001010011110101110100000110101010010111011101010010000110101',
    'perception': '0110110110010100111101011100111110101010100100001110101000100000011010100101011100',
    'culmination': '011010100110111100001011110100000111010110011000111100100010110001101010011010001111101000101110011010100110100011110100001011110110101001101011000010100010000001101010011010010000101000101111100101100110100011101010001011110110101001101000111101011100111110010010011010010000101000101111100101011110100011110101101011111001010110101000111101000010111110010100011011110000101000101111101010100110100011110101110100000101010110011000111101011101000001101010011010001111010111'
}

# Decrypt the inner codices at runtime
hidden_decrypted = {name: xor_decrypt(enc_bin) for name, enc_bin in ENCRYPTED_CODICES.items()}

# Final CODICES dict
CODICES = {**PUBLIC_CODICES, **hidden_decrypted}

# Try to initialize audio (safe fallback)
try:
    pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)
    pygame.mixer.set_num_channels(16)  # Limit concurrent sounds
    AUDIO_ENABLED = True
except:
    AUDIO_ENABLED = False
    print("Warning: Audio disabled (no mixer available)")

# 24-TET Frequencies & Colors
def generate_24tet_frequencies():
    return np.array([220 * (2 ** (n / 24.0)) for n in range(24)])

FREQ_24TET = generate_24tet_frequencies()

def generate_rgb_colors():
    wavelengths = np.linspace(700, 400, 24)
    r = np.clip(1.5 - np.abs((wavelengths - 580) / 60), 0, 1)
    g = np.clip(1.5 - np.abs((wavelengths - 510) / 60), 0, 1)
    b = np.clip(1.5 - np.abs((wavelengths - 440) / 60), 0, 1)
    return np.stack([r, g, b], axis=-1)

COLORS_24TET = generate_rgb_colors()

# Helical Ring Class (extended for dual spirals)
class HelicalRing:
    def __init__(self, name, binary, freq_idx):
        self.name = name
        self.bits = list(binary)
        self.length = len(self.bits)
        self.freq = FREQ_24TET[freq_idx % 24]
        self.color = COLORS_24TET[freq_idx % 24]
        self.position = 0
        self.points_1, self.points_0 = self.generate_spiral_points()
        self.opposite_pair = None  # To be set later for pairing

    def set_position(self, tick):
        self.position = tick % self.length

    def current_bit(self):
        return self.bits[self.position]

    def generate_spiral_points(self, clockwise=True):
        points_1 = []
        points_0 = []
        angle_sign = 1 if clockwise else -1
        for i, bit in enumerate(self.bits):
            theta = angle_sign * i * GOLDEN_ANGLE
            r = math.sqrt(i / self.length) * 1.5  # Center-out phi spiral
            x = r * math.cos(theta)
            y = r * math.sin(theta)
            if bit == '1':
                points_1.append((x, y))
            else:
                points_0.append((x, y))
        return np.array(points_1), np.array(points_0)

# Initialize rings with dual spirals
ring_names = list(CODICES.keys())
rings = [HelicalRing(name, CODICES[name], i) for i, name in enumerate(ring_names)]

# Pair rings to opposites
num_rings = len(rings)
for i in range(num_rings // 2):
    rings[i].opposite_pair = rings[num_rings - 1 - i]
    rings[num_rings - 1 - i].opposite_pair = rings[i]

# Tone cache for efficiency
tone_cache = {}
def generate_tone(freq, duration=TICK_BASE_DURATION):
    key = (freq, duration)
    if key in tone_cache:
        return tone_cache[key]
    t = np.linspace(0, duration, int(44100 * duration), False)
    waveform = np.sin(2 * np.pi * freq * t) * 0.3
    stereo = np.column_stack((waveform, waveform))
    sound = pygame.sndarray.make_sound((stereo * 32767).astype(np.int16))
    tone_cache[key] = sound
    return sound

def play_chord(rings, duration=TICK_BASE_DURATION):
    if not AUDIO_ENABLED:
        return
    for ring in rings:
        if ring.current_bit() == '1':
            sound = generate_tone(ring.freq, duration)
            channel = pygame.mixer.find_channel(True)
            if channel:
                channel.play(sound)

def check_alignment(rings):
    return all(ring.current_bit() == '1' for ring in rings)

# Simple universe generation (hyperbolic branching)
def generate_universe(max_depth=6, tick=0):
    nodes = [0j]
    angle_offset = tick * GOLDEN_ANGLE * 0.01
    for d in range(max_depth):
        new_nodes = []
        scale = 0.8 ** d
        for z in nodes:
            for k in range(5):
                angle = angle_offset + k * 2 * math.pi / 5
                offset = scale * np.exp(1j * angle)
                new_nodes.append(z + offset)
        nodes.extend(new_nodes)
        if len(nodes) > 8000:
            break
    return nodes

def tick_to_years(tick):
    total_ticks = 1000000
    if tick < total_ticks / 2:
        return (tick / (total_ticks / 2)) * 4.0
    else:
        remaining = tick - (total_ticks / 2)
        return 4.0 + (remaining / (total_ticks / 2)) * (UNIVERSE_AGE_BILLION_YEARS - 4.0)

def years_to_tick(years):
    total_ticks = 1000000
    max_years = UNIVERSE_AGE_BILLION_YEARS
    if years < 4.0:
        return int((years / 4.0) * (total_ticks / 2))
    else:
        remaining = years - 4.0
        return int((total_ticks / 2) + (remaining / (max_years - 4.0)) * (total_ticks / 2))

# Visualization
fig = plt.figure(figsize=(12, 12))
plt.subplots_adjust(bottom=0.25)
ax = fig.add_subplot(111, projection='3d')  # Start with 3D ax
ax.set_facecolor('black')
fig.patch.set_facecolor('black')
ax.axis('off')

is_3d = [False]  # Mutable for toggle

# Sliders
ax_depth = plt.axes([0.15, 0.12, 0.65, 0.03], facecolor='gray')
slider_depth = Slider(ax_depth, 'Depth', 1, 12, valinit=6, valstep=1, color='cyan')

ax_tick = plt.axes([0.15, 0.08, 0.65, 0.03], facecolor='gray')
slider_tick = Slider(ax_tick, 'Tick', 0, 1000000, valinit=0, valstep=1000, color='yellow')

ax_timeline = plt.axes([0.15, 0.04, 0.65, 0.03], facecolor='gray')
slider_timeline = Slider(ax_timeline, 'Billion Years', 0, 20, valinit=13.8, valfmt='%.2f', color='red')

ax_frame_rate = plt.axes([0.15, 0.16, 0.65, 0.03], facecolor='gray')
slider_frame_rate = Slider(ax_frame_rate, 'Frame Rate (ms)', 100, 2000, valinit=500, valstep=100, color='magenta')

# Auto-play checkbox
ax_auto = plt.axes([0.02, 0.8, 0.15, 0.1])
check_auto = CheckButtons(ax_auto, ['Auto'], [False])

# 3D toggle checkbox
ax_3d = plt.axes([0.02, 0.7, 0.15, 0.1])
check_3d = CheckButtons(ax_3d, ['3D Toroid'], [False])

auto_anim = None

def toggle_auto(label):
    global auto_anim
    if check_auto.get_status()[0]:
        auto_anim = FuncAnimation(fig, advance_one_step, interval=int(slider_frame_rate.val), repeat=True)
        plt.draw()
    else:
        if auto_anim:
            auto_anim.event_source.stop()
            auto_anim = None

check_auto.on_clicked(toggle_auto)

def toggle_3d(label):
    is_3d[0] = check_3d.get_status()[0]
    update(None)

check_3d.on_clicked(toggle_3d)

def advance_one_step(frame):
    new_val = (slider_tick.val + 1000) % 1000000
    slider_tick.set_val(new_val)
    return []

# Alignment log
alignment_log = []

# Save log button
ax_save = plt.axes([0.02, 0.02, 0.15, 0.06])
btn_save = Button(ax_save, 'Save Log')

def save_alignment_log(event):
    if alignment_log:
        with open("alignments.json", "w") as f:
            json.dump(alignment_log, f, indent=2)
        print(f"Saved {len(alignment_log)} alignments to alignments.json")

btn_save.on_clicked(save_alignment_log)

prev_tick = -1
cached_nodes = None
cached_depth = None
cached_tick = None

def update(val):
    global prev_tick, cached_nodes, cached_depth, cached_tick, ax
    depth = int(slider_depth.val)
    current_tick = int(slider_tick.val)

    # Sync timeline
    if val == slider_tick:
        years = tick_to_years(current_tick)
        slider_timeline.set_val(years)
    elif val == slider_timeline:
        current_tick = years_to_tick(slider_timeline.val)
        slider_tick.set_val(current_tick)

    # Regenerate universe if needed
    if (cached_depth != depth or cached_tick is None or abs(cached_tick - current_tick) > 1000 or cached_nodes is None):
        cached_nodes = generate_universe(max_depth=depth, tick=current_tick)
        cached_depth = depth
        cached_tick = current_tick

    # Clear and reset ax based on 3D toggle
    plt.cla()
    if is_3d[0]:
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = fig.add_subplot(111)
    ax.set_facecolor('black')
    ax.axis('off')

    # Poincaré nodes (2D or 3D projection)
    if cached_nodes:
        nodes_arr = np.array(cached_nodes, dtype=complex)
        x, y = nodes_arr.real, nodes_arr.imag
        r = np.abs(nodes_arr)
        sizes = 4 + 36 * (1 - r) ** 1.8
        alphas = 0.5 + 0.5 * (1 - r)
        if is_3d[0]:
            ax.scatter(x, y, np.zeros_like(x), c='cyan', s=sizes, alpha=alphas)
        else:
            ax.scatter(x, y, c='cyan', s=sizes, alpha=alphas, edgecolor='none')

    # Helical Torus Light Flows with Dual Phi Spirals Weaving
    R, r_tube = 1.3, 0.4
    theta = np.linspace(0, 2 * np.pi, 150)
    for i, ring in enumerate(rings):
        ring.set_position(current_tick)
        active = ring.current_bit() == '1'
        color = ring.color if active else np.array([0.05, 0.05, 0.1])
        alpha = 1.0 if active else 0.2
        phi = np.linspace(0, 4 * np.pi, 150) + i * 0.6 + current_tick * 0.01
        x_h = (R + r_tube * np.cos(phi)) * np.cos(theta)
        y_h = (R + r_tube * np.cos(phi)) * np.sin(theta)
        z_h = np.sin(theta) * 0.1  # Slight z-twist for 3D
        for j in range(len(theta) - 1):
            if is_3d[0]:
                ax.plot(x_h[j:j+2], y_h[j:j+2], z_h[j:j+2], color=color, alpha=alpha, lw=1.8)
            else:
                ax.plot(x_h[j:j+2], y_h[j:j+2], color=color, alpha=alpha, lw=1.8)

        # Dual Opposite Phi Spirals from Center-Out
        cw_points_1, cw_points_0 = ring.generate_spiral_points(clockwise=True)
        ccw_points_1, ccw_points_0 = ring.generate_spiral_points(clockwise=False)
        offset_x = (i % 3 - 1) * 0.5
        offset_y = (i // 3 - 1) * 0.5
        if len(cw_points_1) > 1:
            if is_3d[0]:
                phi_cw = np.arctan2(cw_points_1[:,1], cw_points_1[:,0])
                theta_cw = np.linspace(0, 2*np.pi, len(cw_points_1))
                x_cw = (R + r_tube * np.cos(phi_cw)) * np.cos(theta_cw) + offset_x
                y_cw = (R + r_tube * np.cos(phi_cw)) * np.sin(theta_cw) + offset_y
                z_cw = r_tube * np.sin(phi_cw)
                ax.plot(x_cw, y_cw, z_cw, color=ring.color, alpha=0.3, lw=1)
                ax.scatter(x_cw, y_cw, z_cw, c=ring.color, s=5, alpha=0.8)
            else:
                ax.plot(cw_points_1[:,0] + offset_x, cw_points_1[:,1] + offset_y, color=ring.color, alpha=0.3, lw=1)
                ax.scatter(cw_points_1[:,0] + offset_x, cw_points_1[:,1] + offset_y, c=ring.color, s=5, alpha=0.8)
        if len(ccw_points_1) > 1:
            if is_3d[0]:
                phi_ccw = np.arctan2(ccw_points_1[:,1], ccw_points_1[:,0])
                theta_ccw = np.linspace(0, 2*np.pi, len(ccw_points_1))
                x_ccw = (R + r_tube * np.cos(phi_ccw)) * np.cos(theta_ccw) + offset_x
                y_ccw = (R + r_tube * np.cos(phi_ccw)) * np.sin(theta_ccw) + offset_y
                z_ccw = r_tube * np.sin(phi_ccw)
                ax.plot(x_ccw, y_ccw, z_ccw, color=ring.color, alpha=0.3, lw=1)
                ax.scatter(x_ccw, y_ccw, z_ccw, c=ring.color, s=5, alpha=0.8)
            else:
                ax.plot(ccw_points_1[:,0] + offset_x, ccw_points_1[:,1] + offset_y, color=ring.color, alpha=0.3, lw=1)
                ax.scatter(ccw_points_1[:,0] + offset_x, ccw_points_1[:,1] + offset_y, c=ring.color, s=5, alpha=0.8)

        # Weave Dual Spirals (connect CW and CCW)
        min_len = min(len(cw_points_1), len(ccw_points_1))
        for j in range(min_len):
            p_cw = cw_points_1[j] + np.array([offset_x, offset_y])
            p_ccw = ccw_points_1[j] + np.array([offset_x, offset_y])
            if is_3d[0]:
                ax.plot([p_cw[0], p_ccw[0]], [p_cw[1], p_ccw[1]], [0, 0.1], color='white', alpha=0.15, lw=0.8)
            else:
                ax.plot([p_cw[0], p_ccw[0]], [p_cw[1], p_ccw[1]], color='white', alpha=0.15, lw=0.8)

        # Pair to Opposite Neighbors
        if ring.opposite_pair:
            opp = ring.opposite_pair
            opp_offset_x = ((num_rings - 1 - i) % 3 - 1) * 0.5
            opp_offset_y = ((num_rings - 1 - i) // 3 - 1) * 0.5
            for j in range(min(len(ring.points_1), len(opp.points_1))):
                p1 = ring.points_1[j] + np.array([offset_x, offset_y])
                p2 = opp.points_1[j] + np.array([opp_offset_x, opp_offset_y])
                if is_3d[0]:
                    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [0, 0.2], color='yellow', alpha=0.1, lw=0.5)
                else:
                    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='yellow', alpha=0.1, lw=0.5)

    # Audio & alignment
    if current_tick != prev_tick:
        play_chord(rings)
        if check_alignment(rings):
            current_age = tick_to_years(current_tick)
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            event = {
                "tick": current_tick,
                "age_byr": current_age,
                "timestamp": timestamp,
                "configuration": [ring.current_bit() for ring in rings]
            }
            alignment_log.append(event)
            print(f"ALIGNMENT #{len(alignment_log)} at {current_age:.3f} billion years (tick {current_tick:,})")
            ax.text(0, 0, 0 if is_3d[0] else None, 'ALIGNMENT!', color='red', fontsize=24, ha='center', va='center')
        prev_tick = current_tick

    title = f"RHA — Dual Phi Spiral Weave (Tick {current_tick:,} | Frame {int(slider_frame_rate.val)}ms | 3D: {is_3d[0]})"
    fig.suptitle(title, color='white', fontsize=16)
    fig.canvas.draw_idle()

slider_depth.on_changed(update)
slider_tick.on_changed(update)
slider_timeline.on_changed(update)
slider_frame_rate.on_changed(update)

# Initial draw
update(None)
plt.show()
