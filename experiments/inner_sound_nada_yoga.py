# Snippet idea for inner sound mode
def update_inner_sound(frame):
    t = frame * 0.05
    coherence = current_coherence  # From breath/heart/brain sync
    
    # Buzz intensity
    buzz_alpha = coherence
    buzz_freq = 10000 + 2000 * np.sin(t * 0.5)  # Slight wobble for realism
    
    # Visual ripples from crown
    for r in np.linspace(0.1, 1.2, 8):
        alpha = buzz_alpha * (1 - r)
        if alpha > 0:
            ripple = plt.Circle((0, 0.65), r, color='white', fill=False, lw=3, alpha=alpha)
            ax.add_patch(ripple)
    
    # Title
    if coherence > 0.7:
        title.set_text(f"Inner Sound Activated — ~{buzz_freq/1000:.1f} kHz Nada\nThe unstruck sound of the field")
    else:
        title.set_text("Listening inward... coherence rising")

# Add toggle + frequency slider
