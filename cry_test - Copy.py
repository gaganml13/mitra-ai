"""
cry_test.py — Cry Detection Diagnostic
Run this to find the RIGHT thresholds for YOUR mic and room.

Usage:
    python cry_test.py

It will:
1. Listen continuously and print live audio stats
2. Tell you exactly what values your mic is producing
3. Tell you what to set in mitra.py
"""

import sounddevice as sd
import numpy as np
import time

# ── MATCH THESE TO YOUR mitra.py ──
INPUT_DEVICE   = 1
INPUT_CHANNELS = 2
SAMPLE_RATE    = 16000
BLOCK_SIZE     = 1280

print("=" * 60)
print("🔬 CRY DETECTION DIAGNOSTIC")
print("=" * 60)
print()
print("Instructions:")
print("  1. Stay SILENT for 5 seconds first (we measure background noise)")
print("  2. Then play your baby crying sound")
print("  3. Watch the numbers — we'll tell you what to set")
print()
print("Starting in 3 seconds...")
time.sleep(3)
print()

samples_collected = []
window_stats      = []
phase             = "silence"
phase_start       = time.time()
SILENCE_DURATION  = 5.0
CRY_DURATION      = 10.0

def audio_callback(indata, frames, time_info, status):
    mono = indata.mean(axis=1) if indata.ndim > 1 else indata.flatten()
    samples_collected.append(mono.copy())

stream = sd.InputStream(
    samplerate=SAMPLE_RATE,
    channels=INPUT_CHANNELS,
    dtype='float32',
    blocksize=BLOCK_SIZE,
    device=INPUT_DEVICE,
    callback=audio_callback,
)

silence_stats = []
cry_stats     = []

with stream:
    print(f"🔇 PHASE 1: Measuring SILENCE — stay quiet for {SILENCE_DURATION:.0f}s")
    print(f"{'TIME':>6}  {'RMS':>8}  {'PEAK':>8}  {'ZCR':>8}  {'ENERGY_VAR':>12}")
    print("-" * 55)

    while True:
        elapsed = time.time() - phase_start
        now     = phase

        if now == "silence" and elapsed >= SILENCE_DURATION:
            phase       = "cry"
            phase_start = time.time()
            print()
            print(f"🔊 PHASE 2: Now PLAY YOUR CRYING SOUND for {CRY_DURATION:.0f}s")
            print(f"{'TIME':>6}  {'RMS':>8}  {'PEAK':>8}  {'ZCR':>8}  {'ENERGY_VAR':>12}")
            print("-" * 55)
            samples_collected.clear()

        elif now == "cry" and elapsed >= CRY_DURATION:
            break

        # Analyse last 0.5s of audio
        if len(samples_collected) < 6:
            time.sleep(0.1)
            continue

        chunk = np.concatenate(samples_collected[-6:]).flatten()
        samples_collected.clear()

        rms  = float(np.sqrt(np.mean(chunk ** 2)))
        peak = float(np.max(np.abs(chunk)))
        zcr  = float(np.mean(np.abs(np.diff(np.sign(chunk)))) / 2)

        # Energy variability
        frame_size = int(SAMPLE_RATE * 0.1)
        n_frames   = len(chunk) // frame_size
        if n_frames > 2:
            frame_rms  = np.array([
                np.sqrt(np.mean(chunk[i*frame_size:(i+1)*frame_size]**2))
                for i in range(n_frames)
            ])
            mean_e = float(np.mean(frame_rms))
            std_e  = float(np.std(frame_rms))
            var_r  = std_e / (mean_e + 1e-9)
        else:
            var_r = 0.0

        print(f"{elapsed:>5.1f}s  {rms:>8.5f}  {peak:>8.5f}  {zcr:>8.4f}  {var_r:>12.4f}")

        stat = {"rms": rms, "peak": peak, "zcr": zcr, "var_r": var_r}
        if now == "silence":
            silence_stats.append(stat)
        else:
            cry_stats.append(stat)

        time.sleep(0.4)

# ── ANALYSIS ─────────────────────────────────────────────
print()
print("=" * 60)
print("📊 RESULTS")
print("=" * 60)

if silence_stats:
    s_rms  = np.mean([s["rms"]   for s in silence_stats])
    s_zcr  = np.mean([s["zcr"]   for s in silence_stats])
    s_var  = np.mean([s["var_r"] for s in silence_stats])
    print(f"\nSILENCE baseline:")
    print(f"  avg RMS  = {s_rms:.5f}")
    print(f"  avg ZCR  = {s_zcr:.4f}")
    print(f"  avg VAR  = {s_var:.4f}")

if cry_stats:
    c_rms      = np.mean([s["rms"]   for s in cry_stats])
    c_zcr_mean = np.mean([s["zcr"]   for s in cry_stats])
    c_zcr_min  = np.min ([s["zcr"]   for s in cry_stats])
    c_zcr_max  = np.max ([s["zcr"]   for s in cry_stats])
    c_var      = np.mean([s["var_r"] for s in cry_stats])
    c_rms_min  = np.min ([s["rms"]   for s in cry_stats])

    print(f"\nCRYING audio:")
    print(f"  avg RMS  = {c_rms:.5f}   (min={c_rms_min:.5f})")
    print(f"  avg ZCR  = {c_zcr_mean:.4f}  (range: {c_zcr_min:.4f} – {c_zcr_max:.4f})")
    print(f"  avg VAR  = {c_var:.4f}")

    print()
    print("=" * 60)
    print("✅ PASTE THESE VALUES into mitra.py")
    print("   (Replace the CRY_* constants near the top)")
    print("=" * 60)

    # Recommended thresholds — set just above silence, well below cry
    rec_energy = round(s_rms * 2.5, 5) if silence_stats else round(c_rms_min * 0.6, 5)
    rec_zcr_lo = max(0.01, round(c_zcr_min * 0.7, 4))
    rec_zcr_hi = round(c_zcr_max * 1.3, 4)
    rec_var    = max(0.05, round(c_var * 0.6, 4))

    print(f"""
CRY_ENERGY_THRESH    = {rec_energy}   # was 0.018
CRY_ZCR_LOW          = {rec_zcr_lo}    # was 0.05
CRY_ZCR_HIGH         = {rec_zcr_hi}    # was 0.25
CRY_DURATION_SEC     = 1.8            # keep as is
CRY_COOLDOWN_SEC     = 30             # keep as is
CRY_CONFIRM_WINDOWS  = 3             # keep as is (raise to 4 if too sensitive)
""")

    # Sanity check
    gap = c_rms / (s_rms + 1e-9)
    print(f"Signal-to-noise ratio (crying vs silence): {gap:.1f}x")
    if gap < 3:
        print("⚠️  WARNING: Crying is only {gap:.1f}x louder than background noise.")
        print("   → Move the mic CLOSER to where the child will be.")
        print("   → Or reduce background noise (fan, TV, AC) in the room.")
    elif gap < 8:
        print("✅ Good separation — detection should work reliably.")
    else:
        print("✅ Excellent separation — detection will be very reliable.")

else:
    print()
    print("⚠️  No crying audio was detected at all!")
    print("   Possible reasons:")
    print("   1. Wrong INPUT_DEVICE number — check with: python -c \"import sounddevice as sd; print(sd.query_devices())\"")
    print("   2. The crying sound wasn't loud enough through the speakers")
    print("   3. Mic is too far from the speaker playing the sound")
    print()
    print("   Run this to list your devices:")
    print("   python -c \"import sounddevice as sd; print(sd.query_devices())\"")
