"""
mitra_calibrate.py  —  Run this ONCE to find the right thresholds for YOUR mic.

Usage:
    python mitra_calibrate.py

It will measure your room's background noise, then ask you to simulate crying.
At the end it prints exact values to paste into mitra_final.py
"""
import sounddevice as sd
import numpy as np
import time

INPUT_DEVICE   = 1
INPUT_CHANNELS = 2
SAMPLE_RATE    = 16000

def record_seconds(sec, label):
    print(f"\n🎙️  Recording {sec}s of: {label}")
    print("   Starting in 2 seconds...")
    time.sleep(2)
    print("   ▶ RECORDING...")
    audio = sd.rec(int(sec * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                   channels=INPUT_CHANNELS, dtype='float32', device=INPUT_DEVICE)
    sd.wait()
    mono = audio.mean(axis=1) if audio.ndim > 1 else audio.flatten()
    print("   ■ Done.")
    return mono

def analyse(audio, label):
    rms  = float(np.sqrt(np.mean(audio ** 2)))
    peak = float(np.max(np.abs(audio)))
    zcr  = float(np.mean(np.abs(np.diff(np.sign(audio)))) / 2)

    frame_size = int(SAMPLE_RATE * 0.1)
    n_frames   = len(audio) // frame_size
    frame_rms  = np.array([
        np.sqrt(np.mean(audio[i*frame_size:(i+1)*frame_size]**2))
        for i in range(n_frames)
    ])
    cv = float(np.std(frame_rms) / (np.mean(frame_rms) + 1e-9))

    print(f"\n📊 {label}:")
    print(f"   RMS   = {rms:.5f}   peak = {peak:.5f}")
    print(f"   ZCR   = {zcr:.5f}")
    print(f"   CV    = {cv:.4f}  (variability — crying needs > 0.25)")
    return {"rms": rms, "peak": peak, "zcr": zcr, "cv": cv}

print("=" * 60)
print("MITRA CALIBRATION TOOL")
print("=" * 60)
print("This tool measures YOUR mic to set correct cry thresholds.")
print("Follow each instruction carefully.\n")

# Step 1 — fan/background
input("Step 1: Make sure the ROOM IS QUIET (fan running is fine).\n        Press Enter when ready...")
bg = record_seconds(5, "Background noise / fan")
bg_stats = analyse(bg, "BACKGROUND NOISE")

# Step 2 — normal speech
input("\nStep 2: You will speak normally into the mic.\n        Press Enter, then say 'Hello Mitra tell me a story'...")
speech = record_seconds(4, "Normal speech")
speech_stats = analyse(speech, "NORMAL SPEECH")

# Step 3 — crying simulation
input("\nStep 3: Simulate crying — make a 'waaah waaah' sound\n        as loud as a child would. Press Enter then start...")
cry = record_seconds(5, "Crying simulation")
cry_stats = analyse(cry, "CRY SIMULATION")

# Step 4 — compute safe thresholds
print("\n" + "=" * 60)
print("RECOMMENDED THRESHOLDS FOR YOUR MIC:")
print("=" * 60)

# Energy threshold: midpoint between background and cry, biased toward cry
energy_thresh = bg_stats["rms"] * 3.5
if energy_thresh > cry_stats["rms"] * 0.5:
    energy_thresh = cry_stats["rms"] * 0.4   # if bg is very close to cry, use cry*0.4
energy_thresh = max(energy_thresh, 0.012)    # absolute minimum

# ZCR: safe range between speech and cry
zcr_low  = max(speech_stats["zcr"] * 1.1, 0.09)  # 10% above speech
zcr_high = min(cry_stats["zcr"]   * 1.3, 0.50)   # 30% above cry peak

# CV threshold
cv_thresh = max(bg_stats["cv"] * 2.0, 0.22)  # double background CV

print(f"""
# Paste these values into mitra_final.py:

CRY_ENERGY_THRESH   = {energy_thresh:.4f}   # bg={bg_stats['rms']:.4f}, cry={cry_stats['rms']:.4f}
CRY_ZCR_LOW         = {zcr_low:.4f}         # speech ZCR={speech_stats['zcr']:.4f}
CRY_ZCR_HIGH        = {zcr_high:.4f}        # cry ZCR={cry_stats['zcr']:.4f}
CRY_CV_THRESH       = {cv_thresh:.3f}        # bg CV={bg_stats['cv']:.3f}, cry CV={cry_stats['cv']:.3f}
CRY_DURATION_SEC    = 2.5
CRY_COOLDOWN_SEC    = 30
CRY_CONFIRM_WINDOWS = 5

# Also paste your speaker bleed gate (for stop command):
# Mitra speaker bleed at mic = peak ~{speech_stats['peak']*0.3:.4f}–{speech_stats['peak']*0.5:.4f}
# Set INTERRUPT_ENERGY_GATE to {speech_stats['peak']*0.6:.4f}
# (60-70% of your normal speech peak = above bleed, below your voice)
INTERRUPT_ENERGY_GATE = {speech_stats['peak'] * 0.65:.4f}
""")

print("SAFETY CHECK:")
gap = cry_stats['rms'] / (bg_stats['rms'] + 1e-9)
if gap < 3.0:
    print(f"⚠️  WARNING: Cry ({cry_stats['rms']:.4f}) is only {gap:.1f}x louder than background ({bg_stats['rms']:.4f})")
    print("   Your mic/room may be too noisy for reliable cry detection.")
    print("   Consider increasing CRY_CONFIRM_WINDOWS to 8 or 10.")
else:
    print(f"✅ Good gap: cry is {gap:.1f}x louder than background — detection should be reliable.")
