"""
cry_detector_ml.py
===================
Drop this file into your mitra/ folder.
mitra.py imports from here instead of using the old heuristic.

If cry_model.pkl exists  → uses trained ML model (accurate)
If it doesn't exist      → falls back to the old heuristic (always works)

USAGE IN mitra.py — replace the old _is_cry_like with:

    from cry_detector_ml import is_cry_ml
    # then in cry_detector_thread, replace _is_cry_like(audio) with:
    # is_cry_ml(audio)
"""

import os
import numpy as np

# Try to load the trained model
_clf    = None
_scaler = None
_using_ml = False

_MODEL_PATH  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cry_model.pkl")
_SCALER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cry_scaler.pkl")

def _try_load_model():
    global _clf, _scaler, _using_ml
    if os.path.exists(_MODEL_PATH) and os.path.exists(_SCALER_PATH):
        try:
            import joblib
            _clf    = joblib.load(_MODEL_PATH)
            _scaler = joblib.load(_SCALER_PATH)
            _using_ml = True
            print("✅ ML cry model loaded — using trained classifier.")
        except Exception as e:
            print(f"⚠️  Could not load cry model: {e} — using heuristic fallback.")
    else:
        print("ℹ️  No cry_model.pkl found — using heuristic cry detection.")
        print("   Run train_cry_model.py to train a better model.")

_try_load_model()

SAMPLE_RATE = 16000

# ─────────────────────────────────────────
# ML PREDICTION PATH
# ─────────────────────────────────────────

def _extract_features_fast(audio: np.ndarray) -> np.ndarray | None:
    """
    Same features as train_cry_model.py but runs in real-time.
    librosa is imported lazily so mitra.py starts fast even without it.
    """
    try:
        import librosa

        y  = audio.astype(np.float32)
        sr = SAMPLE_RATE

        # Pad or trim to 3 seconds
        target = sr * 3
        if len(y) < target:
            y = np.pad(y, (0, target - len(y)))
        else:
            start = (len(y) - target) // 2
            y = y[start:start + target]

        features = []

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        features.extend(np.mean(mfcc, axis=1))
        features.extend(np.std(mfcc, axis=1))

        mfcc_delta = librosa.feature.delta(mfcc)
        features.extend(np.mean(mfcc_delta, axis=1))
        features.extend(np.std(mfcc_delta, axis=1))

        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        features.extend(np.mean(chroma, axis=1))
        features.extend(np.std(chroma, axis=1))

        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        features.append(float(np.mean(centroid)))
        features.append(float(np.std(centroid)))

        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        features.append(float(np.mean(bandwidth)))
        features.append(float(np.std(bandwidth)))

        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        features.append(float(np.mean(rolloff)))
        features.append(float(np.std(rolloff)))

        zcr = librosa.feature.zero_crossing_rate(y)
        features.append(float(np.mean(zcr)))
        features.append(float(np.std(zcr)))

        rms = librosa.feature.rms(y=y)
        features.append(float(np.mean(rms)))
        features.append(float(np.std(rms)))

        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=32)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        features.extend(np.mean(mel_db, axis=1))
        features.extend(np.std(mel_db, axis=1))

        return np.array(features, dtype=np.float32)

    except Exception as e:
        return None


# ─────────────────────────────────────────
# HEURISTIC FALLBACK (original logic)
# ─────────────────────────────────────────

def _is_cry_heuristic(audio: np.ndarray) -> bool:
    CRY_ENERGY_THRESH = 0.0088
    CRY_ZCR_LOW       = 0.07
    CRY_ZCR_HIGH      = 0.38

    rms = float(np.sqrt(np.mean(audio ** 2)))
    if rms < CRY_ENERGY_THRESH:
        return False

    zcr = float(np.mean(np.abs(np.diff(np.sign(audio)))) / 2)
    if not (CRY_ZCR_LOW <= zcr <= CRY_ZCR_HIGH):
        return False

    frame_size = int(SAMPLE_RATE * 0.1)
    n_frames   = len(audio) // frame_size
    if n_frames < 4:
        return False

    frame_rms = np.array([
        np.sqrt(np.mean(audio[i*frame_size:(i+1)*frame_size]**2))
        for i in range(n_frames)
    ])
    mean_e = float(np.mean(frame_rms))
    std_e  = float(np.std(frame_rms))
    if mean_e < CRY_ENERGY_THRESH:
        return False
    if std_e / (mean_e + 1e-9) < 0.15:
        return False

    # Spectral flatness — reject fan/white noise
    fft_mag  = np.abs(np.fft.rfft(audio))
    eps      = 1e-9
    log_mean = np.mean(np.log(fft_mag + eps))
    lin_mean = np.mean(fft_mag) + eps
    flatness = float(np.exp(log_mean) / lin_mean)
    if flatness > 0.45:
        return False

    return True


# ─────────────────────────────────────────
# PUBLIC API — use this in mitra.py
# ─────────────────────────────────────────

# Confidence threshold — model must be this sure to trigger.
# From live logs, real crying was scoring 0.48–0.66 → old 0.72 blocked all of it.
# Set to 0.45 so genuine cry signals reliably pass.
# Raise toward 0.65 if you get false positives from background noise.
ML_CONFIDENCE_THRESHOLD = 0.45

def is_cry_ml(audio: np.ndarray) -> bool:
    """
    Main function — call this instead of _is_cry_like() in mitra.py.
    Uses trained ML model if available, heuristic if not.
    """
    if not _using_ml:
        return _is_cry_heuristic(audio)

    feat = _extract_features_fast(audio)
    if feat is None:
        # Feature extraction failed — fall back to heuristic
        return _is_cry_heuristic(audio)

    try:
        feat_scaled = _scaler.transform([feat])
        proba       = _clf.predict_proba(feat_scaled)[0]
        cry_prob    = proba[1]

        print(f"   🍼 Cry ML: P(crying)={cry_prob:.2f}", end="")
        if cry_prob >= ML_CONFIDENCE_THRESHOLD:
            print(" → DETECTED ✅")
            return True
        else:
            print(" → not crying")
            return False
    except Exception as e:
        print(f"   ⚠️ ML predict error: {e} — using heuristic")
        return _is_cry_heuristic(audio)
