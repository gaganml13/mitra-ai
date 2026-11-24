"""
train_cry_model.py
==================
Trains a baby/child cry detector from your own audio dataset.

HOW TO USE:
    1. Create this folder structure next to this script:
       dataset/
         crying/      ← .wav clips of baby/child crying (80+ clips)
         not_crying/  ← .wav clips of everything else  (80+ clips)

    2. Run:  python train_cry_model.py

    3. It saves:  cry_model.pkl  and  cry_scaler.pkl
       Copy both files into your mitra/ folder.

    4. mitra.py will automatically use these instead of the old heuristic.

CLIP REQUIREMENTS:
    - Format: .wav  (mp3 also works, librosa handles it)
    - Length: 2-10 seconds each
    - Sample rate: any (script resamples automatically)
    - Quality: real recordings work fine, don't need to be clean
"""

import os
import sys
import numpy as np
import librosa
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble         import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm              import SVC
from sklearn.preprocessing    import StandardScaler
from sklearn.model_selection  import StratifiedKFold, cross_val_score
from sklearn.metrics          import classification_report, confusion_matrix
from sklearn.pipeline         import Pipeline

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
DATASET_DIR   = "dataset"
CRY_DIR       = os.path.join(DATASET_DIR, "crying")
NOT_CRY_DIR   = os.path.join(DATASET_DIR, "not_crying")
MODEL_OUT     = "cry_model.pkl"
SCALER_OUT    = "cry_scaler.pkl"
SAMPLE_RATE   = 16000
CLIP_DURATION = 3.0    # seconds — clips longer than this are chunked

# ─────────────────────────────────────────
# STEP 1: FEATURE EXTRACTION
# We convert each audio clip into a fixed-size feature vector.
# Features used:
#   - MFCC (Mel-Frequency Cepstral Coefficients) — the most
#     important feature for any audio classification task.
#     Captures the "shape" of the sound, like a fingerprint.
#   - Chroma features — pitch class energy
#   - Spectral centroid / bandwidth / rolloff — where energy sits
#   - RMS energy — loudness profile
#   - Zero crossing rate — tonality vs noise
#   - Delta MFCCs — how features change over time
# ─────────────────────────────────────────

def extract_features(file_path: str) -> np.ndarray | None:
    """
    Load one audio file and return a 1D feature vector.
    Returns None if the file can't be loaded or is too short.
    """
    try:
        # Load and resample to 16kHz mono
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)

        # Skip clips shorter than 0.5s
        if len(y) < sr * 0.5:
            print(f"   ⏭ Skipping (too short): {os.path.basename(file_path)}")
            return None

        # If clip is longer than CLIP_DURATION, take the middle chunk
        # (middle of a cry is usually the most characteristic part)
        max_samples = int(CLIP_DURATION * sr)
        if len(y) > max_samples:
            start = (len(y) - max_samples) // 2
            y = y[start:start + max_samples]

        # Pad if shorter than CLIP_DURATION
        if len(y) < max_samples:
            y = np.pad(y, (0, max_samples - len(y)))

        features = []

        # ── MFCCs (40 coefficients × mean + std = 80 values) ───────
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        features.extend(np.mean(mfcc, axis=1))
        features.extend(np.std(mfcc, axis=1))

        # ── Delta MFCCs (captures temporal change) ──────────────────
        mfcc_delta = librosa.feature.delta(mfcc)
        features.extend(np.mean(mfcc_delta, axis=1))
        features.extend(np.std(mfcc_delta, axis=1))

        # ── Chroma (12 pitch classes) ────────────────────────────────
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        features.extend(np.mean(chroma, axis=1))
        features.extend(np.std(chroma, axis=1))

        # ── Spectral centroid ────────────────────────────────────────
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        features.append(float(np.mean(centroid)))
        features.append(float(np.std(centroid)))

        # ── Spectral bandwidth ───────────────────────────────────────
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        features.append(float(np.mean(bandwidth)))
        features.append(float(np.std(bandwidth)))

        # ── Spectral rolloff ─────────────────────────────────────────
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        features.append(float(np.mean(rolloff)))
        features.append(float(np.std(rolloff)))

        # ── Zero crossing rate ───────────────────────────────────────
        zcr = librosa.feature.zero_crossing_rate(y)
        features.append(float(np.mean(zcr)))
        features.append(float(np.std(zcr)))

        # ── RMS energy ───────────────────────────────────────────────
        rms = librosa.feature.rms(y=y)
        features.append(float(np.mean(rms)))
        features.append(float(np.std(rms)))

        # ── Mel spectrogram summary ──────────────────────────────────
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=32)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        features.extend(np.mean(mel_db, axis=1))
        features.extend(np.std(mel_db, axis=1))

        return np.array(features, dtype=np.float32)

    except Exception as e:
        print(f"   ❌ Error loading {os.path.basename(file_path)}: {e}")
        return None


def load_dataset(cry_dir: str, not_cry_dir: str):
    """
    Load all audio files from both folders.
    Returns X (feature matrix) and y (labels: 1=crying, 0=not_crying).
    """
    X, y = [], []
    SUPPORTED = (".wav", ".mp3", ".ogg", ".flac", ".m4a")

    print("─" * 50)
    print("📂 Loading CRYING clips...")
    cry_files = [f for f in os.listdir(cry_dir)
                 if f.lower().endswith(SUPPORTED)]
    print(f"   Found {len(cry_files)} files")

    for fname in sorted(cry_files):
        path = os.path.join(cry_dir, fname)
        feat = extract_features(path)
        if feat is not None:
            X.append(feat)
            y.append(1)
            print(f"   ✓ {fname}  ({feat.shape[0]} features)")

    print()
    print("📂 Loading NOT CRYING clips...")
    not_files = [f for f in os.listdir(not_cry_dir)
                 if f.lower().endswith(SUPPORTED)]
    print(f"   Found {len(not_files)} files")

    for fname in sorted(not_files):
        path = os.path.join(not_cry_dir, fname)
        feat = extract_features(path)
        if feat is not None:
            X.append(feat)
            y.append(0)
            print(f"   ✓ {fname}")

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


# ─────────────────────────────────────────
# STEP 2: TRAIN MULTIPLE MODELS AND PICK BEST
# We try 3 classifiers and keep whichever scores highest.
# ─────────────────────────────────────────

def train_and_evaluate(X: np.ndarray, y: np.ndarray):
    """
    Scale features, train three classifiers using 5-fold cross validation,
    pick the best, print results.
    """
    print()
    print("─" * 50)
    print(f"📊 Dataset summary:")
    print(f"   Total clips   : {len(y)}")
    print(f"   Crying        : {np.sum(y == 1)}")
    print(f"   Not crying    : {np.sum(y == 0)}")
    print(f"   Feature size  : {X.shape[1]}")

    if len(y) < 20:
        print()
        print("⚠️  WARNING: You have fewer than 20 total clips.")
        print("   Model accuracy will be unreliable with so little data.")
        print("   Aim for at least 80 crying + 80 not_crying clips.")
        print()

    # Scale features to zero mean, unit variance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Cross-validation setup
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    candidates = {
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=None,
            min_samples_split=2, class_weight="balanced",
            random_state=42, n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.1,
            max_depth=5, random_state=42
        ),
        "SVM (RBF)": SVC(
            kernel="rbf", C=10, gamma="scale",
            probability=True, class_weight="balanced",
            random_state=42
        ),
    }

    print()
    print("─" * 50)
    print("🏋️  Training and cross-validating all models...")
    print()

    results = {}
    for name, clf in candidates.items():
        scores = cross_val_score(clf, X_scaled, y, cv=cv,
                                 scoring="f1", n_jobs=-1)
        mean_f1 = scores.mean()
        std_f1  = scores.std()
        results[name] = (mean_f1, std_f1, clf)
        bar = "█" * int(mean_f1 * 30)
        print(f"   {name:<22} F1 = {mean_f1:.3f} ± {std_f1:.3f}  {bar}")

    # Pick best
    best_name = max(results, key=lambda k: results[k][0])
    best_f1, best_std, best_clf = results[best_name]

    print()
    print(f"🏆 Best model: {best_name}  (F1={best_f1:.3f})")
    print()

    # Final fit on all data
    print("🔧 Final fit on full dataset...")
    best_clf.fit(X_scaled, y)

    # Full dataset report
    y_pred = best_clf.predict(X_scaled)
    print()
    print("📋 Classification report (training set):")
    print(classification_report(y, y_pred,
                                target_names=["not_crying", "crying"]))

    cm = confusion_matrix(y, y_pred)
    print("Confusion matrix:")
    print(f"   TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"   FN={cm[1,0]}  TP={cm[1,1]}")

    if cm[0,1] > 0:
        print(f"\n⚠️  {cm[0,1]} false positive(s) — 'not crying' clips predicted as 'crying'")
        print("   These will cause false alarms in Mitra. Add more not_crying clips.")
    if cm[1,0] > 0:
        print(f"⚠️  {cm[1,0]} false negative(s) — crying clips missed entirely")
        print("   Add more varied crying clips.")

    return best_clf, scaler, best_name, best_f1


# ─────────────────────────────────────────
# STEP 3: SAVE MODEL
# ─────────────────────────────────────────

def save_model(clf, scaler, model_path: str, scaler_path: str):
    joblib.dump(clf,    model_path)
    joblib.dump(scaler, scaler_path)
    print()
    print("─" * 50)
    print(f"💾 Saved: {model_path}")
    print(f"💾 Saved: {scaler_path}")
    print()
    print("✅ DONE! Copy both .pkl files into your mitra/ folder.")
    print("   Mitra will automatically load and use them.")


# ─────────────────────────────────────────
# STEP 4: TEST A SINGLE CLIP  (optional)
# ─────────────────────────────────────────

def test_clip(file_path: str, model_path: str = MODEL_OUT,
              scaler_path: str = SCALER_OUT):
    """
    Quick test: predict whether a single audio clip is crying.
    Usage: python train_cry_model.py test path/to/clip.wav
    """
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print("❌ Model files not found. Run training first.")
        return

    clf    = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    feat = extract_features(file_path)
    if feat is None:
        print("❌ Could not extract features from this file.")
        return

    feat_scaled = scaler.transform([feat])
    pred        = clf.predict(feat_scaled)[0]
    proba       = clf.predict_proba(feat_scaled)[0]

    label = "😭 CRYING" if pred == 1 else "✅ NOT CRYING"
    conf  = proba[pred] * 100
    print(f"\nFile   : {os.path.basename(file_path)}")
    print(f"Result : {label}")
    print(f"Confidence : {conf:.1f}%")
    print(f"  P(not_crying) = {proba[0]*100:.1f}%")
    print(f"  P(crying)     = {proba[1]*100:.1f}%")


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

if __name__ == "__main__":

    if len(sys.argv) >= 3 and sys.argv[1] == "test":
        test_clip(sys.argv[2])
        sys.exit(0)

    print("=" * 50)
    print("🍼  CRY DETECTION MODEL TRAINER")
    print("=" * 50)
    print()

    # Check dataset exists
    if not os.path.exists(CRY_DIR):
        print(f"❌ Missing folder: {CRY_DIR}")
        print()
        print("Create this structure:")
        print("  dataset/")
        print("    crying/       ← .wav clips of baby/child crying")
        print("    not_crying/   ← .wav clips of everything else")
        sys.exit(1)

    if not os.path.exists(NOT_CRY_DIR):
        print(f"❌ Missing folder: {NOT_CRY_DIR}")
        sys.exit(1)

    # Load data
    X, y = load_dataset(CRY_DIR, NOT_CRY_DIR)

    if len(y) == 0:
        print("❌ No audio files found in dataset folders.")
        print("   Make sure your clips end in .wav / .mp3 / .ogg")
        sys.exit(1)

    if np.sum(y == 1) == 0:
        print("❌ No crying clips found.")
        sys.exit(1)

    if np.sum(y == 0) == 0:
        print("❌ No not_crying clips found.")
        sys.exit(1)

    # Train
    clf, scaler, best_name, best_f1 = train_and_evaluate(X, y)

    # Save
    save_model(clf, scaler, MODEL_OUT, SCALER_OUT)

    print()
    if best_f1 >= 0.90:
        print("🟢 Excellent accuracy! Model is ready for Mitra.")
    elif best_f1 >= 0.80:
        print("🟡 Good accuracy. Adding more clips will improve it further.")
    elif best_f1 >= 0.70:
        print("🟠 Decent. Add more crying AND not_crying clips for better results.")
    else:
        print("🔴 Low accuracy. You need more data — aim for 150+ clips per class.")
        print("   Also make sure clips are correctly labelled in the right folder.")
