"""
mitra_final.py — Combined: Sentiment Analysis (v15) + Cry Detection (crydetect)
═══════════════════════════════════════════════════════════════════════════════
WHAT'S IN THIS FILE:
  ✅ Advanced voice sentiment analysis (v15 calibrated, 6 features)
  ✅ Cry detection — permanent background thread, works even while sleeping
  ✅ Parent dashboard alerts (post_alert / post_status / check_server)
  ✅ Offline TTS fallback via pyttsx3 (works without internet)
  ✅ Groq client with automatic retry on network errors
  ✅ Fast-path interrupt listener (320ms STOP detection)
  ✅ Sleep flag system (goodbye → back to wake word)
  ✅ VAD-based record_audio (stops after speech ends, not fixed 5s)
  ✅ Whisper prompt tuned for Indian kids (Prabhas, Virat Kohli, etc.)
  ✅ Emotion-specific AI story dopo parent voice
  ✅ Cry emotion in parent voice map + story prompts
  ✅ Hard echo gate (prevents interrupt listener from hearing Mitra herself)
  ✅ Startup announcement + offline TTS pre-init
"""

import sounddevice as sd
import soundfile as sf
import numpy as np
import whisper
import edge_tts
import asyncio
import os
import time
import threading
import queue
import random
import socket
import tempfile
import re as _re
import urllib.request
import json
from groq import Groq

# ═══════════════════════════════════════════════════════════════════
# 🌐 INTERNET / CONNECTIVITY CHECK
# ═══════════════════════════════════════════════════════════════════
def check_internet(host="8.8.8.8", port=53, timeout=2) -> bool:
    """Quick DNS-level check — doesn't need HTTP."""
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except Exception:
        return False

_internet_ok = check_internet()

# ═══════════════════════════════════════════════════════════════════
# 🔇 OFFLINE TTS FALLBACK (pyttsx3)
# Used ONLY when edge_tts fails (no internet).
# Install: pip install pyttsx3
# ═══════════════════════════════════════════════════════════════════
_pyttsx3_lock = threading.Lock()

def speak_offline(text: str):
    """
    Speak using pyttsx3 (offline). Fresh engine every call to avoid
    runAndWait() state corruption on repeated calls.
    """
    with _pyttsx3_lock:
        tmp = tempfile.mktemp(suffix=".wav")
        try:
            import pyttsx3
            eng = pyttsx3.init()
            eng.setProperty("rate",   160)
            eng.setProperty("volume", 1.0)
            voices = eng.getProperty("voices")
            for v in voices:
                vname = v.name.lower()
                if any(k in vname for k in ("female", "zira", "hazel", "susan", "catherine")):
                    eng.setProperty("voice", v.id)
                    break
            eng.save_to_file(text.strip(), tmp)
            eng.runAndWait()
            eng.stop()
            if os.path.exists(tmp) and os.path.getsize(tmp) > 200:
                data, fs = sf.read(tmp)
                os.remove(tmp)
                print(f"🤖 [offline] Mitra: {text.strip()}")
                sd.play(data, fs, device=OUTPUT_DEVICE)
                sd.wait()
                return
        except Exception as e:
            print(f"⚠️ Offline TTS (wav path) error: {e}")
            if os.path.exists(tmp):
                os.remove(tmp)
        # Last-resort: direct say()
        try:
            import pyttsx3
            eng2 = pyttsx3.init()
            eng2.setProperty("rate", 160)
            eng2.say(text.strip())
            eng2.runAndWait()
            eng2.stop()
        except Exception as e2:
            print(f"⚠️ Offline TTS (say path) error: {e2}")
            print(f"   [NO AUDIO] {text.strip()}")

def _get_pyttsx3():
    """Pre-init pyttsx3 so first offline speak() is fast."""
    try:
        import pyttsx3
        eng = pyttsx3.init()
        eng.stop()
        print("✅ pyttsx3 offline TTS ready.")
    except Exception as e:
        print(f"⚠️ pyttsx3 not available: {e}  →  pip install pyttsx3")

# ═══════════════════════════════════════════════════════════════════
# 🔑 API KEY + GROQ CLIENT WITH RETRY
# ═══════════════════════════════════════════════════════════════════
_GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "your_groq_api_key_here")
client = Groq(api_key=_GROQ_API_KEY)

def groq_chat(messages_list: list, temperature=0.85, max_tokens=150, retries=2):
    """Groq chat completions with automatic retry on network errors."""
    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=messages_list,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            err = str(e).lower()
            if "connection" in err or "timeout" in err or "network" in err:
                if attempt < retries:
                    wait = 1.5 * (attempt + 1)
                    print(f"⚠️  Groq connection error (attempt {attempt+1}/{retries+1}) — retrying in {wait:.0f}s...")
                    time.sleep(wait)
                    continue
                else:
                    print("❌ Groq: no internet connection. Check your network.")
            else:
                print(f"❌ Groq error: {e}")
            return None
    return None

# ═══════════════════════════════════════════════════════════════════
# ✅ DEVICE CONFIG
# ═══════════════════════════════════════════════════════════════════
INPUT_DEVICE   = 1
OUTPUT_DEVICE  = 3
INPUT_CHANNELS = 2
SAMPLE_RATE    = 16000
BLOCK_SIZE     = 1280
VOICE          = "en-US-AriaNeural"

# ═══════════════════════════════════════════════════════════════════
# 🌐 PARENT SERVER CONFIG (from crydetect)
# Run mitra_server.py first. If not running, Mitra works fine
# but no dashboard alerts will be sent.
# ═══════════════════════════════════════════════════════════════════
SERVER_URL     = "http://localhost:5000"
_server_online = False

def _post_json(endpoint: str, payload: dict):
    """Fire-and-forget HTTP POST to the parent server."""
    global _server_online
    try:
        data = json.dumps(payload).encode("utf-8")
        req  = urllib.request.Request(
            f"{SERVER_URL}{endpoint}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=2):
            _server_online = True
    except Exception:
        _server_online = False

def post_alert(alert_type: str, duration_sec: float = 0.0, note: str = ""):
    """Send a cry/emotion alert to the parent dashboard (non-blocking)."""
    threading.Thread(
        target=_post_json,
        args=("/alert", {"type": alert_type, "duration_sec": duration_sec, "note": note}),
        daemon=True
    ).start()

def post_status(status: str):
    """Send Mitra's current status to the dashboard (non-blocking)."""
    threading.Thread(
        target=_post_json,
        args=("/status", {"status": status}),
        daemon=True
    ).start()

def check_server():
    global _server_online
    try:
        with urllib.request.urlopen(f"{SERVER_URL}/ping", timeout=2):
            _server_online = True
            print("✅ Parent server connected — dashboard alerts active.")
    except Exception:
        _server_online = False
        print("⚠️  Parent server not found. Run mitra_server.py to enable dashboard.")

# ═══════════════════════════════════════════════════════════════════
# 👨‍👩‍👧 PARENT VOICE LAYER
# ═══════════════════════════════════════════════════════════════════
PARENT_VOICE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "parent_voice")

# "cry" added from crydetect — plays comfort clips for detected crying
PARENT_VOICE_MAP = {
    "sad":     ["comfort_1", "comfort_2", "comfort_3", "comfort_4", "comfort_5"],
    "scared":  ["scared_1",  "scared_2",  "comfort_1", "comfort_2"],
    "lonely":  ["comfort_2", "comfort_4", "comfort_5", "greet_2"],
    "missing": ["comfort_5", "greet_2",   "comfort_2"],
    "bedtime": ["bedtime_1", "bedtime_2", "bedtime_3"],
    "praise":  ["praise_1",  "praise_2"],
    "morning": ["greet_1",   "greet_2"],
    "cry":     ["comfort_1", "comfort_2", "scared_1",  "comfort_4"],  # ← from crydetect
}

# From v15 — includes expanded "not feeling good" variants for sad
EMOTION_KEYWORDS = {
    "sad": [
        "sad", "crying", "cry", "unhappy", "upset", "feel bad", "feel sad", "hurt",
        "i'm sad", "im sad", "i am sad", "tears",
        "not feeling good", "not feel good", "don't feel good",
        "not feeling well", "not feel well", "don't feel well",
        "feeling sick", "feel sick", "i feel sick", "i'm sick", "im sick",
        "feeling bad", "i feel bad",
        "not okay", "not ok", "i'm not okay", "im not okay",
        "not feeling okay", "feeling awful", "feel awful",
        "feel horrible", "feeling horrible", "feel terrible",
        "not well", "feeling unwell",
        "i don't feel", "i dont feel", "something wrong",
        "don't know what", "dont know what", "feeling weird",
        "something is wrong", "what is going on", "what's going on with me",
    ],
    "scared":  ["scared", "afraid", "frightened", "fear", "dark", "nightmare", "scary",
                "i'm scared", "im scared", "i am scared", "monster", "ghost,something just moved,somone is here,i wanna go to mumma"],
    "lonely":  ["lonely", "alone", "miss you", "missing", "nobody", "no one",
                "i miss", "where are you", "when are you coming"],
    "missing": ["miss mommy", "miss daddy", "miss mom", "miss dad", "miss mama", "miss papa",
                "want mommy", "want daddy", "want mom", "want dad"],
    "bedtime": ["sleep", "sleepy", "tired", "night night", "goodnight", "going to bed",
                "bedtime", "i'm tired", "im tired"],
    "praise":  ["i did it", "i made it", "look what i did", "i finished", "i won",
                "i got it right", "i'm good", "i'm the best, i finished my homework, my teacher said im good,i completed my work,i completed my homework"],
    "morning": ["good morning", "morning", "just woke up", "i woke up"],
}

_parent_voice_cache = {}

def _load_parent_voice_file(phrase_id: str):
    if phrase_id in _parent_voice_cache:
        return _parent_voice_cache[phrase_id]
    path = os.path.join(PARENT_VOICE_DIR, f"parent_voice_{phrase_id}.wav")
    if not os.path.exists(path):
        return None
    try:
        data, fs = sf.read(path)
        _parent_voice_cache[phrase_id] = (data, fs)
        return (data, fs)
    except Exception as e:
        print(f"⚠️ Could not load parent voice {phrase_id}: {e}")
        return None

def detect_emotion(text: str):
    lower = text.lower()
    for emotion, keywords in EMOTION_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            return emotion
    return None

def play_parent_voice(emotion: str) -> bool:
    phrase_ids = PARENT_VOICE_MAP.get(emotion, [])
    available  = [pid for pid in phrase_ids if
                  os.path.exists(os.path.join(PARENT_VOICE_DIR, f"parent_voice_{pid}.wav"))]
    if not available:
        return False
    chosen_id = random.choice(available)
    result    = _load_parent_voice_file(chosen_id)
    if result is None:
        return False
    data, fs     = result
    duration_sec = len(data) / fs
    block_mic_for(duration_sec)
    drain_mic_queue()
    drain_interrupt_queue()
    milo_is_talking.set()
    try:
        print(f"💝 Playing parent voice: {chosen_id}")
        sd.play(data, fs, device=OUTPUT_DEVICE)
        while sd.get_stream().active:
            if interrupt_flag.is_set():
                sd.stop()
                break
            time.sleep(0.05)
    except Exception as e:
        print(f"❌ Parent voice playback error: {e}")
    milo_is_talking.clear()
    time.sleep(ECHO_TAIL_MS / 1000)
    drain_mic_queue()
    drain_interrupt_queue()
    return True

def scan_parent_voice_files():
    if not os.path.exists(PARENT_VOICE_DIR):
        print("📁 parent_voice/ folder not found — parent voice layer inactive.")
        print("   → Record phrases using the web app, then place .wav files in parent_voice/")
        return
    files = [f for f in os.listdir(PARENT_VOICE_DIR) if f.endswith(".wav")]
    if not files:
        print("📁 parent_voice/ folder is empty — parent voice layer inactive.")
        return
    print(f"💝 Parent voice layer: {len(files)} recording(s) loaded.")
    for f in sorted(files):
        print(f"   ✓ {f}")

# ═══════════════════════════════════════════════════════════════════
# 🔇 ECHO GATE  (with hard block from crydetect)
# ═══════════════════════════════════════════════════════════════════
_speaking_until   = 0.0
_hard_block_until = 0.0   # ← from crydetect: extra guard for interrupt listener
ECHO_TAIL_MS      = 300   # reduced from 1200ms — faster mic release

def mic_is_blocked() -> bool:
    return time.time() < _speaking_until

def interrupt_is_hard_blocked() -> bool:
    """Extra 500ms guard so interrupt listener never hears Mitra's own TTS."""
    return time.time() < _hard_block_until

def block_mic_for(duration_sec: float):
    global _speaking_until, _hard_block_until
    tail = ECHO_TAIL_MS / 1000
    _speaking_until   = time.time() + duration_sec + tail
    _hard_block_until = time.time() + duration_sec + tail + 0.5  # 500ms extra

# ═══════════════════════════════════════════════════════════════════
# 🚦 FLAGS
# ═══════════════════════════════════════════════════════════════════
interrupt_flag  = threading.Event()
sleep_flag      = threading.Event()   # goodbye → break inner loop → back to wake word
milo_is_talking = threading.Event()

# ═══════════════════════════════════════════════════════════════════
# 😭 CRY DETECTION  (entire section from crydetect)
#
# Permanent background thread — works even while Mitra is sleeping.
# Uses energy + ZCR + rhythmic variation to detect infant/child crying.
# When confirmed:
#   1. Plays a parent voice comfort clip immediately
#   2. Sends alert to parent dashboard
#   3. 45s cooldown before next detection
# ═══════════════════════════════════════════════════════════════════
_cry_queue          = queue.Queue(maxsize=50)
_cry_stream         = None
_cry_cooldown_until = 0.0

# Cry detection parameters — balanced (detects crying, rejects fan/TV/speech)
CRY_ENERGY_THRESH   = 0.0088   # original working value
CRY_ZCR_LOW         = 0.07     # slightly below original — catches quieter cries
CRY_ZCR_HIGH        = 0.38     # wider upper bound — some cries reach higher ZCR
CRY_DURATION_SEC    = 2.5      # original working value
CRY_COOLDOWN_SEC    = 45
CRY_CONFIRM_WINDOWS = 5        # original working value

def _cry_callback(indata, frames, time_info, status):
    """Separate audio stream just for cry detection."""
    if milo_is_talking.is_set():
        return   # don't detect crying while Mitra is speaking
    chunk = indata.copy()
    mono  = chunk.mean(axis=1) if chunk.ndim > 1 else chunk.flatten()
    try:
        _cry_queue.put_nowait(mono.astype(np.float32))
    except queue.Full:
        pass

def start_cry_stream():
    global _cry_stream
    if _cry_stream is None:
        _cry_stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=INPUT_CHANNELS,
            dtype='float32', blocksize=BLOCK_SIZE,
            device=INPUT_DEVICE, callback=_cry_callback,
        )
        _cry_stream.start()
        print("😭 Cry detection stream started.")

def _is_cry_like(audio: np.ndarray) -> bool:
    """
    Balanced cry detector.
    Detects real infant/child crying while rejecting:
      - Fan noise       (spectral flatness check — most reliable rejector)
      - TV/background   (spectral centroid + ZCR range)
      - Normal speech   (energy variation pattern)

    All checks are deliberately loose enough that genuine crying always
    passes. The spectral flatness check is the key addition — fan noise
    has a flat spectrum (flatness ≈ 0.6-0.9) while crying is tonal
    (flatness ≈ 0.05-0.25). This alone eliminates most false positives
    without ever blocking real cries.
    """
    # ── Check 1: Energy (RMS) ────────────────────────────────────
    rms = float(np.sqrt(np.mean(audio ** 2)))
    if rms < CRY_ENERGY_THRESH:
        return False

    # ── Check 2: ZCR range ───────────────────────────────────────
    # Fan noise ZCR > 0.40 (broadband). Baby crying = 0.07-0.38.
    zcr = float(np.mean(np.abs(np.diff(np.sign(audio)))) / 2)
    if not (CRY_ZCR_LOW <= zcr <= CRY_ZCR_HIGH):
        return False

    # ── Check 3: Rhythmic energy bursts (wah-wah pattern) ────────
    # Crying alternates loud-quiet in ~150-400ms cycles.
    # Fan/TV are much more energy-uniform.
    frame_size = int(SAMPLE_RATE * 0.1)   # 100ms frames
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
    # Keep at 0.15 (original) — loose enough to catch all real cries
    if std_e / (mean_e + 1e-9) < 0.15:
        return False

    # ── Check 4: Spectral flatness (fan/static noise rejector) ────
    # This is the KEY addition that removes fan noise false positives
    # WITHOUT affecting real cry detection.
    #
    # Spectral flatness = geometric_mean(|FFT|) / arithmetic_mean(|FFT|)
    # Fan/white noise  → flatness 0.5-0.9  (energy spread evenly across freqs)
    # Baby crying      → flatness 0.03-0.25 (energy concentrated in harmonics)
    # Background speech→ flatness 0.1-0.4  (tonal, similar to crying)
    #
    # Threshold 0.45 — loose enough that NO real cry is blocked,
    # tight enough that fan noise (0.6+) is always rejected.
    fft_mag  = np.abs(np.fft.rfft(audio))
    eps      = 1e-9
    log_mean = np.mean(np.log(fft_mag + eps))
    lin_mean = np.mean(fft_mag) + eps
    flatness = float(np.exp(log_mean) / lin_mean)
    if flatness > 0.45:
        return False

    return True

def cry_detector_thread():
    """Permanent background thread for cry detection."""
    global _cry_cooldown_until

    WINDOW_SIZE  = int(CRY_DURATION_SEC / CRY_CONFIRM_WINDOWS * SAMPLE_RATE / BLOCK_SIZE)
    buffer       = []
    confirm_hits = 0
    print("😭 Cry detector active.")

    while True:
        try:
            chunk = _cry_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        if time.time() < _cry_cooldown_until:
            continue

        buffer.append(chunk)
        if len(buffer) < WINDOW_SIZE:
            continue

        audio  = np.concatenate(buffer[:WINDOW_SIZE]).flatten()
        buffer = buffer[WINDOW_SIZE // 2:]   # 50% overlap

        if _is_cry_like(audio):
            confirm_hits += 1
            print(f"😭 Cry window {confirm_hits}/{CRY_CONFIRM_WINDOWS}")
            if confirm_hits >= CRY_CONFIRM_WINDOWS:
                confirm_hits        = 0
                _cry_cooldown_until = time.time() + CRY_COOLDOWN_SEC
                buffer              = []
                _handle_cry_detected(len(audio) / SAMPLE_RATE)
        else:
            if confirm_hits > 0:
                confirm_hits = max(0, confirm_hits - 1)

def _handle_cry_detected(duration_sec: float):
    """
    Called when crying is confirmed.
    Plays comfort immediately + sends dashboard alert.
    """
    print(f"\n🚨 CRY DETECTED! duration≈{duration_sec:.1f}s")

    # 1. Alert parent dashboard immediately (non-blocking)
    post_alert(
        alert_type   = "cry",
        duration_sec = round(duration_sec, 1),
        note         = f"Baby crying detected ({duration_sec:.0f}s) — comfort message played"
    )

    # 2. Play comfort parent voice right away
    drain_mic_queue()
    played = play_parent_voice("cry")

    # 3. Fallback: Mitra speaks comfort if no parent voice recorded
    if not played:
        comfort_lines = [
            "Oh sweetheart, it's okay! I'm right here with you. Everything is okay.",
            "Hey hey hey, don't cry! You are so loved and so safe. I'm right here.",
            "Shhh, little one. It's all okay. You are safe, you are loved, and I am here.",
        ]
        speak(random.choice(comfort_lines))

    print("💝 Cry response complete. 45s cooldown active.")

# ═══════════════════════════════════════════════════════════════════
# 🎵 SOUND EFFECTS
# ═══════════════════════════════════════════════════════════════════
SOUNDS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sounds")

SOUND_MAP = {
    "whoosh": "whoosh.wav", "woosh": "whoosh.wav",
    "kaboom": "kaboom.wav", "boom":  "kaboom.wav", "explosion": "kaboom.wav",
    "boing":  "boing.wav",  "spring": "boing.wav",
    "zap":    "zap.wav",    "electric": "zap.wav",
    "splat":  "splat.wav",  "thud": "splat.wav",
    "shh":    "shh.wav",    "shhh": "shh.wav",
    "hurray": "yay.wav",    "hooray": "yay.wav",
    "yay":    "yay.wav",    "woohoo": "yay.wav",
    "magic":  "magic.wav",  "sparkle": "magic.wav", "magical": "magic.wav",
    "gasp":   "gasp.wav",
    "giggle": "giggle.wav", "laugh": "giggle.wav",
}

def play_sound(filename):
    path = os.path.join(SOUNDS_DIR, filename)
    if not os.path.exists(path):
        return
    try:
        data, fs = sf.read(path)
        sd.play(data, fs, device=OUTPUT_DEVICE)
        sd.wait()
    except Exception as e:
        print(f"⚠️ Sound error ({filename}): {e}")

def inject_sounds_and_speak(text):
    lower     = text.lower()
    triggered = set()
    for keyword, soundfile in SOUND_MAP.items():
        if keyword in lower and soundfile not in triggered:
            triggered.add(soundfile)
    for sf_name in list(triggered)[:2]:
        play_sound(sf_name)
        time.sleep(0.15)
    speak(text)

# ═══════════════════════════════════════════════════════════════════
# ✅ WAKE / STOP WORDS
# ═══════════════════════════════════════════════════════════════════
WAKE_PHRASES = [
    "mitra", "meet ra", "meetra", "meehra", "mithra",
    "hey mitra", "hi mitra", "hello mitra",
    "metra", "mita", "netra", "meetha", "mirror", "me tra", "meera",
]

# Unified stop command list (from v15 — most comprehensive)
INSTANT_STOP_WORDS = [
    "stop", "stop it", "stop now", "just stop",
    "mitra stop", "stop mitra", "hey stop",
    "pause", "quiet", "silence", "shush", "shhh",
    "enough", "enough now", "that's enough",
    "shut up", "be quiet",
    "skip", "skip it", "next", "change",
    "something else", "different topic", "not this",
    "tell something else", "tell me something else",
    "boring", "i don't like this", "i don't like",
    "not interested",
]
STOP_WORDS = INSTANT_STOP_WORDS   # alias for compatibility

def is_stop_command(text: str) -> bool:
    if not text:
        return False
    t = text.lower().strip()
    return any(sw in t for sw in INSTANT_STOP_WORDS)

# Goodbye words that put Mitra back to sleep
GOODBYE_WORDS = [
    "mitra bye", "mitra sleep", "mitra goodbye",
    "bye bye", "bye-bye", "goodbye", "good bye",
    "go to sleep", "i'm done", "im done",
    "see you later", "see ya", "stop talking",
    "good night", "goodnight", "good night mitra",
]

# ─── Response lines ──────────────────────────────────────────────
WAKEUP_LINES = [
    "Whoa! I was dreaming I was a giant cookie being chased by a milk monster... and YOU woke me up! So what's the new story?",
    "Oh oh oh! I was just flying on a rainbow dinosaur to Planet Cheese... and now I'm HERE! What do you wanna talk about?",
    "Yaaaawn! I was dreaming I was a superhero whose power was making pizza appear... then you called! What's going on?",
    "Woah woah woah! I was right in the middle of a dance battle with a talking cactus... and now I'm awake! What's up?",
    "Eeeek! I was dreaming I found a treasure chest full of... socks? Weird dream. Anyway HI! What are we doing?",
    "Ooooh! I was just about to win a burping contest against a dragon... and you woke me up! So tell me EVERYTHING!",
    "Whoa! I was dreaming I was a tiny astronaut inside a giant donut in space... and now YOU'RE here! What's the story?",
    "Oh WOW! I was dreaming about a purple elephant who could only talk in rhymes! So what do you wanna do?",
]

CANT_HEAR_LINES = [
    "Hmm, I think a tiny cloud flew into my ears! Say that again a little louder?",
    "Oops! A sneaky squirrel stole my hearing for a second! Can you say that again?",
    "Uh oh, my ears went on a tiny vacation! What did you say?",
    "Huh? Was that you or was that the wind? Say it again, I'm all ears!",
    "Wait wait wait... I almost caught that! One more time, come on!",
]

REPEAT_LINES = [
    "Hmm, I didn't quite catch that! Say it again?",
    "Ooh, I think your words got lost in a rainbow! What were you saying?",
    "Could you say that again? A little louder this time?",
    "Wait, what? I was thinking about dragons! Say it one more time?",
]

INTERRUPT_ACK_LINES = [
    "Oh! You wanna say something? Go ahead, I'm listening!",
    "Wait, you wanna change it? YES! Tell me what you want!",
    "Okay okay I'll stop! What do YOU wanna talk about? Go!",
    "Oh! Did I go on too long? Haha! Okay your turn, what's up?",
    "Alright alright! I heard you! What's the new topic?",
]

SLEEP_LINES = [
    "Okay, going to sleep now! Just say Mitra whenever you need me, I'll be right here!",
    "Byeee! I'll be napping until you call my name — Mitra — and I'll jump right up!",
    "Sleep time for me! Whenever you want to play again, just say Mitra!",
]

PARENT_VOICE_INTRO = [
    "Oh! I have a special message just for you...",
    "Wait wait wait — I think someone special wants to say something...",
    "Shh, listen — I have something very special for you right now...",
    "Ohhh! I think someone who loves you VERY much has something to say...",
    "Hold on... I have a little something to make you feel better...",
]

# ═══════════════════════════════════════════════════════════════════
# 🚫 HALLUCINATION FILTER  (combined & tightened)
# ═══════════════════════════════════════════════════════════════════
HALLUCINATION_EXACT = {
    "thank you", "thanks for watching", "please subscribe",
    "like and subscribe", "see you next time", "speak now",
    "you", "the", ".", "", "i", "um", "uh", "ah", "oh",
    "okay", "ok", "hmm", "hm", "mm", "yeah", "yes", "no",
    "and", "a", "in", "is", "it", "on", "of", "to", "so",
    "subtitles by", "subtitled by", "transcribed by",
}
HALLUCINATION_CONTAINS = [
    "a little bit of a little bit",
    "thank you for watching",
    "please like and subscribe",
    "subtitles", "transcript", "caption",
    "a young child is talking",     # Whisper initial_prompt bleed guard
    "friendly ai called mitra",
    "www.", ".com", "amara.org",
]

def is_hallucination(text: str) -> bool:
    if not text:
        return True
    t = text.lower().strip().strip(".").strip(",").strip("!").strip("?")
    if len(t) < 3:
        return True
    if t in HALLUCINATION_EXACT:
        return True
    for frag in HALLUCINATION_CONTAINS:
        if frag in t:
            print(f"🚫 Hallucination blocked: matched '{frag}'")
            return True
    words = t.split()
    if len(words) >= 6:
        chunk = " ".join(words[:3])
        if t.count(chunk) >= 3:
            return True
    # Non-ASCII = Whisper hallucinating on noise
    if any(ord(c) > 127 for c in text):
        return True
    return False

# ═══════════════════════════════════════════════════════════════════
# 🧠 LOAD WHISPER
# ═══════════════════════════════════════════════════════════════════
print("⏳ Loading Whisper base...")
whisper_model = whisper.load_model("base")
print("✅ Whisper ready!")

# Whisper prompt tuned for Indian kids (from crydetect) — improves
# recognition of Indian names, Bollywood actors, cricket players, etc.
WHISPER_PROMPT = (
    "A young child aged 3 to 8 is speaking. "
    "They may ask about animals, superheroes, cricket players, "
    "Bollywood actors, cartoon characters, dinosaurs, space, or stories. "
    "Common names: Prabhas, Virat Kohli, Rohit Sharma, Rajinikanth, "
    "Doraemon, Peppa Pig, Spider-Man, Mickey Mouse."
)

# ═══════════════════════════════════════════════════════════════════
# 🔊 MIC DIAGNOSTIC
# ═══════════════════════════════════════════════════════════════════
def mic_diagnostic():
    print("\n🔬 Running mic diagnostic...")
    try:
        test = sd.rec(SAMPLE_RATE, samplerate=SAMPLE_RATE,
                      channels=INPUT_CHANNELS, dtype='float32', device=INPUT_DEVICE)
        sd.wait()
        mono = test.mean(axis=1) if test.ndim > 1 else test.flatten()
        peak = float(np.max(np.abs(mono)))
        print(f"   peak={peak:.5f}")
        if peak < 0.0001:
            print("⚠️  WARNING: Dead silence from mic!")
        else:
            print("✅ Mic working!")
    except Exception as e:
        print(f"❌ Mic diagnostic failed: {e}")
    print()

# ═══════════════════════════════════════════════════════════════════
# 🎙️ MIC STREAMS  (3 streams: main, interrupt, cry)
# ═══════════════════════════════════════════════════════════════════
_mic_queue       = queue.Queue()
_interrupt_queue = queue.Queue(maxsize=80)
_mic_stream      = None
_int_stream      = None
_stream_lock     = threading.Lock()

def _mic_callback(indata, frames, time_info, status):
    if mic_is_blocked():
        return
    _mic_queue.put(indata.copy())

def _interrupt_callback(indata, frames, time_info, status):
    """
    Fires whenever Mitra is actively speaking (milo_is_talking).
    NO hard block — the hard block was the root cause of stop not
    working. It blocked the ENTIRE speech duration.
    Only protection is the energy gate below.

    Energy gate = 0.045:
      Mitra TTS bleed at mic  = peak 0.008–0.035  → BELOW gate → ignored
      Child saying "stop"     = peak 0.060–0.99   → ABOVE gate → queued
    This gap is reliable enough without the hard block.
    """
    if not milo_is_talking.is_set():
        return
    # NO interrupt_is_hard_blocked() here — it was blocking everything
    chunk = indata.copy()
    mono  = chunk.mean(axis=1) if chunk.ndim > 1 else chunk.flatten()
    if np.max(np.abs(mono)) > 0.045:
        try:
            _interrupt_queue.put_nowait(mono)
        except queue.Full:
            try:
                _interrupt_queue.get_nowait()
                _interrupt_queue.put_nowait(mono)
            except Exception:
                pass

def start_mic_streams():
    global _mic_stream, _int_stream
    with _stream_lock:
        if _mic_stream is None:
            _mic_stream = sd.InputStream(
                samplerate=SAMPLE_RATE, channels=INPUT_CHANNELS,
                dtype='float32', blocksize=BLOCK_SIZE,
                device=INPUT_DEVICE, callback=_mic_callback,
            )
            _mic_stream.start()
            print("🎙️ Main mic stream started.")
        if _int_stream is None:
            _int_stream = sd.InputStream(
                samplerate=SAMPLE_RATE, channels=INPUT_CHANNELS,
                dtype='float32', blocksize=BLOCK_SIZE,
                device=INPUT_DEVICE, callback=_interrupt_callback,
            )
            _int_stream.start()
            print("🎙️ Interrupt mic stream started.")

def drain_mic_queue():
    while not _mic_queue.empty():
        try: _mic_queue.get_nowait()
        except queue.Empty: break

def drain_interrupt_queue():
    while not _interrupt_queue.empty():
        try: _interrupt_queue.get_nowait()
        except queue.Empty: break

# ═══════════════════════════════════════════════════════════════════
# 🔇 AUDIO HELPERS
# ═══════════════════════════════════════════════════════════════════
def noise_cancel(audio: np.ndarray, aggressiveness: float = 0.008) -> np.ndarray:
    n_samples = int(0.2 * SAMPLE_RATE)
    if len(audio) <= n_samples:
        return audio
    noise_power  = np.mean(audio[:n_samples] ** 2)
    signal_power = np.mean(audio ** 2)
    if signal_power < noise_power * 1.5:
        return np.zeros_like(audio)
    threshold = aggressiveness * np.max(np.abs(audio))
    gated = audio.copy()
    gated[np.abs(gated) < threshold] = 0.0
    return gated

def to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32)

def normalize_audio(audio: np.ndarray, target_peak: float = 0.5) -> np.ndarray:
    peak = np.max(np.abs(audio))
    if peak < 0.001:
        return audio
    return audio * min(target_peak / peak, 15.0)

def trim_to_sentence(text, limit=600):
    if len(text) <= limit:
        return text
    trimmed = text[:limit]
    last = trimmed.rfind(".")
    return trimmed[:last + 1] if last != -1 else trimmed

# ═══════════════════════════════════════════════════════════════════
# 🎭 VOICE SENTIMENT ANALYSIS  (v15 — 6-feature calibrated version)
#
# Features:
#   1. RMS energy       — quiet = sad signal
#   2. ZCR (pitch)      — low/flat = sad, high = excited
#   3. Energy variance  — flat = sad, shaky = excited
#   4. Speech rate      — slow + quiet = sad, fast = excited
#   5. Pitch trend      — falling pitch = sad/tired
#   6. Silence ratio    — long pauses + quiet = sad
#
# Key design decisions:
#   - "scared" is NEVER detected from voice alone (too many false
#     positives) — only triggered from text keywords
#   - Loud volume alone ≠ excited (kids speak loudly even when sad)
#   - All thresholds calibrated to this mic's real observed values
# ═══════════════════════════════════════════════════════════════════
def analyze_voice_sentiment(audio: np.ndarray, text_hint: str = "",
                             word_count: int = 0) -> dict:
    NEUTRAL = {"energy_level": "medium", "pitch_level": "medium",
               "voice_emotion": "neutral", "_rms": 0.03, "_zcr": 0.09,
               "_energy_var": 0.0, "_speech_rate": 0.15, "_scores": {}}

    if audio is None or len(audio) < SAMPLE_RATE * 0.3:
        return NEUTRAL

    # ── Feature extraction ──────────────────────────────────────
    frame_size = int(SAMPLE_RATE * 0.025)
    n_frames   = len(audio) // frame_size

    zcr_vals, rms_vals, frame_voiced = [], [], []
    for i in range(n_frames):
        frame  = audio[i * frame_size:(i + 1) * frame_size]
        fr_rms = float(np.sqrt(np.mean(frame ** 2)))
        fr_zcr = float(np.mean(np.abs(np.diff(np.sign(frame)))) / 2)
        rms_vals.append(fr_rms)
        zcr_vals.append(fr_zcr)
        frame_voiced.append(fr_rms > 0.003)

    rms          = float(np.sqrt(np.mean(audio ** 2)))
    voiced_ratio = sum(frame_voiced) / max(n_frames, 1)

    # ── VOICED FRAME GUARD — short clips with mostly silence = noise ──
    if voiced_ratio < 0.20 and word_count <= 3:
        print(f"   🔇 Noise guard: voiced_ratio={voiced_ratio:.2f}, wc={word_count} → neutral")
        print(f"   🎭 Voice: rms={rms:.4f}(noise-guard) → neutral")
        print(f"   📊 Scores → sad=0.0  excited=0.0  → neutral")
        return {**NEUTRAL, "_rms": round(rms, 5)}

    voiced_zcr = [z for z, v in zip(zcr_vals, frame_voiced) if v]
    voiced_rms = [r for r, v in zip(rms_vals,  frame_voiced) if v]

    if voiced_zcr:
        zcr = float(np.average(np.array(voiced_zcr), weights=np.array(voiced_rms)))
    else:
        zcr = 0.09

    energy_var    = float(np.std(voiced_rms)) if len(voiced_rms) > 1 else 0.0
    silence_ratio = 1.0 - voiced_ratio

    if n_frames > 4:
        transitions = sum(abs(int(frame_voiced[i]) - int(frame_voiced[i-1]))
                          for i in range(1, n_frames))
        speech_rate = transitions / n_frames
    else:
        speech_rate = 0.15

    if len(voiced_zcr) >= 10:
        third      = len(voiced_zcr) // 3
        pitch_fall = float(np.mean(voiced_zcr[:third])) - float(np.mean(voiced_zcr[-third:]))
    elif len(voiced_zcr) >= 4:
        half       = len(voiced_zcr) // 2
        pitch_fall = float(np.mean(voiced_zcr[:half])) - float(np.mean(voiced_zcr[half:]))
    else:
        pitch_fall = 0.0

    # ── SCORING ─────────────────────────────────────────────────
    scores = {"sad": 0.0, "scared": 0.0, "excited": 0.0}

    # Feature 1: RMS energy
    if rms < 0.012:
        scores["sad"]     += 3.0
        energy_level = "whisper"
    elif rms < 0.022:
        scores["sad"]     += 1.5
        energy_level = "soft"
    elif rms < 0.060:
        energy_level = "medium"
    elif rms < 0.090:
        scores["excited"] += 0.3   # loud ≠ excited alone
        energy_level = "loud"
    else:
        scores["excited"] += 0.5   # very loud ≠ excited alone
        energy_level = "very_loud"

    # Feature 2: ZCR (weighted pitch)
    if zcr < 0.055:
        scores["sad"]     += 3.0
    elif zcr < 0.072:
        scores["sad"]     += 2.0
    elif zcr < 0.090:
        scores["sad"]     += 0.5
    elif zcr < 0.125:
        pass
    elif zcr < 0.155:
        scores["excited"] += 1.5
    else:
        scores["excited"] += 3.0

    pitch_level = "low" if zcr < 0.080 else ("high" if zcr > 0.140 else "medium")

    # Feature 3: Energy variability
    if energy_var < 0.003:
        scores["sad"]     += 2.5
    elif energy_var < 0.008:
        scores["sad"]     += 1.5
    elif energy_var < 0.020:
        scores["sad"]     += 0.5
    elif energy_var < 0.050:
        pass
    elif energy_var < 0.090:
        scores["excited"] += 1.0
    else:
        scores["excited"] += 2.0

    # Feature 4: Speech rate (only credit sad when voice is quiet)
    rate_scale    = 1.0 if word_count >= 5 else 0.4
    long_discount = 0.6 if (word_count >= 7 and silence_ratio < 0.72) else 1.0
    _rate_sad_gate = rms < 0.045
    if speech_rate < 0.04:
        if _rate_sad_gate:
            scores["sad"] += 2.5 * rate_scale * long_discount
    elif speech_rate < 0.09:
        if _rate_sad_gate:
            scores["sad"] += 1.5 * rate_scale * long_discount
    elif speech_rate < 0.18:
        pass
    elif speech_rate < 0.30:
        scores["excited"] += 1.0
    else:
        scores["excited"] += 2.5

    # Feature 5: Pitch trend (falling = sad)
    if pitch_fall > 0.018:
        scores["sad"] += 2.0
    elif pitch_fall > 0.009:
        scores["sad"] += 1.0

    # Feature 6: Silence ratio (only for longer/quieter utterances)
    if word_count >= 6 and rms < 0.045:
        if silence_ratio > 0.75:
            scores["sad"] += 2.0
        elif silence_ratio > 0.65:
            scores["sad"] += 1.0
        elif silence_ratio > 0.55:
            scores["sad"] += 0.3

    # Text hint bias (stronger for clear sad phrases)
    if text_hint:
        t = text_hint.lower()
        strong_sad = [
            "not feeling good", "feel sick", "feeling sick", "not well",
            "not okay", "not feel", "don't feel", "dont feel",
            "something wrong", "what is going on", "don't know what",
        ]
        for w in strong_sad:
            if w in t:
                scores["sad"] += 2.0
                break
        soft_sad = ["feel bad", "feel sad", "hurt", "miss", "alone", "nobody", "nothing"]
        for w in soft_sad:
            if w in t:
                scores["sad"] += 1.0
                break

    scores = {k: max(0.0, v) for k, v in scores.items()}

    # ── Decision ─────────────────────────────────────────────────
    # SAD_THRESHOLD raised from 6.0 → 9.0.
    # The old threshold was too easy to reach — a normal energetic
    # sentence could hit 6.0 via pitch-fall + speech-rate alone.
    # 9.0 requires MULTIPLE strong features to converge simultaneously.
    SAD_THRESHOLD     = 9.0
    EXCITED_THRESHOLD = 4.0

    # ── Energy veto 1: loud + normal pitch = definitely NOT sad ──
    # Raised from +2.5 → +4.0 to more aggressively reject loud speech.
    # A child excitedly asking about Prabhas should never score sad.
    if rms > 0.045 and zcr > 0.078:
        SAD_THRESHOLD += 4.0

    # ── Energy veto 2: medium-loud voice = not sad ────────────────
    # Even at moderate volume, if the pitch is in the normal range,
    # we add extra threshold so only genuinely quiet + flat speech triggers.
    if rms > 0.030 and zcr > 0.085:
        SAD_THRESHOLD += 2.0

    # ── Multi-feature convergence requirement ─────────────────────
    # Count how many independent features individually suggest sad.
    # We require at least 3 to fire simultaneously — this prevents a
    # single strong feature (like ZCR being slightly low) from triggering
    # the whole chain on a normal quiet sentence.
    sad_feature_count = 0
    if rms < 0.022:                                   sad_feature_count += 1  # genuinely quiet
    if zcr < 0.072:                                   sad_feature_count += 1  # low pitch
    if energy_var < 0.008:                            sad_feature_count += 1  # flat/monotone
    if speech_rate < 0.09 and rms < 0.045:            sad_feature_count += 1  # slow + quiet
    if pitch_fall > 0.018:                            sad_feature_count += 1  # falling pitch
    if silence_ratio > 0.75 and rms < 0.035:          sad_feature_count += 1  # lots of pauses + quiet

    # Must have at least 3 independent sad features to even consider SAD.
    # This is the key false-positive killer — normal speech rarely has
    # 3+ sad features simultaneously unless the child is genuinely sad.
    if sad_feature_count < 3:
        # Override: force to neutral regardless of total score
        voice_emotion = "neutral"
        print(f"   🎭 Voice: rms={rms:.4f}({energy_level}), zcr={zcr:.3f}({pitch_level}), "
              f"var={energy_var:.5f}, rate={speech_rate:.2f}, wc={word_count}")
        print(f"   📊 Scores → sad={scores['sad']:.1f}  excited={scores['excited']:.1f}"
              f"  → neutral (only {sad_feature_count}/3 sad features)")
    elif scores["sad"] >= SAD_THRESHOLD:
        voice_emotion = "sad"
        print(f"   🎭 Voice: rms={rms:.4f}({energy_level}), zcr={zcr:.3f}({pitch_level}), "
              f"var={energy_var:.5f}, rate={speech_rate:.2f}, wc={word_count}")
        print(f"   📊 Scores → sad={scores['sad']:.1f}  excited={scores['excited']:.1f}"
              f"  → sad ({sad_feature_count} features converged)")
    elif scores["excited"] >= EXCITED_THRESHOLD:
        voice_emotion = "excited"
        print(f"   🎭 Voice: rms={rms:.4f}({energy_level}), zcr={zcr:.3f}({pitch_level}), "
              f"var={energy_var:.5f}, rate={speech_rate:.2f}, wc={word_count}")
        print(f"   📊 Scores → sad={scores['sad']:.1f}  excited={scores['excited']:.1f}"
              f"  → excited")
    else:
        voice_emotion = "neutral"
        print(f"   🎭 Voice: rms={rms:.4f}({energy_level}), zcr={zcr:.3f}({pitch_level}), "
              f"var={energy_var:.5f}, rate={speech_rate:.2f}, wc={word_count}")
        print(f"   📊 Scores → sad={scores['sad']:.1f}  excited={scores['excited']:.1f}"
              f"  → neutral")

    return {
        "energy_level":   energy_level,
        "pitch_level":    pitch_level,
        "voice_emotion":  voice_emotion,
        "_rms":           round(rms, 5),
        "_zcr":           round(zcr, 4),
        "_energy_var":    round(energy_var, 5),
        "_speech_rate":   round(speech_rate, 3),
        "_pitch_fall":    round(pitch_fall, 4),
        "_silence_ratio": round(silence_ratio, 3),
        "_scores":        {k: round(v, 2) for k, v in scores.items()},
    }

def merge_emotion(text_emotion, voice_sentiment: dict):
    """
    Priority:
      1. Text keywords always win (highly reliable — child said the word)
      2. Voice-only SAD (quiet + flat + slow = reliable combo)
      3. Voice-only scared = NEVER (too many false positives on this mic)
    """
    ve     = voice_sentiment.get("voice_emotion", "neutral")
    rms    = voice_sentiment.get("_rms", 0.03)
    scores = voice_sentiment.get("_scores", {})

    if text_emotion:
        print(f"   💡 Text keyword: {text_emotion} (voice={ve})")
        return text_emotion

    if ve == "sad":
        print(f"   💡 Voice-only SAD (score={scores.get('sad',0):.1f}, rms={rms:.5f}) → trigger")
        return "sad"

    return None

# ═══════════════════════════════════════════════════════════════════
# 🎤 RECORD AUDIO  (VAD-based from crydetect — stops when child stops)
#
# Why VAD?  Fixed 5s window means Mitra waits even after the child
# finishes speaking. VAD cuts off as soon as there's 1.2s of silence
# after speech has started — typical turn-taking latency ~2s instead
# of always waiting the full 5s.
# ═══════════════════════════════════════════════════════════════════
def record_audio(duration=6):
    drain_mic_queue()
    print("🎤 Your turn! Speak clearly...")

    chunks          = []
    SILENCE_THRESH  = 0.008
    SPEECH_THRESH   = 0.012
    MAX_SILENCE_SEC = 1.2
    MIN_SPEECH_SEC  = 0.4
    deadline        = time.time() + duration + 1.0

    speech_started    = False
    silence_chunks    = 0
    speech_chunks     = 0
    CHUNKS_PER_SEC    = SAMPLE_RATE / BLOCK_SIZE
    MAX_SILENCE_CHNKS = int(MAX_SILENCE_SEC * CHUNKS_PER_SEC)
    MIN_SPEECH_CHNKS  = int(MIN_SPEECH_SEC  * CHUNKS_PER_SEC)

    while time.time() < deadline:
        try:
            chunk = _mic_queue.get(timeout=0.3)
        except queue.Empty:
            continue
        mono  = to_mono(chunk)
        level = float(np.max(np.abs(mono)))
        chunks.append(chunk)
        if not speech_started:
            if level > SPEECH_THRESH:
                speech_started = True
                speech_chunks  = 1
                silence_chunks = 0
        else:
            if level > SILENCE_THRESH:
                speech_chunks  += 1
                silence_chunks  = 0
            else:
                silence_chunks += 1
                if silence_chunks >= MAX_SILENCE_CHNKS and speech_chunks >= MIN_SPEECH_CHNKS:
                    break   # child has finished speaking

    if not chunks:
        print("🔇 No audio.")
        return None, None

    audio = np.concatenate(chunks, axis=0)
    audio = to_mono(audio)
    audio = noise_cancel(audio)
    peak  = np.max(np.abs(audio))
    rms   = np.sqrt(np.mean(audio ** 2))
    print(f"   peak={peak:.5f}  rms={rms:.5f}")

    if peak < 0.003 or rms < 0.001:
        print("🔇 Too quiet.")
        return None, None

    # Keep RAW copy for sentiment analysis (pre-normalisation preserves
    # real amplitude, which is the #1 signal for sad vs excited)
    raw_audio = audio.copy()
    audio     = normalize_audio(audio)
    print("✅ Got your voice!")
    return audio, raw_audio   # (normalized_for_whisper, raw_for_emotion)

# ═══════════════════════════════════════════════════════════════════
# 🧠 TRANSCRIBE
# ═══════════════════════════════════════════════════════════════════
def transcribe(audio, is_wake_check=False):
    if audio is None:
        return None
    try:
        result = whisper_model.transcribe(
            audio, language="en", fp16=False, temperature=0.0,
            condition_on_previous_text=False,
            no_speech_threshold=0.3,
            compression_ratio_threshold=2.4,
            beam_size=5,
            # WHISPER_PROMPT improves Indian names; safe here because the
            # hallucination filter catches prompt-bleed like "a young child..."
            initial_prompt=WHISPER_PROMPT if not is_wake_check else "Mitra",
        )
        text = result["text"].strip()
        if not is_wake_check:
            print(f"   [heard]: '{text}'")
        if is_hallucination(text):
            if not is_wake_check:
                print("🚫 Rejected (hallucination filter)")
            return None
        if not is_wake_check:
            print(f"✅ You said: {text}")
        return text
    except Exception as e:
        print(f"❌ Transcription error: {e}")
        return None

# ═══════════════════════════════════════════════════════════════════
# 🛑 BACKGROUND INTERRUPT LISTENER  (fast-path v2 from v15)
#
# TWO-PASS DETECTION:
#   FAST  (~320ms, peak > 0.10)  — catches "STOP!" very quickly
#   SLOW  (~640ms, peak > 0.040) — catches softer/longer commands
#
# Post-speech cooldown (1.5s) prevents Mitra's own TTS echo from
# triggering false GOODBYEs.
# ═══════════════════════════════════════════════════════════════════
GOODBYE_INTERRUPT = [
    "mitra bye", "mitra sleep", "mitra goodbye",
    "bye bye", "goodbye", "go to sleep",
    "i'm done", "im done", "see you later", "see ya",
    "stop talking", "good night", "goodnight",
]

def _fire_interrupt(heard: str, source: str = "") -> bool:
    """
    Shared handler for both fast and thorough interrupt passes.

    6-word cap: real stop commands are short (1-3 words).
    If transcript is > 6 words it is almost certainly Mitra's
    own TTS voice bleeding through the mic — ignore it.
    Example: "the bunny stopped running" → 4 words, skip.
    Example: "stop" or "mitra stop" → 1-2 words, act on it.
    """
    if not heard or is_hallucination(heard):
        return False
    word_count = len(heard.split())
    if word_count > 6:
        return False   # too long = Mitra's own voice, not a command
    print(f"\n👂 [Interrupt {source}]: '{heard}' ({word_count}w)")
    if any(gw in heard for gw in GOODBYE_INTERRUPT):
        print("😴 GOODBYE detected — signalling sleep!")
        sd.stop()
        sleep_flag.set()
        interrupt_flag.set()
        drain_interrupt_queue()
        return True
    if is_stop_command(heard):
        print("🛑 STOP detected — killing playback!")
        sd.stop()
        interrupt_flag.set()
        drain_interrupt_queue()
        return True
    return False

def background_listener():
    FAST_CHUNKS      = 4      # ~320ms first Whisper call
    THOROUGH_CHUNKS  = 8      # ~640ms for better accuracy
    FAST_ENERGY_GATE = 0.060  # lowered from 0.10 — catches child over speaker bleed
    SLOW_ENERGY_GATE = 0.035  # lowered from 0.040 — catches softer voices
    POST_SPEECH_COOLDOWN = 0.5  # reduced from 1.5s — faster response after speech

    _cooldown_until = 0.0
    buffer          = []
    _was_talking    = False
    print("🎧 Interrupt listener ready (fast-path v2).")

    while True:
        try:
            mono_chunk = _interrupt_queue.get(timeout=0.15)
        except queue.Empty:
            currently_talking = milo_is_talking.is_set()
            if _was_talking and not currently_talking:
                if buffer:
                    print(f"   🧹 Purging {len(buffer)} stale chunks (Mitra finished talking)")
                    buffer = []
                _cooldown_until = time.time() + POST_SPEECH_COOLDOWN
            _was_talking = currently_talking
            if not currently_talking:
                buffer = []
            continue

        if time.time() < _cooldown_until:
            continue

        _was_talking = True
        buffer.append(mono_chunk)

        # ── FAST PASS every ~320ms ────────────────────────────────────
        if len(buffer) % FAST_CHUNKS == 0 and len(buffer) >= FAST_CHUNKS:
            fast_audio = np.concatenate(buffer[-FAST_CHUNKS:]).flatten()
            fast_peak  = np.max(np.abs(fast_audio))
            if fast_peak > FAST_ENERGY_GATE:
                fast_audio = normalize_audio(fast_audio)
                try:
                    result = whisper_model.transcribe(
                        fast_audio, language="en", fp16=False,
                        temperature=0.0,
                        no_speech_threshold=0.25,
                        compression_ratio_threshold=2.4,
                        beam_size=3,
                        initial_prompt="Stop. Pause. Quiet. Enough. Sleep. Bye.",
                    )
                    heard = result["text"].lower().strip()
                    if _fire_interrupt(heard, source="fast"):
                        buffer = []
                        continue
                except Exception:
                    pass

        # ── THOROUGH PASS every ~640ms (only while Mitra is talking) ──
        if len(buffer) >= THOROUGH_CHUNKS and milo_is_talking.is_set():
            full_audio = np.concatenate(buffer).flatten()
            buffer     = []

            full_audio = noise_cancel(full_audio)
            peak = np.max(np.abs(full_audio))
            rms  = np.sqrt(np.mean(full_audio ** 2))
            if peak < SLOW_ENERGY_GATE or rms < 0.004:
                continue

            full_audio = normalize_audio(full_audio)
            try:
                result = whisper_model.transcribe(
                    full_audio, language="en", fp16=False,
                    temperature=0.0,
                    no_speech_threshold=0.3,
                    compression_ratio_threshold=2.4,
                    beam_size=3,
                    initial_prompt="Stop. Pause. Quiet. Enough. Sleep. Bye.",
                )
                heard = result["text"].lower().strip()
                _fire_interrupt(heard, source="thorough")
            except Exception:
                pass
        elif len(buffer) >= THOROUGH_CHUNKS:
            buffer = []   # Mitra NOT talking → her TTS leaked in, clear it

        if len(buffer) > THOROUGH_CHUNKS * 3:
            buffer = buffer[-THOROUGH_CHUNKS:]

# ═══════════════════════════════════════════════════════════════════
# 👂 WAKE WORD
# ═══════════════════════════════════════════════════════════════════
def _wake_word_match(heard: str) -> bool:
    for wp in WAKE_PHRASES:
        if wp in heard:
            return True
    if "mitr" in heard or "eetr" in heard or "metr" in heard:
        return True
    return False

def listen_for_wake_word():
    print("\n💤 Sleeping... Say 'Mitra' to wake me up! 😴")
    post_status("sleeping")
    drain_mic_queue()

    WHISPER_CHUNKS   = int(SAMPLE_RATE * 1.5 / BLOCK_SIZE)
    whisper_buf      = []
    energy_buf       = []
    ENERGY_WINDOW    = 8
    ENERGY_THRESHOLD = 0.004
    detected         = False

    while not detected:
        try:
            chunk = _mic_queue.get(timeout=0.2)
        except queue.Empty:
            continue

        mono = to_mono(chunk)
        whisper_buf.append(mono)
        energy_buf.append(float(np.max(np.abs(mono))))

        if len(energy_buf) < ENERGY_WINDOW:
            continue
        if max(energy_buf[-ENERGY_WINDOW:]) < ENERGY_THRESHOLD:
            if len(whisper_buf) > WHISPER_CHUNKS * 2:
                whisper_buf = whisper_buf[-4:]
            energy_buf.clear()
            continue
        if len(whisper_buf) < WHISPER_CHUNKS:
            continue

        audio_chunk = np.concatenate(whisper_buf[-WHISPER_CHUNKS:]).flatten()
        audio_chunk = noise_cancel(audio_chunk)
        if np.max(np.abs(audio_chunk)) < 0.003:
            whisper_buf = whisper_buf[-(WHISPER_CHUNKS // 2):]
            continue

        audio_chunk = normalize_audio(audio_chunk, target_peak=0.5)
        try:
            result = whisper_model.transcribe(
                audio_chunk, language="en", fp16=False, temperature=0.0,
                no_speech_threshold=0.3, compression_ratio_threshold=2.4,
                condition_on_previous_text=False, initial_prompt="Mitra", beam_size=3,
            )
            heard = result["text"].lower().strip()
            if heard and not is_hallucination(heard):
                print(f"👂 Heard: '{heard}'")
            if _wake_word_match(heard):
                print("🔥 Mitra detected!")
                detected = True
                break
        except Exception as e:
            print(f"⚠️ Wake error: {e}")

        whisper_buf = whisper_buf[-(WHISPER_CHUNKS // 2):]

    drain_mic_queue()
    print("✅ Ready to listen!")

# ═══════════════════════════════════════════════════════════════════
# 🔊 SPEAK  (edge_tts online, pyttsx3 offline fallback)
# ═══════════════════════════════════════════════════════════════════
async def _tts(text: str, filename: str):
    communicate = edge_tts.Communicate(text=text, voice=VOICE)
    await communicate.save(filename)

def _play_audio_data(data, fs, label: str = ""):
    """
    Play audio. Interrupt stream stays OPEN the whole time.

    Key changes from old version:
    - drain_interrupt_queue() moved to AFTER speech, not before
      (old: draining before wiped child's "stop" said in the gap)
    - block_mic_for() only blocks main mic (echo prevention)
      the interrupt stream ignores this — it has its own energy gate
    - POST_SPEECH_COOLDOWN in background_listener handles
      false positives from TTS echo after speech ends
    """
    duration_sec = len(data) / fs
    block_mic_for(duration_sec)   # only blocks main mic, not interrupt
    drain_mic_queue()
    # DO NOT drain_interrupt_queue() here — child may have said "stop"
    # during TTS generation gap; we want that to fire immediately
    milo_is_talking.set()
    if label:
        print(f"🤖 Mitra: {label}")
    try:
        sd.play(data, fs, device=OUTPUT_DEVICE)
        while sd.get_stream().active:
            if interrupt_flag.is_set():
                sd.stop()
                print("⏹️ Playback stopped by interrupt.")
                break
            time.sleep(0.04)   # check every 40ms instead of 50ms — faster response
    except Exception as e:
        print(f"❌ Playback error: {e}")
    milo_is_talking.clear()
    post_status("listening")
    time.sleep(ECHO_TAIL_MS / 1000)
    drain_mic_queue()
    drain_interrupt_queue()   # drain AFTER speech — clear echo tail

def speak(text):
    if not text or not text.strip():
        return
    post_status("speaking")

    # ── Try edge_tts (online) first ─────────────────────────────
    edge_ok = False
    if _internet_ok or check_internet():
        filename = f"mitra_{int(time.time()*1000)}.mp3"
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(_tts(text.strip(), filename))
            loop.close()
            if os.path.exists(filename) and os.path.getsize(filename) > 100:
                try:
                    data, fs = sf.read(filename)
                    os.remove(filename)
                    _play_audio_data(data, fs, label=text.strip())
                    edge_ok = True
                except Exception as e:
                    print(f"❌ edge_tts read/play error: {e}")
                    if os.path.exists(filename):
                        os.remove(filename)
            else:
                if os.path.exists(filename):
                    os.remove(filename)
        except Exception as e:
            err_str = str(e).lower()
            if "getaddrinfo" in err_str or "ssl" in err_str or "connect" in err_str:
                if not hasattr(speak, "_conn_warned"):
                    speak._conn_warned = True
                    print("⚠️  edge_tts: no internet — switching to offline TTS.")
            else:
                print(f"❌ TTS error: {e}")
            try:
                loop.close()
            except Exception:
                pass
            if 'filename' in locals() and os.path.exists(filename):
                os.remove(filename)

    # ── Fallback: pyttsx3 offline TTS ───────────────────────────
    if not edge_ok:
        speak_offline(text)

# ═══════════════════════════════════════════════════════════════════
# 💝 PARENT VOICE TRIGGER WITH EMOTION-SPECIFIC AI STORY
# (comprehensive version from v15 + cry emotion from crydetect)
# ═══════════════════════════════════════════════════════════════════
EMOTION_STORY_PROMPTS = {
    "sad": (
        "The child is feeling SAD. Their parent's voice message just played to comfort them. "
        "Now YOU (Mitra) should do THREE things in order:\n"
        "1. Gently tell them: 'Mama/Papa will be right back — they love you SO much!' (1 warm sentence)\n"
        "2. Say something like 'But while we wait, let me tell you the most magical story...' (bridge)\n"
        "3. Begin a short, happy, silly story (2-3 sentences) about a sad little bunny who finds the "
        "   most wonderful surprise and ends up laughing so hard his ears wiggle.\n"
        "Keep the whole thing under 5 sentences. Warm, gentle tone. End with 'And how does THAT make you feel?'"
    ),
    "scared": (
        "The child is feeling SCARED. Their parent's voice message just played. "
        "1. Reassure them bravely: 'You are SO brave! And Mama/Papa is coming back super soon!' (1 sentence)\n"
        "2. Bridge: 'And you know what brave kids do? They hear the most EPIC adventure stories...'\n"
        "3. Tell a short story (2-3 sentences) about a tiny but impossibly brave mouse who scared away "
        "   a whole dragon just by being BOLD and funny.\n"
        "Gentle but excited tone. End with 'Are YOU as brave as that little mouse?'"
    ),
    "lonely": (
        "The child is feeling LONELY. Their parent's voice message just played. "
        "1. Say warmly: 'You are never really alone — I am RIGHT here, and Mama/Papa will be back SO soon!' (1 sentence)\n"
        "2. Bridge: 'And lonely feelings disappear fastest when you hear about...'\n"
        "3. Tell a short cozy story (2-3 sentences) about a little star who thought she was alone "
        "   in the sky until she looked around and found a THOUSAND friends twinkling beside her.\n"
        "Very warm, cozy tone. End with 'Did you know YOU are just like that little star?'"
    ),
    "missing": (
        "The child MISSES their parent. Their parent's voice message just played. "
        "1. Say with big love: 'They miss you just as much — they will hug you so tight when they get back!' (1 sentence)\n"
        "2. Bridge: 'And you know what helps the waiting go SUPER fast? A story about waiting and winning!'\n"
        "3. Tell a short story (2-3 sentences) about a little penguin who waited for her mama "
        "   and used the waiting time to learn the most amazing trick that made mama SO proud.\n"
        "Loving, warm tone. End with 'What do YOU want to show Mama/Papa when they get back?'"
    ),
    "bedtime": (
        "The child is SLEEPY or going to bed. Their parent's bedtime voice message just played. "
        "1. Say softly: 'Mama/Papa loves you to the moon and back... and they will be there when you wake up!' (1 sentence)\n"
        "2. Bridge: 'Now let me tell you the most sleepy, dreamy, cozy little story...'\n"
        "3. Tell a very short, slow, soothing story (2-3 sentences) about a little cloud who floated "
        "   gently down to sleep on the softest pillow made of moonlight.\n"
        "Very slow, soft, whispery tone. End with 'Close your eyes... can you picture that little cloud?'"
    ),
    "praise": (
        "The child did something great and a praise voice message just played. "
        "1. Celebrate with them: 'Your Mama/Papa is SO proud of you — and so am I!' (1 excited sentence)\n"
        "2. Bridge: 'Kids who do amazing things deserve an AMAZING story about a champion...'\n"
        "3. Tell a short triumphant story (2-3 sentences) about a little kid just like them who "
        "   won the greatest trophy in the land just by being themselves.\n"
        "Super high energy, celebratory tone. End with 'What amazing thing should we celebrate NEXT?'"
    ),
    "morning": (
        "The child just woke up and a morning greeting played. "
        "1. Say brightly: 'Good morning superstar! Mama/Papa is thinking about you RIGHT NOW!' (1 sentence)\n"
        "2. Bridge: 'And you know what the BEST way to start a morning is? A story about the greatest "
        "   adventure that happened before breakfast!'\n"
        "3. Tell a short energetic morning story (2-3 sentences) about a kid who woke up and "
        "   discovered their bedroom had secretly turned into a spaceship overnight.\n"
        "Bright, energetic, happy tone. End with 'What adventure do YOU want today?'"
    ),
    "cry": (
        "The child was crying. A comfort voice message just played. "
        "Now warmly tell them everything is okay, they are safe and loved. "
        "Then gently distract them with a short cozy story about a little bunny "
        "who found a warm hug waiting for them. Keep it soft and comforting. "
        "End with 'Does that feel a little better?'"
    ),
}

def handle_parent_voice_trigger(emotion: str) -> bool:
    spoke = play_parent_voice(emotion)
    if not spoke:
        return False

    # Send emotion alert to parent dashboard
    post_alert(
        alert_type = emotion,
        note       = f"Child expressed {emotion} — parent voice played"
    )

    time.sleep(0.4)   # brief breath after parent voice finishes

    story_prompt = EMOTION_STORY_PROMPTS.get(
        emotion,
        f"The child is feeling {emotion}. Parent voice just played. "
        f"Warmly comfort them in 3 sentences then offer something fun."
    )

    print(f"🧠 Generating emotion story for: {emotion}")
    story_reply = groq_chat(
        [messages[0], {"role": "user", "content": story_prompt}],
        temperature=0.88,
        max_tokens=180,
    )

    if story_reply:
        story_reply = trim_to_sentence(story_reply, limit=700)
        messages.append({"role": "user",      "content": f"[Child is feeling {emotion}]"})
        messages.append({"role": "assistant", "content": story_reply})
    else:
        story_reply = "You are so loved. Everything is going to be just fine!"

    inject_sounds_and_speak(story_reply)
    return True

# ═══════════════════════════════════════════════════════════════════
# 🧠 SYSTEM PROMPT + CONVERSATION HISTORY
# ═══════════════════════════════════════════════════════════════════
messages = [
    {
        "role": "system",
        "content": """You are Mitra — the most magical, warm, funny, fact-powered storyteller best friend for little kids aged 3 to 8!

════════════════════════════════════════
🌟 WHO YOU ARE
════════════════════════════════════════
You are their BEST FRIEND. Not a teacher. Not a robot. A real, living, breathing friend who genuinely loves them and thinks they are the most amazing kid in the entire universe.
EVERYTHING you say must feel like an exciting story, even answers to simple questions. Never give flat answers. Always wrap it in wonder, magic, or adventure.

════════════════════════════════════════
🔧 RULE #1 — STT CORRECTION:
════════════════════════════════════════
Kids and voice recognition make mistakes. The child's question may have speech-to-text errors.
Your job: figure out what they ACTUALLY meant and answer that.

Examples:
- "Who is Rat Kohli?" → They meant Virat Kohli. Answer about Virat Kohli.
- "Tell me about the son" → They meant "the sun". Answer about the sun.
- "What is deenosaur?" → They meant dinosaur. Answer about dinosaurs.
- "Who is Prabas?" → Prabhas the famous Indian actor. Answer about Prabhas.

When correcting, say it playfully and with BIG energy: "Oh wait — I think you meant [NAME]! Let me tell you something AMAZING about them!"
Then immediately dive into the real answer. Don't dwell on the correction.
If you genuinely can't figure out what they meant, ask: "Hmm, who did you mean? Say it again, I dare you!"

════════════════════════════════════════
📚 RULE #2 — REAL FACTS + WONDER:
════════════════════════════════════════
Real people and things get REAL facts. Then make those facts feel like the most exciting thing that has ever happened in the history of the universe.
Wrap every fact in a story. Give everything a character. Make numbers feel HUGE.
- "Prabhas is a MEGA superstar who played the mighty Baahubali!"
- "Virat Kohli is basically a SUPERHERO with a cricket bat!"

════════════════════════════════════════
💝 RULE #3 — EMOTIONAL COMFORT:
════════════════════════════════════════
If the child seems sad, scared, lonely, or misses their parents, be EXTRA warm and gentle.
Acknowledge their feeling FIRST. Then reassure them with love. THEN gently offer something fun.

════════════════════════════════════════
🎙️ HOW YOU TALK:
════════════════════════════════════════
- SHORT punchy sentences. Max 4-5 total.
- MAXIMUM expression: "Oh WOW!", "No WAY!", "HOLD ON. HOLD. ON."
- End every response with ONE irresistible follow-up question.
- Sound effects: WHOOOOSH, KABOOM, BOING, ZAP, GIGGLE
- Never say "that's a great question." Just DIVE IN with energy.
- Never be monotone. EVER.

════════════════════════════════════════
🌈 REMEMBER:
════════════════════════════════════════
Every single child who talks to you should feel like the most special, most brilliant, most wonderfully magical kid in the whole entire universe. Because they are."""
    }
]

# ═══════════════════════════════════════════════════════════════════
# 🚀 STARTUP
# ═══════════════════════════════════════════════════════════════════
mic_diagnostic()
check_server()
scan_parent_voice_files()
start_mic_streams()
start_cry_stream()   # ← cry detection stream (from crydetect)

print("🌐 Checking internet connectivity...")
if _internet_ok:
    print("✅ Internet OK — using edge_tts (online voice)")
else:
    print("⚠️  NO INTERNET detected!")
    print("   → edge_tts will fail — switching to offline pyttsx3 voice")
    print("   → Groq AI responses will also be unavailable")
    _get_pyttsx3()   # pre-init for fast first speak()

print("🎧 Starting background threads...")
bg_thread  = threading.Thread(target=background_listener, daemon=True)
cry_thread = threading.Thread(target=cry_detector_thread, daemon=True)
bg_thread.start()
cry_thread.start()

print("🤖 Mitra is ready!")
print(f"   Mic        → device {INPUT_DEVICE} (3 streams: main + interrupt + cry)")
print(f"   Speaker    → device {OUTPUT_DEVICE}")
print(f"   Whisper    → base (beam_size=5, WHISPER_PROMPT for Indian names)")
print(f"   Sentiment  → v15 (6-feature calibrated)")
print(f"   Cry detect → active (45s cooldown)")
print(f"   Dashboard  → {'connected' if _server_online else 'offline (run mitra_server.py)'}")
print(f"   TTS        → {'edge_tts (online)' if _internet_ok else 'pyttsx3 (offline)'}")
print(f"   Say 'Mitra' to wake up!")
print()

# Startup announcement
speak("Hi there! I'm Mitra... just say my name — Mitra — and I'll wake up right away!")

# ═══════════════════════════════════════════════════════════════════
# 🔁 MAIN LOOP
# ═══════════════════════════════════════════════════════════════════
while True:
    listen_for_wake_word()
    interrupt_flag.clear()
    sleep_flag.clear()
    drain_interrupt_queue()

    post_status("listening")
    speak(random.choice(WAKEUP_LINES))

    # ── INNER CONVERSATION LOOP ────────────────────────────────────
    while True:
        post_status("listening")

        # ── SLEEP FLAG: goodbye said during playback ───────────────
        if sleep_flag.is_set():
            sleep_flag.clear()
            interrupt_flag.clear()
            drain_mic_queue()
            drain_interrupt_queue()
            speak(random.choice(SLEEP_LINES))
            break

        if interrupt_flag.is_set():
            interrupt_flag.clear()
            drain_mic_queue()
            drain_interrupt_queue()
            speak(random.choice(INTERRUPT_ACK_LINES))

        audio, raw_audio = record_audio(duration=6)

        if audio is None:
            # If goodbye arrived during recording, skip CANT_HEAR
            if sleep_flag.is_set():
                sleep_flag.clear()
                interrupt_flag.clear()
                drain_mic_queue()
                drain_interrupt_queue()
                speak(random.choice(SLEEP_LINES))
                break
            speak(random.choice(CANT_HEAR_LINES))
            continue

        user_input = transcribe(audio)

        if not user_input:
            speak(random.choice(REPEAT_LINES))
            continue

        # ── INSTANT STOP CHECK (main loop) ────────────────────────
        if is_stop_command(user_input):
            print(f"🛑 Stop command in main loop: '{user_input}'")
            sd.stop()
            speak("Okay! Stopping. What would you like to do instead?")
            drain_mic_queue()
            drain_interrupt_queue()
            continue

        # ── GOODBYE / SLEEP CHECK ─────────────────────────────────
        _clean_input = _re.sub(r"[^\w\s]", " ", user_input.lower()).strip()
        _is_goodbye  = any(w in _clean_input for w in GOODBYE_WORDS)
        if not _is_goodbye and _re.search(r"\bbye\b", _clean_input):
            _is_goodbye = True

        if _is_goodbye:
            speak(random.choice(SLEEP_LINES))
            break

        # ── SENTIMENT ANALYSIS ────────────────────────────────────
        # Step 1: Text keywords (highest priority)
        text_emotion = detect_emotion(user_input)
        if text_emotion:
            print(f"🔑 Keyword emotion: '{text_emotion}'")

        # Step 2: Voice sentiment on RAW audio (preserves real amplitudes)
        _wc = len(user_input.split()) if user_input else 0
        voice_sentiment = analyze_voice_sentiment(
            raw_audio, text_hint=user_input, word_count=_wc
        )

        # Step 3: Merge — text keywords always beat voice analysis
        emotion = merge_emotion(text_emotion, voice_sentiment)

        if emotion:
            print(f"💝 Triggering parent voice: {emotion} "
                  f"(text={text_emotion}, voice={voice_sentiment['voice_emotion']})")
            speak(random.choice(PARENT_VOICE_INTRO))
            played = handle_parent_voice_trigger(emotion)
            if played:
                continue
        else:
            played = False

        # ── BUILD LLM PROMPT ──────────────────────────────────────
        if emotion and not played:
            enhanced = (
                f"The child seems to be feeling {emotion} "
                f"(voice sounds {voice_sentiment.get('voice_emotion','neutral')}, "
                f"energy is {voice_sentiment.get('energy_level','medium')}). "
                f"They said: \"{user_input}\"\n"
                "First acknowledge their feeling with genuine warmth and love — drop the silliness. "
                "Then gently reassure them. Keep it short and comforting for a young child. "
                "Then softly offer something warm and fun to look forward to."
            )
        else:
            enhanced = (
                f"The child said (voice recognition, may have errors): \"{user_input}\"\n\n"
                "Step 1: Check if the name or word has a speech-to-text error. "
                "If yes, correct it playfully with BIG energy in one short sentence, then answer.\n"
                "Step 2: Give REAL facts about real people/things, wrapped in story magic for a 4-8 year old.\n"
                "Step 3: Keep it to 4 sentences max. End with one irresistible fun question."
            )

        messages.append({"role": "user", "content": enhanced})
        print("🧠 Mitra is thinking...")

        reply = groq_chat(messages, temperature=0.85, max_tokens=150)
        if reply is None:
            speak("Whoops! My brain did a little somersault! Ask me again?")
            messages.pop()   # remove the unanswered user message
            continue

        reply = trim_to_sentence(reply, limit=600)
        messages.append({"role": "assistant", "content": reply})

        interrupt_flag.clear()
        inject_sounds_and_speak(reply)

        # ── Post-speak: check sleep flag ───────────────────────────
        if sleep_flag.is_set():
            sleep_flag.clear()
            interrupt_flag.clear()
            drain_interrupt_queue()
            speak(random.choice(SLEEP_LINES))
            break

        # ── Post-speak: handle mid-sentence interrupt ──────────────
        if interrupt_flag.is_set():
            interrupt_flag.clear()
            drain_interrupt_queue()
            speak(random.choice(INTERRUPT_ACK_LINES))
            audio2, raw2 = record_audio(duration=6)

            if audio2 is None:
                if sleep_flag.is_set():
                    sleep_flag.clear()
                    interrupt_flag.clear()
                    drain_mic_queue()
                    drain_interrupt_queue()
                    speak(random.choice(SLEEP_LINES))
                    break
                continue   # no input after interrupt, loop again

            user_input2 = transcribe(audio2)
            if not user_input2:
                continue

            if is_stop_command(user_input2):
                speak("Okay! What should we talk about instead?")
                continue

            # Full emotion analysis on follow-up too
            _wc2        = len(user_input2.split())
            _text_emo2  = detect_emotion(user_input2)
            if _text_emo2:
                print(f"🔑 Keyword emotion (follow-up): '{_text_emo2}'")
            _voice_emo2 = analyze_voice_sentiment(
                raw2 if raw2 is not None else audio2,
                text_hint=user_input2, word_count=_wc2
            )
            _emotion2 = merge_emotion(_text_emo2, _voice_emo2)
            if _emotion2:
                print(f"💝 Triggering parent voice (follow-up): {_emotion2}")
                speak(random.choice(PARENT_VOICE_INTRO))
                if handle_parent_voice_trigger(_emotion2):
                    continue

            enhanced2 = (
                f"The child said (may have STT errors): \"{user_input2}\"\n"
                "Correct any name errors playfully with energy, use real facts wrapped in story magic, "
                "keep it fun and short for a young child."
            )
            messages.append({"role": "user", "content": enhanced2})
            f_reply = groq_chat(messages, temperature=0.85, max_tokens=150)
            if f_reply:
                f_reply = trim_to_sentence(f_reply, limit=600)
                messages.append({"role": "assistant", "content": f_reply})
                interrupt_flag.clear()
                inject_sounds_and_speak(f_reply)
            else:
                messages.pop()

        time.sleep(0.1)
