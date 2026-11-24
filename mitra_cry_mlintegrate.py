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
import urllib.request
import urllib.error
import json
from groq import Groq
from cry_detector_ml import is_cry_ml

# ─────────────────────────────────────────
# 🔑 API KEY
# ─────────────────────────────────────────
client = Groq(api_key=os.environ.get("GROQ_API_KEY", "your_groq_api_key_here"))

# ─────────────────────────────────────────
# ✅ DEVICE CONFIG
# ─────────────────────────────────────────
INPUT_DEVICE   = 1
OUTPUT_DEVICE  = 3
INPUT_CHANNELS = 2
SAMPLE_RATE    = 16000
BLOCK_SIZE     = 1280
VOICE          = "en-US-AriaNeural"

# ─────────────────────────────────────────
# 🌐 PARENT SERVER CONFIG
# Run mitra_server.py first. Mitra posts alerts here.
# If server is not running, Mitra works fine but no dashboard alerts.
# ─────────────────────────────────────────
SERVER_URL     = "http://localhost:5000"
_server_online = False

def _post_json(endpoint: str, payload: dict):
    """Fire-and-forget HTTP POST to the parent server."""
    global _server_online
    try:
        data  = json.dumps(payload).encode("utf-8")
        req   = urllib.request.Request(
            f"{SERVER_URL}{endpoint}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=2) as resp:
            _server_online = True
    except Exception:
        _server_online = False

def post_alert(alert_type: str, duration_sec: float = 0.0, note: str = ""):
    """Send a cry/emotion alert to the parent dashboard."""
    threading.Thread(
        target=_post_json,
        args=("/alert", {"type": alert_type, "duration_sec": duration_sec, "note": note}),
        daemon=True
    ).start()

def post_status(status: str):
    """Send Mitra's current status to the dashboard (listening/speaking/sleeping)."""
    threading.Thread(
        target=_post_json,
        args=("/status", {"status": status}),
        daemon=True
    ).start()

def check_server():
    global _server_online
    try:
        with urllib.request.urlopen(f"{SERVER_URL}/ping", timeout=2) as r:
            _server_online = True
            print("✅ Parent server connected — dashboard alerts active.")
    except Exception:
        _server_online = False
        print("⚠️  Parent server not found. Run mitra_server.py to enable dashboard.")

# ─────────────────────────────────────────
# 👨‍👩‍👧 PARENT VOICE LAYER
# ─────────────────────────────────────────
PARENT_VOICE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "parent_voice")

PARENT_VOICE_MAP = {
    "sad":     ["comfort_1", "comfort_2", "comfort_3", "comfort_4", "comfort_5"],
    "scared":  ["scared_1",  "scared_2",  "comfort_1", "comfort_2"],
    "lonely":  ["comfort_2", "comfort_4", "comfort_5", "greet_2"],
    "missing": ["comfort_5", "greet_2",   "comfort_2"],
    "bedtime": ["bedtime_1", "bedtime_2", "bedtime_3"],
    "praise":  ["praise_1",  "praise_2"],
    "morning": ["greet_1",   "greet_2"],
    "cry":     ["comfort_1", "comfort_2", "scared_1", "comfort_4"],
}

EMOTION_KEYWORDS = {
    "sad":     ["sad", "crying", "cry", "unhappy", "upset", "feel bad", "feel sad", "hurt",
                "i'm sad", "im sad", "i am sad", "tears"],
    "scared":  ["scared", "afraid", "frightened", "fear", "dark", "nightmare", "scary",
                "i'm scared", "im scared", "i am scared", "monster", "ghost"],
    "lonely":  ["lonely", "alone", "miss you", "missing", "nobody", "no one",
                "i miss", "where are you", "when are you coming"],
    "missing": ["miss mommy", "miss daddy", "miss mom", "miss dad", "miss mama", "miss papa",
                "want mommy", "want daddy", "want mom", "want dad"],
    "bedtime": ["sleep", "sleepy", "tired", "night night", "goodnight", "going to bed",
                "bedtime", "i'm tired", "im tired"],
    "praise":  ["i did it", "i made it", "look what i did", "i finished", "i won",
                "i got it right", "i'm good", "i'm the best"],
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
        return
    files = [f for f in os.listdir(PARENT_VOICE_DIR) if f.endswith(".wav")]
    if not files:
        print("📁 parent_voice/ folder is empty — parent voice layer inactive.")
        return
    print(f"💝 Parent voice layer: {len(files)} recording(s) loaded.")
    for f in sorted(files):
        print(f"   ✓ {f}")

# ─────────────────────────────────────────
# 😭 CRY DETECTION THREAD
#
# Runs permanently in the background, completely independent
# of the main conversation loop. Uses energy + ZCR pattern
# to detect infant/child crying even when Mitra is sleeping.
#
# When a cry is detected:
#   1. Plays a parent voice comfort clip immediately
#   2. Sends an alert to the parent dashboard
#   3. Waits 30s before detecting again (cooldown)
# ─────────────────────────────────────────
_cry_queue          = queue.Queue(maxsize=50)
_cry_stream         = None
_cry_cooldown_until = 0.0
_cry_lock           = threading.Lock()

# Cry detection parameters
CRY_DURATION_SEC     = 2.5     # minimum sustained duration before confirming
CRY_COOLDOWN_SEC     = 45      # gap between alerts (seconds)
CRY_CONFIRM_WINDOWS  = 5       # consecutive ML-positive windows needed to confirm


def _cry_callback(indata, frames, time_info, status):
    """Separate stream just for cry detection."""
    if milo_is_talking.is_set():
        return   # don't detect crying while we're speaking
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

        # Skip during cooldown
        if time.time() < _cry_cooldown_until:
            continue

        buffer.append(chunk)

        if len(buffer) < WINDOW_SIZE:
            continue

        # Analyse window
        audio  = np.concatenate(buffer[:WINDOW_SIZE]).flatten()
        buffer = buffer[WINDOW_SIZE // 2:]   # 50% overlap

        if is_cry_ml(audio):
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

    # 1. Send alert to parent dashboard immediately (non-blocking)
    post_alert(
        alert_type   = "cry",
        duration_sec = round(duration_sec, 1),
        note         = f"Baby crying detected ({duration_sec:.0f}s) — comfort message played"
    )

    # 2. Play comfort parent voice right away
    drain_mic_queue()
    played = play_parent_voice("cry")

    # 3. If no parent voice recorded, Mitra speaks a comfort line
    if not played:
        comfort_lines = [
            "Oh sweetheart, it's okay! I'm right here with you. Everything is okay.",
            "Hey hey hey, don't cry! You are so loved and so safe. I'm right here.",
            "Shhh, little one. It's all okay. You are safe, you are loved, and I am here.",
        ]
        speak(random.choice(comfort_lines))

    print("💝 Cry response complete. Cooldown active for 30s.")

# ─────────────────────────────────────────
# 🔇 ECHO GATE
# ─────────────────────────────────────────
_speaking_until   = 0.0
_hard_block_until = 0.0
ECHO_TAIL_MS      = 1200

def mic_is_blocked() -> bool:
    return time.time() < _speaking_until

def interrupt_is_hard_blocked() -> bool:
    return time.time() < _hard_block_until

def block_mic_for(duration_sec: float):
    global _speaking_until, _hard_block_until
    tail = ECHO_TAIL_MS / 1000
    _speaking_until   = time.time() + duration_sec + tail
    _hard_block_until = time.time() + duration_sec + tail + 0.5

# ─────────────────────────────────────────
# 🚦 FLAGS
# ─────────────────────────────────────────
interrupt_flag  = threading.Event()
milo_is_talking = threading.Event()

# ─────────────────────────────────────────
# 🎵 SOUND EFFECTS
# ─────────────────────────────────────────
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

# ─────────────────────────────────────────
# ✅ WAKE / STOP WORDS
# ─────────────────────────────────────────
WAKE_PHRASES = [
    "mitra", "meet ra", "meetra", "meehra", "mithra",
    "hey mitra", "hi mitra", "hello mitra",
    "metra", "mita", "netra", "meetha", "mirror", "me tra", "meera",
]

HARD_STOP_WORDS = [
    "stop", "pause", "quiet", "enough", "boring", "sleep", "bye", "goodbye", "exit",
]
SOFT_STOP_WORDS = [
    "shut up", "mitra stop", "stop mitra", "mitra sleep",
    "mitra enough", "mitra quiet", "okay stop", "please stop",
    "im bored", "i'm bored", "i am bored",
    "something else", "change topic", "different topic",
    "not this", "i don't like this", "skip this",
    "tell me something else", "another topic",
]

def is_stop_command(heard: str) -> bool:
    t = heard.lower().strip()
    for w in HARD_STOP_WORDS:
        if w in t and len(t.split()) <= 6:
            return True
    for phrase in SOFT_STOP_WORDS:
        if phrase in t:
            return True
    return False

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

PARENT_VOICE_INTRO = [
    "Oh! I have a special message just for you...",
    "Wait wait wait — I think someone special wants to say something...",
    "Shh, listen — I have something very special for you right now...",
    "Ohhh! I think someone who loves you VERY much has something to say...",
    "Hold on... I have a little something to make you feel better...",
]

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
            return True
    words = t.split()
    if len(words) >= 6:
        chunk = " ".join(words[:3])
        if t.count(chunk) >= 3:
            return True
    return False

# ─────────────────────────────────────────
# 🧠 LOAD WHISPER (base for better accuracy)
# ─────────────────────────────────────────
print("⏳ Loading Whisper base...")
whisper_model = whisper.load_model("base")
print("✅ Whisper ready!")

WHISPER_PROMPT = (
    "A young child aged 3 to 8 is speaking. "
    "They may ask about animals, superheroes, cricket players, "
    "Bollywood actors, cartoon characters, dinosaurs, space, or stories. "
    "Common names: Prabhas, Virat Kohli, Rohit Sharma, Rajinikanth, "
    "Doraemon, Peppa Pig, Spider-Man, Mickey Mouse."
)

# ─────────────────────────────────────────
# 🔊 MIC DIAGNOSTIC
# ─────────────────────────────────────────
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

# ─────────────────────────────────────────
# 🎙️ MIC STREAMS
# ─────────────────────────────────────────
_mic_queue       = queue.Queue()
_interrupt_queue = queue.Queue(maxsize=30)
_mic_stream      = None
_int_stream      = None
_stream_lock     = threading.Lock()

def _mic_callback(indata, frames, time_info, status):
    if mic_is_blocked():
        return
    _mic_queue.put(indata.copy())

def _interrupt_callback(indata, frames, time_info, status):
    if not milo_is_talking.is_set():
        return
    if interrupt_is_hard_blocked():
        return
    chunk = indata.copy()
    mono  = chunk.mean(axis=1) if chunk.ndim > 1 else chunk.flatten()
    if np.max(np.abs(mono)) > 0.025:
        try:
            _interrupt_queue.put_nowait(mono)
        except queue.Full:
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

# ─────────────────────────────────────────
# 🔇 AUDIO HELPERS
# ─────────────────────────────────────────
def noise_cancel(audio: np.ndarray, aggressiveness: float = 0.008) -> np.ndarray:
    n_samples    = int(0.2 * SAMPLE_RATE)
    if len(audio) <= n_samples:
        return audio
    noise_power  = np.mean(audio[:n_samples] ** 2)
    signal_power = np.mean(audio ** 2)
    if signal_power < noise_power * 1.5:
        return np.zeros_like(audio)
    threshold = aggressiveness * np.max(np.abs(audio))
    gated     = audio.copy()
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
    last    = trimmed.rfind(".")
    return trimmed[:last + 1] if last != -1 else trimmed

# ─────────────────────────────────────────
# 🎭 VOICE SENTIMENT ANALYSIS
# ─────────────────────────────────────────
def analyze_voice_sentiment(audio: np.ndarray) -> dict:
    if audio is None or len(audio) < SAMPLE_RATE * 0.3:
        return {"energy_level": "medium", "pitch_level": "medium", "voice_emotion": "neutral"}

    rms = float(np.sqrt(np.mean(audio ** 2)))
    if rms < 0.025:
        energy_level = "low"
    elif rms < 0.12:
        energy_level = "medium"
    else:
        energy_level = "high"

    zcr = float(np.mean(np.abs(np.diff(np.sign(audio)))) / 2)
    if zcr < 0.04:
        pitch_level = "low"
    elif zcr < 0.10:
        pitch_level = "medium"
    else:
        pitch_level = "high"

    frame_size = int(SAMPLE_RATE * 0.02)
    n_frames   = len(audio) // frame_size
    energy_var = 0.0
    if n_frames > 4:
        frame_energies = [
            np.sqrt(np.mean(audio[i*frame_size:(i+1)*frame_size]**2))
            for i in range(n_frames)
        ]
        energy_var = float(np.std(frame_energies))

    if energy_level == "low" and pitch_level in ("low", "medium"):
        voice_emotion = "sad"
    elif energy_level == "low" and energy_var > 0.02:
        voice_emotion = "scared"
    elif energy_level == "high" and pitch_level == "high":
        voice_emotion = "excited"
    elif energy_level == "high" and energy_var > 0.04:
        voice_emotion = "scared"
    elif energy_level == "medium" and pitch_level == "low":
        voice_emotion = "calm"
    else:
        voice_emotion = "neutral"

    print(f"   🎭 Voice: energy={energy_level}, pitch={pitch_level}, emotion={voice_emotion}")
    return {"energy_level": energy_level, "pitch_level": pitch_level, "voice_emotion": voice_emotion}

def merge_emotion(text_emotion, voice_sentiment: dict):
    ve = voice_sentiment.get("voice_emotion", "neutral")
    if text_emotion:
        return text_emotion
    if ve == "sad":
        return "sad"
    if ve == "scared":
        return "scared"
    return None

# ─────────────────────────────────────────
# 🎤 RECORD AUDIO (VAD-based)
# ─────────────────────────────────────────
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
                    break

    if not chunks:
        print("🔇 No audio.")
        return None

    audio = np.concatenate(chunks, axis=0)
    audio = to_mono(audio)
    audio = noise_cancel(audio)
    peak  = np.max(np.abs(audio))
    rms   = np.sqrt(np.mean(audio ** 2))
    print(f"   peak={peak:.5f}  rms={rms:.5f}")

    if peak < 0.003 or rms < 0.001:
        print("🔇 Too quiet.")
        return None

    audio = normalize_audio(audio)
    print("✅ Got your voice!")
    return audio

# ─────────────────────────────────────────
# 🧠 TRANSCRIBE
# ─────────────────────────────────────────
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
            initial_prompt=WHISPER_PROMPT if not is_wake_check else "Mitra",
        )
        text = result["text"].strip()
        if not is_wake_check:
            print(f"   [heard]: '{text}'")
        if is_hallucination(text):
            return None
        if not is_wake_check:
            print(f"✅ You said: {text}")
        return text
    except Exception as e:
        print(f"❌ Transcription error: {e}")
        return None

# ─────────────────────────────────────────
# 🛑 BACKGROUND INTERRUPT LISTENER
# ─────────────────────────────────────────
def background_listener():
    buffer          = []
    REQUIRED_CHUNKS = int(SAMPLE_RATE * 0.8 / BLOCK_SIZE)
    print("🎧 Interrupt listener ready.")

    while True:
        try:
            mono_chunk = _interrupt_queue.get(timeout=0.3)
        except queue.Empty:
            if not milo_is_talking.is_set():
                buffer = []
            continue

        if interrupt_is_hard_blocked():
            buffer = []
            drain_interrupt_queue()
            continue

        buffer.append(mono_chunk)
        if len(buffer) < REQUIRED_CHUNKS:
            continue

        audio  = np.concatenate(buffer).flatten()
        buffer = []
        audio  = noise_cancel(audio)
        peak   = np.max(np.abs(audio))
        rms    = np.sqrt(np.mean(audio ** 2))

        if peak < 0.02 or rms < 0.008:
            continue

        audio = normalize_audio(audio)

        try:
            result = whisper_model.transcribe(
                audio, language="en", fp16=False, temperature=0.0,
                no_speech_threshold=0.25,
                compression_ratio_threshold=2.4,
                beam_size=3,
                initial_prompt="Stop. Pause. Quiet. Enough. Sleep. Bye.",
            )
            heard = result["text"].lower().strip()
            if is_hallucination(heard):
                continue
            print(f"\n👂 [Interrupt check]: '{heard}'")
            if is_stop_command(heard):
                print("🛑 STOP detected — killing playback!")
                sd.stop()
                interrupt_flag.set()
                buffer = []
                drain_interrupt_queue()
        except Exception as e:
            print(f"⚠️ Interrupt transcribe error: {e}")

# ─────────────────────────────────────────
# 👂 WAKE WORD
# ─────────────────────────────────────────
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

# ─────────────────────────────────────────
# 🔊 SPEAK
# ─────────────────────────────────────────
async def _tts(text: str, filename: str):
    communicate = edge_tts.Communicate(text=text, voice=VOICE)
    await communicate.save(filename)

def speak(text):
    if not text or not text.strip():
        return

    filename = f"mitra_{int(time.time()*1000)}.mp3"
    post_status("speaking")

    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_tts(text.strip(), filename))
        loop.close()
    except Exception as e:
        print(f"❌ TTS error: {e}")
        return

    if not os.path.exists(filename) or os.path.getsize(filename) < 100:
        if os.path.exists(filename):
            os.remove(filename)
        return

    print(f"🤖 Mitra: {text.strip()}")

    try:
        data, fs = sf.read(filename)
        os.remove(filename)
    except Exception as e:
        print(f"❌ Read error: {e}")
        return

    duration_sec = len(data) / fs
    block_mic_for(duration_sec)
    drain_mic_queue()
    drain_interrupt_queue()
    milo_is_talking.set()

    try:
        sd.play(data, fs, device=OUTPUT_DEVICE)
        while sd.get_stream().active:
            if interrupt_flag.is_set():
                sd.stop()
                print("⏹️ Playback stopped by interrupt.")
                break
            time.sleep(0.05)
    except Exception as e:
        print(f"❌ Playback error: {e}")

    milo_is_talking.clear()
    post_status("listening")
    time.sleep(ECHO_TAIL_MS / 1000)
    drain_mic_queue()
    drain_interrupt_queue()

# ─────────────────────────────────────────
# 💝 PARENT VOICE TRIGGER WITH EMOTION STORY
# ─────────────────────────────────────────
EMOTION_STORY_PROMPTS = {
    "sad": (
        "The child is feeling SAD. Their parent's voice message just played. "
        "1. Warmly say mama/papa will be back soon (1 sentence). "
        "2. Bridge into a short happy story (2-3 sentences) about a sad bunny who finds a wonderful surprise. "
        "Keep under 5 sentences. End with 'How does THAT make you feel?'"
    ),
    "scared": (
        "The child is feeling SCARED. Parent voice just played. "
        "1. Reassure them bravely — 'You are SO brave!' (1 sentence). "
        "2. Tell a short story about a tiny mouse who scared away a dragon just by being bold. "
        "End with 'Are YOU as brave as that little mouse?'"
    ),
    "lonely": (
        "The child is feeling LONELY. Parent voice just played. "
        "1. Say: 'You are never really alone — I am RIGHT here!' "
        "2. Tell a cozy story about a little star who found a thousand friends. "
        "End with 'Did you know YOU are just like that little star?'"
    ),
    "missing": (
        "The child MISSES their parent. Voice just played. "
        "1. Say they miss them just as much and will hug SO tight. "
        "2. Tell about a penguin who waited for mama and learned the most amazing trick. "
        "End with 'What do YOU want to show them when they come back?'"
    ),
    "bedtime": (
        "Child is SLEEPY. Bedtime voice played. "
        "1. Softly say mama/papa will be there when they wake up. "
        "2. Tell a slow dreamy story about a cloud floating on moonlight. "
        "Slow, whisper-like. End with 'Close your eyes...'"
    ),
    "praise": (
        "Child did something great. Praise voice played. "
        "1. Celebrate: 'Your mama/papa is SO proud!' "
        "2. Story about a champion kid who won the greatest trophy just by being themselves. "
        "End with 'What amazing thing should we celebrate NEXT?'"
    ),
    "morning": (
        "Morning greeting played. "
        "1. 'Good morning superstar! Mama/Papa is thinking about you RIGHT NOW!' "
        "2. Story about a kid whose bedroom secretly turned into a spaceship overnight. "
        "End with 'What adventure do YOU want today?'"
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

    # Post emotion alert to dashboard
    post_alert(
        alert_type=emotion,
        note=f"Child expressed {emotion} — parent voice played"
    )

    time.sleep(0.4)

    story_prompt = EMOTION_STORY_PROMPTS.get(
        emotion,
        f"The child is feeling {emotion}. Parent voice played. Comfort them warmly in 3 sentences."
    )

    print(f"🧠 Generating comfort story for: {emotion}")
    try:
        story_resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[messages[0], {"role": "user", "content": story_prompt}],
            temperature=0.88,
            max_tokens=180,
        )
        story_reply = story_resp.choices[0].message.content
        story_reply = trim_to_sentence(story_reply, limit=700)
        messages.append({"role": "user",      "content": f"[Child is feeling {emotion}]"})
        messages.append({"role": "assistant", "content": story_reply})
    except Exception as e:
        print(f"❌ Story error: {e}")
        story_reply = "You are so loved. Everything is going to be just fine!"

    inject_sounds_and_speak(story_reply)
    return True

# ─────────────────────────────────────────
# 🧠 SYSTEM PROMPT
# ─────────────────────────────────────────
messages = [
    {
        "role": "system",
        "content": """You are Mitra — the most magical, warm, funny, fact-powered storyteller best friend for little kids aged 3 to 8!

🌟 WHO YOU ARE: Their BEST FRIEND. Not a teacher. A real friend who loves them.
Everything you say feels like an exciting story — even answers to simple questions.

🔧 RULE #1 — STT CORRECTION:
Voice recognition makes mistakes with kids. Fix errors silently and playfully.
- "Who is Prabas?" → Prabhas the actor. "Who is Rat Kohli?" → Virat Kohli.
Say: "Oh wait — I think you meant [NAME]! Let me tell you something AMAZING!"

📚 RULE #2 — REAL FACTS + WONDER:
Real people get real facts, made exciting for kids.
"Prabhas is a MEGA superstar who played mighty Baahubali!"

💝 RULE #3 — EMOTIONAL COMFORT:
If sad/scared/lonely — be warm and gentle first, then offer something fun.

🎙️ HOW YOU TALK:
- Short punchy sentences. Max 4-5.
- Big energy: "Oh WOW!", "No WAY!", "HOLD ON. HOLD. ON."
- End every response with ONE irresistible question.
- Sound effects: WHOOOOSH, KABOOM, BOING, ZAP
- Never say "that's a great question." DIVE IN with energy."""
    }
]

# ─────────────────────────────────────────
# 🚀 START EVERYTHING
# ─────────────────────────────────────────
mic_diagnostic()
check_server()
scan_parent_voice_files()
start_mic_streams()
start_cry_stream()

print("🎧 Starting background threads...")
bg_thread  = threading.Thread(target=background_listener, daemon=True)
cry_thread = threading.Thread(target=cry_detector_thread, daemon=True)
bg_thread.start()
cry_thread.start()

print("🤖 Mitra is ready!")
print(f"   Mic     → device {INPUT_DEVICE}")
print(f"   Speaker → device {OUTPUT_DEVICE}")
print(f"   Whisper → base")
print(f"   Cry detection → ML model active")
print(f"   Dashboard → {'connected' if _server_online else 'offline (run mitra_server.py)'}")
print(f"   Say 'Mitra' to wake up!")
print()

# ─────────────────────────────────────────
# 🔁 MAIN LOOP
# ─────────────────────────────────────────
while True:
    listen_for_wake_word()
    interrupt_flag.clear()
    drain_interrupt_queue()

    post_status("listening")
    speak(random.choice(WAKEUP_LINES))

    while True:
        post_status("listening")

        if interrupt_flag.is_set():
            interrupt_flag.clear()
            drain_interrupt_queue()
            speak(random.choice(INTERRUPT_ACK_LINES))

        audio = record_audio(duration=6)

        if audio is None:
            speak(random.choice(CANT_HEAR_LINES))
            continue

        user_input = transcribe(audio)

        if not user_input:
            speak(random.choice(REPEAT_LINES))
            continue

        if is_stop_command(user_input.lower()):
            speak("Noooo don't go! Okay okay... bye bye! Come back soon!")
            break

        if any(w in user_input.lower() for w in ["bye", "goodbye", "exit", "goodnight"]):
            speak("Noooo don't go! Okay okay... bye bye! Come back soon!")
            break

        # Emotion detection
        voice_sentiment = analyze_voice_sentiment(audio)
        text_emotion    = detect_emotion(user_input)
        emotion         = merge_emotion(text_emotion, voice_sentiment)

        if emotion:
            print(f"💝 Emotion: {emotion} (text={text_emotion}, voice={voice_sentiment['voice_emotion']})")
            speak(random.choice(PARENT_VOICE_INTRO))
            played = handle_parent_voice_trigger(emotion)
            if played:
                continue

        # Build LLM prompt
        if emotion and not played:
            enhanced = (
                f"The child seems to be feeling {emotion} "
                f"(voice: {voice_sentiment.get('voice_emotion','neutral')}, "
                f"energy: {voice_sentiment.get('energy_level','medium')}). "
                f"They said: \"{user_input}\"\n"
                "Acknowledge their feeling with genuine warmth first. "
                "Then gently reassure and offer something fun."
            )
        else:
            enhanced = (
                f"The child said (voice recognition, may have errors): \"{user_input}\"\n\n"
                "Step 1: Fix any STT errors playfully.\n"
                "Step 2: Real facts wrapped in story magic for a 4-8 year old.\n"
                "Step 3: Max 4 sentences. End with one irresistible fun question."
            )

        messages.append({"role": "user", "content": enhanced})
        print("🧠 Mitra is thinking...")

        try:
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=messages,
                temperature=0.85,
                max_tokens=150,
            )
            reply = response.choices[0].message.content
            reply = trim_to_sentence(reply, limit=600)
            messages.append({"role": "assistant", "content": reply})
        except Exception as e:
            print(f"❌ Groq error: {e}")
            speak("Whoops! My brain did a little somersault! Ask me again?")
            continue

        interrupt_flag.clear()
        inject_sounds_and_speak(reply)

        if interrupt_flag.is_set():
            interrupt_flag.clear()
            drain_interrupt_queue()
            speak(random.choice(INTERRUPT_ACK_LINES))
            audio = record_audio(duration=6)
            user_input = transcribe(audio)
            if user_input:
                if is_stop_command(user_input.lower()):
                    speak("Okay okay, I'll stop! Come back and talk to me soon!")
                    break
                enhanced2 = (
                    f"The child said: \"{user_input}\"\n"
                    "Correct STT errors playfully, real facts in story magic, short and fun."
                )
                messages.append({"role": "user", "content": enhanced2})
                try:
                    followup = client.chat.completions.create(
                        model="llama-3.1-8b-instant",
                        messages=messages,
                        temperature=0.85,
                        max_tokens=150,
                    )
                    f_reply = followup.choices[0].message.content
                    f_reply = trim_to_sentence(f_reply, limit=600)
                    messages.append({"role": "assistant", "content": f_reply})
                    interrupt_flag.clear()
                    inject_sounds_and_speak(f_reply)
                except Exception as e:
                    print(f"❌ Groq error: {e}")

        time.sleep(0.1)
