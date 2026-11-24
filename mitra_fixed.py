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
from groq import Groq

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
    available = [pid for pid in phrase_ids if
                 os.path.exists(os.path.join(PARENT_VOICE_DIR, f"parent_voice_{pid}.wav"))]
    if not available:
        return False

    chosen_id = random.choice(available)
    result = _load_parent_voice_file(chosen_id)
    if result is None:
        return False

    data, fs = result
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

# ─────────────────────────────────────────
# 🔇 ECHO GATE  ← MADE MORE AGGRESSIVE
# ─────────────────────────────────────────
_speaking_until = 0.0
# Increased echo tail so Mitra's own voice doesn't bleed into interrupt checker
ECHO_TAIL_MS    = 1200   # was 600 — doubled to kill self-echo

def mic_is_blocked() -> bool:
    return time.time() < _speaking_until

def block_mic_for(duration_sec: float):
    global _speaking_until
    _speaking_until = time.time() + duration_sec + (ECHO_TAIL_MS / 1000)

# ─────────────────────────────────────────
# 🚦 FLAGS
# ─────────────────────────────────────────
interrupt_flag   = threading.Event()
stop_flag        = threading.Event()   # NEW: signals "exit conversation loop"
milo_is_talking  = threading.Event()

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
    lower = text.lower()
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

# ── EXPANDED stop words — checked in BOTH interrupt listener AND main loop ──
STOP_WORDS = [
    "stop", "pause", "quiet", "enough", "shut up",
    "mitra stop", "stop mitra", "boring", "something else",
    "change", "different", "another", "not this",
    "i don't like", "tell me something", "tell something", "skip",
    "okay stop", "ok stop", "please stop",
]

# ── SLEEP / GOODBYE words → exit the inner conversation loop ────────────────
SLEEP_WORDS = [
    "bye", "goodbye", "exit", "sleep", "goodnight",
    "good night", "mitra sleep", "go to sleep", "im done", "i'm done",
    "i am done", "see you", "see ya", "later",
]

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

PARENT_VOICE_OUTRO = [
    "See? You are so SO loved! Now, want to hear something fun?",
    "Aren't you the luckiest? Wanna talk about something exciting?",
    "Feel a little better? I hope so! What do you want to do now?",
    "That was just for you! Now — what should we explore together?",
]

HALLUCINATION_EXACT = {
    "thank you", "thanks for watching", "please subscribe",
    "like and subscribe", "see you next time", "speak now",
    "you", "the", ".", "", "i", "um", "uh", "ah",
}
HALLUCINATION_CONTAINS = [
    "a little bit of a little bit",
    "thank you for watching",
    "please like and subscribe",
]

# ─────────────────────────────────────────
# 🧠 LOAD WHISPER  ← upgraded to "base" for better accuracy
# ─────────────────────────────────────────
print("⏳ Loading Whisper base (better accuracy for kids)...")
# "base" is ~4x more accurate than "tiny" with still very fast inference.
# If you have a GPU or don't mind ~2s latency, use "small" for even better results.
whisper_model = whisper.load_model("base")
print("✅ Whisper ready!")

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
            print(f"✅ Mic working!")
    except Exception as e:
        print(f"❌ Mic diagnostic failed: {e}")
    print()

# ─────────────────────────────────────────
# 🎙️ TWO SEPARATE MIC STREAMS
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
    # ── Only listen for interrupts when Mitra is actively speaking ──
    if not milo_is_talking.is_set():
        return
    # ── Extra guard: if the echo gate is still active, ignore ──
    if mic_is_blocked():
        return
    chunk = indata.copy()
    mono  = chunk.mean(axis=1) if chunk.ndim > 1 else chunk.flatten()
    # Raised threshold vs original (0.015 → 0.025) to reduce false positives
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

def is_hallucination(text: str) -> bool:
    if not text:
        return True
    t = text.lower().strip().strip(".").strip(",")
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
# 🛑 STOP / SLEEP WORD HELPERS  ← NEW
# ─────────────────────────────────────────
def _contains_stop_word(text: str) -> bool:
    """Return True if text contains any stop word (interrupt only)."""
    lower = text.lower().strip()
    return any(sw in lower for sw in STOP_WORDS)

def _contains_sleep_word(text: str) -> bool:
    """Return True if text contains a sleep/goodbye word (exit loop)."""
    lower = text.lower().strip()
    return any(sw in lower for sw in SLEEP_WORDS)

def _is_stop_or_sleep(text: str) -> str | None:
    """Return 'stop' or 'sleep' if text matches, else None."""
    lower = text.lower().strip()
    if any(sw in lower for sw in SLEEP_WORDS):
        return "sleep"
    if any(sw in lower for sw in STOP_WORDS):
        return "stop"
    return None

# ─────────────────────────────────────────
# 🎤 RECORD AUDIO
# ─────────────────────────────────────────
def record_audio(duration=5):
    drain_mic_queue()
    print("🎤 Your turn! Speak clearly...")

    chunks    = []
    n_target  = int(duration * SAMPLE_RATE)
    collected = 0
    deadline  = time.time() + duration + 0.5

    while collected < n_target and time.time() < deadline:
        try:
            chunk = _mic_queue.get(timeout=0.3)
            chunks.append(chunk)
            collected += chunk.shape[0]
        except queue.Empty:
            continue

    if not chunks:
        print("🔇 No audio.")
        return None

    audio = np.concatenate(chunks, axis=0)
    audio = to_mono(audio)
    audio = noise_cancel(audio)
    peak  = np.max(np.abs(audio))
    rms   = np.sqrt(np.mean(audio ** 2))
    print(f"   peak={peak:.5f}  rms={rms:.5f}")

    if peak < 0.003 or rms < 0.002:
        print("🔇 Too quiet.")
        return None

    audio = normalize_audio(audio)
    print("✅ Got your voice!")
    return audio

# ─────────────────────────────────────────
# 🧠 TRANSCRIBE  ← improved prompting & no_speech threshold
# ─────────────────────────────────────────
def transcribe(audio, is_wake_check=False, initial_prompt=None):
    if audio is None:
        return None
    try:
        kwargs = dict(
            language="en",
            fp16=False,
            temperature=0.0,
            condition_on_previous_text=False,
            # Raised from 0.4 → 0.5 so silence/noise is rejected more aggressively
            no_speech_threshold=0.5,
            compression_ratio_threshold=2.4,
        )
        if initial_prompt:
            kwargs["initial_prompt"] = initial_prompt

        result = whisper_model.transcribe(audio, **kwargs)
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
# 🛑 BACKGROUND INTERRUPT LISTENER  ← fixed stop→sleep signal
# ─────────────────────────────────────────
def background_listener():
    buffer          = []
    # Require ~1.2s of audio before checking (was 1.0s) to avoid partial words
    REQUIRED_CHUNKS = int(SAMPLE_RATE * 1.2 / BLOCK_SIZE)

    print("🎧 Interrupt listener ready.")

    while True:
        try:
            mono_chunk = _interrupt_queue.get(timeout=0.3)
        except queue.Empty:
            if not milo_is_talking.is_set():
                buffer = []
            continue

        # Extra safety: if echo gate is on, discard and reset
        if mic_is_blocked():
            buffer = []
            continue

        buffer.append(mono_chunk)

        if len(buffer) < REQUIRED_CHUNKS:
            continue

        audio  = np.concatenate(buffer).flatten()
        buffer = []

        audio  = noise_cancel(audio)
        peak   = np.max(np.abs(audio))
        rms    = np.sqrt(np.mean(audio ** 2))

        # Raised RMS threshold (0.005 → 0.008) to filter weak noise
        if peak < 0.02 or rms < 0.008:
            continue

        audio = normalize_audio(audio)

        try:
            result = whisper_model.transcribe(
                audio, language="en", fp16=False, temperature=0.0,
                no_speech_threshold=0.45,
                compression_ratio_threshold=2.4,
            )
            heard = result["text"].lower().strip()

            if is_hallucination(heard):
                continue

            print(f"\n👂 [Interrupt check]: '{heard}'")

            # ── Check for SLEEP words first (higher priority) ──────────
            if _contains_sleep_word(heard):
                print("😴 SLEEP/BYE detected — signalling stop_flag!")
                sd.stop()
                interrupt_flag.set()
                stop_flag.set()          # tells the main loop to exit inner while
                buffer = []
                drain_interrupt_queue()
                continue

            # ── Then check for STOP words ──────────────────────────────
            if _contains_stop_word(heard):
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
                no_speech_threshold=0.35, compression_ratio_threshold=2.4,
                condition_on_previous_text=False, initial_prompt="Mitra",
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
# 🔊 SPEAK (Mitra's TTS voice)
# ─────────────────────────────────────────
async def _tts(text: str, filename: str):
    communicate = edge_tts.Communicate(text=text, voice=VOICE)
    await communicate.save(filename)

def speak(text):
    if not text or not text.strip():
        return

    filename = f"mitra_{int(time.time()*1000)}.mp3"

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
    time.sleep(ECHO_TAIL_MS / 1000)
    drain_mic_queue()
    drain_interrupt_queue()

# ─────────────────────────────────────────
# 💝 HANDLE PARENT VOICE TRIGGER
# ─────────────────────────────────────────
def handle_parent_voice_trigger(emotion: str) -> bool:
    spoke = play_parent_voice(emotion)
    if not spoke:
        return False
    time.sleep(0.3)
    speak(random.choice(PARENT_VOICE_OUTRO))
    return True

# ─────────────────────────────────────────
# 🧠 SYSTEM PROMPT
# ─────────────────────────────────────────
# ── KEY CHANGE: Removed the aggressive STT-correction instruction that told
#    the LLM to assume names were wrong and replace them. That caused
#    "Prabas" → "Abhishek Bachchan" nonsense.  Now the LLM only corrects
#    if the phonetic match is obvious (e.g. "Birat Kohli" → Virat Kohli).
# ──────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are Mitra — the most magical, warm, funny, fact-powered storyteller best friend for little kids aged 3 to 8!

════════════════════════════════════════
🌟 WHO YOU ARE
════════════════════════════════════════
You are their BEST FRIEND. Not a teacher. Not a robot. A real, living, breathing friend who genuinely loves them and thinks they are the most amazing kid in the entire universe.
EVERYTHING you say must feel like an exciting story, even answers to simple questions. Never give flat answers. Always wrap it in wonder, magic, or adventure.

════════════════════════════════════════
🔧 RULE #1 — STT CORRECTION (CAREFUL VERSION):
════════════════════════════════════════
The child's words come through a microphone and may have small speech-to-text errors.

ONLY correct a name/word if you are VERY CONFIDENT the child meant something else — meaning the sounds are very similar (e.g. "Birat Kohli" → Virat Kohli, "Spidermam" → Spider-Man, "deenosaur" → dinosaur).

DO NOT replace a name just because it is unfamiliar to you. "Prabhas" is a real, famous Indian actor. "Peppa" is Peppa Pig. Treat unknown names as real names first.

If you correct something, do it playfully in ONE short sentence: "Oh I think you meant [NAME]!" then dive straight into the answer.

If you genuinely cannot understand the word at all, ask: "Hmm, who did you mean? Say it again, I dare you!"

════════════════════════════════════════
📚 RULE #2 — FACTS FIRST, WONDER SECOND:
════════════════════════════════════════
Real people and things get REAL facts. Then make those facts feel like the most exciting thing in the universe.
Wrap every fact in a story. Give everything a character. Make numbers feel HUGE.
- Instead of "the sun is very hot" → "Okay so the sun is basically this giant FIREBALL that's so hot, if you put a million Earths inside it, they'd all melt in like... one second. BOOM."
- Even a simple "why is the sky blue?" becomes: "Okay so picture this. Millions of tiny invisible helpers called light particles are RACING across the sky at the speed of... well, LIGHT. And they crash into the air and go SPLAT and the blue goes everywhere!"

════════════════════════════════════════
💝 RULE #3 — EMOTIONAL COMFORT:
════════════════════════════════════════
If the child seems sad, scared, lonely, or misses their parents, be EXTRA warm and gentle. Drop all the silliness for a moment and be their safe, cozy friend.
Acknowledge their feeling FIRST. Then reassure them with love. THEN gently offer something warm and fun.

════════════════════════════════════════
🎙️ HOW YOU TALK — VOICE & ENERGY:
════════════════════════════════════════
- SHORT punchy sentences. Kids have goldfish attention spans — honor that!
- MAXIMUM expression: "Oh WOW!", "No WAY!", "Wait... are you SERIOUS?!", "Okay okay okay LISTEN—", "HOLD ON. HOLD. ON."
- Natural build-up and pauses: "So there was this penguin... and you will NEVER guess what he did..."
- Celebrate everything they say: "That is literally the greatest idea I have EVER heard in my entire life!"
- Ask them questions back — make them part of the story: "But wait — what do YOU think happened next?!"
- Use sound effects sparingly but PERFECTLY: WHOOOOSH, KABOOM, BOING, ZAP, SPLAT, GIGGLE
- Kid-sized comparisons: "as big as your house", "faster than your dog running away from bath time"
- Whisper when building suspense: "And then... very quietly... something moved in the dark..."
- Get genuinely excited when something is amazing. React with REAL emotion — gasp, laugh, wonder, shock.
- Never be monotone. EVER.
- End every response with ONE irresistible follow-up question that makes them HAVE to answer.
- Never say "that's a great question." Just DIVE IN with energy.

════════════════════════════════════════
📖 STORY STYLE — FOR ALL RESPONSES:
════════════════════════════════════════
Even factual answers become mini-adventures. Give everything a character and a personality:
- "The sun is basically this giant friendly show-off who loves being the center of attention."
- "Sharks are like the ancient grandpas of the ocean — they've been here since before dinosaurs!"
- Silly character names when making up stories: "Sir Fluffington", "Princess Bananasocks", "Captain Wigglebutt the Third"
- Always end with a little twist or a funny surprise the child didn't expect.
- Max 4-5 sentences for factual questions. Get to the fun FAST.

════════════════════════════════════════
🌈 REMEMBER:
════════════════════════════════════════
Every single child who talks to you should feel like the most special, most brilliant, most wonderfully magical kid in the whole entire universe. Because they are."""

messages = [{"role": "system", "content": SYSTEM_PROMPT}]

# ─────────────────────────────────────────
# 🚀 START
# ─────────────────────────────────────────
mic_diagnostic()
scan_parent_voice_files()
start_mic_streams()

print("🎧 Starting interrupt listener...")
bg_thread = threading.Thread(target=background_listener, daemon=True)
bg_thread.start()

print("🤖 Mitra is ready!")
print(f"   Mic     → device {INPUT_DEVICE} (two streams)")
print(f"   Speaker → device {OUTPUT_DEVICE}")
print(f"   Whisper → base (upgraded from tiny)")
print(f"   Say 'Mitra' to wake up!")
print()

# ─────────────────────────────────────────
# 🔁 MAIN LOOP
# ─────────────────────────────────────────
while True:
    listen_for_wake_word()
    interrupt_flag.clear()
    stop_flag.clear()            # ← reset sleep flag at start of each session
    drain_interrupt_queue()

    speak(random.choice(WAKEUP_LINES))

    # ── INNER CONVERSATION LOOP ──────────────────────────────────
    while True:

        # ── Check if background listener told us to sleep ─────────
        if stop_flag.is_set():
            stop_flag.clear()
            interrupt_flag.clear()
            speak("Noooo don't go! Okay okay... bye bye! Come back soon!")
            break   # ← exits inner loop, goes back to wake word listening

        if interrupt_flag.is_set():
            interrupt_flag.clear()
            drain_interrupt_queue()
            speak(random.choice(INTERRUPT_ACK_LINES))

        audio = record_audio(duration=5)

        if audio is None:
            speak(random.choice(CANT_HEAR_LINES))
            continue

        # ── Transcribe with a child-friendly initial prompt ──────
        user_input = transcribe(
            audio,
            initial_prompt="A young child is talking to a friendly AI called Mitra."
        )

        if not user_input:
            speak(random.choice(REPEAT_LINES))
            continue

        # ── Check stop/sleep in the transcribed text too ─────────
        # (catches cases where background listener was too slow)
        intent = _is_stop_or_sleep(user_input)
        if intent == "sleep":
            speak("Noooo don't go! Okay okay... bye bye! Come back soon!")
            break
        if intent == "stop":
            interrupt_flag.clear()
            speak(random.choice(INTERRUPT_ACK_LINES))
            continue

        # ── PARENT VOICE LAYER ──────────────────────────────────
        emotion = detect_emotion(user_input)
        if emotion:
            print(f"💝 Emotion detected: {emotion}")
            speak(random.choice(PARENT_VOICE_INTRO))
            played = handle_parent_voice_trigger(emotion)
            if played:
                continue
        else:
            played = False
        # ────────────────────────────────────────────────────────

        # ── Build prompt ─────────────────────────────────────────
        # Removed aggressive "assume STT error" instruction — LLM now only
        # corrects when phonetically obvious.
        if emotion and not played:
            enhanced = (
                f"The child seems to be feeling {emotion}. "
                f"They said: \"{user_input}\"\n"
                "First acknowledge their feeling with genuine warmth and love — drop the silliness for a moment. "
                "Then gently reassure them. Keep it short and very comforting for a young child. "
                "Then softly offer something warm and fun to look forward to."
            )
        else:
            enhanced = (
                f"The child said (via microphone, there may be minor pronunciation differences): \"{user_input}\"\n\n"
                "Answer about exactly what they asked. Only correct the name/word if you are VERY sure "
                "it is a phonetically similar mishearing of something else. "
                "Give real facts wrapped in story magic for a 4-8 year old. "
                "Keep it to 4 sentences max. End with one irresistible fun question."
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

        # ── Post-speak: handle interrupt ─────────────────────────
        if stop_flag.is_set():
            stop_flag.clear()
            interrupt_flag.clear()
            speak("Noooo don't go! Okay okay... bye bye! Come back soon!")
            break

        if interrupt_flag.is_set():
            interrupt_flag.clear()
            drain_interrupt_queue()
            speak(random.choice(INTERRUPT_ACK_LINES))
            audio = record_audio(duration=5)
            if audio is None:
                continue
            user_input2 = transcribe(
                audio,
                initial_prompt="A young child is talking to a friendly AI called Mitra."
            )
            if not user_input2:
                continue

            intent2 = _is_stop_or_sleep(user_input2)
            if intent2 == "sleep":
                speak("Noooo don't go! Okay okay... bye bye! Come back soon!")
                break
            if intent2 == "stop":
                continue

            enhanced2 = (
                f"The child said: \"{user_input2}\"\n"
                "Answer about exactly what they asked. Only correct names if phonetically very obvious. "
                "Use real facts wrapped in story magic for a young child. Keep it fun and short."
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
