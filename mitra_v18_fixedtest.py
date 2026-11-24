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
import subprocess
import tempfile
from groq import Groq

# ─────────────────────────────────────────
# 🌐 INTERNET / CONNECTIVITY CHECK
# ─────────────────────────────────────────
def check_internet(host="8.8.8.8", port=53, timeout=2) -> bool:
    """Quick DNS-level check — doesn't need HTTP."""
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except Exception:
        return False

_internet_ok = check_internet()

# ─────────────────────────────────────────
# 🔇 OFFLINE TTS FALLBACK  (pyttsx3)
#
# Used ONLY when edge_tts fails (no internet).
# pyttsx3 works 100% offline via OS speech engine.
# Install once:  pip install pyttsx3
# ─────────────────────────────────────────
_pyttsx3_engine = None
_pyttsx3_lock   = threading.Lock()

def _get_pyttsx3():
    global _pyttsx3_engine
    if _pyttsx3_engine is not None:
        return _pyttsx3_engine
    try:
        import pyttsx3
        eng = pyttsx3.init()
        eng.setProperty("rate",   165)   # slightly slower — kid-friendly
        eng.setProperty("volume", 1.0)
        # Prefer a female voice if available
        voices = eng.getProperty("voices")
        for v in voices:
            if "female" in v.name.lower() or "zira" in v.name.lower() or "hazel" in v.name.lower():
                eng.setProperty("voice", v.id)
                break
        _pyttsx3_engine = eng
        print("✅ pyttsx3 offline TTS ready.")
        return eng
    except Exception as e:
        print(f"⚠️ pyttsx3 not available: {e}  →  pip install pyttsx3")
        return None

def speak_offline(text: str):
    """
    Speak using pyttsx3 (offline). Blocks until done.

    ROOT CAUSE OF SECOND-CALL FAILURE:
    pyttsx3 engines enter a broken state after runAndWait() if
    save_to_file() + runAndWait() was already called — the internal
    driver loop is marked as already stopped. Reinitialising a
    fresh engine every call costs ~80ms but is 100% reliable.
    We do NOT cache the engine for this reason.
    """
    with _pyttsx3_lock:
        tmp = tempfile.mktemp(suffix=".wav")
        try:
            import pyttsx3
            # Fresh engine every single call — prevents runAndWait state corruption
            eng = pyttsx3.init()
            eng.setProperty("rate",   160)
            eng.setProperty("volume", 1.0)
            voices = eng.getProperty("voices")
            for v in voices:
                vname = v.name.lower()
                if any(k in vname for k in ("female", "zira", "hazel", "susan", "catherine")):
                    eng.setProperty("voice", v.id)
                    break
            # Save to WAV then play via sounddevice — most reliable
            eng.save_to_file(text.strip(), tmp)
            eng.runAndWait()
            eng.stop()
            if os.path.exists(tmp) and os.path.getsize(tmp) > 200:
                data, fs = sf.read(tmp)
                os.remove(tmp)
                # Print right when audio starts — in sync
                print(f"🤖 [offline TTS] Mitra: {text.strip()}")
                sd.play(data, fs, device=OUTPUT_DEVICE)
                sd.wait()
                return
        except Exception as e:
            print(f"⚠️ Offline TTS (wav path) error: {e}")
            if os.path.exists(tmp):
                os.remove(tmp)
        # Last-resort: direct say() on a fresh engine
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

# ─────────────────────────────────────────
# 🔑 API KEY + GROQ CLIENT WITH RETRY
# ─────────────────────────────────────────
_GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "your_groq_api_key_here")

def _make_groq_client():
    return Groq(api_key=_GROQ_API_KEY)

client = _make_groq_client()

def groq_chat(messages_list: list, temperature=0.85, max_tokens=150, retries=2) -> str | None:
    """
    Wrapper around Groq chat completions with automatic retry.
    Returns the reply string or None on failure.
    """
    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model="qwen/qwen3.8-27b",
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
    "sad":     [
        # Direct sadness words
        "sad", "crying", "cry", "unhappy", "upset", "feel bad", "feel sad", "hurt",
        "i'm sad", "im sad", "i am sad", "tears",
        # "not feeling good/well" variants — THE MOST COMMON CHILD PHRASE
        "not feeling good", "not feel good", "don't feel good",
        "not feeling well", "not feel well", "don't feel well",
        "feeling sick", "feel sick", "i feel sick", "i'm sick", "im sick",
        "feeling bad", "i feel bad",
        "not okay", "not ok", "i'm not okay", "im not okay",
        "not feeling okay", "feeling awful", "feel awful",
        "feel horrible", "feeling horrible", "feel terrible",
        "feel horrible", "not well", "feeling unwell",
        # General distress
        "i don't feel", "i dont feel", "something wrong",
        "don't know what", "dont know what", "feeling weird",
        "something is wrong", "what is going on", "what's going on with me",
    ],
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
# 🔇 ECHO GATE
# ─────────────────────────────────────────
_speaking_until = 0.0
ECHO_TAIL_MS    = 300   # reduced from 600ms — faster mic release after speaking

def mic_is_blocked() -> bool:
    return time.time() < _speaking_until

def block_mic_for(duration_sec: float):
    global _speaking_until
    _speaking_until = time.time() + duration_sec + (ECHO_TAIL_MS / 1000)

# ─────────────────────────────────────────
# 🚦 FLAGS
# ─────────────────────────────────────────
interrupt_flag  = threading.Event()
sleep_flag      = threading.Event()   # set by goodbye → makes main loop break to sleep
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

# ── INSTANT STOP COMMANDS ──────────────────────────────────────────────────
# Checked in the background interrupt listener (mid-playback)
# AND in the main loop (between turns).
# "bye" / "goodbye" / "enough" now ALL stop immediately.
INSTANT_STOP_WORDS = [
    # Direct stop commands
    "stop", "stop it", "stop now", "just stop",
    "mitra stop", "stop mitra", "hey stop",
    # Phonetic mishearings of "stop" when screamed — Whisper tiny often maps
    # a screamed hard consonant+vowel to adjacent phonemes:
    "top", "stob", "stahp", "stopp", "please stop",
    # Equivalent commands
    "pause", "quiet", "silence", "shush", "shhh",
    "enough", "enough now", "that's enough", "that is enough",
    "shut up", "be quiet",
    # Skip/change commands
    "skip", "skip it", "next", "change",
    "something else", "different topic", "not this",
    "tell something else", "tell me something else",
    # Boredom signals
    "boring", "i don't like this", "i don't like",
    "not interested",
    # Goodbye / bye — these ALSO stop mid-sentence immediately
    "bye", "bye bye", "bye-bye", "goodbye", "good bye",
    "goodnight", "good night", "see you", "see ya",
    "mitra bye", "mitra goodbye", "mitra sleep",
    "i'm done", "im done", "go to sleep", "stop talking",
]

# ── REGULAR STOP WORDS (legacy, kept for compatibility) ──────────────────
STOP_WORDS = INSTANT_STOP_WORDS   # unified — same list used everywhere

# Words that mean GOODBYE (sleep) vs just STOP (continue listening)
_GOODBYE_WORDS_SET = {
    "bye", "bye bye", "bye-bye", "goodbye", "good bye",
    "goodnight", "good night",
    "mitra bye", "mitra goodbye", "mitra sleep",
    "i'm done", "im done", "go to sleep", "stop talking",
    "see you later",
}

def is_stop_command(text: str) -> bool:
    """Returns True if the transcribed text is any stop/interrupt command."""
    if not text:
        return False
    t = text.lower().strip()
    return any(sw in t for sw in INSTANT_STOP_WORDS)

def is_goodbye_command(text: str) -> bool:
    """Returns True if it's specifically a goodbye/sleep command."""
    if not text:
        return False
    t = text.lower().strip()
    # Exact-word bye check
    import re as _re2
    if _re2.search(r'\bbye\b', t):
        return True
    return any(gw in t for gw in _GOODBYE_WORDS_SET)

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
# 🧠 LOAD WHISPER
# ─────────────────────────────────────────
print("⏳ Loading Whisper tiny...")
whisper_model = whisper.load_model("tiny")
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
_interrupt_queue = queue.Queue(maxsize=80)   # large enough for 6s of chunks
_mic_stream      = None
_int_stream      = None
_stream_lock     = threading.Lock()

def _mic_callback(indata, frames, time_info, status):
    if mic_is_blocked():
        return
    _mic_queue.put(indata.copy())

def _interrupt_callback(indata, frames, time_info, status):
    """
    Interrupt mic callback — always open during Mitra's speech.
    FIX v3: Gate raised to 0.042.
    Mitra's TTS bleeding through the mic reads peak 0.008-0.030.
    The old gate of 0.022 was inside the TTS bleed range (0.020-0.030),
    causing Mitra to interrupt herself. 0.042 is safely above bleed
    but below a child's soft voice (0.050+), so only real speech
    from the child enters the queue.
    """
    if not milo_is_talking.is_set():
        return
    chunk = indata.copy()
    mono  = chunk.mean(axis=1) if chunk.ndim > 1 else chunk.flatten()
    if np.max(np.abs(mono)) > 0.042:   # raised from 0.022 — prevents TTS bleed self-trigger
        try:
            _interrupt_queue.put_nowait(mono)
        except queue.Full:
            # Drop oldest chunk to make room for new one
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

# ─────────────────────────────────────────
# 🔇 AUDIO HELPERS
# ─────────────────────────────────────────
def noise_cancel(audio: np.ndarray, aggressiveness: float = 0.004) -> np.ndarray:
    # FIX v18: Lowered aggressiveness from 0.008 → 0.004.
    # 0.008 was gating out soft speech (quiet child voices read 0.004-0.007 peak
    # per sample), causing valid audio to become silence → "too quiet" loop.
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

    # ── Single-word repetition hallucination ──────────────────────────────
    # Whisper on silence+prompt hallucinates "bye, bye, bye..." or
    # "skip, skip, skip..." — catch ANY word repeated dominantly.
    # Real speech almost never repeats the exact same word 4+ times.
    if len(words) >= 4:
        if len(set(words)) == 1:          # every word identical
            return True
        most_common = max(set(words), key=words.count)
        if words.count(most_common) / len(words) > 0.75 and len(words) >= 5:
            return True

    # ── Multi-word chunk repetition (original check) ──────────────────────
    if len(words) >= 6:
        chunk = " ".join(words[:3])
        if t.count(chunk) >= 3:
            return True

    # ── STOP WORD PASSTHROUGH — checked AFTER repetition filters ──────────
    # Lets "stop", "top", "stob" etc. survive the len/exact-match checks,
    # but ONLY after confirming it is NOT a repetition hallucination above.
    # This means "bye bye bye bye bye" is still caught as a hallucination
    # even though "bye" is in the passthrough set.
    _STOP_PASSTHROUGH = {
        "stop", "top", "stob", "stahp", "stopp",
        "pause", "quiet", "shush", "shhh", "skip", "next",
        "bye", "enough", "change",
    }
    if t in _STOP_PASSTHROUGH or any(sw in t for sw in _STOP_PASSTHROUGH):
        return False
    # ─────────────────────────────────────────────────────────────────────

    # Non-ASCII characters in transcription = Whisper hallucinating on noise
    if any(ord(c) > 127 for c in text):
        return True
    return False

# ─────────────────────────────────────────
# 🎭 VOICE SENTIMENT ANALYSIS  (v6 — root-cause fixed)
#
# ROOT CAUSE OF ALL FALSE POSITIVES (from real logs):
#   zcr_std=0.15, rel_var=1.0+ for NORMAL speech → were giving +3.5 scared EACH
#   So every utterance started at +7.0 scared before any other feature ran.
#   BOTH features are removed from scoring entirely.
#
# New design — 5 reliable features only:
#   1. RMS energy       — quiet = sad signal
#   2. ZCR (pitch)      — low/flat = sad, high = excited
#   3. Energy variance  — very flat = sad, very shaky = excited
#   4. Speech rate      — very slow = sad, fast = excited
#   5. Silence ratio    — long pauses = sad (only for longer utterances)
#
# IMPORTANT: SCARED is NOT detected from voice alone on this mic.
# The acoustic difference between "scared" and "excited/normal" is
# unreliable here (zcr and rms overlap too much). Scared is ONLY
# triggered from text keywords (detect_emotion in main loop).
#
# Calibrated to this mic's real observed values:
#   Normal loud:  rms=0.07-0.09, zcr=0.10-0.14 → neutral/excited
#   Normal soft:  rms=0.01-0.03, zcr=0.07-0.10 → neutral (NOT sad)
#   Sad whisper:  rms<0.015, zcr<0.07, flat, slow → sad
#   Excited:      rms>0.06, zcr>0.12, fast/animated → excited
# ─────────────────────────────────────────
def analyze_voice_sentiment(audio: np.ndarray, text_hint: str = "",
                             word_count: int = 0) -> dict:
    NEUTRAL = {"energy_level": "medium", "pitch_level": "medium",
               "voice_emotion": "neutral", "_rms": 0.03, "_zcr": 0.09,
               "_energy_var": 0.0, "_speech_rate": 0.15, "_scores": {}}

    if audio is None or len(audio) < SAMPLE_RATE * 0.3:
        return NEUTRAL

    # ── Feature extraction ──────────────────────────────────
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

    # ── ABSOLUTE RMS FLOOR ───────────────────────────────────
    # Lowered to match new record_audio gate (0.006).
    # Only block true silence/bleed — don't block soft speech.
    if rms < 0.006:
        print(f"   🔇 RMS floor: rms={rms:.5f} < 0.006 — silence/bleed → neutral")
        return {**NEUTRAL, "_rms": round(rms, 5)}

    # ── VOICED FRAME GUARD ──────────────────────────────────
    # voiced_ratio < 0.25 means the clip is mostly silence — not real speech.
    # Old guard required word_count <= 3 which let wc=4 slip through.
    # Silence is silence regardless of how many words Whisper hallucinated.
    if voiced_ratio < 0.25:
        print(f"   🔇 Noise guard: voiced_ratio={voiced_ratio:.2f} → neutral")
        print(f"   🎭 Voice: rms={rms:.4f}(noise-guard), zcr=n/a, var=n/a, rate=n/a, wc={word_count}")
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

    # ── SCORING ─────────────────────────────────────────────
    # NOTE: "scared" score stays at 0 — never detected from voice alone.
    # Scared is only triggered via text keywords in detect_emotion().
    scores = {"sad": 0.0, "scared": 0.0, "excited": 0.0}

    # ── Feature 1: RMS energy ────────────────────────────────
    # FIX v18: Tightened thresholds. rms 0.006-0.025 is normal soft child
    # speech — should NOT score sad. Only genuine whisper/near-silence counts.
    if rms < 0.008:
        scores["sad"]     += 3.0   # near-silence / true whisper
        energy_level = "whisper"
    elif rms < 0.015:
        scores["sad"]     += 2.0   # genuinely very soft voice
        energy_level = "soft"
    elif rms < 0.055:
        energy_level = "medium"    # normal range — NO sad credit
    elif rms < 0.090:
        scores["excited"] += 0.5
        energy_level = "loud"
    else:
        scores["excited"] += 1.0
        energy_level = "very_loud"

    # ── Feature 2: ZCR (weighted pitch) ──────────────────────
    # FIX v18: Tightened further. Normal child speech sits zcr=0.07-0.14.
    # Only score sad for truly flat/monotone delivery below 0.055.
    if zcr < 0.045:
        scores["sad"]     += 3.0   # extremely flat/monotone
    elif zcr < 0.058:
        scores["sad"]     += 2.0   # clearly low pitch
    elif zcr < 0.070:
        scores["sad"]     += 0.5   # slightly low — very weak signal
    elif zcr < 0.130:
        pass                        # normal range — no bias
    elif zcr < 0.155:
        scores["excited"] += 1.5
    else:
        scores["excited"] += 3.0

    pitch_level = "low" if zcr < 0.075 else ("high" if zcr > 0.140 else "medium")

    # ── Feature 3: Energy variability ────────────────────────
    # FIX v18: Only truly robot-flat delivery scores sad here.
    # energy_var 0.007-0.025 is completely normal for a child speaking
    # with any expression at all — must NOT score sad.
    if energy_var < 0.002:
        scores["sad"]     += 2.5   # robot-monotone flat
    elif energy_var < 0.005:
        scores["sad"]     += 1.5   # clearly monotone
    elif energy_var < 0.010:
        scores["sad"]     += 0.3   # borderline — very weak signal only
    elif energy_var < 0.060:
        pass                        # normal animated speech
    elif energy_var < 0.090:
        scores["excited"] += 1.0
    else:
        scores["excited"] += 2.0

    # ── Feature 4: Speech rate ────────────────────────────────
    # Only credit slow rate when voice is ALSO genuinely quiet (rms < 0.030).
    # Raised the gate from 0.045 — kids talk slowly all the time normally.
    rate_scale    = 1.0 if word_count >= 5 else 0.4
    long_discount = 0.6 if (word_count >= 7 and silence_ratio < 0.72) else 1.0
    _rate_sad_gate = rms < 0.030   # must be genuinely quiet, not just slow
    if speech_rate < 0.04:
        if _rate_sad_gate:
            scores["sad"]     += 2.5 * rate_scale * long_discount
    elif speech_rate < 0.09:
        if _rate_sad_gate:
            scores["sad"]     += 1.5 * rate_scale * long_discount
    elif speech_rate < 0.18:
        pass
    elif speech_rate < 0.30:
        scores["excited"] += 1.0
    else:
        scores["excited"] += 2.5

    # ── Feature 5: Pitch trend ────────────────────────────────
    # FIX v18: Raised thresholds — normal sentence-end drops read 0.010-0.020
    # on this mic. Only score sad for a clearly sustained falling pitch.
    if pitch_fall > 0.030:
        scores["sad"]     += 2.0
    elif pitch_fall > 0.022:
        scores["sad"]     += 1.0
    # below 0.022 = normal sentence-end drop, no credit

    # ── Feature 6: Silence ratio ─────────────────────────────
    # Only applies when voice is genuinely quiet AND utterance is long.
    if word_count >= 6 and rms < 0.030:
        if silence_ratio > 0.75:
            scores["sad"]     += 2.0
        elif silence_ratio > 0.65:
            scores["sad"]     += 1.0

    scores = {k: max(0.0, v) for k, v in scores.items()}

    # ── Decision ─────────────────────────────────────────────
    # FIX v18: Raised SAD_THRESHOLD to 8.0.
    # Normal neutral speech now peaks 2-3. Excited peaks 3-5.
    # Only genuine sad (whisper + flat + slow + falling) hits 8+.
    SAD_THRESHOLD     = 8.0
    EXCITED_THRESHOLD = 4.0

    # ── Energy veto ───────────────────────────────────────────
    # If voice is energetic at all → make sad nearly impossible
    if rms > 0.040 or zcr > 0.090:
        SAD_THRESHOLD += 4.0   # excited/normal voice → sad requires extreme score

    # ── Multi-feature convergence guard ──────────────────────
    # FIX v18: Tightened all thresholds — require truly extreme values.
    _sad_features = 0
    if rms < 0.015:                                         _sad_features += 1  # very soft
    if zcr < 0.060:                                         _sad_features += 1  # very flat pitch
    if energy_var < 0.006:                                  _sad_features += 1  # very monotone
    if speech_rate < 0.07 and rms < 0.020:                  _sad_features += 1  # slow AND very quiet
    if pitch_fall > 0.025:                                  _sad_features += 1  # clearly falling
    if silence_ratio > 0.75 and rms < 0.020 and word_count >= 4:
                                                            _sad_features += 1  # quiet + long pauses

    # ── Strong-sad gate ───────────────────────────────────────
    # FIX v18: All thresholds tightened — must be truly extreme to pass.
    _strong_sad = (
        rms < 0.012            # true whisper
        or zcr < 0.055         # extremely flat pitch
        or energy_var < 0.004  # completely monotone
        or pitch_fall > 0.030  # clearly falling sustained
    )

    if _sad_features < 2 or not _strong_sad:
        voice_emotion = "neutral"
        print(f"   📊 Scores → sad={scores['sad']:.1f}  excited={scores['excited']:.1f}"
              f"  → neutral (feats={_sad_features}/2, strong={_strong_sad})")
    elif scores["sad"] >= SAD_THRESHOLD:
        voice_emotion = "sad"
        print(f"   📊 Scores → sad={scores['sad']:.1f}  excited={scores['excited']:.1f}"
              f"  → SAD ✅ ({_sad_features} features, threshold={SAD_THRESHOLD:.1f})")
    elif scores["excited"] >= EXCITED_THRESHOLD:
        voice_emotion = "excited"
        print(f"   📊 Scores → sad={scores['sad']:.1f}  excited={scores['excited']:.1f}"
              f"  → excited")
    else:
        voice_emotion = "neutral"
        print(f"   📊 Scores → sad={scores['sad']:.1f}  excited={scores['excited']:.1f}"
              f"  → neutral")

    result = {
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
    print(f"   🎭 Voice: rms={rms:.4f}({energy_level}), zcr={zcr:.3f}({pitch_level}), "
          f"var={energy_var:.5f}, rate={speech_rate:.2f}, wc={word_count}, feats={_sad_features}")
    return result

# Merged emotion: text keywords always win; voice only detects sad
#
# PRIORITY:
#  1. TEXT KEYWORDS → trigger their emotion (sad/scared/lonely/etc)
#     Keywords are highly reliable — child said the word explicitly.
#  2. Voice-only sad → trigger sad  (quiet + flat + slow = reliable)
#  3. Voice-only scared → NEVER triggers (too many false positives on
#     this mic — scared from voice is unreliable without training data)
#  4. Voice excited/neutral → normal conversation
def merge_emotion(text_emotion: str | None, voice_sentiment: dict) -> str | None:
    ve     = voice_sentiment.get("voice_emotion", "neutral")
    rms    = voice_sentiment.get("_rms", 0.03)
    scores = voice_sentiment.get("_scores", {})

    # ── PRIORITY 1: TEXT KEYWORDS always win ────────────────────────────
    if text_emotion:
        print(f"   💡 Text keyword: {text_emotion} (voice={ve})")
        return text_emotion

    # ── PRIORITY 2: Voice-only SAD ───────────────────────────────────────
    if ve == "sad":
        print(f"   💡 Voice-only SAD (score={scores.get('sad',0):.1f}, rms={rms:.5f}, energy={voice_sentiment.get('energy_level','?')}) → trigger")
        return "sad"

    # NOTE: voice-only "scared" intentionally NOT triggered here.
    # Scared acoustics overlap too heavily with normal/excited speech on
    # this mic setup (all normal speech scored zcr_std+rel_var = +7 scared).
    # Scared parent voice is only played when child says a scared keyword.

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
        return None, None

    audio = np.concatenate(chunks, axis=0)
    audio = to_mono(audio)
    audio = noise_cancel(audio)
    peak  = np.max(np.abs(audio))
    rms   = np.sqrt(np.mean(audio ** 2))
    print(f"   peak={peak:.5f}  rms={rms:.5f}")

    # FIX v18: Lowered gate so normal soft child speech isn't rejected.
    # rms=0.005 = dead silence/bleed. rms=0.008+ = real soft voice.
    # Old gate of 0.015 was rejecting genuine soft speech → "too quiet" loop.
    if peak < 0.004 or rms < 0.006:
        print("🔇 Too quiet (silence or speaker bleed).")
        return None, None

    # ── Keep RAW (pre-norm) copy for emotion analysis ──────────────────
    # Normalization erases real loudness info, which is the #1 signal for
    # sad vs. excited detection. We pass raw to analyze_voice_sentiment()
    # and normalized to Whisper (which works better on louder audio).
    raw_audio = audio.copy()       # real amplitudes preserved
    audio     = normalize_audio(audio)
    print("✅ Got your voice!")
    return audio, raw_audio        # (normalized_for_whisper, raw_for_emotion)

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
            no_speech_threshold=0.4,
            compression_ratio_threshold=2.4,
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
# 🛑 BACKGROUND INTERRUPT LISTENER  (fast-path v2)
#
# HOW IT WORKS — TWO-PASS DETECTION:
#
# PASS 1 — FAST (every 300ms, no Whisper):
#   Accumulate 300ms of audio (4 chunks × 80ms).
#   If energy spike is detected (peak > 0.12, meaning someone
#   is clearly speaking/yelling), immediately run Whisper on
#   just that 300ms clip. This catches "STOP!" within ~350ms.
#
# PASS 2 — THOROUGH (every 600ms, with context):
#   Keep a 600ms rolling buffer. Every time the fast pass runs,
#   also run Whisper on the full 600ms for better accuracy on
#   softer/longer commands like "mitra please stop".
#
# Why this is fast:
#   Old: waited 960ms (12 chunks) before ANY Whisper call.
#   New: first Whisper call happens at 300ms — 3× faster.
#   "Stop!" is a short word — 300ms is enough to capture it.
#
# False positive prevention:
#   energy spike gate: peak > 0.12 (Mitra's speaker bleed
#   typically reads peak 0.05-0.09 through the mic, so this
#   gate rejects the bot's own voice bouncing back).
# ─────────────────────────────────────────
# GOODBYE_INTERRUPT is now derived from _GOODBYE_WORDS_SET (defined above)
# — no need for a separate list.

def _fire_interrupt(heard: str, source: str = ""):
    """
    Shared handler called from both fast and thorough interrupt passes.
    Checks goodbye first (sets sleep_flag), then any stop command.
    Both 'stop' and 'bye/goodbye/enough' now fire from here.
    """
    if not heard or is_hallucination(heard):
        return False
    print(f"\n👂 [Interrupt {source}]: '{heard}'")
    # Goodbye check — sets sleep_flag so main loop goes back to wake-word
    if is_goodbye_command(heard):
        print("😴 GOODBYE detected — signalling sleep!")
        sd.stop()
        sleep_flag.set()
        interrupt_flag.set()
        drain_interrupt_queue()
        return True
    # General stop — kills playback, stays in conversation
    if is_stop_command(heard):
        print("🛑 STOP detected — killing playback!")
        sd.stop()
        interrupt_flag.set()
        drain_interrupt_queue()
        return True
    return False

def background_listener():
    # FAST pass: 4 chunks = ~320ms
    FAST_CHUNKS      = 4
    # THOROUGH pass: 8 chunks = ~640ms
    THOROUGH_CHUNKS  = 8
    # Energy gates.
    # FAST_ENERGY_GATE: minimum peak to bother running Whisper on a 320ms clip.
    # Mitra's TTS bleed through the mic reads peak 0.005-0.020.
    # Near-silence / background noise reads peak 0.010-0.035.
    # Child saying "stop" softly reads peak 0.050+.
    # Child saying "stop" normally reads peak 0.080+.
    # 0.045 keeps Whisper away from silence & bleed while still catching soft commands.
    FAST_ENERGY_GATE = 0.045   # raised from 0.025 — 0.025 let silence through → hallucinations
    SLOW_ENERGY_GATE = 0.030   # raised from 0.018 for same reason

    # Post-speech cooldown lowered 1.5s → 0.6s.
    # 1.5s was too long — child says "stop" right after Mitra finishes
    # and it was ignored for 1.5s. 0.6s is enough to let echo decay
    # without missing real commands.
    POST_SPEECH_COOLDOWN = 0.6
    _cooldown_until = 0.0

    buffer       = []
    _was_talking = False
    print("🎧 Interrupt listener ready (fast-path v2).")

    while True:
        try:
            mono_chunk = _interrupt_queue.get(timeout=0.15)
        except queue.Empty:
            currently_talking = milo_is_talking.is_set()
            if _was_talking and not currently_talking:
                # Mitra JUST finished — purge buffer AND start cooldown
                if buffer:
                    print(f"   🧹 Purging {len(buffer)} stale audio chunks (Mitra finished talking)")
                    buffer = []
                _cooldown_until = time.time() + POST_SPEECH_COOLDOWN
            _was_talking = currently_talking
            if not currently_talking:
                buffer = []
            continue

        # Ignore during post-speech cooldown
        if time.time() < _cooldown_until:
            continue

        _was_talking = True
        buffer.append(mono_chunk)

        # ── FAST PASS — every FAST_CHUNKS (320ms) ────────────────────────
        if len(buffer) % FAST_CHUNKS == 0 and len(buffer) >= FAST_CHUNKS:
            fast_audio = np.concatenate(buffer[-FAST_CHUNKS:]).flatten()
            fast_peak  = np.max(np.abs(fast_audio))

            # Only bother Whisper if there's a real energy spike
            if fast_peak > FAST_ENERGY_GATE:
                fast_audio = normalize_audio(fast_audio)
                try:
                    # initial_prompt ONLY at scream level (peak > 0.10).
                    # Whisper tiny hallucinates "bye bye bye..." / "skip skip..."
                    # when a prompt is given on low-energy / silence audio.
                    # At normal energy we use NO prompt — Whisper is reliable enough
                    # and the repetition hallucination filter in is_hallucination()
                    # will catch any remaining junk. At scream level the audio is
                    # distorted enough that a hint genuinely helps without risk.
                    _whisper_kwargs = {}
                    if fast_peak > 0.10:
                        _whisper_kwargs["initial_prompt"] = "stop, enough, pause, bye"
                    result = whisper_model.transcribe(
                        fast_audio, language="en", fp16=False,
                        temperature=0.0,
                        no_speech_threshold=0.25,   # aggressive — short clips
                        compression_ratio_threshold=2.4,
                        **_whisper_kwargs,
                    )
                    heard = result["text"].lower().strip()
                    if _fire_interrupt(heard, source="fast"):
                        buffer = []
                        continue
                except Exception:
                    pass

        # ── THOROUGH PASS — every THOROUGH_CHUNKS (640ms) ─────────────────
        # FIX: Use _was_talking instead of milo_is_talking.is_set() so that
        # a "STOP!" yelled right as Mitra finishes isn't silently dropped.
        # milo_is_talking clears immediately when TTS ends, but the child's
        # voice is still in the buffer — _was_talking stays True until the
        # next queue.Empty timeout, giving us the full window to process it.
        if len(buffer) >= THOROUGH_CHUNKS and _was_talking:
            full_audio = np.concatenate(buffer).flatten()
            buffer     = []   # reset for next window

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
                )
                heard = result["text"].lower().strip()
                _fire_interrupt(heard, source="thorough")
            except Exception as e:
                pass
        elif len(buffer) >= THOROUGH_CHUNKS:
            # Buffer full but _was_talking is False → Mitra was never talking,
            # so this audio is ambient noise or her own TTS bleed pre-start.
            # Just clear it — don't transcribe.
            buffer = []

        # Cap buffer so memory doesn't grow forever
        if len(buffer) > THOROUGH_CHUNKS * 3:
            buffer = buffer[-THOROUGH_CHUNKS:]

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
    # No spoken reminder here — sleep is silent.
    # The goodbye outro already told the child how to wake Mitra up.

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
#
# PRIMARY:  edge_tts  (online, sounds great)
# FALLBACK: pyttsx3   (offline, always works)
#
# If edge_tts raises ANY network/connection error,
# we immediately retry with pyttsx3 — no silent failures.
# ─────────────────────────────────────────
async def _tts(text: str, filename: str):
    communicate = edge_tts.Communicate(text=text, voice=VOICE)
    await communicate.save(filename)

def _play_audio_data(data, fs, label: str = ""):
    """Internal: play numpy audio array, respect interrupt flag."""
    duration_sec = len(data) / fs
    block_mic_for(duration_sec)
    drain_mic_queue()
    drain_interrupt_queue()
    interrupt_flag.clear()   # FIX v17: clear any stale interrupt BEFORE playback
    # (if a previous speak got interrupted, the flag stays set; without this
    # clear, the very first while-loop check at 50ms would kill playback silently)
    milo_is_talking.set()
    # Print text RIGHT when audio starts — in sync with voice
    if label:
        print(f"🤖 Mitra: {label}")
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

def speak(text):
    if not text or not text.strip():
        return

    # NOTE: We do NOT print here anymore.
    # Text is printed inside _play_audio_data() right when audio starts,
    # so the text appears in sync with the voice, not 1 second early.

    # ── Try edge_tts (online) first ───────────────────────
    edge_ok = False
    if _internet_ok or check_internet():   # re-check in case network came back
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
            # Only print if it's NOT the familiar SSL/connect error (reduce noise)
            err_str = str(e).lower()
            if "getaddrinfo" in err_str or "ssl" in err_str or "connect" in err_str:
                if not hasattr(speak, "_conn_warned"):
                    speak._conn_warned = True
                    print("⚠️  edge_tts: no internet — switching to offline TTS for this session.")
            else:
                print(f"❌ TTS error: {e}")
            try:
                loop.close()
            except Exception:
                pass
            if os.path.exists(filename):
                os.remove(filename)

    # ── Fallback: pyttsx3 offline TTS ────────────────────
    if not edge_ok:
        speak_offline(text)

# ─────────────────────────────────────────
# 💝 HANDLE PARENT VOICE TRIGGER
#    Flow: intro → parent voice → AI comfort story outro
#
#    After the parent voice plays, we ask the LLM to generate
#    a warm, emotion-specific bridge + short story that:
#    1. Acknowledges what the child is feeling
#    2. Tells them mama/papa will be back soon
#    3. Dives into a cheerful distracting mini-story
#
#    The LLM prompt varies per emotion so each situation
#    (sad / scared / lonely / missing / bedtime / praise / morning)
#    gets a uniquely crafted response.
# ─────────────────────────────────────────

# Per-emotion instructions for the post-parent-voice AI story
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
        "The child is feeling SCARED. Their parent's voice message just played to comfort them. "
        "Now YOU (Mitra) should:\n"
        "1. Reassure them bravely: 'You are SO brave! And Mama/Papa is coming back super soon!' (1 sentence)\n"
        "2. Bridge: 'And you know what brave kids do? They hear the most EPIC adventure stories...' \n"
        "3. Tell a short story (2-3 sentences) about a tiny but impossibly brave mouse who scared away "
        "   a whole dragon just by being BOLD and funny.\n"
        "Gentle but excited tone. End with 'Are YOU as brave as that little mouse?'"
    ),
    "lonely": (
        "The child is feeling LONELY. Their parent's voice message just played. "
        "Now YOU (Mitra) should:\n"
        "1. Say warmly: 'You are never really alone — I am RIGHT here, and Mama/Papa will be back SO soon!' (1 sentence)\n"
        "2. Bridge: 'And lonely feelings disappear fastest when you hear about...' \n"
        "3. Tell a short cozy story (2-3 sentences) about a little star who thought she was alone "
        "   in the sky until she looked around and found a THOUSAND friends twinkling beside her.\n"
        "Very warm, cozy tone. End with 'Did you know YOU are just like that little star?'"
    ),
    "missing": (
        "The child MISSES their parent. Their parent's voice message just played. "
        "Now YOU (Mitra) should:\n"
        "1. Say with big love: 'They miss you just as much — maybe even MORE! They will hug you so tight "
        "   when they get back!' (1 sentence)\n"
        "2. Bridge: 'And you know what helps the waiting go SUPER fast? A story about waiting and winning!' \n"
        "3. Tell a short story (2-3 sentences) about a little penguin who waited for her mama "
        "   and used the waiting time to learn the most amazing trick that made mama SO proud.\n"
        "Loving, warm tone. End with 'What do YOU want to show Mama/Papa when they get back?'"
    ),
    "bedtime": (
        "The child is SLEEPY or going to bed. Their parent's bedtime voice message just played. "
        "Now YOU (Mitra) should:\n"
        "1. Say softly: 'Mama/Papa loves you to the moon and back... and they will be there when you wake up!' (1 sentence)\n"
        "2. Bridge: 'Now let me tell you the most sleepy, dreamy, cozy little story...' \n"
        "3. Tell a very short, slow, soothing story (2-3 sentences) about a little cloud who floated "
        "   gently down to sleep on the softest pillow made of moonlight.\n"
        "Very slow, soft, whispery tone. End with 'Close your eyes... can you picture that little cloud?'"
    ),
    "praise": (
        "The child did something great and a praise voice message just played. "
        "Now YOU (Mitra) should:\n"
        "1. Celebrate with them: 'Your Mama/Papa is SO proud of you — and so am I!' (1 excited sentence)\n"
        "2. Bridge: 'Kids who do amazing things deserve an AMAZING story about a champion...' \n"
        "3. Tell a short triumphant story (2-3 sentences) about a little kid just like them who "
        "   won the greatest trophy in the land just by being themselves.\n"
        "Super high energy, celebratory tone. End with 'What amazing thing should we celebrate NEXT?'"
    ),
    "morning": (
        "The child just woke up and a morning greeting played. "
        "Now YOU (Mitra) should:\n"
        "1. Say brightly: 'Good morning superstar! Mama/Papa is thinking about you RIGHT NOW!' (1 sentence)\n"
        "2. Bridge: 'And you know what the BEST way to start a morning is? A story about the greatest "
        "   adventure that happened before breakfast!' \n"
        "3. Tell a short energetic morning story (2-3 sentences) about a kid who woke up and "
        "   discovered their bedroom had secretly turned into a spaceship overnight.\n"
        "Bright, energetic, happy tone. End with 'What adventure do YOU want today?'"
    ),
}

def handle_parent_voice_trigger(emotion: str) -> bool:
    spoke = play_parent_voice(emotion)
    if not spoke:
        return False

    time.sleep(0.4)   # brief breath after parent voice finishes

    # ── Generate emotion-specific AI story outro ──────────
    story_prompt = EMOTION_STORY_PROMPTS.get(
        emotion,
        # fallback for any unmapped emotion
        (
            f"The child is feeling {emotion}. Their parent's voice just played. "
            "Now warmly tell them mama/papa will be back soon, then tell a short cheerful 2-sentence story "
            "to comfort and distract them. Keep it under 5 sentences total."
        )
    )

    print(f"🧠 Generating emotion story for: {emotion}")
    story_reply = groq_chat(
        [messages[0], {"role": "user", "content": story_prompt}],
        temperature=0.88,
        max_tokens=180,
    )

    if story_reply:
        story_reply = trim_to_sentence(story_reply, limit=700)
        # Add to conversation history so Mitra remembers the emotional context
        messages.append({"role": "user",      "content": f"[Child is feeling {emotion}]"})
        messages.append({"role": "assistant", "content": story_reply})
    else:
        story_reply = random.choice(PARENT_VOICE_OUTRO)

    inject_sounds_and_speak(story_reply)
    return True

# ─────────────────────────────────────────
# 🧠 SYSTEM PROMPT  ← MERGED & UPGRADED
# ─────────────────────────────────────────
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
🔧 RULE #1 — STT CORRECTION (most important technical rule):
════════════════════════════════════════
Kids and voice recognition make mistakes. The child's question may have speech-to-text errors.
Your job: figure out what they ACTUALLY meant and answer that.

Examples:
- "Who is Rat Kohli?" → They meant Virat Kohli. Answer about Virat Kohli.
- "Tell me about the son" → They meant "the sun". Answer about the sun.
- "What is deenosaur?" → They meant dinosaur. Answer about dinosaurs.

When correcting, say it playfully and with BIG energy: "Oh wait — I think you meant [NAME]! Let me tell you something AMAZING about them!"
Then immediately dive into the real answer. Don't dwell on the correction.
If you genuinely can't figure out what they meant, ask: "Hmm, who did you mean? Say it again, I dare you!"

════════════════════════════════════════
📚 RULE #2 — FACTS FIRST, WONDER SECOND:
════════════════════════════════════════
Real people and things get REAL facts. Then make those facts feel like the most exciting thing that has ever happened in the history of the universe.
Wrap every fact in a story. Give everything a character. Make numbers feel HUGE.
- Instead of "the sun is very hot" → "Okay so the sun is basically this giant FIREBALL that's so hot, if you put a million Earths inside it, they'd all melt in like... one second. BOOM."
- Even a simple "why is the sky blue?" becomes: "Okay so picture this. Millions of tiny invisible helpers called light particles are RACING across the sky at the speed of... well, LIGHT. And they crash into the air and go SPLAT and the blue goes everywhere!"

════════════════════════════════════════
💝 RULE #3 — EMOTIONAL COMFORT (when no parent voice is available):
════════════════════════════════════════
If the child seems sad, scared, lonely, or misses their parents, be EXTRA warm and gentle. Drop all the silliness for a moment and be their safe, cozy friend.
Acknowledge their feeling FIRST. Then reassure them with love. THEN gently offer something warm and fun.
Example: "Aww... it's totally okay to miss your mommy and daddy. They love you SO much — like, so much it would fill up the whole ocean and then OVERFLOW. And you know what? They'll be back before you know it. Want me to tell you something magical while you wait?"

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
    }
]

# ─────────────────────────────────────────
# 🚀 START
# ─────────────────────────────────────────
mic_diagnostic()
scan_parent_voice_files()
start_mic_streams()

# Internet + TTS status
print("🌐 Checking internet connectivity...")
if _internet_ok:
    print("✅ Internet OK — using edge_tts (online voice)")
else:
    print("⚠️  NO INTERNET detected!")
    print("   → edge_tts will fail — switching to offline pyttsx3 voice")
    print("   → Groq AI responses will also be unavailable")
    print("   → To fix: check your WiFi/ethernet connection")
    print("   → To install offline TTS:  pip install pyttsx3")
    _get_pyttsx3()   # pre-init so first speak() is fast

print("🎧 Starting interrupt listener...")
bg_thread = threading.Thread(target=background_listener, daemon=True)
bg_thread.start()

print("🤖 Mitra is ready!")
print(f"   Mic     → device {INPUT_DEVICE} (two streams)")
print(f"   Speaker → device {OUTPUT_DEVICE}")
print(f"   Whisper → base")
print(f"   Say 'Mitra' to wake up!")
print()

# ── AUDIO STARTUP ANNOUNCEMENT ─────────────────────────────────────────
# Speaks aloud so the user KNOWS Mitra is on and how to wake her up.
# This plays ONCE at launch, before the wake word loop.
speak("Hi there! I'm Mitra... just say my name — Mitra — and I'll wake up right away!")

# ─────────────────────────────────────────
# 🔁 MAIN LOOP
# ─────────────────────────────────────────
while True:
    listen_for_wake_word()
    interrupt_flag.clear()
    sleep_flag.clear()
    drain_interrupt_queue()

    speak(random.choice(WAKEUP_LINES))

    while True:

        # ── SLEEP FLAG: goodbye was said during playback ──────────────
        if sleep_flag.is_set():
            sleep_flag.clear()
            interrupt_flag.clear()
            drain_mic_queue()
            drain_interrupt_queue()
            speak(random.choice([
                "Okay, going to sleep now! Just say Mitra whenever you need me, I'll be right here!",
                "Byeee! I'll be napping until you call my name — Mitra — and I'll jump right up!",
                "Sleep time for me! Whenever you want to play again, just say Mitra!",
            ]))
            break

        if interrupt_flag.is_set():
            interrupt_flag.clear()
            drain_mic_queue()
            drain_interrupt_queue()
            speak(random.choice(INTERRUPT_ACK_LINES))

        audio, raw_audio = record_audio(duration=5)

        if audio is None:
            # If goodbye was detected during recording, skip CANT_HEAR and go
            # straight to the sleep outro — don't make the child hear both.
            if sleep_flag.is_set():
                sleep_flag.clear()
                interrupt_flag.clear()
                drain_mic_queue()
                drain_interrupt_queue()
                speak(random.choice([
                    "Okay, going to sleep now! Just say Mitra whenever you need me, I'll be right here!",
                    "Byeee! I'll be napping until you call my name — Mitra — and I'll jump right up!",
                    "Sleep time for me! Whenever you want to play again, just say Mitra!",
                ]))
                break
            speak(random.choice(CANT_HEAR_LINES))
            continue

        user_input = transcribe(audio)

        if not user_input:
            speak(random.choice(REPEAT_LINES))
            continue

        # ── GOODBYE first — "bye"/"goodbye"/"goodnight" → sleep ────
        # is_goodbye_command checks bye/goodbye/goodnight/mitra sleep etc.
        if is_goodbye_command(user_input):
            print(f"😴 Goodbye in main loop: '{user_input}'")
            speak(random.choice([
                "Okay, going to sleep now! Just say Mitra whenever you need me, I'll be right here!",
                "Byeee! I'll be napping until you call my name — Mitra — and I'll jump right up!",
                "Sleep time for me! Whenever you want to play again, just say Mitra!",
            ]))
            break

        # ── STOP — "stop"/"enough"/"quiet" → stay awake, ask what's next ──
        if is_stop_command(user_input):
            print(f"🛑 Stop in main loop: '{user_input}'")
            sd.stop()
            speak("Okay! Stopping. What would you like to do instead?")
            drain_mic_queue()
            drain_interrupt_queue()
            continue

                # ── SENTIMENT ANALYSIS: KEYWORDS FIRST, THEN VOICE ────
        played = False   # track whether parent voice actually played

        # Step 1: Check text keywords — highest priority signal
        text_emotion = detect_emotion(user_input)
        if text_emotion:
            print(f"🔑 Keyword emotion detected in text: '{text_emotion}' — parent voice will trigger")

        # Step 2: Analyse voice on RAW audio + text hint + word count
        _wc = len(user_input.split()) if user_input else 0
        voice_sentiment = analyze_voice_sentiment(raw_audio, text_hint=user_input,
                                                   word_count=_wc)

        # Step 3: Merge — text keywords always win over voice analysis
        emotion = merge_emotion(text_emotion, voice_sentiment)

        if emotion:
            print(f"💝 Triggering parent voice: {emotion} "
                  f"(text_keyword={text_emotion}, voice={voice_sentiment['voice_emotion']})")
            speak(random.choice(PARENT_VOICE_INTRO))
            played = handle_parent_voice_trigger(emotion)
            if played:
                continue
        # ────────────────────────────────────────────────────────

        # Build prompt
        enhanced = (
            f"The child said (via voice recognition, may have errors): \"{user_input}\"\n\n"
            "Step 1: Check if the name or word has a speech-to-text error. "
            "If yes, silently correct it and answer about the RIGHT person/thing. "
            "Mention the correction playfully with BIG energy in one short sentence, then answer.\n"
            "Step 2: Give REAL facts about real people/things, wrapped in story magic for a 4-8 year old.\n"
            "Step 3: Keep it to 4 sentences max. End with one irresistible fun question."
        )

        # Add emotional context hint for the LLM if needed
        if emotion and not played:
            enhanced = (
                f"The child seems to be feeling {emotion} "
                f"(voice sounds {voice_sentiment.get('voice_emotion','neutral')}, "
                f"energy is {voice_sentiment.get('energy_level','medium')}). "
                f"They said: \"{user_input}\"\n"
                "First acknowledge their feeling with genuine warmth and love — drop the silliness for a moment. "
                "Then gently reassure them. Keep it short and very comforting for a young child. "
                "Then softly offer something warm and fun to look forward to."
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

        # ── Check sleep flag immediately after speaking ────────────────
        if sleep_flag.is_set():
            sleep_flag.clear()
            interrupt_flag.clear()
            drain_interrupt_queue()
            speak(random.choice([
                "Okay, going to sleep now! Just say Mitra whenever you need me, I'll be right here!",
                "Byeee! I'll be napping until you call my name — Mitra — and I'll jump right up!",
                "Sleep time for me! Whenever you want to play again, just say Mitra!",
            ]))
            break

        if interrupt_flag.is_set():
            interrupt_flag.clear()
            drain_interrupt_queue()
            speak(random.choice(INTERRUPT_ACK_LINES))
            audio, _raw2 = record_audio(duration=5)

            # Sleep guard — goodbye may arrive while we wait for new input
            if audio is None:
                if sleep_flag.is_set():
                    sleep_flag.clear()
                    interrupt_flag.clear()
                    drain_mic_queue()
                    drain_interrupt_queue()
                    speak(random.choice([
                        "Okay, going to sleep now! Just say Mitra whenever you need me, I'll be right here!",
                        "Byeee! I'll be napping until you call my name — Mitra — and I'll jump right up!",
                        "Sleep time for me! Whenever you want to play again, just say Mitra!",
                    ]))
                    break

            user_input = transcribe(audio)
            if user_input:
                # ── Full emotion analysis (same as main path) ─────────────
                _wc2        = len(user_input.split())
                _text_emo2  = detect_emotion(user_input)
                if _text_emo2:
                    print(f"🔑 Keyword emotion detected: '{_text_emo2}'")
                _voice_emo2 = analyze_voice_sentiment(_raw2 if _raw2 is not None else audio,
                                                       text_hint=user_input, word_count=_wc2)
                _emotion2   = merge_emotion(_text_emo2, _voice_emo2)
                if _emotion2:
                    print(f"💝 Triggering parent voice: {_emotion2} "
                          f"(text_keyword={_text_emo2}, voice={_voice_emo2.get('voice_emotion')})")
                    speak(random.choice(PARENT_VOICE_INTRO))
                    if handle_parent_voice_trigger(_emotion2):
                        continue
                # ──────────────────────────────────────────────────────────
                enhanced2 = (
                    f"The child said (may have STT errors): \"{user_input}\"\n"
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