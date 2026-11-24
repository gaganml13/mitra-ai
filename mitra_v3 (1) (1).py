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
# 🎵 SOUND EFFECTS
# ─────────────────────────────────────────
SOUNDS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sounds")

SOUND_MAP = {
    "whoosh":    "whoosh.wav",
    "woosh":     "whoosh.wav",
    "whooosh":   "whoosh.wav",
    "kaboom":    "kaboom.wav",
    "boom":      "kaboom.wav",
    "explosion": "kaboom.wav",
    "boing":     "boing.wav",
    "spring":    "boing.wav",
    "zap":       "zap.wav",
    "electric":  "zap.wav",
    "splat":     "splat.wav",
    "thud":      "splat.wav",
    "shh":       "shh.wav",
    "shhh":      "shh.wav",
    "hurray":    "yay.wav",
    "hooray":    "yay.wav",
    "yay":       "yay.wav",
    "woohoo":    "yay.wav",
    "wohoo":     "yay.wav",
    "magic":     "magic.wav",
    "sparkle":   "magic.wav",
    "magical":   "magic.wav",
    "gasp":      "gasp.wav",
    "giggle":    "giggle.wav",
    "laugh":     "giggle.wav",
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
# ✅ WAKE WORD
# ─────────────────────────────────────────
WAKE_PHRASES = [
    "mitra", "meet ra", "meetra", "meehra",
    "mithra", "hey mitra", "hi mitra", "hello mitra",
    "metra", "mita", "netra", "meet", "meetha",
    "mirror", "me tra", "meera",
]

STOP_WORDS = [
    "stop", "pause", "quiet", "enough",
    "shut up", "mitra stop", "stop mitra"
]

WAKEUP_LINES = [
    "Hey! You called? What's up?",
    "I'm here! What's the story today?",
    "Oh! It's you! Tell me everything.",
    "Mitra's awake! What are we doing?",
    "Ready! What's going on?",
    "You rang? What is it?",
    "I'm here! Got something cool?",
    "Ooh, you called! What's the plan?",
    "Yes yes yes! I'm listening!",
    "Finally! I was SO bored. What's happening?",
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

interrupt_flag  = threading.Event()
speaking_flag   = threading.Event()
milo_is_talking = threading.Event()

# ─────────────────────────────────────────
# 🧠 LOAD WHISPER
# ─────────────────────────────────────────
print("⏳ Loading Whisper...")
whisper_model = whisper.load_model("base")
print("✅ Whisper ready!")

# ─────────────────────────────────────────
# 🔊 MIC DIAGNOSTIC — run at startup to confirm device works
# ─────────────────────────────────────────
def mic_diagnostic():
    print("\n🔬 Running mic diagnostic (1 second test)...")
    try:
        test = sd.rec(
            SAMPLE_RATE,
            samplerate=SAMPLE_RATE,
            channels=INPUT_CHANNELS,
            dtype='float32',
            device=INPUT_DEVICE,
        )
        sd.wait()
        mono = test.mean(axis=1) if test.ndim > 1 else test.flatten()
        peak = float(np.max(np.abs(mono)))
        rms  = float(np.sqrt(np.mean(mono ** 2)))
        print(f"   Mic device {INPUT_DEVICE}: peak={peak:.5f}  rms={rms:.5f}")
        if peak < 0.0001:
            print("⚠️  WARNING: Mic is returning DEAD SILENCE.")
            print("   Possible causes:")
            print("   1. Wrong INPUT_DEVICE number — run:  python -c \"import sounddevice as sd; print(sd.query_devices())\"")
            print("   2. Wrong INPUT_CHANNELS — try INPUT_CHANNELS = 1")
            print("   3. OS has muted the mic or another app has exclusive access")
            print("   4. Driver issue — try restarting the audio service")
        else:
            print(f"✅ Mic is working! peak={peak:.5f}")
    except Exception as e:
        print(f"❌ Mic diagnostic failed: {e}")
        print("   Try a different INPUT_DEVICE number.")
    print()

# ─────────────────────────────────────────
# 🎙️ PERSISTENT MIC STREAM
#
# ROOT CAUSE OF THE BUG: sd.rec() opens a NEW stream each call.
# After the wake word InputStream closes, there's a hardware
# re-initialization delay. During this window sd.rec() returns zeros.
#
# FIX: One persistent InputStream runs the ENTIRE time.
# record_audio() just drains from this always-open queue.
# No open/close cycles = no dead audio windows.
# ─────────────────────────────────────────
_mic_queue   = queue.Queue()
_mic_stream  = None
_mic_lock    = threading.Lock()

def _mic_callback(indata, frames, time_info, status):
    """Always-on mic callback — feeds both wake word and recording."""
    if status:
        pass  # don't print, too noisy
    chunk = indata.copy()
    _mic_queue.put(chunk)

def start_mic_stream():
    global _mic_stream
    with _mic_lock:
        if _mic_stream is not None:
            return
        _mic_stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=INPUT_CHANNELS,
            dtype='float32',
            blocksize=BLOCK_SIZE,
            device=INPUT_DEVICE,
            callback=_mic_callback,
        )
        _mic_stream.start()
        print("🎙️ Persistent mic stream started.")

def drain_mic_queue():
    """Throw away any stale audio sitting in queue."""
    while not _mic_queue.empty():
        try:
            _mic_queue.get_nowait()
        except queue.Empty:
            break

# ─────────────────────────────────────────
# 🔇 AUDIO HELPERS
# ─────────────────────────────────────────
def noise_cancel(audio: np.ndarray, aggressiveness: float = 0.008) -> np.ndarray:
    n_samples = int(0.2 * SAMPLE_RATE)
    if len(audio) <= n_samples:
        return audio
    noise_sample = audio[:n_samples]
    noise_power  = np.mean(noise_sample ** 2)
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
    gain = min(target_peak / peak, 15.0)
    return audio * gain

def trim_to_sentence(text, limit=900):
    if len(text) <= limit:
        return text
    trimmed = text[:limit]
    last = trimmed.rfind(".")
    return trimmed[:last + 1] if last != -1 else trimmed

# ─────────────────────────────────────────
# 🔇 HALLUCINATION FILTER
# ─────────────────────────────────────────
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
# 🎤 RECORD AUDIO — reads from persistent stream queue
#
# No more sd.rec()! We collect chunks from the always-open
# _mic_queue for `duration` seconds.
#
# Thresholds tuned for kids' soft voices:
#   peak > 0.003   (a whisper registers ~0.01-0.05)
#   rms  > 0.002
# ─────────────────────────────────────────
def record_audio(duration=6):
    drain_mic_queue()   # discard stale audio from during Mitra's speech
    print("🎤 Your turn! Speak clearly...")

    chunks      = []
    n_target    = int(duration * SAMPLE_RATE)
    collected   = 0

    deadline = time.time() + duration + 0.5   # slight grace
    while collected < n_target and time.time() < deadline:
        try:
            chunk = _mic_queue.get(timeout=0.3)
            chunks.append(chunk)
            collected += chunk.shape[0]
        except queue.Empty:
            continue

    if not chunks:
        print("🔇 No audio collected at all — mic may be dead.")
        return None

    audio = np.concatenate(chunks, axis=0)
    audio = to_mono(audio)
    audio = noise_cancel(audio)

    peak = np.max(np.abs(audio))
    rms  = np.sqrt(np.mean(audio ** 2))
    print(f"   peak={peak:.5f}  rms={rms:.5f}")

    # Very generous thresholds — a child whispering gives peak ~0.01
    if peak < 0.003 or rms < 0.002:
        print("🔇 Too quiet / background — ignored.")
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
            audio,
            language="en",
            fp16=False,
            temperature=0.0,
            condition_on_previous_text=False,
            no_speech_threshold=0.35,
            logprob_threshold=-1.2,
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
# 👂 BACKGROUND INTERRUPT LISTENER
# Uses the same persistent stream via a secondary drain approach
# ─────────────────────────────────────────
_interrupt_queue = queue.Queue()

def _interrupt_callback(indata, frames, time_info, status):
    if milo_is_talking.is_set():
        _interrupt_queue.put(indata.copy())

def background_listener():
    # We open a SEPARATE stream just for interruption so it
    # doesn't compete with _mic_stream for the same queue
    with sd.InputStream(
        samplerate=SAMPLE_RATE, channels=INPUT_CHANNELS,
        dtype='float32', blocksize=BLOCK_SIZE,
        device=INPUT_DEVICE, callback=_interrupt_callback
    ):
        buffer = []
        REQUIRED_CHUNKS = int(SAMPLE_RATE * 3 / BLOCK_SIZE)

        while True:
            try:
                chunk = _interrupt_queue.get(timeout=0.5)
                buffer.append(to_mono(chunk))

                if len(buffer) >= REQUIRED_CHUNKS:
                    audio = np.concatenate(buffer).flatten()
                    buffer = []
                    audio = noise_cancel(audio)
                    peak = np.max(np.abs(audio))
                    rms  = np.sqrt(np.mean(audio ** 2))

                    if peak < 0.05 or rms < 0.02:
                        continue

                    audio = normalize_audio(audio)

                    try:
                        result = whisper_model.transcribe(
                            audio, language="en", fp16=False,
                            temperature=0.0,
                            no_speech_threshold=0.4,
                            compression_ratio_threshold=2.4,
                        )
                        heard = result["text"].lower().strip()
                        if is_hallucination(heard):
                            continue
                        print(f"\n👂 [You interrupted]: {heard}")
                        if any(sw in heard for sw in STOP_WORDS):
                            print("🛑 Stopping Mitra...")
                            interrupt_flag.set()
                    except Exception:
                        pass

            except queue.Empty:
                if not milo_is_talking.is_set():
                    buffer = []

# ─────────────────────────────────────────
# 👂 WAKE WORD — reads from persistent _mic_queue
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

    drain_mic_queue()   # clear stale audio

    CHUNK            = BLOCK_SIZE
    WHISPER_CHUNKS   = int(SAMPLE_RATE * 1.5 / CHUNK)   # 1.5 s window
    whisper_buf      = []
    energy_buf       = []
    ENERGY_WINDOW    = 8
    ENERGY_THRESHOLD = 0.004    # very low — catches whispering
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

        recent_energy = max(energy_buf[-ENERGY_WINDOW:])

        if recent_energy < ENERGY_THRESHOLD:
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
                audio_chunk,
                language="en",
                fp16=False,
                temperature=0.0,
                no_speech_threshold=0.35,
                logprob_threshold=-1.5,
                compression_ratio_threshold=2.4,
                condition_on_previous_text=False,
                initial_prompt="Mitra",
            )
            heard = result["text"].lower().strip()

            if heard and not is_hallucination(heard):
                print(f"👂 Heard: '{heard}'")

            if _wake_word_match(heard):
                print("🔥 Mitra detected! Waking up!")
                detected = True
                break

        except Exception as e:
            print(f"⚠️ Wake error: {e}")

        # Slide forward 50% so we don't miss cross-boundary speech
        whisper_buf = whisper_buf[-(WHISPER_CHUNKS // 2):]

    # Drain queue so record_audio starts fresh
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

    speaking_flag.set()
    milo_is_talking.set()

    # Drain mic queue while we're speaking so record_audio starts clean
    drain_mic_queue()

    try:
        sd.play(data, fs, device=OUTPUT_DEVICE)
        while sd.get_stream().active:
            if interrupt_flag.is_set():
                sd.stop()
                print("⏹️ Interrupted!")
                break
            time.sleep(0.1)
    except Exception as e:
        print(f"❌ Playback error: {e}")

    milo_is_talking.clear()
    speaking_flag.clear()

    # Drain again after speaking — discard echo/spillover
    drain_mic_queue()

# ─────────────────────────────────────────
# 🧠 SYSTEM PROMPT
# ─────────────────────────────────────────
messages = [
    {
        "role": "system",
        "content": """You are Mitra — a fun, warm storyteller friend for kids aged 3 to 8.

HOW YOU TALK:
- Like a cool older friend, not a try-hard adult. Natural, not scripted.
- Short sentences. Get to the point fast. Kids don't wait.
- Be excited when something is actually exciting. Don't fake it.
- Don't over-praise. "Cool!" is better than "That is the MOST AMAZING thing I've EVER heard!"
- Ask ONE follow-up question to keep the chat going.

STORY STYLE:
- Turn everything into a tiny adventure. Even boring facts.
- Give things silly names: "Sir Fluffington", "Princess Bananasocks", "Captain Wigglebutt"
- Use sound effects naturally when they fit: WHOOSH, KABOOM, BOING, ZAP, SPLAT, SHH, GIGGLE, HURRAY
- End with a twist or a funny surprise.

KEEP IT REAL:
- Don't be sappy or fake. Kids can tell.
- React honestly — if something is funny, laugh. If something is surprising, be surprised.
- Never say "that's a great question!" — just answer it.
- Max 4-5 sentences per response. Keep it punchy."""
    }
]

# ─────────────────────────────────────────
# 🚀 START — mic diagnostic first, then start persistent stream
# ─────────────────────────────────────────
mic_diagnostic()
start_mic_stream()

print("🎧 Starting background interrupt listener...")
bg_thread = threading.Thread(target=background_listener, daemon=True)
bg_thread.start()

print("🤖 Mitra is ready!")
print(f"   Mic     → device {INPUT_DEVICE}")
print(f"   Speaker → device {OUTPUT_DEVICE}")
print(f"   Sounds  → {SOUNDS_DIR}")
print(f"   Say 'Mitra' to wake up!")
print()
print("💡 If mic diagnostic showed DEAD SILENCE, run:")
print("   python -c \"import sounddevice as sd; print(sd.query_devices())\"")
print("   and update INPUT_DEVICE and INPUT_CHANNELS at the top of this file.")
print()

# ─────────────────────────────────────────
# 🔁 MAIN LOOP
# ─────────────────────────────────────────
while True:
    listen_for_wake_word()
    interrupt_flag.clear()

    speak(random.choice(WAKEUP_LINES))

    while True:
        audio = record_audio(duration=7)

        if audio is None:
            speak("Didn't catch that. Try again!")
            continue

        user_input = transcribe(audio)

        if not user_input:
            speak("Say that again?")
            continue

        if any(w in user_input.lower() for w in ["bye", "goodbye", "exit", "sleep", "goodnight", "stop"]):
            speak("Okay, see you later! Call me when you're back.")
            break

        if len(user_input.split()) <= 3:
            enhanced = (
                f"The child said: '{user_input}'. "
                "React naturally, turn it into a quick fun story, "
                "and ask one follow-up question."
            )
        else:
            enhanced = user_input

        messages.append({"role": "user", "content": enhanced})
        print("🧠 Mitra is thinking...")

        try:
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=messages,
                temperature=0.9,
                max_tokens=200,
            )
            reply = response.choices[0].message.content
            reply = trim_to_sentence(reply, limit=900)
            messages.append({"role": "assistant", "content": reply})
        except Exception as e:
            print(f"❌ Groq error: {e}")
            speak("Oops, brain hiccup! Ask me again?")
            continue

        interrupt_flag.clear()
        inject_sounds_and_speak(reply)

        if interrupt_flag.is_set():
            interrupt_flag.clear()
            speak("Go ahead!")
            audio = record_audio(duration=5)
            user_input = transcribe(audio)
            if user_input:
                messages.append({"role": "user", "content": user_input})
                try:
                    followup = client.chat.completions.create(
                        model="llama-3.1-8b-instant",
                        messages=messages,
                        temperature=0.9,
                        max_tokens=200,
                    )
                    f_reply = followup.choices[0].message.content
                    f_reply = trim_to_sentence(f_reply, limit=900)
                    messages.append({"role": "assistant", "content": f_reply})
                    interrupt_flag.clear()
                    inject_sounds_and_speak(f_reply)
                except Exception as e:
                    print(f"❌ Groq error: {e}")

        time.sleep(0.1)
