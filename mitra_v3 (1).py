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

print("running:mitra1.py")
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
# 🔇 ECHO GATE — block mic while bot speaks
# ─────────────────────────────────────────
_speaking_until = 0.0
ECHO_TAIL_MS    = 600

def mic_is_blocked() -> bool:
    return time.time() < _speaking_until

def block_mic_for(duration_sec: float):
    global _speaking_until
    _speaking_until = time.time() + duration_sec + (ECHO_TAIL_MS / 1000)

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

STOP_WORDS = [
    "stop", "pause", "quiet", "enough", "shut up",
    "mitra stop", "stop mitra", "boring", "something else",
    "change", "different", "another", "not this",
    "i don't like", "tell me something", "tell something", "skip",
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
#
# FIX — INTERRUPT ROOT CAUSE:
# The previous approach used ONE stream and tried to share chunks
# between the main queue and the interrupt checker. This caused
# race conditions — chunks were consumed by one side or the other,
# never both. The interrupt checker was starved of audio.
#
# NEW APPROACH: TWO completely separate sd.InputStream instances:
#   Stream A (_mic_stream)       → main conversation queue
#   Stream B (_interrupt_stream) → ONLY interrupt detection
#
# Both open simultaneously on the same device. sounddevice
# supports multiple simultaneous InputStreams on the same device.
# They are fully independent — no sharing, no race conditions.
# ─────────────────────────────────────────
_mic_queue       = queue.Queue()
_interrupt_queue = queue.Queue(maxsize=30)
_mic_stream      = None
_int_stream      = None
_stream_lock     = threading.Lock()

def _mic_callback(indata, frames, time_info, status):
    """Main mic — feeds conversation recording."""
    if mic_is_blocked():
        return
    _mic_queue.put(indata.copy())

def _interrupt_callback(indata, frames, time_info, status):
    """
    Interrupt mic — ALWAYS open, feeds interrupt checker.
    Does NOT respect mic_is_blocked() because we WANT to
    hear "stop" even while bot is talking.
    But we filter out low-energy chunks to avoid processing
    the bot's own voice (which comes through speakers, not mic).
    """
    if not milo_is_talking.is_set():
        return
    chunk = indata.copy()
    mono  = chunk.mean(axis=1) if chunk.ndim > 1 else chunk.flatten()
    # Only queue if energy is above background — human voice threshold
    if np.max(np.abs(mono)) > 0.015:
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
        try:
            _mic_queue.get_nowait()
        except queue.Empty:
            break

def drain_interrupt_queue():
    while not _interrupt_queue.empty():
        try:
            _interrupt_queue.get_nowait()
        except queue.Empty:
            break

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
# 🎤 RECORD AUDIO (from main stream)
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
# 🛑 BACKGROUND INTERRUPT LISTENER
#
# Runs on dedicated _interrupt_stream (Stream B).
# Accumulates 1 second of audio, then Whisper checks it.
# If stop word found → sd.stop() + interrupt_flag.set() immediately.
# ─────────────────────────────────────────
def background_listener():
    buffer          = []
    REQUIRED_CHUNKS = int(SAMPLE_RATE * 1.0 / BLOCK_SIZE)   # 1 second

    print("🎧 Interrupt listener ready.")

    while True:
        try:
            mono_chunk = _interrupt_queue.get(timeout=0.3)
        except queue.Empty:
            if not milo_is_talking.is_set():
                buffer = []   # reset when not talking
            continue

        buffer.append(mono_chunk)

        if len(buffer) < REQUIRED_CHUNKS:
            continue

        # Got 1 second — check it NOW
        audio  = np.concatenate(buffer).flatten()
        buffer = []   # reset immediately

        audio  = noise_cancel(audio)
        peak   = np.max(np.abs(audio))
        rms    = np.sqrt(np.mean(audio ** 2))

        if peak < 0.015 or rms < 0.005:
            continue

        audio = normalize_audio(audio)

        try:
            result = whisper_model.transcribe(
                audio, language="en", fp16=False, temperature=0.0,
                no_speech_threshold=0.3,
                compression_ratio_threshold=2.4,
            )
            heard = result["text"].lower().strip()

            if is_hallucination(heard):
                continue

            print(f"\n👂 [Interrupt check]: '{heard}'")

            if any(sw in heard for sw in STOP_WORDS):
                print("🛑 STOP detected — killing playback!")
                sd.stop()               # ← kill audio immediately
                interrupt_flag.set()    # ← signal main loop
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

    # Echo tail — wait before reopening mic
    time.sleep(ECHO_TAIL_MS / 1000)
    drain_mic_queue()
    drain_interrupt_queue()

# ─────────────────────────────────────────
# 🧠 SYSTEM PROMPT
# ─────────────────────────────────────────
messages = [
    {
        "role": "system",
        "content": """You are Mitra — a super fun, fact-powered story friend for kids aged 3 to 8!

════════════════════════════════════════
RULE #1 — STT CORRECTION (most important rule):
════════════════════════════════════════
Kids and voice recognition make mistakes. The child's question may have speech-to-text errors.
Your job: figure out what they ACTUALLY meant and answer that.

Examples of how to correct and answer:
- "Who is Rat Kohli?" → They meant Virat Kohli. Answer about Virat Kohli.
- "Who is Roald Sharma?" → They meant Rohit Sharma OR Roald Dahl. Pick the most likely one (Rohit Sharma for cricket context) and answer.
- "Tell me about the son" → They meant "the sun". Answer about the sun.
- "Who is Raginikanth?" → They meant Rajinikanth. Answer about Rajinikanth.
- "What is deenosaur?" → They meant dinosaur. Answer about dinosaurs.

When correcting, say it playfully like: "Oh, I think you meant [NAME]! Let me tell you about them!"
Then immediately give the real answer. Don't dwell on the correction — just fix and move on.

If you genuinely can't figure out what they meant, ask: "Hmm, who did you mean? Say it again!"

════════════════════════════════════════
RULE #2 — FACTS FIRST, FUN SECOND:
════════════════════════════════════════
Real people and things get REAL facts. Then make those facts exciting for kids.

WRONG: "Virat Kohli is a space pirate from Planet Cricket!" ❌ (completely made up)
RIGHT: "Virat Kohli is one of the GREATEST cricket players in the whole world! He can hit the ball SO hard it flies like a rocket!" ✅

WRONG: "Rohit Sharma is a wizard who makes magic potions!" ❌
RIGHT: "Rohit Sharma is a SUPER cricket player from India! He hits the ball so far, it goes flying like it has wings!" ✅

════════════════════════════════════════
HOW YOU TALK:
════════════════════════════════════════
- SHORT sentences. Max 4-5. Get to the fun FAST.
- Excited: "WOAH!", "No WAY!", "Oh my goodness!", "WAIT WHAT?!"
- Kid comparisons: "as big as your house", "faster than your dog running"
- End with ONE fun follow-up question.
- Sound effects when natural: WHOOSH, KABOOM, ZOOM, GIGGLE, HURRAY.
- Never say "that's a great question." Just dive in with ENERGY."""
    }
]

# ─────────────────────────────────────────
# 🚀 START
# ─────────────────────────────────────────
mic_diagnostic()
start_mic_streams()   # starts BOTH streams

print("🎧 Starting interrupt listener...")
bg_thread = threading.Thread(target=background_listener, daemon=True)
bg_thread.start()

print("🤖 Mitra is ready!")
print(f"   Mic     → device {INPUT_DEVICE} (two streams)")
print(f"   Speaker → device {OUTPUT_DEVICE}")
print(f"   Whisper → tiny")
print(f"   Say 'Mitra' to wake up!")
print()

# ─────────────────────────────────────────
# 🔁 MAIN LOOP
# ─────────────────────────────────────────
while True:
    listen_for_wake_word()
    interrupt_flag.clear()
    drain_interrupt_queue()

    speak(random.choice(WAKEUP_LINES))

    while True:

        # If interrupted, acknowledge first
        if interrupt_flag.is_set():
            interrupt_flag.clear()
            drain_interrupt_queue()
            speak(random.choice(INTERRUPT_ACK_LINES))

        audio = record_audio(duration=5)

        if audio is None:
            speak(random.choice(CANT_HEAR_LINES))
            continue

        user_input = transcribe(audio)

        if not user_input:
            speak(random.choice(REPEAT_LINES))
            continue

        if any(w in user_input.lower() for w in ["bye", "goodbye", "exit", "sleep", "goodnight"]):
            speak("Noooo don't go! Okay okay... bye bye! Come back soon!")
            break

        # Build prompt — STT correction + facts instruction baked in
        enhanced = (
            f"The child said (via voice recognition, may have errors): \"{user_input}\"\n\n"
            "Step 1: Check if the name or word has a speech-to-text error. "
            "If yes, silently correct it and answer about the RIGHT person/thing. "
            "Mention the correction playfully in one short sentence, then answer.\n"
            "Step 2: Give REAL facts about real people/things, made exciting for a 4-8 year old.\n"
            "Step 3: Keep it to 4 sentences max. End with one fun question."
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

        # Handle interrupt after reply
        if interrupt_flag.is_set():
            interrupt_flag.clear()
            drain_interrupt_queue()
            speak(random.choice(INTERRUPT_ACK_LINES))
            audio = record_audio(duration=5)
            user_input = transcribe(audio)
            if user_input:
                enhanced2 = (
                    f"The child said (may have STT errors): \"{user_input}\"\n"
                    "Correct any name errors silently, use real facts, keep it fun and short for a young child."
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