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
import urllib.request
from urllib.error import URLError, HTTPError
from groq import Groq
try:
    from emotion_display import set_emotion, update_display
except Exception:
    def set_emotion(*args, **kwargs): pass
    def update_display(*args, **kwargs): pass
try:
    import pygame
except Exception:
    pygame = None

import json
import datetime

# ─────────────────────────────────────────
# 📊 DASHBOARD SERVER  (local WiFi only)
#
# Runs a tiny Flask HTTP server on port 5050 in a background thread.
# The parent opens  http://<PI_IP>:5050  in any browser on the same WiFi.
#
# Endpoints:
#   GET /api/status   → Pi online flag, child name, session start
#   GET /api/session  → full session summary (counts, duration)
#   GET /api/events   → chronological event feed
#
# Install once:  pip install flask flask-cors
# Zero changes to existing features — purely additive.
# ─────────────────────────────────────────

_dashboard_events   = []
_session_start_time = None
_dashboard_lock     = threading.Lock()

_dash_emotion_counts  = {}
_dash_rhymes_played   = []
_dash_langs_taught    = []
_dash_call_count      = 0
_dash_auto_call_count = 0

# Forward-declared — filled in after _child_name and _distress_count are defined
_child_name    = None
_distress_count = 0
_target_pi_ip   = None

def _ts():
    return datetime.datetime.now().isoformat(timespec="seconds")

def _now_str():
    return datetime.datetime.now().strftime("%H:%M:%S")

def log_event(event_type: str, label: str, detail: str = "", severity: str = "info"):
    global _dash_call_count, _dash_auto_call_count
    with _dashboard_lock:
        evt = {
            "id":        len(_dashboard_events) + 1,
            "timestamp": _ts(),
            "time":      _now_str(),
            "type":      event_type,
            "label":     label,
            "detail":    detail,
            "severity":  severity,
        }
        _dashboard_events.append(evt)
        if event_type == "emotion":
            _dash_emotion_counts[detail] = _dash_emotion_counts.get(detail, 0) + 1
        elif event_type == "rhyme" and detail:
            _dash_rhymes_played.append(detail)
        elif event_type == "language" and detail:
            if detail not in _dash_langs_taught:
                _dash_langs_taught.append(detail)
        elif event_type == "call":
            _dash_call_count += 1
        elif event_type == "auto_call":
            _dash_auto_call_count += 1

def reset_dashboard_session():
    global _session_start_time, _dashboard_events
    global _dash_emotion_counts, _dash_rhymes_played
    global _dash_langs_taught, _dash_call_count, _dash_auto_call_count
    with _dashboard_lock:
        _session_start_time   = _ts()
        _dashboard_events     = []
        _dash_emotion_counts  = {}
        _dash_rhymes_played   = []
        _dash_langs_taught    = []
        _dash_call_count      = 0
        _dash_auto_call_count = 0
    log_event("session_start", "Session started", severity="info")

def _get_session_summary():
    with _dashboard_lock:
        dur_str = "—"
        if _session_start_time:
            try:
                start   = datetime.datetime.fromisoformat(_session_start_time)
                secs    = int((datetime.datetime.now() - start).total_seconds())
                dur_str = f"{secs // 60}m {secs % 60}s"
            except Exception:
                pass
        dominant = max(_dash_emotion_counts, key=_dash_emotion_counts.get) \
                   if _dash_emotion_counts else "neutral"
        return {
            "session_start":    _session_start_time or "—",
            "duration":         dur_str,
            "child_name":       _child_name or "—",
            "dominant_mood":    dominant,
            "emotion_counts":   dict(_dash_emotion_counts),
            "rhymes_played":    list(_dash_rhymes_played),
            "languages_taught": list(_dash_langs_taught),
            "call_count":       _dash_call_count,
            "auto_call_count":  _dash_auto_call_count,
            "distress_count":   _distress_count,
            "total_events":     len(_dashboard_events),
        }

def _start_dashboard_server():
    try:
        from flask import Flask, jsonify, request
        try:
            from flask_cors import CORS
            _has_cors = True
        except ImportError:
            _has_cors = False
    except ImportError:
        print("⚠️  Dashboard unavailable — install: pip install flask flask-cors")
        return

    app = Flask(__name__)
    if _has_cors:
        from flask_cors import CORS as _CORS
        _CORS(app)

    def _fetch_pi_json(path):
        if not _target_pi_ip:
            raise RuntimeError("Pi IP not configured")
        url = f"http://{_target_pi_ip}:5050/api{path}"
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            payload = resp.read().decode("utf-8")
            return json.loads(payload)

    @app.route("/api/config", methods=["GET", "POST"])
    def api_config():
        global _target_pi_ip
        if request.method == "POST":
            payload = request.get_json(silent=True) or {}
            pi_ip = payload.get("pi_ip") if isinstance(payload, dict) else None
            if not pi_ip or not isinstance(pi_ip, str) or not pi_ip.strip():
                return jsonify({"ok": False, "error": "pi_ip is required"}), 400
            _target_pi_ip = pi_ip.strip()
            return jsonify({"ok": True, "pi_ip": _target_pi_ip})
        return jsonify({"ok": True, "pi_ip": _target_pi_ip})

    @app.route("/api/status")
    def api_status():
        if _target_pi_ip:
            try:
                return jsonify(_fetch_pi_json("/status"))
            except Exception:
                pass
        return jsonify({
            "pi_online":          True,
            "child_name":         _child_name or "—",
            "session_start":      _session_start_time or "—",
            "distress_count":     _distress_count,
            "distress_threshold": _DISTRESS_CALL_AT,
            "internet_ok":        _internet_ok,
        })

    @app.route("/api/session")
    def api_session():
        if _target_pi_ip:
            try:
                return jsonify(_fetch_pi_json("/session"))
            except Exception:
                pass
        return jsonify(_get_session_summary())

    @app.route("/api/events")
    def api_events():
        if _target_pi_ip:
            try:
                return jsonify(_fetch_pi_json("/events"))
            except Exception:
                pass
        with _dashboard_lock:
            return jsonify(list(reversed(_dashboard_events)))

    @app.route("/api/clear", methods=["POST"])
    def api_clear():
        reset_dashboard_session()
        return jsonify({"ok": True})

    # ── PARENT VOICE UPLOAD ──────────────────────────────────────────
    _ALLOWED_IDS = [
        "comfort_1","comfort_2","comfort_3","comfort_4","comfort_5",
        "scared_1","scared_2","greet_1","greet_2",
        "bedtime_1","bedtime_2","bedtime_3","praise_1","praise_2",
    ]

    @app.route("/api/upload_voice", methods=["POST"])
    def api_upload_voice():
        try:
            phrase_id  = request.form.get("phrase_id", "").strip()
            if phrase_id not in _ALLOWED_IDS:
                return jsonify({"ok": False, "error": "unknown phrase_id"}), 400
            audio_file = request.files.get("audio")
            if not audio_file:
                return jsonify({"ok": False, "error": "audio file required"}), 400
            os.makedirs(PARENT_VOICE_DIR, exist_ok=True)
            dest = os.path.join(PARENT_VOICE_DIR, f"parent_voice_{phrase_id}.wav")
            audio_file.save(dest)
            _parent_voice_cache.pop(phrase_id, None)
            log_event("voice_upload", f"Parent voice saved: {phrase_id}", detail=phrase_id, severity="info")
            return jsonify({"ok": True, "saved": f"parent_voice_{phrase_id}.wav"})
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/voice_files", methods=["GET"])
    def api_voice_files():
        result = {}
        for pid in _ALLOWED_IDS:
            path = os.path.join(PARENT_VOICE_DIR, f"parent_voice_{pid}.wav")
            result[pid] = os.path.exists(path)
        return jsonify(result)

    @app.route("/api/delete_voice", methods=["POST"])
    def api_delete_voice():
        payload   = request.get_json(silent=True) or {}
        phrase_id = (payload.get("phrase_id") or "").strip()
        if phrase_id not in _ALLOWED_IDS:
            return jsonify({"ok": False, "error": "unknown phrase_id"}), 400
        path = os.path.join(PARENT_VOICE_DIR, f"parent_voice_{phrase_id}.wav")
        if os.path.exists(path):
            os.remove(path)
            _parent_voice_cache.pop(phrase_id, None)
        return jsonify({"ok": True})

    # ── SSE PUSH NOTIFICATIONS ───────────────────────────────────────
    _sse_subs  = []
    _sse_lk    = threading.Lock()

    def _broadcast_alert(event_type, label, detail, severity):
        if severity not in ("warning", "alert"):
            return
        import queue as _sq
        payload = json.dumps({
            "type": event_type, "label": label,
            "detail": detail, "severity": severity,
            "time": _now_str(),
        })
        msg = f"data: {payload}\n\n"
        with _sse_lk:
            dead = []
            for q in _sse_subs:
                try: q.put_nowait(msg)
                except Exception: dead.append(q)
            for q in dead:
                try: _sse_subs.remove(q)
                except Exception: pass

    _real_log = log_event
    def _log_push(et, lb, dt="", sv="info"):
        _real_log(et, lb, dt, sv)
        _broadcast_alert(et, lb, dt, sv)
    app._log_push = _log_push

    @app.route("/api/stream")
    def api_stream():
        import queue as _sq2
        from flask import Response, stream_with_context
        cq = _sq2.Queue(maxsize=50)
        with _sse_lk:
            _sse_subs.append(cq)
        def _gen():
            yield 'data: {"type":"connected"}\n\n'
            while True:
                try: yield cq.get(timeout=20)
                except Exception: yield 'data: {"type":"ping"}\n\n'
        return Response(
            stream_with_context(_gen()),
            mimetype="text/event-stream",
            headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"},
        )

    @app.route("/api/trigger_call", methods=["POST"])
    def api_trigger_call():
        log_event("call", "Call triggered from dashboard", severity="warning")
        return jsonify({"ok": True})

    def _run():
        import logging as _lg
        _lg.getLogger("werkzeug").setLevel(_lg.ERROR)
        try:
            app.run(host="0.0.0.0", port=5050, debug=False, use_reloader=False)
        except OSError as e:
            print(f"⚠️  Dashboard port 5050 in use: {e}")
    t = threading.Thread(target=_run, daemon=True)
    t.start()
    print("📊 Dashboard server → http://<PI_IP>:5050  (same WiFi)")

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
    """Stub — pyttsx3 replaced by espeak-ng on Raspberry Pi."""
    global _pyttsx3_engine
    if _pyttsx3_engine is not None:
        return _pyttsx3_engine
    # Check espeak-ng is available instead
    try:
        subprocess.run(["espeak-ng", "--version"], capture_output=True, timeout=3)
        print("✅ espeak-ng offline TTS ready.")
    except Exception:
        print("⚠️ espeak-ng not found — run: sudo apt-get install espeak-ng")
    return None

def speak_offline(text: str):
    """
    Offline TTS using espeak-ng — pre-installed on Raspberry Pi OS.
    No pyttsx3 needed. Install: sudo apt-get install espeak-ng
    """
    with _pyttsx3_lock:
        tmp = tempfile.mktemp(suffix=".wav")
        try:
            subprocess.run(
                ["espeak-ng", "-v", "en+f3", "-s", "150", "-a", "200",
                 "-w", tmp, text.strip()],
                check=True, capture_output=True, timeout=10
            )
            if os.path.exists(tmp) and os.path.getsize(tmp) > 200:
                data, fs = sf.read(tmp)
                os.remove(tmp)
                print(f"\U0001f916 [offline TTS] Mitra: {text.strip()}")
                _safe_play(data, fs)
                sd.wait()
                return
            else:
                print("\u26a0\ufe0f espeak-ng produced empty file")
                if os.path.exists(tmp): os.remove(tmp)
        except FileNotFoundError:
            print("\u26a0\ufe0f espeak-ng not found — run: sudo apt-get install espeak-ng")
        except Exception as e:
            print(f"\u26a0\ufe0f Offline TTS error: {e}")
            if os.path.exists(tmp):
                try: os.remove(tmp)
                except: pass
        print(f"   [NO AUDIO] {text.strip()}")
_GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "your_groq_api_key_here")

def _make_groq_client():
    return Groq(api_key=_GROQ_API_KEY)

client = _make_groq_client()

# ─────────────────────────────────────────
# 😀 DISPLAY INIT
# ─────────────────────────────────────────
def init_emotion_display():
    try:
        if pygame:
            pygame.init()
        t = threading.Thread(target=update_display, daemon=True)
        t.start()
        set_emotion("sleeping")
        print("✅ Emotion display ready.")
    except Exception as e:
        print(f"⚠️ Emotion display unavailable: {e}")

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
# Hardware detected on YOUR Raspberry Pi:
#   USB Mic : Texas Instruments PCM2902 Audio Codec (ALSA card 3)
#   Speaker : bcm2835 Headphones — Pi 3.5mm jack   (ALSA card 0)
#
# We match by device name so this survives reboots.
# ─────────────────────────────────────────

def _show_devices():
    import sounddevice as _sd
    print("\n🔍 Audio devices on this Pi:")
    for i, d in enumerate(_sd.query_devices()):
        i_tag = f"MIC ch={d['max_input_channels']}"   if d['max_input_channels']  > 0 else ""
        o_tag = f"SPKR ch={d['max_output_channels']}" if d['max_output_channels'] > 0 else ""
        tags  = "  ".join(t for t in [i_tag, o_tag] if t)
        print(f"  [{i:2d}] {d['name']:<40} {tags}")
    print()

def _find_by_name(kw, want_input):
    import sounddevice as _sd
    ck = 'max_input_channels' if want_input else 'max_output_channels'
    for i, d in enumerate(_sd.query_devices()):
        if d[ck] > 0 and kw.lower() in d['name'].lower():
            return i
    for i, d in enumerate(_sd.query_devices()):
        if d[ck] > 0:
            return i
    return 0

_show_devices()

INPUT_DEVICE    = _find_by_name("PCM2902",  want_input=True)   # TI USB mic
OUTPUT_DEVICE   = _find_by_name("bcm2835",  want_input=False)  # Pi 3.5mm jack

print(f"🎤 USB Mic  → sounddevice [{INPUT_DEVICE}]")
print(f"🔊 Speaker  → sounddevice [{OUTPUT_DEVICE}]")

INPUT_CHANNELS  = 1       # PCM2902 USB mic is mono
SAMPLE_RATE     = 16000
BLOCK_SIZE      = 1024    # lower than default = less latency on Pi 4
VOICE           = "en-US-AriaNeural"

def _out_channels():
    import sounddevice as _sd
    try:
        return min(_sd.query_devices(OUTPUT_DEVICE)['max_output_channels'], 2)
    except Exception:
        return 2
OUTPUT_CHANNELS = _out_channels()
print(f"   Output channels : {OUTPUT_CHANNELS}\n")

# ─────────────────────────────────────────
# 🔊 SAFE PLAY — normalises channels to match output device
# Fixes PaErrorCode -9998 (invalid channel count) and
# _CallbackContext channels conflict on AUX output.
# ─────────────────────────────────────────
def _safe_play(data, fs):
    """
    Play numpy audio through OUTPUT_DEVICE, always matching
    the device's channel count so PortAudio never rejects it.
    - Mono   (1-ch) data → duplicate to stereo if device needs 2ch
    - Stereo (2-ch) data → mix down to mono if device needs 1ch
    - Resample to SAMPLE_RATE so playback speed is correct
    """
    import numpy as np
    import sounddevice as _sd

    # ── 1. Ensure float32 ────────────────────────────────────────
    if data.dtype != np.float32:
        data = data.astype(np.float32)

    # ── 2. Normalise shape to 2-D (samples, channels) ───────────
    if data.ndim == 1:
        data = data[:, np.newaxis]   # (N,) → (N,1)

    src_ch = data.shape[1]
    tgt_ch = OUTPUT_CHANNELS

    # ── 3. Channel-count conversion ─────────────────────────────
    if src_ch == tgt_ch:
        out = data
    elif src_ch == 1 and tgt_ch == 2:
        out = np.hstack([data, data])          # mono → stereo
    elif src_ch == 2 and tgt_ch == 1:
        out = data.mean(axis=1, keepdims=True) # stereo → mono
    else:
        # More than 2 channels — mix everything down to tgt_ch
        out = data[:, :tgt_ch] if src_ch > tgt_ch else np.hstack([data] * (tgt_ch // src_ch + 1))[:, :tgt_ch]

    # ── 4. Play and wait ─────────────────────────────────────────
    _sd.play(out, fs, device=OUTPUT_DEVICE)


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
        _safe_play(data, fs)
        while sd.get_stream().active:
            if interrupt_flag.is_set():
                sd.stop()
                break
            time.sleep(0.05)
    except Exception as e:
        print(f"❌ Parent voice playback error: {e}")

    milo_is_talking.clear()
    set_emotion("happy")
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
# 😢 DISTRESS ESCALATION COUNTER
# ─────────────────────────────────────────
# Tracks how many times sad/scared has triggered in this session.
# After 2 triggers → auto-call parent on the 3rd.
# Resets each time Mitra goes back to sleep (new session).
_distress_count   = 0
_DISTRESS_CALL_AT = 3   # call parent on the Nth distress trigger

# ─────────────────────────────────────────
# 👤 NAME MEMORY
# ─────────────────────────────────────────
# Mitra asks the child's name on the very first wake of the session.
# Stored here and injected into the system prompt so LLM uses it.
_child_name      = None   # None = not yet asked
_name_asked      = False  # True after we've asked this session

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
        _safe_play(data, fs)
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
# These stop Mitra MID-SENTENCE during playback AND immediately end
# her current response from the main loop. Checked in BOTH the
# background interrupt listener AND at every main loop iteration.
# Designed to be short, distinct, hard to mishear.
INSTANT_STOP_WORDS = [
    # Direct stop commands
    "stop", "stop it", "stop now", "just stop",
    "mitra stop", "stop mitra", "hey stop",
    # Equivalent commands
    "pause", "quiet", "silence", "shush", "shhh",
    "enough", "enough now", "that's enough",
    "shut up", "be quiet",
    # Skip/change commands
    "skip", "skip it", "next", "change",
    "something else", "different topic", "not this",
    "tell something else", "tell me something else",
    # Boredom signals
    "boring", "i don't like this", "i don't like",
    "not interested",
    # ── GOODBYE / SLEEP — also act as stop + sleep trigger ──────────
    "bye", "bye bye", "bye-bye", "goodbye", "good bye",
    "mitra bye", "mitra goodbye", "mitra sleep",
    "see ya", "see you", "see you later",
    "good night", "goodnight", "good night mitra",
    "go to sleep", "i'm done", "im done", "stop talking",
]

# ── REGULAR STOP WORDS (legacy, kept for compatibility) ──────────────────
STOP_WORDS = INSTANT_STOP_WORDS   # unified — same list used everywhere

# Words that mean GOODBYE (sleep) — subset of INSTANT_STOP_WORDS
_GOODBYE_WORDS_SET = {
    "bye", "bye bye", "bye-bye", "goodbye", "good bye",
    "mitra bye", "mitra goodbye", "mitra sleep",
    "see ya", "see you", "see you later",
    "good night", "goodnight", "good night mitra",
    "go to sleep", "i'm done", "im done", "stop talking",
}

def is_stop_command(text: str) -> bool:
    """Returns True if the transcribed text is a stop/interrupt command."""
    if not text:
        return False
    t = text.lower().strip()
    return any(sw in t for sw in INSTANT_STOP_WORDS)

def is_goodbye_command(text: str) -> bool:
    """Returns True if the text is specifically a goodbye/sleep command."""
    if not text:
        return False
    import re as _re2
    t = text.lower().strip()

    # Reject hallucinations FIRST — e.g. "we'll see you in the next video"
    # contains "see you" but is NOT a goodbye from the child.
    if is_hallucination(t):
        return False

    # "bye" with word boundary — very reliable, even alone
    if _re2.search(r'\bbye\b', t):
        return True

    # "see you" / "see ya" — only count if the utterance is SHORT
    # (a long sentence containing "see you" is almost certainly a hallucination)
    for sv in ("see you", "see ya"):
        if sv in t and len(t) <= len(sv) + 20:
            return True

    # All other goodbye phrases — direct substring match
    _GOODBYE_EXACT = {
        "bye bye", "bye-bye", "goodbye", "good bye",
        "mitra bye", "mitra goodbye", "mitra sleep",
        "good night", "goodnight", "good night mitra",
        "go to sleep", "i'm done", "im done", "stop talking",
    }
    return any(gw in t for gw in _GOODBYE_EXACT)

WAKEUP_LINES = [
    "Whoa! I was dreaming I was a giant cookie being chased by a milk monster... and YOU woke me up! So what's the new story?",
    "Oh oh oh! I was just flying on a rainbow dinosaur to Planet Cheese... and now I'm HERE! What do you wanna talk about?",
    "Yaaaawn! I was dreaming I was a superhero whose power was making pizza appear... then you called! What's going on?",
    "Woah woah woah! I was right in the middle of a dance battle with a talking cactus... and now I'm awake! What's up?",
    "Eeeek! I was dreaming I found a treasure chest full of... socks? Weird dream. Anyway HI! What are we doing?",
    "Ooooh! I was just about to win a burping contest against a dragon... and you called! So tell me EVERYTHING!",
    "Whoa! I was dreaming I was a tiny astronaut inside a giant donut in space... and now YOU'RE here! What's the story?",
    "Oh WOW! I was dreaming about a purple elephant who could only talk in rhymes! So what do you wanna do?",
]

# ── SLEEP OUTROS — deliberately contain NO goodbye/bye/see ya words ──────
# Mitra's speaker bleed can make Whisper hear her own words.
# Removing trigger words from estos lines prevents self-triggering sleep.
SLEEP_OUTRO_LINES = [
    "Okay, nap time for me! Just say Mitra whenever you need me, I'll be right here!",
    "I'll be napping until you call my name — Mitra — and I'll jump right up!",
    "Rest time for me! Whenever you want to play again, just say Mitra!",
    "I'm going to close my eyes now! Call my name — Mitra — and I'll wake up instantly!",
    "Snooze time! Just say Mitra and I'll be back before you can say supercalifragilistic!",
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
    "thanks", "thank", "okay", "ok", "bye", "hi",
    "yes", "no", "oh", "wow", "hey",
}
# Whisper hallucinates these phrases from TTS speaker bleed.
# When the mic picks up Mitra's own voice, Whisper often outputs
# YouTube/podcast clichés because they appear constantly in training data.
# Any match here → treated as silence, NEVER as a stop/goodbye command.
HALLUCINATION_CONTAINS = [
    # YouTube phrases — most common Whisper bleed hallucinations
    "a little bit of a little bit",
    "thank you for watching",
    "please like and subscribe",
    "we'll see you in the next",
    "see you in the next video",
    "we'll see you next",
    "see you next time",
    "don't forget to subscribe",
    "like and subscribe",
    "hit the subscribe",
    "bell icon",
    "click the link",
    "in the next episode",
    "stay tuned",
    "that's all for today",
    "until next time",
    "see you soon",                 # too generic — catches bleed but not goodbyes
    "thanks for listening",
    "this has been",
    "join us next",
    "coming up next",
    "don't forget to",
    "if you enjoyed",
    "leave a comment",
    "share this video",
    # Repetition hallucination
    "you you you",
    "the the the",
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
# 🎙️ SINGLE SHARED MIC STREAM
# (USB 2.0 mics only support ONE open stream at a time.
#  We use ONE stream with ONE combined callback that feeds
#  BOTH queues — same behaviour, zero hardware conflict.)
# ─────────────────────────────────────────
_mic_queue       = queue.Queue()
_interrupt_queue = queue.Queue(maxsize=80)   # large enough for 6s of chunks
_mic_stream      = None
_int_stream      = None          # kept for API compatibility — not used
_stream_lock     = threading.Lock()

def _combined_callback(indata, frames, time_info, status):
    """
    Single callback for the one physical USB mic.

    Feeds _mic_queue  → used by record_audio / listen_for_wake_word
    Feeds _interrupt_queue → used by background_listener while Mitra speaks

    kid voice peak ranges (measured):
      3-5 yr old soft:   0.08 - 0.15
      6-8 yr old normal: 0.12 - 0.25
      TTS speaker bleed: 0.05 - 0.18 (handled by POST_SPEECH_COOLDOWN)
    Gate set at 0.10 — cooldown is the real bleed guard, not this gate.
    """
    chunk = indata.copy()
    # Mono-safe: flatten regardless of channel count
    mono  = chunk.mean(axis=1) if chunk.ndim > 1 else chunk.flatten()

    # ── Feed main recording queue (when mic not blocked) ──────────
    if not mic_is_blocked():
        _mic_queue.put(chunk)

    # ── Feed interrupt queue (only while Mitra is speaking) ───────
    if milo_is_talking.is_set() and np.max(np.abs(mono)) > 0.10:
        try:
            _interrupt_queue.put_nowait(mono)
        except queue.Full:
            try:
                _interrupt_queue.get_nowait()
                _interrupt_queue.put_nowait(mono)
            except Exception:
                pass

def start_mic_streams():
    global _mic_stream
    with _stream_lock:
        if _mic_stream is None:
            _mic_stream = sd.InputStream(
                samplerate=SAMPLE_RATE, channels=INPUT_CHANNELS,
                dtype='float32', blocksize=BLOCK_SIZE,
                device=INPUT_DEVICE, callback=_combined_callback,
            )
            _mic_stream.start()
            print("🎙️ Single shared mic stream started (USB 2.0 compatible).")

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
    # Non-ASCII characters in transcription = Whisper hallucinating on noise
    # Real child speech in English is always pure latin/ASCII characters
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

    # ── VOICED FRAME GUARD ──────────────────────────────────
    # Short clips with mostly silence = noise/background, not an emotion
    if voiced_ratio < 0.20 and word_count <= 3:
        print(f"   🔇 Noise guard: voiced_ratio={voiced_ratio:.2f}, wc={word_count} → neutral")
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
    # KEY FIX: Loud volume alone does NOT mean excited.
    # Kids speak loudly even when sad, sick, or upset.
    # Loud speech only gets a tiny excited nudge — real excitement
    # is detected via ZCR (pitch) and speech rate instead.
    # Only give sad credit for GENUINELY quiet speech (< 0.018)
    if rms < 0.012:
        scores["sad"]     += 3.0   # near silence = strongly sad/withdrawn
        energy_level = "whisper"
    elif rms < 0.022:
        scores["sad"]     += 1.5   # genuinely quiet voice
        energy_level = "soft"
    elif rms < 0.060:
        # Normal speech range — no bias in either direction
        energy_level = "medium"
    elif rms < 0.090:
        scores["excited"] += 0.3   # FIX: was +1.0, reduced — loud ≠ excited
        energy_level = "loud"
    else:
        scores["excited"] += 0.5   # FIX: was +2.5, reduced — very loud ≠ excited
        energy_level = "very_loud"

    # ── Feature 2: ZCR (weighted pitch) ──────────────────────
    # FIX v17: Tightened further — logs show zcr=0.056 on normal loud speech
    # (rms=0.09) still scoring +2.0 sad because 0.056 < 0.072.
    # Normal range on this mic is 0.065-0.14. Only truly flat (<0.050) = sad.
    if zcr < 0.042:
        scores["sad"]     += 3.0
    elif zcr < 0.052:
        scores["sad"]     += 2.0
    elif zcr < 0.065:
        scores["sad"]     += 0.5   # borderline — needs ALL other features too
    elif zcr < 0.125:
        pass                        # normal range — no bias
    elif zcr < 0.155:
        scores["excited"] += 1.5
    else:
        scores["excited"] += 3.0

    pitch_level = "low" if zcr < 0.065 else ("high" if zcr > 0.140 else "medium")

    # ── Feature 3: Energy variability ────────────────────────
    # Very flat (low var) = monotone = sad
    # Only credit sad for VERY flat delivery
    if energy_var < 0.003:
        scores["sad"]     += 2.5   # near-perfectly flat = strong sad signal
    elif energy_var < 0.008:
        scores["sad"]     += 1.5
    elif energy_var < 0.020:
        scores["sad"]     += 0.5
    elif energy_var < 0.050:
        pass                        # normal animated speech
    elif energy_var < 0.090:
        scores["excited"] += 1.0
    else:
        scores["excited"] += 2.0

    # ── Feature 4: Speech rate ────────────────────────────────
    # Scale down for short clips (short phrases are naturally slow).
    # ROOT-CAUSE FIX: slow rate on a LOUD voice just means a short sentence
    # was spoken inside a 5-second window — not sadness.  Apply sad credit
    # for slow rate ONLY when the voice is genuinely quiet (rms < 0.045).
    rate_scale    = 1.0 if word_count >= 5 else 0.4
    long_discount = 0.6 if (word_count >= 7 and silence_ratio < 0.72) else 1.0
    _rate_sad_gate = rms < 0.045   # loud voice → slow rate ≠ sad
    if speech_rate < 0.04:
        if _rate_sad_gate:
            scores["sad"]     += 2.5 * rate_scale * long_discount
    elif speech_rate < 0.09:
        if _rate_sad_gate:
            scores["sad"]     += 1.5 * rate_scale * long_discount
    elif speech_rate < 0.18:
        pass                        # normal rate
    elif speech_rate < 0.30:
        scores["excited"] += 1.0
    else:
        scores["excited"] += 2.5

    # ── Feature 5: Pitch trend ────────────────────────────────
    # Positive = falling pitch = sad/tired
    # Only credit clearly falling, not ambiguous small changes
    if pitch_fall > 0.018:
        scores["sad"]     += 2.0
    elif pitch_fall > 0.009:
        scores["sad"]     += 1.0

    # ── Feature 6: Silence ratio (quiet + longer utterances only) ──────
    # ROOT-CAUSE FIX: in a 5-second recording, any short sentence has high
    # silence ratio — that's not sadness.  Only credit sad silence when the
    # voice is genuinely quiet (rms < 0.045) AND the child said enough words.
    if word_count >= 6 and rms < 0.045:
        if silence_ratio > 0.75:
            scores["sad"]     += 2.0
        elif silence_ratio > 0.65:
            scores["sad"]     += 1.0
        elif silence_ratio > 0.55:
            scores["sad"]     += 0.3

    # ── Text hint soft bias (from actual words said) ──────────
    # FIX: Stronger bias (+2.0 per match instead of +1.0) because
    # text content is a highly reliable signal for sadness.
    if text_hint:
        t = text_hint.lower()
        # Strong sad phrases — multiple matches stack
        strong_sad = [
            "not feeling good", "feel sick", "feeling sick", "not well",
            "not okay", "not feel", "don't feel", "dont feel",
            "something wrong", "what is going on", "don't know what",
        ]
        for w in strong_sad:
            if w in t:
                scores["sad"] += 2.0
                break
        # Softer sad phrases
        soft_sad = ["feel bad", "feel sad", "hurt", "miss", "alone", "nobody", "nothing"]
        for w in soft_sad:
            if w in t:
                scores["sad"] += 1.0
                break

    scores = {k: max(0.0, v) for k, v in scores.items()}

    # ── Decision ─────────────────────────────────────────────
    # FIX v17: Raised SAD_THRESHOLD to 7.5.
    # Logs show normal loud speech hitting sad=3.0-4.0 (zcr=0.056 low + energy_var).
    # A genuinely sad child (whisper + flat + slow + falling) should score 8-12.
    # 7.5 ensures only true convergence of ALL sad features triggers.
    SAD_THRESHOLD     = 7.5
    EXCITED_THRESHOLD = 4.0

    # FIX v17: Stronger energy veto.
    # Logs: rms=0.09 (very_loud), zcr=0.056 → still scoring sad=3-4.
    # If rms > 0.045 (medium-loud child voice) → add +3.5 to threshold.
    # A loud voice is almost never sad — makes sad nearly impossible on loud speech.
    if rms > 0.045:
        SAD_THRESHOLD += 3.5
    elif rms > 0.030:
        SAD_THRESHOLD += 1.5   # medium voice — slight raise

    # SCARED_THRESHOLD intentionally unused — scored as 0, never triggers from voice
    if scores["sad"] >= SAD_THRESHOLD:
        voice_emotion = "sad"
    elif scores["excited"] >= EXCITED_THRESHOLD:
        voice_emotion = "excited"
    else:
        voice_emotion = "neutral"

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
          f"var={energy_var:.5f}, rate={speech_rate:.2f}, wc={word_count}")
    print(f"   📊 Scores → sad={scores['sad']:.1f}  excited={scores['excited']:.1f}"
          f"  → {voice_emotion}")
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
    set_emotion("listening")
    print("🎤 Your turn! Speak clearly...")

    chunks    = []
    n_target  = int(duration * SAMPLE_RATE)
    collected = 0
    # FIX: extra 1.0s grace period — sometimes the mic queue lags slightly
    deadline  = time.time() + duration + 1.0

    while collected < n_target and time.time() < deadline:
        try:
            chunk = _mic_queue.get(timeout=0.3)
            chunks.append(chunk)
            collected += chunk.shape[0]
        except queue.Empty:
            continue

    if not chunks:
        print("🔇 No audio chunks received at all.")
        return None, None

    audio = np.concatenate(chunks, axis=0)
    audio = to_mono(audio)

    # FIX: check RAW peak BEFORE noise_cancel so we don't lose
    # the signal level info (noise_cancel can zero out soft speech)
    raw_peak = np.max(np.abs(audio))
    raw_rms  = np.sqrt(np.mean(audio ** 2))
    print(f"   peak={raw_peak:.5f}  rms={raw_rms:.5f}")

    # FIX: Lowered silence gate — old 0.003/0.002 was catching real soft speech.
    # Only reject if it's truly dead silence (peak < 0.001).
    if raw_peak < 0.001:
        print("🔇 Dead silence — nothing recorded.")
        return None, None

    # FIX: if rms is very low but peak is OK, still proceed —
    # child may have spoken just one short word.
    if raw_rms < 0.001 and raw_peak < 0.005:
        print("🔇 Too quiet — only background noise detected.")
        return None, None

    # Keep RAW copy before noise cancel for emotion analysis
    raw_audio = audio.copy()

    # Apply noise cancel AFTER raw copy
    audio = noise_cancel(audio)

    # FIX: if noise_cancel zeroed everything out (over-aggressive),
    # fall back to the raw audio for Whisper
    if np.max(np.abs(audio)) < 0.001:
        print("   ⚠️ noise_cancel zeroed audio — using raw for Whisper")
        audio = raw_audio.copy()

    audio = normalize_audio(audio)
    print("✅ Got your voice!")
    return audio, raw_audio

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
GOODBYE_INTERRUPT = [
    "mitra bye", "mitra sleep", "mitra goodbye",
    "bye bye", "goodbye", "go to sleep",
    "i'm done", "im done", "see you later", "see ya",
    "stop talking", "good night", "goodnight",
    "bye", "see you", "good bye",
]

def _fire_interrupt(heard: str, source: str = ""):
    """Shared handler — called from both fast and thorough pass.
    Checks goodbye first (sets sleep_flag), then any stop command."""
    if not heard or is_hallucination(heard):
        return False
    print(f"\n👂 [Interrupt {source}]: '{heard}'")
    # Goodbye check first — sets sleep_flag so main loop breaks to sleep
    if is_goodbye_command(heard):
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
    FAST_CHUNKS     = 4   # ~320ms
    THOROUGH_CHUNKS = 8   # ~640ms

    # ENERGY GATES — tuned for 3-8 year old kids:
    # 3-5 yr soft voice peak: 0.08-0.15
    # 6-8 yr normal voice peak: 0.12-0.25
    # TTS speaker bleed at mic: 0.05-0.18
    #
    # Strategy: gate set BELOW child voice floor (0.12) so even a
    # quiet 3yr old "stop" gets through. The POST_SPEECH_COOLDOWN
    # (4.0s) is the PRIMARY bleed guard — not this gate.
    FAST_ENERGY_GATE = 0.12   # catches soft 3yr old voice
    SLOW_ENERGY_GATE = 0.10   # thorough pass slightly more lenient

    # PRIMARY BLEED GUARD: 4.0s cooldown after Mitra finishes speaking.
    # During this window ALL chunks are silently discarded, regardless
    # of energy. This is what stops self-triggering — not a high gate.
    # Why 4.0s: Mitra's TTS echo on this device decays in ~2-3s.
    # Extra 1s margin = safe on any room acoustics.
    POST_SPEECH_COOLDOWN = 4.0
    _cooldown_until = 0.0

    buffer       = []
    _was_talking = False
    print("🎧 Interrupt listener ready (kid-voice v5).")

    while True:
        try:
            mono_chunk = _interrupt_queue.get(timeout=0.15)
        except queue.Empty:
            currently_talking = milo_is_talking.is_set()
            if _was_talking and not currently_talking:
                if buffer:
                    print(f"   🧹 Purging {len(buffer)} stale audio chunks (Mitra finished talking)")
                    buffer = []
                # Start cooldown the moment Mitra stops talking
                _cooldown_until = time.time() + POST_SPEECH_COOLDOWN
            _was_talking = currently_talking
            if not currently_talking:
                buffer = []
            continue

        # ── PRIMARY BLEED GUARD: cooldown window ──────────────────────
        # Discard ALL audio for 4.0s after Mitra stops — this is the
        # definitive fix for self-triggering. No gate can be more reliable.
        if time.time() < _cooldown_until:
            continue

        # ── SECONDARY GUARD: only process while Mitra is actively talking ─
        if not milo_is_talking.is_set():
            buffer = []
            continue

        _was_talking = True
        buffer.append(mono_chunk)

        # ── FAST PASS — every 4 chunks (~320ms) ──────────────────────
        if len(buffer) % FAST_CHUNKS == 0 and len(buffer) >= FAST_CHUNKS and milo_is_talking.is_set():
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
                    )
                    heard = result["text"].lower().strip()
                    if _fire_interrupt(heard, source="fast"):
                        buffer = []
                        continue
                except Exception:
                    pass

        # ── THOROUGH PASS — every 8 chunks (~640ms) ──────────────────
        if len(buffer) >= THOROUGH_CHUNKS and milo_is_talking.is_set():
            full_audio = np.concatenate(buffer).flatten()
            buffer     = []

            full_audio = noise_cancel(full_audio)
            peak = np.max(np.abs(full_audio))
            rms  = np.sqrt(np.mean(full_audio ** 2))

            if peak < SLOW_ENERGY_GATE or rms < 0.003:
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
            except Exception:
                pass
        elif len(buffer) >= THOROUGH_CHUNKS:
            buffer = []   # not talking — discard bleed

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
    set_emotion("sleeping")
    print("\n💤 Sleeping... Say 'Mitra' to wake me up! 😴")
    drain_mic_queue()
    # No spoken reminder here — sleep is silent.
    # The goodbye outro already told the child how to wake Mitra up.

    WHISPER_CHUNKS   = int(SAMPLE_RATE * 1.5 / BLOCK_SIZE)
    whisper_buf      = []
    energy_buf       = []
    ENERGY_WINDOW    = 8
    # FIX v17: Raised from 0.004 → 0.015.
    # 0.004 was triggering Whisper on background noise/silence,
    # wasting CPU and occasionally matching random noise as "mitr".
    # 0.015 = clear voice present, still sensitive enough for soft children's voices.
    ENERGY_THRESHOLD = 0.015
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
                set_emotion("happy")
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
    milo_is_talking.set()
    # Print text RIGHT when audio starts — in sync with voice
    if label:
        print(f"🤖 Mitra: {label}")
        set_emotion("talking")
    try:
        _safe_play(data, fs)
        while sd.get_stream().active:
            if interrupt_flag.is_set():
                sd.stop()
                print("⏹️ Playback stopped by interrupt.")
                break
            time.sleep(0.05)
    except Exception as e:
        print(f"❌ Playback error: {e}")
    milo_is_talking.clear()
    set_emotion("happy")
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
# ─────────────────────────────────────────
# 🎵 RHYMES FEATURE  (Local Audio Files — v2)
#
# HOW IT WORKS:
#   When the child asks for a rhyme, Mitra plays a real pre-downloaded
#   nursery rhyme audio file from the  rhymes/  folder instead of
#   generating text through TTS.
#
# SETUP — download these 6 free MP3 files and save them into a
#   folder called  rhymes/  next to mitra.py:
#
#   File name to save as              →  Direct download URL
#   ─────────────────────────────────────────────────────────────────────
#   twinkle_twinkle.mp3               →  https://archive.org/download/thebestnurseryrhymes1/1-05%20Twinkle%2C%20Twinkle%20Little%20Star.mp3
#   baa_baa_black_sheep.mp3           →  https://archive.org/download/thebestnurseryrhymes1/1-09%20Baa%20Baa%20Black%20Sheep.mp3
#   humpty_dumpty.mp3                 →  https://archive.org/download/thebestnurseryrhymes1/1-10%20Humpty%20Dumpty.mp3
#   hickory_dickory_dock.mp3          →  https://archive.org/download/thebestnurseryrhymes1/1-08%20Hickory%20Dickory%20Dock.mp3
#   mary_had_a_little_lamb.mp3        →  https://archive.org/download/thebestnurseryrhymes1/1-04%20Mary%20Had%20a%20Little%20Lamb.mp3
#   row_row_your_boat.mp3             →  https://archive.org/download/thebestnurseryrhymes1/1-07%20Row%20Row%20Your%20Boat.mp3
#   ─────────────────────────────────────────────────────────────────────
#   All files are public domain from the Internet Archive (archive.org).
#
# FALLBACK:
#   If the rhymes/ folder is missing or empty, Mitra falls back to the
#   original LLM-generated TTS rhyme so nothing breaks.
# ─────────────────────────────────────────

RHYMES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rhymes")

# Maps the local filename (without extension) → display name for logging
RHYME_FILES = {
    "twinkle_twinkle":      "Twinkle Twinkle Little Star",
    "baa_baa_black_sheep":  "Baa Baa Black Sheep",
    "humpty_dumpty":        "Humpty Dumpty",
    "hickory_dickory_dock": "Hickory Dickory Dock",
    "mary_had_a_little_lamb": "Mary Had a Little Lamb",
    "row_row_your_boat":    "Row Row Your Boat",
}

# Maps keywords the child might say → which file key to play
# e.g. "sing twinkle" → plays twinkle_twinkle.mp3
RHYME_KEYWORD_MAP = {
    # Twinkle
    "twinkle": "twinkle_twinkle",
    "star":    "twinkle_twinkle",
    # Baa Baa
    "baa":        "baa_baa_black_sheep",
    "black sheep":"baa_baa_black_sheep",
    "sheep":      "baa_baa_black_sheep",
    # Humpty
    "humpty":   "humpty_dumpty",
    "dumpty":   "humpty_dumpty",
    "wall":     "humpty_dumpty",
    # Hickory
    "hickory":  "hickory_dickory_dock",
    "dickory":  "hickory_dickory_dock",
    "dock":     "hickory_dickory_dock",
    "clock":    "hickory_dickory_dock",
    "mouse":    "hickory_dickory_dock",
    # Mary
    "mary":     "mary_had_a_little_lamb",
    "lamb":     "mary_had_a_little_lamb",
    # Row Row
    "row":      "row_row_your_boat",
    "boat":     "row_row_your_boat",
    "stream":   "row_row_your_boat",
}

RHYME_TRIGGERS = [
    "rhyme", "rhymes", "sing", "sing a song", "song",
    "nursery rhyme", "tell me a rhyme", "say a rhyme",
    "poem", "lullaby", "sing something", "sing for me",
    "rap", "ditty",
    # ── NEW: STT mishear variants kids actually say ───────────────────
    "rime", "rimes", "ryan", "ryans", "ryme", "rymes",
    "grimes", "grines", "grime", "crime", "lime", "time song",
    "nursery", "nursary", "nursey",
    "twinkle", "twinkie", "twinkl",          # specific rhyme names as triggers
    "baa baa", "humpty", "hickory", "mary lamb", "row row",
    "sing twinkle", "sing baa", "sing humpty",
    "play a song", "play song", "play rhyme", "play a rhyme",
    "music", "melody", "tune",
    "little star", "little lamb",
]

# ── FUZZY RHYME DETECTION: STT misheard variants → trigger ──────────────
# If the child says "ryans", "grines", "rimes" etc. — Whisper mishears
# "rhymes". We check phonetic closeness to catch these without regex.
_RHYME_FUZZY = [
    "ryan", "ryans", "rime", "rimes", "ryme", "rymes",
    "grimes", "grines", "grime", "crimes", "limes",
    "nursery", "nursary", "nursey", "nurce",
    "twinkie", "twinkl", "twinkol",
]

def _fuzzy_rhyme_check(text: str) -> bool:
    """Returns True if text contains a rhyme-like mishear word."""
    t = text.lower()
    return any(f in t for f in _RHYME_FUZZY)

RHYME_INTROS = [
    "Ooooh YES! Rhyme time! Get ready, this one is SO good!",
    "Oh oh oh! A rhyme?! THIS IS MY FAVOURITE THING EVER! Here we go!",
    "Okay okay okay, clear your ears because THIS rhyme is about to be LEGENDARY!",
    "Rhyme time! Rhyme time! I have been WAITING for this moment! Listen!",
    "A rhyme?! Yes YES YES! I picked this one just for YOU! Ready?!",
]

RHYME_OUTROS = [
    "Ta-daaa! How was THAT?! Did you love it?! Want another one?!",
    "BOOM! Did that blow your mind a little? Want me to play another?!",
    "And SCENE! What did you think?! Should I play an even BETTER one?!",
    "THE END! Haha! Was that the greatest rhyme you've ever heard or what?!",
]

def scan_rhyme_files():
    """Print available local rhyme files at startup."""
    if not os.path.exists(RHYMES_DIR):
        print("📁 rhymes/ folder not found — rhyme audio layer inactive (LLM fallback active).")
        print("   → Create a  rhymes/  folder next to mitra.py and download the MP3 files.")
        return
    found = []
    for key in RHYME_FILES:
        path = os.path.join(RHYMES_DIR, f"{key}.mp3")
        if os.path.exists(path):
            found.append(RHYME_FILES[key])
    if not found:
        print("📁 rhymes/ folder is empty — LLM fallback active.")
        print("   → Download the MP3 files as described in the RHYMES FEATURE comment above.")
        return
    print(f"🎵 Rhyme audio files: {len(found)} loaded.")
    for name in found:
        print(f"   ✓ {name}")

def _get_available_rhyme_files() -> list[str]:
    """Return list of rhyme file keys that actually exist on disk."""
    if not os.path.exists(RHYMES_DIR):
        return []
    return [
        key for key in RHYME_FILES
        if os.path.exists(os.path.join(RHYMES_DIR, f"{key}.mp3"))
    ]

def play_rhyme_file(file_key: str) -> bool:
    """
    Play a local nursery rhyme MP3 file.
    Properly blocks the mic, sets milo_is_talking, and respects interrupt_flag —
    identical behaviour to the parent voice player.
    Returns True if file played successfully, False otherwise.
    """
    filepath = os.path.join(RHYMES_DIR, f"{file_key}.mp3")
    if not os.path.exists(filepath):
        print(f"⚠️ Rhyme file not found: {filepath}")
        return False
    try:
        data, fs  = sf.read(filepath)
        duration  = len(data) / fs
        block_mic_for(duration)
        drain_mic_queue()
        drain_interrupt_queue()
        milo_is_talking.set()
        print(f"🎵 Playing: {RHYME_FILES.get(file_key, file_key)}")
        _safe_play(data, fs)
        while sd.get_stream().active:
            if interrupt_flag.is_set():
                sd.stop()
                print("⏹️ Rhyme stopped by interrupt.")
                break
            time.sleep(0.05)
    except Exception as e:
        print(f"❌ Rhyme playback error: {e}")
        milo_is_talking.clear()
        return False
    milo_is_talking.clear()
    time.sleep(ECHO_TAIL_MS / 1000)
    drain_mic_queue()
    drain_interrupt_queue()
    return True

def is_rhyme_request(text: str) -> bool:
    t = text.lower()
    return any(tr in t for tr in RHYME_TRIGGERS) or _fuzzy_rhyme_check(t)

def handle_rhyme_request(user_input: str):
    """
    Play a local nursery rhyme audio file when the child asks for a rhyme.
    1. Check if child named a specific rhyme (e.g. "sing twinkle")
       → play that file.
    2. Otherwise pick a random available file.
    3. If no local files exist at all → fall back to LLM-generated TTS rhyme.
    """
    available = _get_available_rhyme_files()

    # ── NO LOCAL FILES: fall back to LLM ─────────────────────────────────
    if not available:
        speak(random.choice(RHYME_INTROS))
        topic = user_input.lower()
        for trigger in RHYME_TRIGGERS:
            topic = topic.replace(trigger, "").strip()
        topic_hint = f"about '{topic}'" if len(topic) > 2 else "about anything silly and fun"
        rhyme_prompt = (
            f"Write a short, fun, rhyming nursery rhyme {topic_hint} for a child aged 3-8. "
            "Rules: Exactly 4-6 lines. Must rhyme (AABB or ABAB). Simple words. Funny and silly. "
            "No intro or outro — JUST the rhyme. Add 1-2 fun sound effects like WHOOOOSH or BOING."
        )
        rhyme_messages = [
            {"role": "system", "content": "You are a fun children's rhyme writer. Write ONLY the rhyme."},
            {"role": "user",   "content": rhyme_prompt},
        ]
        rhyme = groq_chat(rhyme_messages, temperature=1.0, max_tokens=120)
        if rhyme:
            speak(rhyme.strip())
            speak(random.choice(RHYME_OUTROS))
        else:
            speak("Oops! My rhyme brain did a little flip! Roses are red, you are SO cool, talking to Mitra is better than school!")
        return

    # ── PICK WHICH FILE TO PLAY ───────────────────────────────────────────
    t = user_input.lower()

    # 1. Check if the child mentioned a specific rhyme name
    chosen_key = None
    for keyword, file_key in RHYME_KEYWORD_MAP.items():
        if keyword in t and file_key in available:
            chosen_key = file_key
            print(f"🎵 Specific rhyme requested: '{keyword}' → {RHYME_FILES[file_key]}")
            break

    # 2. If no specific match, pick randomly from available files
    if chosen_key is None:
        chosen_key = random.choice(available)
        print(f"🎵 Random rhyme selected: {RHYME_FILES[chosen_key]}")

    # ── PLAY ──────────────────────────────────────────────────────────────
    speak(random.choice(RHYME_INTROS))
    played = play_rhyme_file(chosen_key)

    if played and not interrupt_flag.is_set():
        speak(random.choice(RHYME_OUTROS))
    elif not played:
        # File read failed — graceful fallback message
        speak("Oops! I couldn't find that rhyme right now! Ask a grown-up to check the rhymes folder!")

# ─────────────────────────────────────────
# 🌍 LANGUAGE LEARNING FEATURE
#
# Triggered when child says things like:
#   "teach me French", "how do you say hello in Spanish",
#   "learn Hindi words", "speak Japanese", etc.
#
# Mitra teaches 3 simple words with:
#   1. The foreign word + pronunciation guide
#   2. A SILLY mnemonic that links the sound to the meaning
#   3. A "say it with me!" practice round
#
# Supported languages: French, Spanish, Hindi, German,
#   Italian, Japanese, Mandarin, Arabic, Portuguese
# ─────────────────────────────────────────

SUPPORTED_LANGUAGES = {
    # ── Core languages ────────────────────────────────────────────────
    "french":     "French",
    "spanish":    "Spanish",
    "hindi":      "Hindi",
    "german":     "German",
    "italian":    "Italian",
    "japanese":   "Japanese",
    "mandarin":   "Mandarin Chinese",
    "chinese":    "Mandarin Chinese",
    "arabic":     "Arabic",
    "portuguese": "Portuguese",
    "tamil":      "Tamil",
    "kannada":    "Kannada",
    "telugu":     "Telugu",
    # ── STT MISHEAR ALIASES — maps common mishears to the real language ──
    # Kids say it fast; Whisper often mishears language names.
    "japan":      "Japanese",
    "japanees":   "Japanese",
    "japanes":    "Japanese",
    "japenese":   "Japanese",
    "japonese":   "Japanese",
    "spain":      "Spanish",
    "spanis":     "Spanish",
    "espanol":    "Spanish",
    "france":     "French",
    "franch":     "French",
    "frech":      "French",
    "frence":     "French",
    "hinde":      "Hindi",
    "hindhi":     "Hindi",
    "hindy":      "Hindi",
    "indy":       "Hindi",
    "tami":       "Tamil",
    "tamile":     "Tamil",
    "kanada":     "Kannada",
    "canada":     "Kannada",
    "telugo":     "Telugu",
    "telago":     "Telugu",
    "tellugu":    "Telugu",
    "chin":       "Mandarin Chinese",
    "china":      "Mandarin Chinese",
    "arabi":      "Arabic",
    "arabik":     "Arabic",
    "portu":      "Portuguese",
    "portugese":  "Portuguese",
    "germen":     "German",
    "germon":     "German",
    "italiano":   "Italian",
    "itali":      "Italian",
    "korean":     "Korean",
    "korea":      "Korean",
    "russian":    "Russian",
    "russia":     "Russian",
}

# Canonical language name → key used in LANGUAGE_WORDS lookup
# When an alias maps to a display name, we need the real dict key.
_LANGUAGE_ALIAS_MAP = {
    "Japanese":         "japanese",
    "Spanish":          "spanish",
    "French":           "french",
    "Hindi":            "hindi",
    "Tamil":            "tamil",
    "Kannada":          "kannada",
    "Telugu":           "telugu",
    "Mandarin Chinese": "mandarin",
    "Arabic":           "arabic",
    "Portuguese":       "portuguese",
    "German":           "german",
    "Italian":          "italian",
    "Korean":           "korean",
    "Russian":          "russian",
}

LANGUAGE_TRIGGERS = [
    "teach me", "teach", "learn", "how do you say",
    "say it in", "translate", "speak", "what is hello in",
    "what's hello in", "words in", "language",
    # ── NEW: more natural child phrasings ────────────────────────────
    "i want to learn", "i wanna learn", "show me",
    "how to speak", "how to say", "tell me in",
    "what does hello mean in", "say hello in",
    "teach me words", "teach words", "lingo",
    "how do people say", "what language", "teach language",
    "words from", "speak to me in", "talk in",
]

LANGUAGE_INTROS = [
    "Ooooh YES! {lang} time! I LOVE this language — it sounds SO magical! Ready?!",
    "OH WOW! You want to learn {lang}?! That is the COOLEST thing ever! Let's go!",
    "YESSS! {lang} lesson with Mitra! This is going to blow your tiny mind! Listen carefully!",
    "Oh oh oh! {lang}! Amazing choice! Some of these words sound like SILLY giggles! Here we go!",
    "LANGUAGE TIME! {lang} incoming! Get your ears ready because this is SO much fun!",
]

LANGUAGE_OUTROS = [
    "Ta-daaa! You just learned THREE whole {lang} words! You're basically a genius now! Want three more?!",
    "BOOM! Three {lang} words DOWN! You are SO smart! Should we do three more?!",
    "And THAT is your {lang} lesson for today! Say those words to someone — they will be SO impressed!",
    "You just spoke {lang}! Like a real {lang} speaker! Want to learn even MORE cool words?!",
]

def is_language_request(text: str) -> tuple:
    """
    Returns (True, language_key) if child is asking to learn a language,
    (False, None) otherwise.
    Handles STT mishear aliases (e.g. "japanees" → "japanese").
    """
    t = text.lower()

    # Step 1: find a language name (or alias) in the text
    detected_lang = None
    detected_display = None
    for key in SUPPORTED_LANGUAGES:
        if key in t:
            detected_lang = key
            detected_display = SUPPORTED_LANGUAGES[key]
            break
    if detected_lang is None:
        return False, None

    # Step 2: resolve alias → canonical LANGUAGE_WORDS key
    canonical_key = _LANGUAGE_ALIAS_MAP.get(detected_display, detected_lang)

    # Step 3: confirm there's also a learning/teaching trigger word
    if any(tr in t for tr in LANGUAGE_TRIGGERS):
        return True, canonical_key

    # Step 4: if language name is in text AND it's a short utterance
    # (child just said "teach me telugu" — trigger even without extra words)
    if len(t.split()) <= 6:
        return True, canonical_key

    return False, None

def handle_language_request(language: str):
    """
    Teach 3 words in a natural Mitra voice.
    FIX v20: Completely rewritten prompt that stops the LLM from
    splitting syllables letter-by-letter (D-H-O-O) and instead
    produces correct phonetic pronunciation as a full word sound.
    """
    lang_name = SUPPORTED_LANGUAGES.get(language, language.capitalize())

    speak(random.choice(LANGUAGE_INTROS).format(lang=lang_name))

    # ── HARDCODED CORRECT WORDS PER LANGUAGE ──────────────────────────
    # The LLM was making up words entirely (DHOO-NU for "thank you" in Hindi).
    # For the most common languages we hardcode the correct words so Mitra
    # always teaches real vocabulary. LLM only handles unknown languages.
    # ── LANGUAGE WORDS ─────────────────────────────────────────────────
    # Phonetics written as TTS-readable English sounds (no dashes).
    # TTS reads "nah mah SKAH rum" naturally as separate syllables.
    # Dashes cause TTS to pause awkwardly or spell it out.
    # Each entry: (english_word, foreign_word, tts_phonetic, mnemonic)
    LANGUAGE_WORDS = {
        "hindi": [
            ("hello",     "Namaste",     "nah mah stay",    "like NAH and then you STAY! Say it together: nah mah stay!"),
            ("thank you", "Shukriya",    "shoo kree yah",   "like a SHOE doing a CREEK dance! shoo kree yah!"),
            ("cat",       "Billi",       "bill ee",         "like a cat named BILLY! bill ee!"),
        ],
        "telugu": [
            ("hello",     "Namaskaram",  "nah mas kah rum", "like a car going nah mas kah rum!"),
            ("thank you", "Dhanyavadalu","dhun yah vah dah loo","like DUNE yah vah dah loo — say it fast!"),
            ("dog",       "Kukka",       "koo kah",         "like a dog barking koo kah koo kah!"),
        ],
        "tamil": [
            ("hello",     "Vanakkam",    "vah nah kum",     "like a VAN going nah kum!"),
            ("thank you", "Nandri",      "nun dree",        "like a NUN climbing a TREE — nun dree!"),
            ("elephant",  "Yanai",       "yah nai",         "like YAH nai — the elephant says YAH!"),
        ],
        "french": [
            ("hello",     "Bonjour",     "bon zhoor",       "like BON JOURNEY said super fast — bon zhoor!"),
            ("thank you", "Merci",       "mare see",        "like a MARE horse who can SEE — mare see!"),
            ("cat",       "Chat",        "sha",             "like SHHH but with an A at the end — sha!"),
        ],
        "spanish": [
            ("hello",     "Hola",        "oh lah",          "like OH LA LA but just oh lah!"),
            ("thank you", "Gracias",     "grah see ahs",    "like GRAB some SEAS — grah see ahs!"),
            ("dog",       "Perro",       "pay roh",         "like PAY the ROW — pay roh!"),
        ],
        "japanese": [
            ("hello",     "Konnichiwa",  "kon nee chee wah","like CON KNEE CHEE WAH — imagine a knee going WA!"),
            ("thank you", "Arigatou",    "ah ree gah toe",  "like ah ree gah TOE — touch your toe and say it!"),
            ("dog",       "Inu",         "ee noo",          "like EE noo — the dog says ee noo!"),
        ],
        "kannada": [
            ("hello",     "Namaskara",   "nah mas kah rah", "like a car going nah mas kah rah!"),
            ("thank you", "Dhanyavada",  "dhun yah vah dah","like DUNE yah vah dah — imagine a sand dune!"),
            ("cat",       "Bekku",       "bay koo",         "like BAY and KOO together — bay koo!"),
        ],
        "malayalam": [
            ("hello",     "Namaskaram",  "nah mas kah rum", "like a car going nah mas kah rum!"),
            ("thank you", "Nanni",       "nun nee",         "like saying nun nee really fast!"),
            ("elephant",  "Aana",        "ah nah",          "like AH nah — the elephant says AH!"),
        ],
        "german": [
            ("hello",     "Hallo",       "hah loh",         "like HELLO but say hah loh!"),
            ("thank you", "Danke",       "dahn keh",        "like DONKEY but dahn keh!"),
            ("dog",       "Hund",        "hoont",           "like HOUND but hoont!"),
        ],
        "arabic": [
            ("hello",     "Marhaba",     "mar hah bah",     "like MAR HAH BAH — three big claps!"),
            ("thank you", "Shukran",     "shoo kran",       "like SHOE plus KRAN — shoo kran!"),
            ("cat",       "Qitta",       "kit tah",         "like KITTEN but kit tah!"),
        ],
        "mandarin": [
            ("hello",     "Ni hao",      "nee how",         "like KNEE HOW are you! nee how!"),
            ("thank you", "Xie xie",     "shyeh shyeh",     "like SHYEH SHYEH — say it two times!"),
            ("cat",       "Mao",         "mao",             "like a cat going MEOW but just mao!"),
        ],
        "portuguese": [
            ("hello",     "Ola",         "oh lah",          "like OH LAH — wave and say oh lah!"),
            ("thank you", "Obrigado",    "oh bree gah doo", "like oh bree gah doo — a big thank you dance!"),
            ("dog",       "Cachorro",    "kah shore roh",   "like a dog at the SHORE — kah shore roh!"),
        ],
        "italian": [
            ("hello",     "Ciao",        "chow",            "like CHOW — say it fast — chow!"),
            ("thank you", "Grazie",      "grah tsyeh",      "like grah tsyeh — the cheese says thank you!"),
            ("cat",       "Gatto",       "gah toh",         "like gah toh — a cat named Gatto!"),
        ],
        "russian": [
            ("hello",     "Privet",      "pree vyet",       "like pree VYET — a friendly vyet!"),
            ("thank you", "Spasibo",     "spah see bah",    "like SPAH SEE BAH — spa for bah!"),
            ("cat",       "Koshka",      "kosh kah",        "like kosh kah — a squishy cat!"),
        ],
        "korean": [
            ("hello",     "Annyeong",    "ahn nyung",       "like ahn NYUNG — like saying YOUNG!"),
            ("thank you", "Gamsahamnida","gam sah hum nee dah","like gam SAH — say it with a bow!"),
            ("dog",       "Gae",         "gay",             "like GAY — just one short word!"),
        ],
    }

    words = LANGUAGE_WORDS.get(language)

    if words:
        # ── Build lesson — natural flowing speech, no dashes in phonetics ─
        # Template deliberately avoids "Say it like this: X-Y-Z" pattern
        # because TTS reads dashes as pauses. We write phonetics as plain
        # space-separated syllables so TTS flows naturally.
        lesson_parts = []
        for i, (english, foreign, phonetic, mnemonic) in enumerate(words):
            if i == 0:
                opener = "Okay FIRST word!"
            elif i == 1:
                opener = "Next one!"
            else:
                opener = "And the LAST one!"

            lesson_parts.append(
                f"{opener} {english.capitalize()} in {lang_name} is {foreign}! "
                f"Say it like this: {phonetic}! "
                f"Remember it {mnemonic}"
            )

        lesson   = "\n\n".join(lesson_parts)
        practice = (
            f"NOW SAY THEM ALL WITH ME! "
            f"{words[0][2]}! {words[1][2]}! {words[2][2]}! "
            f"OH MY GOODNESS you are speaking {lang_name} RIGHT NOW!"
        )
        full_lesson = lesson + "\n\n" + practice

        print(f"🌍 Language lesson ({lang_name}) — hardcoded correct words:")
        print(full_lesson)
        speak(full_lesson)
        time.sleep(0.3)
        speak(random.choice(LANGUAGE_OUTROS).format(lang=lang_name))
        messages.append({"role": "user",      "content": f"[Child asked to learn {lang_name}]"})
        messages.append({"role": "assistant", "content": f"I taught them: {words[0][1]}, {words[1][1]}, {words[2][1]}"})

    else:
        # ── LLM fallback for unsupported languages ─────────────────────
        prompt = (
            f"You are Mitra, a super fun AI for kids aged 3-8. "
            f"Teach 3 simple {lang_name} words.\n\n"

            "CRITICAL RULES:\n"
            "1. Use ONLY plain English letters — NO foreign script at all.\n"
            "2. Write pronunciation as plain English syllables separated by SPACES only.\n"
            "   NEVER use hyphens or dashes in phonetics (bad: BON-ZHOOR, good: bon zhoor).\n"
            "3. The pronunciation must sound natural when read aloud by a text-to-speech engine.\n"
            "4. Mnemonic = one silly English sentence using a word that SOUNDS like the foreign word.\n"
            "5. Teach: hello, thank you, one animal.\n"
            "6. End with practice: say all 3 phonetics together.\n\n"

            "GOOD EXAMPLE (copy this style exactly):\n"
            "Okay FIRST word! Hello in French is Bonjour! "
            "Say it like this: bon zhoor! It sounds like BON JOURNEY said super fast! bon zhoor!\n\n"
            "Next one! Thank you in French is Merci! "
            "Say: mare see! Like a MARE horse who can SEE! mare see!\n\n"
            "And the LAST one! Cat in French is Chat! "
            "Say: sha! Like SHHH but with an A — sha!\n\n"
            "NOW SAY THEM ALL! bon zhoor! mare see! sha! You speak French!\n\n"

            f"Now write the same for {lang_name}. "
            "Use SPACES between syllables, NEVER dashes. "
            "Output ONLY the lesson — no intro or explanation."
        )

        lang_messages = [
            {
                "role": "system",
                "content": (
                    "You are Mitra teaching kids a language. "
                    "Use ONLY plain English letters. "
                    "Write pronunciations as full phonetic words, NEVER letter-by-letter. "
                    "Be excited and silly."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        result = groq_chat(lang_messages, temperature=0.75, max_tokens=300)

        if result:
            import re as _re_lang
            result = _re_lang.sub(r'[^\x00-\x7F]+', '', result).strip()
            result = _re_lang.sub(r' +', ' ', result)
            print(f"🌍 Language lesson generated ({lang_name}):\n{result}")
            speak(result)
            time.sleep(0.3)
            speak(random.choice(LANGUAGE_OUTROS).format(lang=lang_name))
            messages.append({"role": "user",      "content": f"[Child asked to learn {lang_name}]"})
            messages.append({"role": "assistant", "content": f"I taught them 3 {lang_name} words!"})
        else:
            speak(f"Oops! My {lang_name} brain did a tiny flip! Can you ask me again?")

# ─────────────────────────────────────────
# 📞 PARENT CALLING FEATURE  (WiFi / Demo)
#
# HOW IT WORKS FOR DEMO:
#   When child says "call mummy/daddy/parents":
#   → Mitra speaks a warm intro
#   → Triggers a WiFi call via the system (demo uses a config file
#     with parent's phone number / SIP address)
#   → Mitra tells the child to wait
#
# FOR REAL DEPLOYMENT:
#   Use Twilio (pip install twilio) for real phone calls over WiFi.
#   Requires a free Twilio account + trial number.
#   The demo below simulates the call for prototype purposes.
#
# SETUP (one time):
#   Create a file called parent_contact.txt next to mitra.py
#   Line 1: parent's phone number e.g. +919876543210
#   Line 2: parent's name e.g. Mama
# ─────────────────────────────────────────
CALL_TRIGGERS = [
    # Direct "call X" — original
    "call mummy", "call mommy", "call mama", "call mom",
    "call daddy", "call papa", "call dad",
    "call my parents", "call my mom", "call my dad",
    # "I want to talk to X"
    "i want to talk to mummy", "i want to talk to mommy",
    "i want to talk to mama", "i want to talk to mom",
    "i want to talk to daddy", "i want to talk to papa",
    "i want to talk to dad", "i want to talk to my parents",
    # "want X"
    "want mummy", "want mommy", "want mama",
    "want daddy", "want papa", "want dad",
    # "phone/ring X"
    "phone mummy", "phone mommy", "phone mama",
    "phone daddy", "phone papa", "phone dad",
    "ring mummy", "ring mommy", "ring mama",
    # ── NEW: natural child distress phrasings ──────────────────────────
    # "I want mom/dad"
    "i want mom", "i want dad", "i want my mom", "i want my dad",
    "i want my mummy", "i want my mommy", "i want my mama",
    "i want my daddy", "i want my papa",
    # "I need mom/dad"
    "i need mom", "i need dad", "i need my mom", "i need my dad",
    "i need mummy", "i need mommy", "i need mama",
    "i need daddy", "i need papa",
    # "where is mom/dad"
    "where is mom", "where is dad", "where is mummy", "where is mommy",
    "where is mama", "where is daddy", "where is papa",
    "where's mom", "where's dad", "where's mummy", "where's mommy",
    "where's mama", "where's daddy", "where's papa",
    # "can you call"
    "can you call mom", "can you call dad", "can you call mummy",
    "can you call mommy", "can you call mama", "can you call daddy",
    "can you call papa", "can you call my parents",
    # "talk to mom/dad"
    "talk to mom", "talk to dad", "talk to mummy", "talk to mommy",
    "talk to mama", "talk to daddy", "talk to papa",
    # "contact mom/dad"
    "contact mom", "contact dad", "contact mummy",
    # "get mom/dad"
    "get mom", "get dad", "get mummy", "get mommy", "get mama",
    "get daddy", "get papa",
    # "tell mom/dad"
    "tell mom", "tell dad", "tell mummy",
    # single word distress (common for young kids)
    "mommy please", "mummy please", "daddy please", "papa please",
    # STT mishear variants
    "call mama now", "call momma", "call moma", "call dad now",
    "goal mom", "goal dad", "goal mummy",   # STT "call" → "goal"
    "call them", "call her", "call him",
]

PARENT_CONTACT_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "parent_contact.txt"
)

def _load_parent_contact():
    """Load parent phone number and name from parent_contact.txt."""
    if not os.path.exists(PARENT_CONTACT_FILE):
        return None, "Mama or Papa"
    try:
        lines = open(PARENT_CONTACT_FILE).read().strip().splitlines()
        number = lines[0].strip() if len(lines) > 0 else None
        name   = lines[1].strip() if len(lines) > 1 else "Mama or Papa"
        return number, name
    except Exception:
        return None, "Mama or Papa"

def is_call_request(text: str) -> bool:
    t = text.lower()
    return any(ct in t for ct in CALL_TRIGGERS)

def handle_call_request():
    """
    Real Twilio WiFi call using hardcoded credentials.
    When parent picks up → Twilio plays a warm message telling them
    their child wants to talk to them through Mitra.
    After call is placed → Mitra engages the child with a fun distraction
    so they are not just sitting quietly waiting.
    """
    # ── Hardcoded Twilio credentials (your account) ──────────────────
    TWILIO_SID   = "ACec34cdd1d09419a91521341dfbd4b7d4"
    TWILIO_TOKEN = "5621c0f0916c0c01de824a795d4a69cf"
    TWILIO_FROM  = "+17542968496"   # your Twilio number
    PARENT_TO    = "+918310693713"  # parent's real phone number
    PARENT_NAME  = "Mama or Papa"   # shown in logs

    # ── Step 1: Warm intro to the child ─────────────────────────────
    speak(
        f"Oh oh oh! You want to talk to {PARENT_NAME}?! "
        "That is the BEST idea! "
        "I am going to call them RIGHT NOW — hold on just one tiny second!"
    )

    # ── Step 2: Place the real call ──────────────────────────────────
    twiml_message = (
        '<Response>'
        '<Say voice="alice" language="en-IN">'
        'Hello! This is Mitra, the magical AI friend of your child. '
        'Your little one is missing you and wants to hear your voice! '
        'They are waiting by the device right now. '
        'Please come to them as soon as you can — they love you so much! '
        'This has been a message from Mitra. Goodbye!'
        '</Say>'
        '</Response>'
    )

    call_placed = False
    try:
        from twilio.rest import Client as TwilioClient
        tw   = TwilioClient(TWILIO_SID, TWILIO_TOKEN)
        call = tw.calls.create(
            to=PARENT_TO,
            from_=TWILIO_FROM,
            twiml=twiml_message,
        )
        print(f"📞 Twilio call placed! SID={call.sid}  to={PARENT_TO}")
        call_placed = True

    except ImportError:
        print("❌ Twilio not installed! Run:  pip install twilio")
        speak(
            "Oops! My phone brain is not ready yet! "
            "Ask a grown-up to help install the phone helper, "
            "and next time I will call them for real!"
        )
        return

    except Exception as e:
        print(f"❌ Twilio call failed: {e}")
        speak(
            f"Oh no! Something went a little wobbly with the phone! "
            "Maybe the internet sneezed! "
            "Ask a grown-up to check the connection and try again!"
        )
        return

    # ── Step 3: Tell child call was placed + start fun distraction ───
    if call_placed:
        speak(
            f"DONE! I just called {PARENT_NAME} and told them you want to talk! "
            "Their phone is ringing RIGHT NOW! "
            "They are going to come running to you SO fast — "
            "maybe even faster than a cheetah on a skateboard!"
        )
        time.sleep(1.0)
        speak(
            "But while we wait for them — "
            "because the BEST things are worth waiting for — "
            "let me keep you company! "
            "I have a VERY important question for you..."
        )

        # ── Step 4: Engaging distraction questions ────────────────────
        distraction_questions = [
            "If YOU could have ANY superpower in the whole world — "
            "and I mean ANY, even something nobody has thought of yet — "
            "what would it be and WHY?! I need DETAILS!",

            "Okay okay okay, VERY important question. "
            "If you could eat only ONE food forever and ever — "
            "just ONE — what would it be? "
            "And remember, this is FOREVER, so think carefully!",

            "Here is a mystery I need YOUR brain to solve. "
            "Why do you think the sky is blue and NOT purple? "
            "Because honestly, purple is a GREAT colour. "
            "What do YOU think happened?",

            "If you were the BOSS of the whole world for one day — "
            "just ONE day — what is the very first thing you would change? "
            "I am writing this down because your ideas are BRILLIANT!",

            "Quick! If you could be any animal — "
            "literally ANY animal in the whole universe including made-up ones — "
            "what would you pick and what amazing thing would you do first?",
        ]

        speak(random.choice(distraction_questions))

def ask_child_name():
    """
    Ask the child their name on first wake. Stores in _child_name.
    Returns the name string (never None — falls back to 'Super Friend').
    """
    global _child_name, _name_asked
    _name_asked = True
    speak("Oh hi hi hi! Before we start our amazing adventure — what's YOUR name? Tell me!")
    audio, _ = record_audio(duration=5)
    if audio is None:
        _child_name = "Super Friend"
        speak("That's okay! I'll just call you my Super Friend for now!")
        return _child_name
    name_text = transcribe(audio)
    if not name_text or len(name_text.strip()) < 2:
        _child_name = "Super Friend"
        speak("Hmm, I didn't quite catch that! I'll call you my Super Friend for now!")
        return _child_name
    import re as _re_name
    cleaned = _re_name.sub(
        r"(?i)(my name is|i am|i'm|its|it's|call me|they call me)\s*", "",
        name_text.strip()
    ).strip().split()
    _child_name = cleaned[0].capitalize() if cleaned and len(cleaned[0]) >= 2 else "Super Friend"
    speak(
        f"OH WOW! {_child_name}! That is literally the BEST name I have EVER heard! "
        f"Hi {_child_name}! Let's have the most AMAZING time together!"
    )
    print(f"👤 Child name learned: '{_child_name}'")
    return _child_name

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
scan_rhyme_files()
start_mic_streams()
init_emotion_display()

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

_start_dashboard_server()   # 📊 start dashboard on port 5050

print("🤖 Mitra is ready!")
print(f"   Mic     → device {INPUT_DEVICE} (USB mic — single stream)")
print(f"   Speaker → device {OUTPUT_DEVICE} (AUX out — {OUTPUT_CHANNELS}ch)")
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
    set_emotion("happy")
    interrupt_flag.clear()
    sleep_flag.clear()
    drain_interrupt_queue()
    _distress_count = 0   # reset per session
    reset_dashboard_session()   # 📊 new session — reset dashboard counters

    # ── NAME MEMORY: ask name on very first wake of the program ──────
    if not _name_asked:
        ask_child_name()
        # After asking, if goodbye was detected during name recording, break
        if sleep_flag.is_set():
            sleep_flag.clear()
            speak(random.choice(SLEEP_OUTRO_LINES))
            continue
    elif _child_name:
        # Personalise the wakeup line with their name
        wakeup = random.choice(WAKEUP_LINES)
        # Insert name into the first sentence naturally
        wakeup = wakeup.replace("YOU woke me up", f"{_child_name} woke me up", 1)
        wakeup = wakeup.replace("YOU'RE here", f"{_child_name} is here", 1)
        speak(wakeup)
    else:
        speak(random.choice(WAKEUP_LINES))

    while True:

        # ── SLEEP FLAG: goodbye was said during playback ──────────────
        if sleep_flag.is_set():
            sleep_flag.clear()
            interrupt_flag.clear()
            drain_mic_queue()
            drain_interrupt_queue()
            speak(random.choice([
                *SLEEP_OUTRO_LINES,
            ]))
            break

        if interrupt_flag.is_set():
            interrupt_flag.clear()
            drain_mic_queue()
            drain_interrupt_queue()
            time.sleep(0.1)   # tiny settle so interrupt_flag stays clear before speak
            if not interrupt_flag.is_set():
                speak(random.choice(INTERRUPT_ACK_LINES))
            interrupt_flag.clear()

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
                    *SLEEP_OUTRO_LINES,
                ]))
                break
            speak(random.choice(CANT_HEAR_LINES))
            continue

        user_input = transcribe(audio)

        if not user_input:
            speak(random.choice(REPEAT_LINES))
            continue

        # ── INSTANT STOP CHECK (main loop) ──────────────────────
        # Catches "stop" even when Mitra is NOT currently speaking
        # (the background listener only fires during playback).
        # NOTE: goodbye words are in INSTANT_STOP_WORDS too — but we
        # check is_goodbye_command first so they route to sleep, not just stop.
        if is_goodbye_command(user_input):
            print(f"😴 Goodbye in main loop: '{user_input}'")
            log_event("session_end", "😴 Session ended — child said goodbye", severity="info")
            speak(random.choice([
                *SLEEP_OUTRO_LINES,
            ]))
            break

        if is_stop_command(user_input):
            print(f"🛑 Stop command in main loop: '{user_input}'")
            sd.stop()
            speak("Okay! Stopping. What would you like to do instead?")
            drain_mic_queue()
            drain_interrupt_queue()
            continue
        # ────────────────────────────────────────────────────────

        # ── 🎵 RHYME REQUEST ────────────────────────────────────
        if is_rhyme_request(user_input):
            print(f"🎵 Rhyme request detected: '{user_input}'")
            # figure out which rhyme will be chosen for the log
            _avail = _get_available_rhyme_files()
            _rhyme_name = "Random rhyme"
            if _avail:
                _t = user_input.lower()
                for _kw, _fk in RHYME_KEYWORD_MAP.items():
                    if _kw in _t and _fk in _avail:
                        _rhyme_name = RHYME_FILES.get(_fk, _fk)
                        break
                else:
                    _rhyme_name = RHYME_FILES.get(_avail[0], "Rhyme")
            log_event("rhyme", f"🎵 Rhyme played: {_rhyme_name}", detail=_rhyme_name, severity="info")
            handle_rhyme_request(user_input)
            continue
        # ────────────────────────────────────────────────────────

        # ── 🌍 LANGUAGE LEARNING REQUEST ────────────────────────
        _is_lang, _lang_key = is_language_request(user_input)
        if _is_lang:
            print(f"🌍 Language request detected: '{_lang_key}'")
            _lang_display = SUPPORTED_LANGUAGES.get(_lang_key, _lang_key.capitalize())
            log_event("language", f"🌍 Language: {_lang_display}", detail=_lang_display, severity="info")
            handle_language_request(_lang_key)
            continue
        # ────────────────────────────────────────────────────────

        # ── 📞 PARENT CALL REQUEST ──────────────────────────────
        if is_call_request(user_input):
            print(f"📞 Call request detected: '{user_input}'")
            log_event("call", "📞 Parent call requested by child", detail=user_input, severity="warning")
            handle_call_request()
            continue
        # ────────────────────────────────────────────────────────

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
            # 📊 Log emotion to dashboard
            _sev = "warning" if emotion in ("sad", "scared", "lonely", "missing") else "info"
            log_event("emotion", f"😶 Emotion: {emotion.capitalize()}", detail=emotion, severity=_sev)

            # ── DISTRESS ESCALATION ───────────────────────────────────────
            # sad/scared/lonely/missing count towards auto-call threshold.
            if emotion in ("sad", "scared", "lonely", "missing"):
                _distress_count += 1
                print(f"   😢 Distress count: {_distress_count}/{_DISTRESS_CALL_AT}")
                log_event("distress", f"⚠️ Distress count: {_distress_count}/{_DISTRESS_CALL_AT}",
                          detail=f"{emotion} #{_distress_count}", severity="warning")
                if _distress_count >= _DISTRESS_CALL_AT:
                    print(f"   📞 Distress threshold reached — auto-calling parent!")
                    log_event("auto_call", "🚨 Auto-call triggered — 3x distress reached",
                              detail=f"Emotion: {emotion}", severity="alert")
                    speak(
                        "Hey... I can hear that you're really not feeling okay. "
                        "I'm going to call Mama or Papa for you RIGHT NOW — "
                        "because they would want to know!"
                    )
                    _distress_count = 0   # reset so it doesn't loop-call
                    handle_call_request()
                    continue
            # ─────────────────────────────────────────────────────────────
            print(f"💝 Triggering parent voice: {emotion} "
                  f"(text_keyword={text_emotion}, voice={voice_sentiment['voice_emotion']})")
            speak(random.choice(PARENT_VOICE_INTRO))
            played = handle_parent_voice_trigger(emotion)
            if played:
                continue
        # ────────────────────────────────────────────────────────

        # Build prompt
        _name_hint = f" The child's name is {_child_name}." if _child_name and _child_name != "Super Friend" else ""
        enhanced = (
            f"The child said (via voice recognition, may have errors): \"{user_input}\"\n\n"
            f"{_name_hint}\n"
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
                *SLEEP_OUTRO_LINES,
            ]))
            break

        if interrupt_flag.is_set():
            interrupt_flag.clear()
            drain_interrupt_queue()
            time.sleep(0.1)
            if not interrupt_flag.is_set():
                speak(random.choice(INTERRUPT_ACK_LINES))
            interrupt_flag.clear()
            audio, _raw2 = record_audio(duration=5)

            # Sleep guard — goodbye may arrive while we wait for new input
            if audio is None:
                if sleep_flag.is_set():
                    sleep_flag.clear()
                    interrupt_flag.clear()
                    drain_mic_queue()
                    drain_interrupt_queue()
                    speak(random.choice([
                        *SLEEP_OUTRO_LINES,
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