"""
mitra_server.py — Local hub between Mitra, the Child Companion and the Parent Dashboard.

Run this FIRST:
    python mitra_server.py          →  http://localhost:5000  (serves parent_dashboard.html)

Everything is local except the providers you explicitly configure in .env
(Groq LLM, Edge-TTS voices, Twilio SMS/calls, ntfy push, Telegram). Missing keys
never crash the server — endpoints report the real configuration state instead.

Routes
  Core (backward compatible with mitra_ultimate.py)
    GET  /                       dashboard
    GET  /ping                   health
    GET  /alerts                 alerts + stats (+ per-alert notification attempts)
    POST /alert                  create alert (JSON, or multipart with an `audio` WAV capture)
    POST /alert/<id>/read        mark read
    POST /alerts/clear           clear alerts (+ their capture files)
    GET|POST /status             device status heartbeat
  Media
    GET  /audio/<file>           parent_voice/ clips
    GET  /audio/tts/<file>       Edge-TTS output
    GET  /audio/parent-msg/<file> parent push-to-talk recordings
    GET  /captures/<file>        distress captures
  Companion
    POST /api/interact           {message, action: chat|cry_simulate|soft_tale, language, queue}
    GET  /api/transcript         shared conversation log ; POST /api/transcript/clear
    POST /api/parent-speak       multipart audio | {text, language} | {phrase_id}  → queued for the child
    GET  /api/pending-audio      outbox status ; POST /api/pending-audio/claim (companion devices)
    GET  /api/parent-voice       clips present on the hub
    POST /api/parent-voice/<id>  upload a 16 kHz mono WAV for an allow-listed phrase id
    GET|POST /api/telemetry      device sound-level samples (if Mitra hardware posts them)
    POST /api/transcribe         voice input → Groq Whisper (multipart `audio`, language hint)
  Live screening (camera + microphone)
    POST /api/screening/analyze  JPEG `frame` + WAV `audio` → local CV/ML + cry model (+ Groq vision)
    GET  /api/screening/state    latest result per device, stress history, engine status
    GET  /api/screening/references  reference frames captured with "Capture Reference Frame"
  Parent alerting
    GET|POST /api/settings       mitra_settings.json   (same-origin only)
    GET|POST /api/settings/credentials  provider keys in .env, masked (same-origin; writes from the hub PC only)
    POST /api/telegram/detect-chat      find your chat id via getUpdates
    POST /api/ntfy/connect       generate a private topic + QR code
    GET  /api/capabilities       which providers are configured (booleans only) + verified Groq models
    GET  /api/models             Groq model discovery (models.list + image-input probe)
    GET  /api/notifications      delivery log           (same-origin only)
    POST /api/notify/test        {channel: sms|call|ntfy|telegram|email}  (same-origin only)
"""

import asyncio
import base64
import io
import json
import math
import os
import re
import secrets
import smtplib
import socket
import ssl
import sys
import threading
import time
import unicodedata
import uuid
import wave
from collections import deque
from datetime import datetime
from email.message import EmailMessage
from xml.sax.saxutils import escape as xml_escape

import requests
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from werkzeug.security import safe_join

try:
    import edge_tts
except ImportError:  # server still runs; TTS reports itself unavailable
    edge_tts = None
try:
    from groq import Groq
except ImportError:
    Groq = None
try:
    import numpy as np
except ImportError:   # screening reports itself unavailable; everything else still works
    np = None
try:
    import segno      # QR codes for the ntfy / LAN links
except ImportError:
    segno = None
# MediaPipe / LiteRT are chatty on stderr; keep startup logs readable.
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


# ════════════════════════════════════════════════════════════════
#  Paths & environment
# ════════════════════════════════════════════════════════════════
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Log lines contain emoji/Indic text; on a cp1252 console or redirected pipe a print()
# would raise UnicodeEncodeError inside a request. Replace unencodable chars instead.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass


def _p(*parts):
    return os.path.join(BASE_DIR, *parts)


def _load_env(path):
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key, value = key.strip(), value.strip().strip('"').strip("'")
            if key:
                os.environ.setdefault(key, value)


ENV_PATH = _p(".env")
_load_env(ENV_PATH)
_PLACEHOLDER = re.compile(r"^(your_|<|changeme|xxx|todo)", re.I)


def env(name):
    """Env var value, or '' when unset / still a placeholder."""
    value = (os.environ.get(name) or "").strip()
    return "" if _PLACEHOLDER.match(value) else value


TTS_CACHE_DIR = _p("tts_cache")
PARENT_VOICE_DIR = _p("parent_voice")
LEGACY_PARENT_VOICE_DIR = _p("mitra trail", "parent_voice")
CAPTURES_DIR = _p("captures")
PARENT_MSG_DIR = _p("parent_messages")
MODELS_DIR = _p("models")
for _d in (TTS_CACHE_DIR, PARENT_VOICE_DIR, CAPTURES_DIR, PARENT_MSG_DIR, MODELS_DIR):
    os.makedirs(_d, exist_ok=True)

DATA_FILE = _p("mitra_alerts.json")
STATUS_FILE = _p("mitra_status.json")
SETTINGS_FILE = _p("mitra_settings.json")
NOTIF_FILE = _p("mitra_notifications.json")
TRANSCRIPT_FILE = _p("mitra_transcript.json")
REFS_FILE = _p("mitra_references.json")

MAX_AUDIO_BYTES = 10 * 1024 * 1024
MAX_IMAGE_BYTES = 4 * 1024 * 1024
NOTIF_LIMIT = 300
TRANSCRIPT_LIMIT = 200
PENDING_TTL_SEC = 600


# ════════════════════════════════════════════════════════════════
#  JSON storage (thread-safe, atomic writes)
# ════════════════════════════════════════════════════════════════
_io_lock = threading.RLock()


def _read_json(path, default_factory):
    with _io_lock:
        if not os.path.exists(path):
            return default_factory()
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return default_factory()


def _write_json(path, data):
    with _io_lock:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)


def _write_bytes_atomic(path, data):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def _prune_dir(directory, keep, prefix=""):
    try:
        files = [os.path.join(directory, n) for n in os.listdir(directory)
                 if n.startswith(prefix) and not n.endswith(".tmp")]
        files.sort(key=os.path.getmtime)
        for path in files[:-keep] if len(files) > keep else []:
            os.remove(path)
    except OSError:
        pass


def _now_iso():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _float(value, default=None, lo=None, hi=None):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    if v != v:  # NaN
        return default
    if lo is not None and v < lo:
        return default
    if hi is not None and v > hi:
        return default
    return v


def _wav_info(data):
    """Parse a PCM WAV header. Returns dict or None when not a readable WAV."""
    try:
        with wave.open(io.BytesIO(data), "rb") as w:
            ch, sw, rate, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
        if rate <= 0 or ch <= 0:
            return None
        return {"channels": ch, "sampwidth": sw, "rate": rate, "frames": n, "duration": n / rate}
    except Exception:
        return None


def _read_upload():
    """Bytes of an uploaded `audio` file (multipart) or raw body. Returns (bytes, error)."""
    f = request.files.get("audio")
    data = f.read(MAX_AUDIO_BYTES + 1) if f else request.get_data(cache=False)
    if not data:
        return None, "No audio received"
    if len(data) > MAX_AUDIO_BYTES:
        return None, "Audio file is larger than 10 MB"
    return data, None


# ════════════════════════════════════════════════════════════════
#  Languages, phrases, replies
# ════════════════════════════════════════════════════════════════
LANGUAGES = {
    "en": {"name": "English", "native": "English", "voice": "en-US-AriaNeural", "script": "Latin"},
    "hi": {"name": "Hindi", "native": "हिन्दी", "voice": "hi-IN-SwaraNeural", "script": "Devanagari"},
    "te": {"name": "Telugu", "native": "తెలుగు", "voice": "te-IN-ShrutiNeural", "script": "Telugu"},
    "ta": {"name": "Tamil", "native": "தமிழ்", "voice": "ta-IN-PallaviNeural", "script": "Tamil"},
    "kn": {"name": "Kannada", "native": "ಕನ್ನಡ", "voice": "kn-IN-SapnaNeural", "script": "Kannada"},
    "es": {"name": "Spanish", "native": "Español", "voice": "es-ES-ElviraNeural", "script": "Latin"},
    "fr": {"name": "French", "native": "Français", "voice": "fr-FR-DeniseNeural", "script": "Latin"},
}

# Allow-list of parent voice phrase ids (file contract: parent_voice/parent_voice_<id>.wav).
PHRASES = {
    "comfort_1": "Hey sweetheart, it's okay. I'm right here with you, always.",
    "comfort_2": "You are so loved. More than all the stars in the sky.",
    "comfort_3": "It's okay to feel sad. I feel it too when I'm away from you.",
    "comfort_4": "You are so brave. I'm proud of you every single day.",
    "comfort_5": "I'll be back before you know it. And I'll give you the biggest hug!",
    "scared_1": "There's nothing to be scared of. I'm always watching over you.",
    "scared_2": "You're safe. You're warm. You're loved. Everything is okay.",
    "bedtime_1": "Time to sleep, little one. Dream of something magical tonight.",
    "bedtime_2": "Close your eyes. I'm sending you a big goodnight kiss right now.",
    "bedtime_3": "Sweet dreams, my love. I'll see you when you wake up.",
    "praise_1": "You did it! I knew you could. That's my amazing kid!",
    "praise_2": "You make me so so SO proud. Every day.",
    "greet_1": "Good morning, sunshine! Today is going to be a great day!",
    "greet_2": "Hi baby! Miss you so much. Be good and have so much fun!",
}
EMOTION_CLIPS = {"cry": "comfort_2", "scared": "scared_1", "sad": "comfort_5"}

REPLIES = {
    "cry": {
        "en": "Mama and Papa love you so much! Does that feel better? Let me tell you a little bedtime story.",
        "hi": "मम्मा और पापा तुमसे बहुत प्यार करते हैं! अब थोड़ा अच्छा लग रहा है? चलो, मैं तुम्हें एक छोटी सी कहानी सुनाती हूँ।",
        "te": "అమ్మా నాన్నలకు నువ్వంటే చాలా ఇష్టం! ఇప్పుడు కొంచెం బాగుందా? రా, నీకు ఒక చిన్న కథ చెబుతాను.",
        "ta": "அம்மாவும் அப்பாவும் உன்னை ரொம்ப நேசிக்கிறாங்க! இப்போ கொஞ்சம் பரவாயில்லையா? வா, உனக்கு ஒரு சின்ன கதை சொல்றேன்.",
        "kn": "ಅಮ್ಮ ಅಪ್ಪ ನಿನ್ನನ್ನು ತುಂಬಾ ಪ್ರೀತಿಸುತ್ತಾರೆ! ಈಗ ಸ್ವಲ್ಪ ಪರವಾಗಿಲ್ಲವಾ? ಬಾ, ನಿನಗೆ ಒಂದು ಪುಟ್ಟ ಕಥೆ ಹೇಳುತ್ತೇನೆ.",
        "es": "¡Mamá y papá te quieren muchísimo! ¿Te sientes un poquito mejor? Te voy a contar un cuento cortito.",
        "fr": "Maman et papa t'aiment très fort ! Tu te sens un peu mieux ? Je vais te raconter une petite histoire.",
    },
    "scared": {
        "en": "You are super brave! The dark is just a blanket the stars sleep under. Shall we tell a funny story to make you smile?",
        "hi": "तुम बहुत बहादुर हो! अंधेरा तो बस एक कंबल है जिसके नीचे तारे सोते हैं। चलो, एक मज़ेदार कहानी सुनें?",
        "te": "నువ్వు చాలా ధైర్యంగా ఉన్నావు! చీకటి అంటే నక్షత్రాలు కప్పుకునే దుప్పటి అంతే. ఒక సరదా కథ విందామా?",
        "ta": "நீ ரொம்ப தைரியசாலி! இருட்டு என்பது நட்சத்திரங்கள் தூங்கும் போர்வை தான். ஒரு வேடிக்கையான கதை கேட்போமா?",
        "kn": "ನೀನು ತುಂಬಾ ಧೈರ್ಯಶಾಲಿ! ಕತ್ತಲೆ ಅಂದರೆ ನಕ್ಷತ್ರಗಳು ಹೊದ್ದು ಮಲಗುವ ಕಂಬಳಿ ಅಷ್ಟೇ. ಒಂದು ತಮಾಷೆ ಕಥೆ ಕೇಳೋಣವಾ?",
        "es": "¡Eres muy valiente! La oscuridad es solo una mantita bajo la que duermen las estrellas. ¿Te cuento un cuento divertido?",
        "fr": "Tu es très courageux ! Le noir, c'est juste une couverture sous laquelle dorment les étoiles. On écoute une histoire rigolote ?",
    },
    "sad": {
        "en": "Mama and Papa are thinking of you right now! Let's play a fun game or sing a song together!",
        "hi": "मम्मा और पापा अभी तुम्हारे बारे में ही सोच रहे हैं! चलो साथ में कोई मज़ेदार खेल खेलें या गाना गाएँ!",
        "te": "అమ్మా నాన్న ఇప్పుడు నీ గురించే ఆలోచిస్తున్నారు! రా, కలిసి ఒక ఆట ఆడదాం లేదా పాట పాడదాం!",
        "ta": "அம்மாவும் அப்பாவும் இப்போ உன்னைப் பத்திதான் நினைச்சுட்டு இருக்காங்க! வா, சேர்ந்து விளையாடலாம் இல்லன்னா பாட்டு பாடலாம்!",
        "kn": "ಅಮ್ಮ ಅಪ್ಪ ಈಗ ನಿನ್ನ ಬಗ್ಗೆಯೇ ಯೋಚಿಸುತ್ತಿದ್ದಾರೆ! ಬಾ, ಒಟ್ಟಿಗೆ ಒಂದು ಆಟ ಆಡೋಣ ಅಥವಾ ಹಾಡು ಹಾಡೋಣ!",
        "es": "¡Mamá y papá están pensando en ti ahora mismo! ¿Jugamos a algo divertido o cantamos una canción?",
        "fr": "Maman et papa pensent à toi en ce moment ! On joue à un jeu rigolo ou on chante une chanson ?",
    },
    "stop": {
        "en": "Okay! I'll stop right now. What would you like to do instead?",
        "hi": "ठीक है! मैं अभी रुक जाती हूँ। अब तुम क्या करना चाहोगे?",
        "te": "సరే! నేను ఇప్పుడే ఆపేస్తాను. ఇప్పుడు ఏం చేద్దాం?",
        "ta": "சரி! நான் இப்போவே நிறுத்திடறேன். இப்போ என்ன செய்யலாம்?",
        "kn": "ಸರಿ! ನಾನು ಈಗಲೇ ನಿಲ್ಲಿಸುತ್ತೇನೆ. ಈಗ ಏನು ಮಾಡೋಣ?",
        "es": "¡Vale! Ya paro. ¿Qué te gustaría hacer ahora?",
        "fr": "D'accord ! J'arrête tout de suite. Qu'est-ce que tu veux faire maintenant ?",
    },
    "greeting": {
        "en": "Hi there! I'm Mitra, your best friend! What would you like to play or talk about?",
        "hi": "नमस्ते! मैं मित्रा हूँ, तुम्हारी सबसे अच्छी दोस्त! आज क्या खेलें या किस बारे में बात करें?",
        "te": "హాయ్! నేను మిత్రా, నీ బెస్ట్ ఫ్రెండ్ని! ఏం ఆడదాం, ఏం మాట్లాడదాం?",
        "ta": "ஹாய்! நான் மித்ரா, உன் பெஸ்ட் ஃப்ரெண்ட்! என்ன விளையாடலாம், எதைப் பத்தி பேசலாம்?",
        "kn": "ಹಾಯ್! ನಾನು ಮಿತ್ರಾ, ನಿನ್ನ ಬೆಸ್ಟ್ ಫ್ರೆಂಡ್! ಏನು ಆಡೋಣ, ಏನು ಮಾತಾಡೋಣ?",
        "es": "¡Hola! Soy Mitra, ¡tu mejor amiga! ¿A qué quieres jugar o de qué quieres hablar?",
        "fr": "Coucou ! Je suis Mitra, ta meilleure amie ! À quoi veux-tu jouer ou de quoi veux-tu parler ?",
    },
    "fallback": {
        "en": "I am listening closely! Tell me more about what you want to do!",
        "hi": "मैं ध्यान से सुन रही हूँ! मुझे और बताओ।",
        "te": "నేను శ్రద్ధగా వింటున్నాను! ఇంకా చెప్పు.",
        "ta": "நான் கவனமா கேட்டுட்டு இருக்கேன்! இன்னும் சொல்லு.",
        "kn": "ನಾನು ಗಮನವಿಟ್ಟು ಕೇಳಿಸಿಕೊಳ್ಳುತ್ತಿದ್ದೇನೆ! ಇನ್ನೂ ಹೇಳು.",
        "es": "¡Te estoy escuchando! Cuéntame más.",
        "fr": "Je t'écoute bien ! Raconte-moi encore.",
    },
}


def reply_for(kind, lang):
    return REPLIES[kind].get(lang) or REPLIES[kind]["en"]


# ── Language detection ──────────────────────────────────────────
_SCRIPT_RANGES = (("hi", 0x0900, 0x097F), ("te", 0x0C00, 0x0C7F),
                  ("ta", 0x0B80, 0x0BFF), ("kn", 0x0C80, 0x0CFF))
_ES_WORDS = {"hola", "quiero", "tengo", "miedo", "cuento", "cuentame", "gracias", "estoy", "triste",
             "por", "favor", "que", "como", "donde", "canta", "cancion", "basta", "mama", "papa",
             "los", "las", "una", "el", "yo", "tu", "me", "muy", "de", "con"}
_FR_WORDS = {"bonjour", "je", "j", "tu", "peur", "merci", "histoire", "raconte", "chante", "maman",
             "suis", "le", "les", "une", "et", "est", "moi", "il", "plait", "arrete", "noir", "du", "des", "avec"}


def _fold(text):
    """Lowercase + strip Latin accents (only used for Latin keyword matching)."""
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def script_language(text):
    """Indic language implied by the script of `text`, or None for Latin/other text."""
    counts = {}
    for ch in text:
        cp = ord(ch)
        for code, lo, hi in _SCRIPT_RANGES:
            if lo <= cp <= hi:
                counts[code] = counts.get(code, 0) + 1
    return max(counts, key=counts.get) if counts else None


def detect_language(text):
    by_script = script_language(text)
    if by_script:
        return by_script
    low = text.lower()
    if any(c in low for c in "ñ¿¡"):
        return "es"
    if any(c in low for c in "çœèêëàâîôûù"):
        return "fr"
    words = set(re.findall(r"[a-z]+", _fold(text)))
    es, fr = len(words & _ES_WORDS), len(words & _FR_WORDS)
    if es >= 2 and es > fr:
        return "es"
    if fr >= 2 and fr > es:
        return "fr"
    return "en"


def resolve_language(requested, text=""):
    """Concrete language code: explicit choice, else detected from text ('auto'), else settings."""
    req = str(requested or "").strip().lower()
    if req in LANGUAGES:
        return req
    if text and text.strip():
        return detect_language(text)
    saved = load_settings().get("language")
    return saved if saved in LANGUAGES else "en"


# ── Multilingual intent classifier ─────────────────────────────
# Order matters: cry → stop → scared → sad (same priority as the original server).
INTENT_KEYWORDS = {
    "cry": [
        # en
        "waaah", "wah wah", "crying", "i am crying", "i'm crying",
        # hi (Devanagari + romanized)
        "रो रहा", "रो रही", "रो रहे", "रोना आ", "रो दिया", "आँसू", "ro raha", "ro rahi", "rona aa",
        # te
        "ఏడుస్తున్నా", "ఏడుస్తు", "ఏడుపు",
        # ta
        "அழுகிறேன்", "அழுகை", "அழுதுட்டு", "அழறேன்",
        # kn
        "ಅಳುತ್ತಿದ್ದೇನೆ", "ಅಳುತ್ತ", "ಅಳು ಬರ",
        # es / fr
        "llorando", "lloro", "quiero llorar", "je pleure", "pleurer", "en pleurs",
    ],
    "stop": [
        "stop", "enough", "hush", "be quiet", "shut up",
        "रुको", "रुक जाओ", "बंद करो", "बस करो", "चुप हो", "ruko", "bas karo", "band karo",
        "ఆపు", "ఆపండి", "ఆగు", "చాలు",
        "நிறுத்து", "நிறுத்துங்க", "போதும்",
        "ನಿಲ್ಲಿಸು", "ನಿಲ್ಲಿಸಿ", "ಸಾಕು",
        "basta", "detente", "para ya", "parate", "callate",
        "arrete", "tais-toi", "ca suffit", "chut",
    ],
    "scared": [
        "scared", "dark", "monster", "afraid", "nightmare", "fear", "frightened",
        "डर", "डरा", "भूत", "अंधेरा", "अँधेरा", "राक्षस", "dar lag", "darr", "dar gaya", "dar gayi",
        "భయ", "చీకటి", "దెయ్యం", "పీడకల",
        "பயம", "பயமா", "இருட்டு", "பேய்", "கெட்ட கனவு",
        "ಭಯ", "ಹೆದರಿಕೆ", "ಕತ್ತಲೆ", "ದೆವ್ವ", "ಕೆಟ್ಟ ಕನಸು",
        "miedo", "asustado", "asustada", "monstruo", "oscuridad", "pesadilla",
        "peur", "monstre", "cauchemar", "le noir", "effraye",
    ],
    "sad": [
        "sad", "lonely", "alone", "miss mama", "miss mommy", "miss papa", "miss daddy", "miss you",
        "उदास", "दुखी", "अकेला", "अकेली", "याद आ रही", "याद आ रहा", "udaas", "akela", "akeli",
        "బాధ", "దిగులు", "ఒంటరి", "గుర్తొస్తు",
        "சோகம", "வருத்தம", "தனியா", "கவலை", "ஞாபகமா",
        "ದುಃಖ", "ಬೇಸರ", "ಒಂಟಿ", "ನೆನಪಾಗ",
        "triste", "me siento solo", "me siento sola", "estoy solo", "estoy sola", "extrano a",
        "je suis seul", "tout seul", "toute seule", "me manque",
    ],
}
INTENT_ORDER = ("cry", "stop", "scared", "sad")


def _compile_intents():
    compiled = {}
    for intent, words in INTENT_KEYWORDS.items():
        latin, native = [], []
        for w in words:
            folded = _fold(w)
            if folded.isascii():
                latin.append(re.escape(folded))
            else:
                native.append(unicodedata.normalize("NFC", w))
        pattern = re.compile(r"(?<![a-z0-9])(?:" + "|".join(latin) + ")") if latin else None
        compiled[intent] = (pattern, native)
    return compiled


_INTENTS = _compile_intents()


def classify_intent(text):
    """Return (intent, matched_keyword) where intent ∈ cry|stop|scared|sad, or (None, None)."""
    if not text:
        return None, None
    folded = _fold(text)
    native_text = unicodedata.normalize("NFC", text.lower())
    for intent in INTENT_ORDER:
        pattern, native = _INTENTS[intent]
        if pattern:
            m = pattern.search(folded)
            if m:
                return intent, m.group(0)
        for kw in native:
            if kw in native_text:
                return intent, kw
    return None, None


# ════════════════════════════════════════════════════════════════
#  Speech (Edge-TTS) & LLM (Groq)
# ════════════════════════════════════════════════════════════════
_THINK_RE = re.compile(r"<think>.*?(</think>|$)", re.S | re.I)


def clean_for_speech(text):
    text = _THINK_RE.sub("", text or "")
    text = re.sub(r"[*#_`~>|]+", "", text)
    return re.sub(r"\s+", " ", text).strip()


async def _synth_async(text, voice, out_path):
    await edge_tts.Communicate(text, voice=voice).save(out_path)


def synthesize(text, language="en"):
    """Edge-TTS → (url, error). Voice follows the language, and the script always wins
    (Devanagari text is never read by an English voice)."""
    if edge_tts is None:
        return None, "edge-tts is not installed (pip install edge-tts)"
    clean = clean_for_speech(text)
    if not clean:
        return None, "Nothing to speak"
    lang = language if language in LANGUAGES else detect_language(clean)
    by_script = script_language(clean)
    if by_script and by_script != lang:
        lang = by_script
    fname = f"mitra_{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}_{lang}.mp3"
    fpath = os.path.join(TTS_CACHE_DIR, fname)
    try:
        asyncio.run(_synth_async(clean, LANGUAGES[lang]["voice"], fpath))
    except Exception as e:
        return None, f"Edge-TTS error: {e}"
    if os.path.exists(fpath) and os.path.getsize(fpath) > 100:
        _prune_dir(TTS_CACHE_DIR, keep=400, prefix="mitra_")
        return f"/audio/tts/{fname}", None
    return None, "Edge-TTS returned no audio"


def generate_tts(text, language="en"):
    """Backward-compatible helper: URL or None."""
    return synthesize(text, language)[0]


GROQ_MODEL_DEFAULT = "qwen/qwen3.8-27b"
_groq_client = None
_groq_key_used = None
_groq_lock = threading.Lock()


def groq_configured():
    return Groq is not None and bool(env("GROQ_API_KEY"))


def _get_groq():
    global _groq_client, _groq_key_used
    with _groq_lock:
        key = env("GROQ_API_KEY")
        if key and (_groq_client is None or key != _groq_key_used) and Groq is not None:
            try:
                _groq_client, _groq_key_used = Groq(api_key=key), key
            except Exception as e:
                print(f"⚠️ Groq client init failed: {e}")
        return _groq_client if key else None


def _chat_create(client, **kwargs):
    """chat.completions.create with hidden reasoning when the model supports it."""
    try:
        return client.chat.completions.create(reasoning_format="hidden", **kwargs)
    except TypeError:
        return client.chat.completions.create(**kwargs)
    except Exception as e:  # model doesn't support reasoning_format → retry plainly
        if "reasoning" not in str(e).lower():
            raise
        return client.chat.completions.create(**kwargs)


# ── Groq model discovery: never trust a hard-coded name ────────────
# The text, vision and speech-to-text models are picked from client.models.list();
# a vision model only counts once it has actually accepted an image in a probe call.
TEXT_MODEL_PREFS = ("qwen/qwen3.8-27b", "openai/gpt-oss-120b", "openai/gpt-oss-20b",
                    "llama-3.3-70b-versatile", "llama-3.1-8b-instant")
VISION_MODEL_HINTS = ("llama-4", "vision", "-vl", "scout", "maverick", "qwen3", "gemma-3")
WHISPER_PREFS = ("whisper-large-v3-turbo", "whisper-large-v3")
_NON_CHAT_HINTS = ("whisper", "tts", "orpheus", "guard", "distil", "playai")
MODEL_TTL_SEC = 6 * 3600
# 64×64 solid red JPEG (693 bytes) used to verify that a model really accepts image input.
_PROBE_JPEG_B64 = (
    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAoHBwgHBgoICAgLCgoLDhgQDg0NDh0VFhEYIx8lJCIfIiEmKzcvJik0KSEiMEEx"
    "NDk7Pj4+JS5ESUM8SDc9Pjv/2wBDAQoLCw4NDhwQEBw7KCIoOzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7"
    "Ozs7Ozs7Ozs7Ozs7Ozv/wAARCABAAEADASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAA"
    "AgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6"
    "Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXG"
    "x8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREA"
    "AgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5"
    "OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPE"
    "xcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwDCoooryD9GCiiigAooooAKKKKACiiigAoo"
    "ooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKAP//Z"
)
_models = {"checked_at": 0.0, "available": [], "text": None, "text_source": None, "text_note": None,
           "vision": None, "vision_probes": [], "whisper": None, "error": None}
_models_lock = threading.Lock()


def _probe_vision(client, model):
    """True when `model` accepts an image (real call with a 64×64 JPEG)."""
    t0 = time.time()
    try:
        _chat_create(client, model=model, max_tokens=60, temperature=0, messages=[{"role": "user", "content": [
            {"type": "text", "text": "What colour is this square? Answer with one word."},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + _PROBE_JPEG_B64}}]}])
        return {"model": model, "ok": True, "ms": int((time.time() - t0) * 1000), "error": None}
    except Exception as e:
        return {"model": model, "ok": False, "ms": int((time.time() - t0) * 1000), "error": _scrub(e)[:200]}


def refresh_models(force=False):
    """Discover usable Groq models (cached for MODEL_TTL_SEC). Returns a snapshot dict."""
    with _models_lock:
        fresh = _models["checked_at"] and time.time() - _models["checked_at"] < MODEL_TTL_SEC
        if fresh and not force and not _models["error"]:
            return dict(_models)
        client = _get_groq()
        if client is None:
            _models.update(checked_at=time.time(), available=[], text=None, vision=None, whisper=None,
                           error="GROQ_API_KEY is not set in .env")
            return dict(_models)
        try:
            listed = client.models.list().data
        except Exception as e:
            _models.update(checked_at=time.time(), error=f"Groq models.list() failed: {_scrub(e)[:200]}")
            return dict(_models)
        ids = sorted(m.id for m in listed if getattr(m, "active", True) is not False)
        chat_ids = [i for i in ids if not any(h in i.lower() for h in _NON_CHAT_HINTS)]
        wanted, note = env("GROQ_MODEL"), None
        if wanted and wanted in ids:
            text, source = wanted, "GROQ_MODEL in .env (verified in models.list)"
        else:
            if wanted:
                note = f"GROQ_MODEL={wanted} is not offered to this API key — using a verified fallback"
            text = next((m for m in TEXT_MODEL_PREFS if m in ids), chat_ids[0] if chat_ids else None)
            source = "preferred model (verified in models.list)" if text in TEXT_MODEL_PREFS else "first available chat model"
        candidates = []
        for m in [env("GROQ_VISION_MODEL")] + [i for i in chat_ids if any(h in i.lower() for h in VISION_MODEL_HINTS)]:
            if m and m in ids and m not in candidates:
                candidates.append(m)
        probes, vision = [], None
        for m in candidates[:5]:
            probe = _probe_vision(client, m)
            probes.append(probe)
            if probe["ok"]:
                vision = m
                break
        whisper = next((m for m in WHISPER_PREFS if m in ids), None)
        _models.update(checked_at=time.time(), available=ids, text=text, text_source=source, text_note=note,
                       vision=vision, vision_probes=probes, whisper=whisper, error=None)
        print(f"🧠 Groq models verified — text: {text} · vision: {vision or 'none'} · speech-to-text: {whisper or 'none'}")
        return dict(_models)


def text_model():
    m = refresh_models()
    return m["text"] or env("GROQ_MODEL") or GROQ_MODEL_DEFAULT


def models_public():
    m = dict(_models)
    return {"checked_at": m["checked_at"] or None, "text": m["text"], "text_source": m["text_source"],
            "text_note": m["text_note"], "vision": m["vision"], "whisper": m["whisper"],
            "vision_probes": m["vision_probes"], "available": m["available"], "error": m["error"]}


def _is_model_gone(err):
    s = str(err).lower()
    return "model_not_found" in s or "decommissioned" in s or "does not exist" in s or "404" in s


DIGEST_MODEL_PREFS = ("openai/gpt-oss-20b", "llama-3.1-8b-instant", "openai/gpt-oss-120b")


def digest_model():
    """Translations use a different verified model when possible, so they don't spend the main
    model's per-minute token budget (shared by chat and vision on free Groq keys)."""
    m = refresh_models()
    return next((x for x in DIGEST_MODEL_PREFS if x in m["available"] and x != m["text"]), None) or text_model()


_RETRY_RE = re.compile(r"try again in ((?:\d+m)?[\d.]+s)", re.I)


def _rate_limit_note(err):
    s = str(err)
    if "429" not in s and "rate limit" not in s.lower():
        return None
    m = _RETRY_RE.search(s)
    limit = re.search(r"\((\w+)\): Limit (\d+)", s)
    what = f"{limit.group(1)} limit {int(limit.group(2)):,}/min" if limit else "per-minute limit"
    return f"Groq rate limit reached ({what} on this key) — retry in {m.group(1) if m else 'a few seconds'}"


def llm_complete(messages, max_tokens=400, temperature=0.7, model=None):
    """Groq chat completion → (clean_text, error). Retries once with a re-verified model on 404."""
    client = _get_groq()
    if client is None:
        return None, "GROQ_API_KEY is not set in .env"
    client = client.with_options(max_retries=1, timeout=45)   # never stall a child's chat on silent retries
    use = model or text_model()
    for attempt in range(2):
        try:
            resp = _chat_create(client, model=use, messages=messages, max_tokens=max_tokens, temperature=temperature)
            text = clean_for_speech(resp.choices[0].message.content or "")
            return (text, None) if text else (None, "Model returned an empty reply")
        except Exception as e:
            if attempt == 0 and not model and _is_model_gone(e):
                use = refresh_models(force=True)["text"] or use
                continue
            return None, _rate_limit_note(e) or f"Groq error: {_scrub(e)}"
    return None, "Groq error: no usable model"


VISION_MAX_SIDE = 448            # Groq bills ≤448 px images as one tile (1,309 input tokens measured)
_vision_blocked_until = 0.0


def _vision_jpeg(jpeg_bytes):
    """Downscale to ≤448 px so a frame costs one image tile instead of two."""
    try:
        import cv2
        img = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        if max(h, w) > VISION_MAX_SIDE:
            s = VISION_MAX_SIDE / max(h, w)
            img = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok:
                return buf.tobytes()
    except Exception:
        pass
    return jpeg_bytes


def vision_complete(prompt, jpeg_bytes, system=None, max_tokens=400):
    """Image + text → (raw_text, model, error) using the verified vision model. No silent retries:
    on a rate limit the caller gets the real reason and when to retry."""
    global _vision_blocked_until
    client = _get_groq()
    if client is None:
        return None, None, "GROQ_API_KEY is not set in .env"
    model = refresh_models()["vision"]
    if not model:
        return None, None, "No vision-capable model is available to this Groq API key"
    wait = _vision_blocked_until - time.time()
    if wait > 0:
        return None, model, f"Groq rate limit cooling down — vision available again in {wait:.0f}s"
    url = "data:image/jpeg;base64," + base64.b64encode(_vision_jpeg(jpeg_bytes)).decode()
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": [
        {"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": url}}]}]
    try:
        resp = _chat_create(client.with_options(max_retries=0, timeout=30), model=model, messages=messages,
                            max_tokens=max_tokens, temperature=0.2)
        text = _THINK_RE.sub("", resp.choices[0].message.content or "").strip()
        return (text, model, None) if text else (None, model, "Vision model returned an empty reply")
    except Exception as e:
        note = _rate_limit_note(e)
        if note:
            retry = getattr(getattr(e, "response", None), "headers", {}).get("retry-after")
            _vision_blocked_until = time.time() + (_float(retry, 5, 0, 600) or 5)
            return None, model, note
        return None, model, f"Groq vision error: {_scrub(e)}"


SYSTEM_PROMPT = (
    "You are Mitra, a smart, caring, and cheerful AI companion for young children (aged 3 to 8). "
    "Rules for your responses:\n"
    "1. Always address what the child or parent is actually asking or saying with precision.\n"
    "2. If asked for a story, tell an engaging, well-crafted, imaginative story in 3-4 sentences.\n"
    "3. If asked an educational or factual question, explain it simply, accurately, and clearly.\n"
    "4. If the user mentions an issue (like overlapping voices), acknowledge it politely and clearly.\n"
    "5. Keep responses concise (2 to 4 sentences), kind, helpful, and grammatically complete.\n"
    "6. Your words are read aloud by a voice engine: never use markdown, asterisks, hashes, bullet "
    "points, emojis or stage directions."
)


def language_rule(lang):
    info = LANGUAGES.get(lang, LANGUAGES["en"])
    if lang == "en":
        return "Reply in simple, warm English."
    return (f"Always reply ONLY in {info['name']} ({info['native']}), written in {info['script']} script, "
            "even if the child mixes in English words. Use simple, warm, everyday words a small child "
            "understands. Do not add translations or transliterations.")


_chat_history = deque(maxlen=10)   # user/assistant turns (system prompt is built per request)
_history_lock = threading.Lock()


# ════════════════════════════════════════════════════════════════
#  Settings
# ════════════════════════════════════════════════════════════════
DEFAULT_SETTINGS = {
    "parent_phone": "",
    "secondary_phone": "",
    "parent_label": "Mama/Papa",
    "language": "en",
    "enable_sms": False,
    "enable_call": False,
    "enable_ntfy": False,
    "ntfy_topic": "",
    "enable_telegram": False,
    "enable_email": False,
    "enable_browser_push": True,
    "enable_escalation_alarm": False,
    "triggers": {"cry": True, "scared": True, "sad": True, "loud": True, "high_stress": True, "absent": False},
    "call_cooldown_sec": 60,
    "monitor_threshold_db": 65,
    "monitor_calibration_db": 90,
    "room_baseline_dbfs": None,     # measured by "Calibrate Quiet Room (3s)"
    "stress_threshold": 66,         # stress index (0–100) that raises a high_stress alert
    # cry_model.pkl probability needed (plus loudness and 2 windows in a row). Measured on the hub:
    # 98% of 150 real cries score ≥ 0.9, but so do 25% of 3-s speech windows (all speech ≥ 0.67).
    "cry_threshold": 0.9,
    "absent_after_sec": 60,         # camera absence before an "absent" alert (when enabled)
}
# Alert type → trigger switch that controls parent notifications for it.
TRIGGER_FOR_TYPE = {"cry": "cry", "scared": "scared", "sad": "sad", "lonely": "sad", "missing": "sad",
                    "loud": "loud", "high_stress": "high_stress", "absent": "absent"}
E164_RE = re.compile(r"^\+[1-9]\d{7,14}$")
TOPIC_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_settings_lock = threading.Lock()


def load_settings():
    raw = _read_json(SETTINGS_FILE, dict)
    merged = json.loads(json.dumps(DEFAULT_SETTINGS))
    if isinstance(raw, dict):
        for k, v in raw.items():
            if k == "triggers" and isinstance(v, dict):
                merged["triggers"].update({t: bool(on) for t, on in v.items() if t in merged["triggers"]})
            elif k in merged:
                merged[k] = v
    return merged


def validate_settings_patch(patch):
    """Return (clean_patch, errors). Unknown keys are ignored."""
    clean, errors = {}, {}
    if not isinstance(patch, dict):
        return {}, {"_": "Expected a JSON object"}
    for key in ("parent_phone", "secondary_phone"):
        if key in patch:
            phone = re.sub(r"[\s\-().]", "", str(patch[key] or ""))
            if phone and not E164_RE.match(phone):
                errors[key] = "Use international E.164 format, e.g. +919876543210"
            else:
                clean[key] = phone
    if "parent_label" in patch:
        label = str(patch["parent_label"] or "").strip()[:24]
        clean["parent_label"] = label or DEFAULT_SETTINGS["parent_label"]
    if "language" in patch:
        lang = str(patch["language"] or "").lower()
        if lang in LANGUAGES or lang == "auto":
            clean["language"] = lang
        else:
            errors["language"] = "Unsupported language"
    for key in ("enable_sms", "enable_call", "enable_ntfy", "enable_telegram", "enable_email",
                "enable_browser_push", "enable_escalation_alarm"):
        if key in patch:
            clean[key] = bool(patch[key])
    if "room_baseline_dbfs" in patch:
        if patch["room_baseline_dbfs"] is None:
            clean["room_baseline_dbfs"] = None
        else:
            v = _float(patch["room_baseline_dbfs"], None, -100, 0)
            if v is None:
                errors["room_baseline_dbfs"] = "Must be between -100 and 0 dBFS"
            else:
                clean["room_baseline_dbfs"] = round(v, 1)
    if "cry_threshold" in patch:
        v = _float(patch["cry_threshold"], None, 0.2, 0.99)
        if v is None:
            errors["cry_threshold"] = "Must be between 0.20 and 0.99"
        else:
            clean["cry_threshold"] = round(v, 2)
    if "ntfy_topic" in patch:
        topic = str(patch["ntfy_topic"] or "").strip()
        if topic and not TOPIC_RE.match(topic):
            errors["ntfy_topic"] = "Letters, numbers, - and _ only (max 64)"
        else:
            clean["ntfy_topic"] = topic
    if "triggers" in patch:
        if isinstance(patch["triggers"], dict):
            clean["triggers"] = {t: bool(v) for t, v in patch["triggers"].items() if t in DEFAULT_SETTINGS["triggers"]}
        else:
            errors["triggers"] = "Expected an object"
    for key, lo, hi in (("call_cooldown_sec", 0, 3600), ("monitor_threshold_db", 40, 90),
                        ("monitor_calibration_db", 60, 130), ("stress_threshold", 30, 95),
                        ("absent_after_sec", 15, 1800)):
        if key in patch:
            v = _float(patch[key], None, lo, hi)
            if v is None:
                errors[key] = f"Must be a number between {lo} and {hi}"
            else:
                clean[key] = int(round(v))
    return clean, errors


def _same_origin():
    """Block cross-site browser requests to sensitive endpoints (phones, paid telephony).
    Non-browser clients (no Origin header) and the dashboard's own origin are allowed."""
    origin = request.headers.get("Origin")
    return not origin or origin.rstrip("/") == request.host_url.rstrip("/")


def _forbidden():
    return jsonify({"ok": False, "error": "Open the dashboard from this Mitra hub (same origin) to change alert settings."}), 403


def _from_hub_machine():
    """Provider keys may only be written from the hub computer itself (anyone on the Wi-Fi can
    open the dashboard; they must not be able to redirect alerts). MITRA_ALLOW_LAN_ADMIN=1 lifts this."""
    addr = (request.remote_addr or "").replace("::ffff:", "")
    return addr in ("127.0.0.1", "::1") or env("MITRA_ALLOW_LAN_ADMIN") == "1"


# ════════════════════════════════════════════════════════════════
#  Parent notifications (Twilio SMS/Voice, ntfy, Telegram)
# ════════════════════════════════════════════════════════════════
ALERT_LABELS = {
    "cry": "Crying detected",
    "scared": "Your child seems scared",
    "sad": "Your child seems sad",
    "lonely": "Your child feels lonely",
    "missing": "Your child is missing you",
    "loud": "Sustained loud sound detected",
    "high_stress": "High stress detected",
    "absent": "Child not seen on camera",
}
_notif_lock = threading.Lock()
_cooldown_lock = threading.Lock()
_last_phone_at = {"sms": 0.0, "call": 0.0}
_last_test_at = {}


def _twilio_creds():
    return env("TWILIO_ACCOUNT_SID"), env("TWILIO_AUTH_TOKEN"), env("TWILIO_FROM_NUMBER")


def twilio_configured():
    return all(_twilio_creds())


def telegram_configured():
    return bool(env("TELEGRAM_BOT_TOKEN") and env("TELEGRAM_CHAT_ID"))


def email_configured():
    return bool(env("SMTP_HOST") and env("SMTP_USER") and env("SMTP_PASS") and env("PARENT_EMAIL"))


def ntfy_server():
    return (env("NTFY_SERVER") or "https://ntfy.sh").rstrip("/")


def ntfy_topic(settings):
    return (settings.get("ntfy_topic") or "").strip() or env("NTFY_TOPIC")


SECRET_ENV = ("GROQ_API_KEY", "TWILIO_AUTH_TOKEN", "TELEGRAM_BOT_TOKEN", "NTFY_TOKEN", "SMTP_PASS")


def _scrub(text):
    """Remove secrets from provider error strings before logging/returning them."""
    text = str(text)
    for name in SECRET_ENV:
        secret = env(name)
        if secret:
            text = text.replace(secret, "***")
    return text[:400]


# ── Credentials in .env (written from the dashboard, applied without restart) ──
EMAIL_RE = re.compile(r"^[^@\s]{1,64}@[A-Za-z0-9.-]{1,190}\.[A-Za-z]{2,}$")
CREDENTIAL_SPECS = {
    # name: (validator regex, secret?, human hint)
    "TWILIO_ACCOUNT_SID": (re.compile(r"^AC[0-9a-fA-F]{32}$"), False, "starts with AC + 32 hex characters"),
    "TWILIO_AUTH_TOKEN": (re.compile(r"^[0-9a-fA-F]{32}$"), True, "32 hex characters"),
    "TWILIO_FROM_NUMBER": (E164_RE, False, "your Twilio number in E.164, e.g. +15551234567"),
    "TELEGRAM_BOT_TOKEN": (re.compile(r"^\d{5,}:[A-Za-z0-9_-]{30,}$"), True, "looks like 123456789:AA…"),
    "TELEGRAM_CHAT_ID": (re.compile(r"^(-?\d{1,20}|@[A-Za-z0-9_]{5,32})$"), False, "a number such as 123456789"),
    "SMTP_HOST": (re.compile(r"^[A-Za-z0-9.-]{3,253}$"), False, "e.g. smtp.gmail.com"),
    "SMTP_PORT": (re.compile(r"^\d{2,5}$"), False, "587 (STARTTLS) or 465 (SSL)"),
    "SMTP_USER": (re.compile(r"^\S{3,200}$"), False, "usually your full email address"),
    "SMTP_PASS": (re.compile(r"^[^\r\n]{4,200}$"), True, "for Gmail: a 16-character App Password"),
    "PARENT_EMAIL": (EMAIL_RE, False, "where alert emails are sent"),
    "NTFY_TOKEN": (re.compile(r"^\S{8,200}$"), True, "optional access token for a protected topic"),
}
_env_lock = threading.Lock()


def _mask(value, secret):
    if not value:
        return ""
    if "@" in value and not secret:
        name, _, domain = value.partition("@")
        return (name[:2] + "•••@" + domain) if len(name) > 2 else "•••@" + domain
    keep = 4 if len(value) > 8 else 2
    return "•" * min(12, max(4, len(value) - keep)) + value[-keep:]


def _env_file_keys():
    keys = set()
    if os.path.exists(ENV_PATH):
        with open(ENV_PATH, "r", encoding="utf-8") as f:
            for raw in f:
                m = re.match(r"^\s*([A-Z0-9_]+)\s*=", raw)
                if m:
                    keys.add(m.group(1))
    return keys


def _gitignored_env():
    try:
        with open(_p(".gitignore"), "r", encoding="utf-8") as f:
            return any(line.strip() in (".env", "/.env", ".env*") for line in f)
    except OSError:
        return False


def credentials_public():
    in_file = _env_file_keys()
    out = {}
    for name, (_, secret, hint) in CREDENTIAL_SPECS.items():
        value = env(name)
        out[name] = {"set": bool(value), "masked": _mask(value, secret), "secret": secret, "hint": hint,
                     "source": ("env file" if name in in_file else "environment") if value else None}
    return out


def write_env_values(updates):
    """Update/insert/remove KEY=value lines in .env atomically, preserving everything else,
    then apply the same values to os.environ so they work immediately."""
    with _env_lock:
        lines = []
        if os.path.exists(ENV_PATH):
            with open(ENV_PATH, "r", encoding="utf-8") as f:
                lines = f.read().splitlines()
        done = set()
        out = []
        for line in lines:
            m = re.match(r"^\s*([A-Z0-9_]+)\s*=", line)
            key = m.group(1) if m else None
            if key in updates:
                if key in done:
                    continue           # drop duplicate definitions of an updated key
                done.add(key)
                if updates[key]:
                    out.append(f"{key}={updates[key]}")
                continue               # empty value → remove the line
            out.append(line)
        missing = [k for k, v in updates.items() if v and k not in done]
        if missing:
            if out and out[-1].strip():
                out.append("")
            out.append("# Added from the Mitra dashboard (Alert Channels)")
            out.extend(f"{k}={updates[k]}" for k in missing)
        tmp = ENV_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(out).rstrip("\n") + "\n")
        os.replace(tmp, ENV_PATH)
        for k, v in updates.items():
            if v:
                os.environ[k] = v
            else:
                os.environ.pop(k, None)


def validate_credentials(patch):
    clean, errors = {}, {}
    if not isinstance(patch, dict):
        return {}, {"_": "Expected a JSON object"}
    for name, raw in patch.items():
        if name not in CREDENTIAL_SPECS:
            continue
        value = str(raw if raw is not None else "").strip()
        if name == "TWILIO_FROM_NUMBER":
            value = re.sub(r"[\s\-().]", "", value)
        if value and not CREDENTIAL_SPECS[name][0].match(value):
            errors[name] = f"Doesn't look right — {CREDENTIAL_SPECS[name][2]}"
        elif "\n" in value or "\r" in value:
            errors[name] = "Must be a single line"
        else:
            clean[name] = value
    return clean, errors


def _mask_phone(phone):
    if not phone:
        return ""
    return phone[:3] + "•" * max(0, len(phone) - 7) + phone[-4:] if len(phone) > 7 else phone


def _attempt(channel, status, detail, target="", sid=None, http_status=None, alert_id=None, test=False):
    return {
        "id": uuid.uuid4().hex[:12], "channel": channel, "status": status, "detail": detail,
        "target": target, "sid": sid, "http_status": http_status, "alert_id": alert_id,
        "test": test, "at": _now_iso(), "ts": time.time(),
    }


def record_attempts(attempts):
    if not attempts:
        return
    with _notif_lock:
        items = _read_json(NOTIF_FILE, list)
        if not isinstance(items, list):
            items = []
        items.extend(attempts)
        _write_json(NOTIF_FILE, items[-NOTIF_LIMIT:])


def load_notifications():
    items = _read_json(NOTIF_FILE, list)
    return items if isinstance(items, list) else []


def _twilio_post(resource, data):
    sid, token, _ = _twilio_creds()
    url = f"https://api.twilio.com/2010-04-01/Accounts/{sid}/{resource}.json"
    r = requests.post(url, data=data, auth=(sid, token), timeout=15)
    try:
        body = r.json()
    except ValueError:
        body = {}
    return r.status_code, body


def _twilio_result(channel, to, code, body, meta):
    if code in (200, 201):
        return _attempt(channel, "sent", f"Accepted by Twilio (status: {body.get('status', 'queued')})",
                        _mask_phone(to), sid=body.get("sid"), http_status=code, **meta)
    msg = body.get("message") or "Unknown Twilio error"
    if body.get("code"):
        msg += f" (Twilio error {body['code']})"
    return _attempt(channel, "failed", f"Twilio HTTP {code}: {msg}", _mask_phone(to), http_status=code, **meta)


def _twilio_precheck(channel, to, meta):
    if not twilio_configured():
        return _attempt(channel, "unconfigured",
                        "Add your Twilio Account SID, Auth Token and From number in Alert Channels → Twilio",
                        _mask_phone(to), **meta)
    if not to:
        return _attempt(channel, "unconfigured", "No parent phone number saved in Alert settings", **meta)
    return None


def send_sms(to, text, **meta):
    blocked = _twilio_precheck("sms", to, meta)
    if blocked:
        return blocked
    try:
        code, body = _twilio_post("Messages", {"To": to, "From": _twilio_creds()[2], "Body": text[:1500]})
    except requests.RequestException as e:
        return _attempt("sms", "failed", _scrub(e), _mask_phone(to), **meta)
    return _twilio_result("sms", to, code, body, meta)


def place_call(to, text, **meta):
    blocked = _twilio_precheck("call", to, meta)
    if blocked:
        return blocked
    spoken = xml_escape(text + " Please check on your child.")
    twiml = (f'<Response><Say voice="alice">{spoken}</Say><Pause length="1"/>'
             f'<Say voice="alice">Repeating. {spoken}</Say></Response>')
    try:
        code, body = _twilio_post("Calls", {"To": to, "From": _twilio_creds()[2], "Twiml": twiml})
    except requests.RequestException as e:
        return _attempt("call", "failed", _scrub(e), _mask_phone(to), **meta)
    return _twilio_result("call", to, code, body, meta)


def _rfc2047(value):
    """HTTP header value that survives non-ASCII text (ntfy decodes RFC 2047 encoded-words)."""
    if value.isascii():
        return value
    return "=?UTF-8?B?" + base64.b64encode(value.encode("utf-8")).decode() + "?="


def send_ntfy(topic, title, text, image=None, click=None, **meta):
    """Publish to ntfy. With `image` (JPEG bytes) the snapshot is uploaded as an attachment."""
    if not topic:
        return _attempt("ntfy", "unconfigured", "Generate a private topic in Alert Channels (or set NTFY_TOPIC in .env)", **meta)
    headers = {"Title": _rfc2047(title or "Mitra alert"), "Priority": "urgent", "Tags": "rotating_light"}
    if click:
        headers["Click"] = click
    if env("NTFY_TOKEN"):
        headers["Authorization"] = f"Bearer {env('NTFY_TOKEN')}"
    try:
        if image:
            headers.update({"Message": _rfc2047(text), "Filename": "mitra_snapshot.jpg"})
            r = requests.put(f"{ntfy_server()}/{topic}", data=image, headers=headers, timeout=20)
        else:
            r = requests.post(f"{ntfy_server()}/{topic}", data=text.encode("utf-8"), headers=headers, timeout=10)
    except requests.RequestException as e:
        return _attempt("ntfy", "failed", _scrub(e), topic, **meta)
    if r.ok:
        try:
            body = r.json()
        except ValueError:
            body = {}
        att = " with camera snapshot" if body.get("attachment") else ""
        return _attempt("ntfy", "sent", f"Published to {ntfy_server()}/{topic}{att}", topic, sid=body.get("id"),
                        http_status=r.status_code, **meta)
    return _attempt("ntfy", "failed", f"ntfy HTTP {r.status_code}: {_scrub(r.text.strip())[:200]}", topic,
                    http_status=r.status_code, **meta)


def send_telegram(text, image=None, **meta):
    """Telegram Bot API: sendPhoto (snapshot + caption) when an image exists, else sendMessage."""
    if not telegram_configured():
        return _attempt("telegram", "unconfigured", "Add your bot token and chat id in Alert Channels → Telegram", **meta)
    token, chat_id = env("TELEGRAM_BOT_TOKEN"), env("TELEGRAM_CHAT_ID")
    target = "chat …" + chat_id[-4:]
    try:
        if image:
            r = requests.post(f"https://api.telegram.org/bot{token}/sendPhoto",
                              data={"chat_id": chat_id, "caption": text[:1024]},
                              files={"photo": ("mitra_snapshot.jpg", image, "image/jpeg")}, timeout=20)
        else:
            r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                              json={"chat_id": chat_id, "text": text}, timeout=10)
        body = r.json() if r.content else {}
    except (requests.RequestException, ValueError) as e:
        return _attempt("telegram", "failed", _scrub(e), target, **meta)
    if r.ok and body.get("ok"):
        how = "Photo + caption delivered" if image else "Delivered"
        return _attempt("telegram", "sent", f"{how} by Telegram Bot API", target,
                        sid=str(body.get("result", {}).get("message_id")), http_status=r.status_code, **meta)
    return _attempt("telegram", "failed", f"Telegram HTTP {r.status_code}: {_scrub(body.get('description', r.text))}",
                    target, http_status=r.status_code, **meta)


def send_email(subject, text, image=None, **meta):
    """SMTP (e.g. Gmail with an App Password). STARTTLS on 587, implicit TLS on 465."""
    if not email_configured():
        return _attempt("email", "unconfigured", "Add SMTP host, user, password and parent email in Alert Channels → Email", **meta)
    host, user, password, to = env("SMTP_HOST"), env("SMTP_USER"), env("SMTP_PASS"), env("PARENT_EMAIL")
    port = int(env("SMTP_PORT") or 587)
    msg = EmailMessage()
    msg["Subject"], msg["From"], msg["To"] = subject, env("SMTP_FROM") or user, to
    msg.set_content(text + "\n\n— Sent by your Mitra hub")
    if image:
        msg.add_attachment(image, maintype="image", subtype="jpeg", filename="mitra_snapshot.jpg")
    try:
        ctx = ssl.create_default_context()
        if port == 465:
            with smtplib.SMTP_SSL(host, port, timeout=20, context=ctx) as s:
                s.login(user, password)
                refused = s.send_message(msg)
        else:
            with smtplib.SMTP(host, port, timeout=20) as s:
                s.starttls(context=ctx)
                s.login(user, password)
                refused = s.send_message(msg)
    except smtplib.SMTPAuthenticationError as e:
        hint = " — Gmail needs an App Password (Google Account → Security → App passwords)" if "gmail" in host else ""
        return _attempt("email", "failed", f"SMTP {e.smtp_code}: {_scrub(e.smtp_error.decode(errors='replace') if isinstance(e.smtp_error, bytes) else e.smtp_error)}{hint}", _mask(to, False), http_status=e.smtp_code, **meta)
    except (smtplib.SMTPException, OSError) as e:
        return _attempt("email", "failed", f"SMTP error: {_scrub(e)}", _mask(to, False), **meta)
    if refused:
        return _attempt("email", "failed", f"Recipient refused: {_scrub(refused)}", _mask(to, False), **meta)
    att = " with snapshot" if image else ""
    return _attempt("email", "sent", f"Accepted by {host}:{port}{att}", _mask(to, False), **meta)


def _cooldown_remaining(channel, cooldown):
    with _cooldown_lock:
        now = time.time()
        remaining = cooldown - (now - _last_phone_at.get(channel, 0.0))
        if remaining > 0:
            return remaining
        _last_phone_at[channel] = now
        return 0


def alert_message(alert):
    label = ALERT_LABELS.get(alert.get("type"), "Mitra event")
    text = f"Mitra alert: {label} at {alert.get('timestamp', '')}."
    if alert.get("note"):
        text += f" {alert['note']}"
    if alert.get("note_en") and alert.get("note_en") != alert.get("note"):
        text += f" (English: {alert['note_en']})"
    if alert.get("observation"):
        text += f" Camera: {alert['observation']}"
    return label, text


def _alert_frame_bytes(alert):
    url = alert.get("frame_url") or ""
    if not url.startswith("/captures/"):
        return None
    path = safe_join(CAPTURES_DIR, url.rsplit("/", 1)[-1])
    try:
        with open(path, "rb") as f:
            return f.read(MAX_IMAGE_BYTES)
    except (OSError, TypeError):
        return None


def _dispatch_worker(alert):
    try:
        s = load_settings()
        if not s["triggers"].get(TRIGGER_FOR_TYPE.get(alert.get("type"), alert.get("type")), False):
            return
        title, text = alert_message(alert)
        image = _alert_frame_bytes(alert)
        meta = {"alert_id": alert["id"]}
        attempts = []
        phones = [p for p in (s["parent_phone"], s["secondary_phone"]) if p] or [""]
        phone_ready = twilio_configured() and bool(s["parent_phone"] or s["secondary_phone"])
        for channel, sender in (("sms", send_sms), ("call", place_call)):
            if not s.get(f"enable_{channel}"):
                continue
            # Cooldown only applies to attempts that can actually reach Twilio, so an
            # unconfigured setup keeps reporting "unconfigured" rather than "cooldown".
            wait = _cooldown_remaining(channel, s["call_cooldown_sec"]) if phone_ready else 0
            if wait > 0:
                attempts.append(_attempt(channel, "skipped", f"Cooldown active — {int(wait)}s left", **meta))
                continue
            for phone in phones:
                attempts.append(sender(phone, text, **meta))
        if s.get("enable_ntfy"):
            attempts.append(send_ntfy(ntfy_topic(s), f"Mitra: {title}", text, image=image, **meta))
        if s.get("enable_telegram"):
            attempts.append(send_telegram("🚨 " + text, image=image, **meta))
        if s.get("enable_email"):
            attempts.append(send_email(f"Mitra alert: {title}", text, image=image, **meta))
        record_attempts(attempts)
        for a in attempts:
            print(f"   📣 {a['channel']}: {a['status']} — {a['detail']}")
    except Exception as e:
        print(f"⚠️ Notification dispatch error: {_scrub(e)}")


def dispatch_parent_notifications(alert):
    """Fire-and-forget: never blocks /alert or /api/interact."""
    threading.Thread(target=_dispatch_worker, args=(dict(alert),), daemon=True,
                     name=f"notify-{alert.get('id')}").start()


# ════════════════════════════════════════════════════════════════
#  Alerts
# ════════════════════════════════════════════════════════════════
_alerts_lock = threading.RLock()
ALERT_TYPE_RE = re.compile(r"^[a-z_]{1,24}$")


def load_data():
    data = _read_json(DATA_FILE, lambda: {"alerts": [], "stats": {"total_cries": 0, "today_cries": 0, "last_cry": None}})
    data.setdefault("alerts", [])
    data.setdefault("stats", {})
    return data


def save_data(data):
    _write_json(DATA_FILE, data)


def recalc_stats(data):
    today = datetime.now().strftime("%Y-%m-%d")
    data["stats"]["total_cries"] = len(data["alerts"])
    data["stats"]["today_cries"] = sum(1 for a in data["alerts"] if a.get("date", "") == today)
    data["stats"]["last_cry"] = data["alerts"][-1]["timestamp"] if data["alerts"] else None


def _delete_capture(alert):
    for key in ("capture_url", "frame_url"):
        url = alert.get(key) or ""
        if url.startswith("/captures/"):
            path = safe_join(CAPTURES_DIR, url.rsplit("/", 1)[-1])
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass


def create_alert(alert_type, note="", duration_sec=0.0, extra=None, capture=None, notify=True, frame=None):
    """Persist an alert (optionally with WAV capture bytes and a JPEG camera frame) and dispatch
    parent notifications. Evidence lands in captures/capture_<id>.wav and captures/frame_<id>.jpg."""
    with _alerts_lock:
        now = datetime.now()
        data = load_data()
        alert_id = int(time.time() * 1000)
        if data["alerts"] and alert_id <= int(data["alerts"][-1].get("id", 0)):
            alert_id = int(data["alerts"][-1]["id"]) + 1
        alert = {
            "id": alert_id,
            "type": alert_type,
            "duration_sec": round(float(duration_sec or 0), 1),
            "note": note,
            "timestamp": now.strftime("%H:%M:%S"),
            "date": now.strftime("%Y-%m-%d"),
            "datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
            "read": False,
        }
        alert.update(extra or {})
        if capture:
            fname = f"capture_{alert_id}.wav"
            _write_bytes_atomic(os.path.join(CAPTURES_DIR, fname), capture["bytes"])
            alert["capture_url"] = f"/captures/{fname}"
            alert["capture_duration_sec"] = round(capture["duration"], 1)
        if frame:
            fname = f"frame_{alert_id}.jpg"
            _write_bytes_atomic(os.path.join(CAPTURES_DIR, fname), frame)
            alert["frame_url"] = f"/captures/{fname}"
        data["alerts"].append(alert)
        for old in data["alerts"][:-200]:
            _delete_capture(old)
        data["alerts"] = data["alerts"][-200:]
        recalc_stats(data)
        save_data(data)
    print(f"🚨 Alert received: {alert['type']} at {alert['timestamp']}"
          + (" (with capture)" if capture else "") + (" (with camera frame)" if frame else ""))
    if notify:
        dispatch_parent_notifications(alert)
    return alert


def update_alert(alert_id, fields):
    """Merge fields into a stored alert (used by the background English digest)."""
    with _alerts_lock:
        data = load_data()
        for a in data["alerts"]:
            if a.get("id") == alert_id:
                a.update(fields)
                save_data(data)
                return True
    return False


def recent_distress_count(seconds=600):
    cutoff = time.time() - seconds
    n = 0
    for a in load_data()["alerts"][-40:]:
        if a.get("type") in DISTRESS_TYPES and a.get("id", 0) / 1000 >= cutoff:
            n += 1
    return n


def set_status(status):
    """Heartbeat from the Mitra hardware (mitra_ultimate.py → POST /status)."""
    try:
        _write_json(STATUS_FILE, {"status": status, "time": time.time()})
    except OSError:
        pass


# ── Hub activity: what the hub and its browser components are really doing ──
DISTRESS_TYPES = {"cry", "scared", "sad", "lonely", "missing", "loud", "high_stress", "absent"}
_activity_lock = threading.Lock()
_activity = {"state": "idle", "until": 0.0}
_companion_state = {}     # companion device → {"t": last claim, "state": idle|listening|thinking|speaking|parent}
_screen_devices = {}      # screening device → last analyze/telemetry time


def mark_activity(state, seconds=8):
    with _activity_lock:
        _activity.update(state=state, until=time.time() + seconds)


def hub_status():
    now = time.time()
    dev = _read_json(STATUS_FILE, lambda: {"status": "offline", "time": 0})
    device_online = now - float(dev.get("time") or 0) <= 10
    with _pending_lock:
        # Present if seen in 30 s; its state only counts while fresh (companions report every 2 s).
        companions = [{"device": d, "state": v["state"] if now - v["t"] <= 6 else "idle", "last_seen": v["t"]}
                      for d, v in _companion_state.items() if now - v["t"] <= 30]
    screening = [{"device": d, "last_seen": t} for d, t in list(_screen_devices.items()) if now - t <= 15]
    with _activity_lock:
        hub_state = _activity["state"] if now < _activity["until"] else "idle"
    comp_states = {c["state"] for c in companions}
    if device_online and dev.get("status") not in (None, "", "unknown", "offline"):
        overall = dev["status"]
    elif comp_states & {"speaking", "parent"} or hub_state == "speaking":
        overall = "speaking"
    elif "listening" in comp_states:
        overall = "listening"
    elif "thinking" in comp_states or hub_state == "thinking":
        overall = "thinking"
    elif screening:
        overall = "screening"
    else:
        overall = "idle"
    return {
        "status": overall, "hub": "online", "time": now,
        "device": {"online": device_online, "status": dev.get("status") if device_online else "offline",
                   "last_seen": dev.get("time") or None},
        "companions": companions, "screening": screening, "hub_activity": hub_state,
    }


# ════════════════════════════════════════════════════════════════
#  Transcript, pending audio queue, telemetry
# ════════════════════════════════════════════════════════════════
_transcript_lock = threading.Lock()


def log_transcript(role, text, language=None, kind="chat", url=None, source="companion"):
    entry = {"id": uuid.uuid4().hex[:12], "role": role, "text": (text or "")[:1200], "ts": int(time.time() * 1000),
             "language": language, "kind": kind, "source": source}
    if url:
        entry["url"] = url
    with _transcript_lock:
        items = _read_json(TRANSCRIPT_FILE, list)
        if not isinstance(items, list):
            items = []
        items.append(entry)
        _write_json(TRANSCRIPT_FILE, items[-TRANSCRIPT_LIMIT:])
    return entry


def update_transcript(updates):
    """updates: {entry_id: {field: value}} — merged into stored transcript entries."""
    with _transcript_lock:
        items = _read_json(TRANSCRIPT_FILE, list)
        if not isinstance(items, list):
            return
        changed = False
        for e in items:
            if e.get("id") in updates:
                e.update(updates[e["id"]])
                changed = True
        if changed:
            _write_json(TRANSCRIPT_FILE, items)


def last_child_message(within_sec=30):
    items = _read_json(TRANSCRIPT_FILE, list)
    if not isinstance(items, list):
        return None
    cutoff = time.time() * 1000 - within_sec * 1000
    for e in reversed(items[-20:]):
        if e.get("role") == "child" and e.get("ts", 0) >= cutoff and e.get("kind") == "chat":
            return e
    return None


_pending_lock = threading.Lock()
_pending = deque(maxlen=50)
_pending_seq = 0


def enqueue_audio(kind, text, audio_url, language=None, label=None, duration_sec=None, error=None):
    """Queue something for the Child Companion to play (claimed exactly once)."""
    global _pending_seq
    with _pending_lock:
        _pending_seq += 1
        item = {"id": _pending_seq, "kind": kind, "text": text, "audio_url": audio_url, "language": language,
                "label": label, "duration_sec": duration_sec, "error": error,
                "created_at": time.time(), "claimed_at": None, "claimed_by": None}
        _pending.append(item)
        return dict(item)


_telemetry_lock = threading.Lock()
_telemetry = deque(maxlen=3000)


# ════════════════════════════════════════════════════════════════
#  Audio ML — cry_model.pkl (SVC) on librosa features
# ════════════════════════════════════════════════════════════════
CRY_SR = 16000


def _clamp01(v):
    try:
        return max(0.0, min(1.0, float(v)))
    except (TypeError, ValueError):
        return 0.0


class CryEngine:
    """Loads cry_model.pkl + cry_scaler.pkl and extracts exactly the 258 features of
    cry_detector_ml.py (MFCC-40 + deltas, chroma, centroid, bandwidth, roll-off, ZCR, RMS, mel-32)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.clf = self.scaler = self.librosa = None
        self.error, self.error_at, self.info = None, 0.0, {}

    @property
    def ready(self):
        return self.clf is not None

    def ensure(self):
        if self.clf is not None:
            return True
        if self.error and time.time() - self.error_at < 60:
            return False
        with self.lock:
            if self.clf is not None:
                return True
            try:
                if np is None:
                    raise ImportError(name="numpy")
                import warnings
                import joblib
                import librosa
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")   # pickled with scikit-learn 1.8.0
                    clf = joblib.load(_p("cry_model.pkl"))
                    scaler = joblib.load(_p("cry_scaler.pkl"))
                self.librosa = librosa
                self.info = {"model": type(clf).__name__, "kernel": getattr(clf, "kernel", None),
                             "features": int(getattr(scaler, "n_features_in_", 0) or 0),
                             "files": "cry_model.pkl + cry_scaler.pkl"}
                self.clf, self.scaler, self.error = clf, scaler, None
                self._features(np.zeros(CRY_SR * 3, dtype=np.float32))   # JIT-compile librosa now, not mid-request
                print(f"🍼 Cry model loaded — {self.info['model']} ({self.info['kernel']}), {self.info['features']} features")
            except FileNotFoundError:
                self.error = "cry_model.pkl / cry_scaler.pkl not found next to mitra_server.py"
            except ImportError as e:
                self.error = f"Python package missing: {e.name or e} (pip install librosa scikit-learn joblib)"
            except Exception as e:
                self.error = f"Could not load the cry model: {e}"
            if self.error:
                self.error_at = time.time()
                print(f"⚠️ {self.error}")
        return self.clf is not None

    def _features(self, y):
        lb, sr, target = self.librosa, CRY_SR, CRY_SR * 3
        y = y.astype(np.float32)
        if len(y) < target:
            y = np.pad(y, (0, target - len(y)))
        else:
            start = (len(y) - target) // 2
            y = y[start:start + target]
        f = []
        mfcc = lb.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        f += list(np.mean(mfcc, axis=1)) + list(np.std(mfcc, axis=1))
        d = lb.feature.delta(mfcc)
        f += list(np.mean(d, axis=1)) + list(np.std(d, axis=1))
        ch = lb.feature.chroma_stft(y=y, sr=sr)
        f += list(np.mean(ch, axis=1)) + list(np.std(ch, axis=1))
        for fn in (lb.feature.spectral_centroid, lb.feature.spectral_bandwidth, lb.feature.spectral_rolloff):
            v = fn(y=y, sr=sr)
            f += [float(np.mean(v)), float(np.std(v))]
        z = lb.feature.zero_crossing_rate(y)
        f += [float(np.mean(z)), float(np.std(z))]
        r = lb.feature.rms(y=y)
        f += [float(np.mean(r)), float(np.std(r))]
        mel = lb.power_to_db(lb.feature.melspectrogram(y=y, sr=sr, n_mels=32), ref=np.max)
        f += list(np.mean(mel, axis=1)) + list(np.std(mel, axis=1))
        return np.array(f, dtype=np.float32)

    def predict(self, y):
        """P(crying) for 16 kHz mono float audio → (probability, error)."""
        if not self.ensure():
            return None, self.error
        try:
            proba = self.clf.predict_proba(self.scaler.transform([self._features(y)]))[0]
            classes = list(getattr(self.clf, "classes_", [0, 1]))
            return float(proba[classes.index(1) if 1 in classes else -1]), None
        except Exception as e:
            return None, f"Cry model error: {e}"


cry_engine = CryEngine()


def decode_wav_16k(data):
    """PCM WAV bytes → (float32 mono samples at 16 kHz, error)."""
    if np is None:
        return None, "numpy is not installed"
    try:
        with wave.open(io.BytesIO(data), "rb") as w:
            ch, sw, rate, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
            raw = w.readframes(n)
    except Exception:
        return None, "Audio must be a PCM WAV file"
    if sw == 2:
        y = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sw == 4:
        y = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    elif sw == 1:
        y = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        return None, f"Unsupported WAV sample width ({sw * 8}-bit)"
    if ch > 1:
        y = y[: len(y) // ch * ch].reshape(-1, ch).mean(axis=1)
    if rate != CRY_SR and len(y):
        if cry_engine.librosa is not None:
            y = cry_engine.librosa.resample(y, orig_sr=rate, target_sr=CRY_SR)
        else:
            grid = np.linspace(0, len(y) - 1, int(len(y) * CRY_SR / rate))
            y = np.interp(grid, np.arange(len(y)), y)
    return y.astype(np.float32), None


def _to_db(v):
    return max(-100.0, 20 * math.log10(v)) if v > 1e-10 else -100.0


def audio_levels(y):
    """Measured loudness of a clip in dBFS: overall RMS, 90th-percentile 100 ms frame, peak."""
    if y is None or not len(y):
        return None
    frame = CRY_SR // 10
    n = len(y) // frame
    frames = np.sqrt(np.mean(np.square(y[: n * frame].reshape(n, frame)), axis=1)) if n else np.array([0.0])
    return {"dbfs": round(_to_db(float(np.sqrt(np.mean(np.square(y))))), 1),
            "p90_dbfs": round(_to_db(float(np.percentile(frames, 90))), 1),
            "peak_dbfs": round(_to_db(float(np.max(np.abs(y)))), 1),
            "seconds": round(len(y) / CRY_SR, 2)}


def pcm16_wav(y):
    """float32 mono 16 kHz → 16-bit PCM WAV bytes."""
    pcm = (np.clip(y, -1, 1) * 32767).astype("<i2").tobytes()
    b = io.BytesIO()
    with wave.open(b, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(CRY_SR)
        w.writeframes(pcm)
    return b.getvalue()


# ════════════════════════════════════════════════════════════════
#  Vision — MediaPipe Tasks (fallback: OpenCV Haar cascades)
# ════════════════════════════════════════════════════════════════
MP_MODEL_URLS = {
    "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
    "pose_landmarker_lite.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
}


def _ensure_model_file(name):
    path = os.path.join(MODELS_DIR, name)
    if os.path.exists(path) and os.path.getsize(path) > 10000:
        return path
    r = requests.get(MP_MODEL_URLS[name], timeout=60)
    r.raise_for_status()
    _write_bytes_atomic(path, r.content)
    print(f"⬇️  Downloaded MediaPipe model {name} ({len(r.content) // 1024} KB)")
    return path


class VisionEngine:
    """Local camera analysis: MediaPipe Face Landmarker (52 expression blendshapes) + Pose
    Landmarker when available, otherwise OpenCV Haar cascades for face / eyes / smile."""

    def __init__(self):
        self.lock = threading.Lock()
        self.cv2 = self.mp = self.face = self.pose = None
        self.haar = {}
        self.backend, self.error, self.error_at, self.notes = None, None, 0.0, []

    @property
    def ready(self):
        return self.backend is not None

    def describe(self):
        if self.backend == "mediapipe":
            return "MediaPipe Face Landmarker (blendshapes)" + (" + Pose Landmarker lite" if self.pose else "")
        if self.backend == "opencv-haar":
            return "OpenCV Haar cascades (face, eyes, smile)"
        return None

    def ensure(self):
        if self.backend:
            return True
        if self.error and time.time() - self.error_at < 60:
            return False
        with self.lock:
            if self.backend:
                return True
            self.notes = []
            if np is None:
                self.error, self.error_at = "numpy is not installed", time.time()
                return False
            try:
                import cv2
                self.cv2 = cv2
                for key, fname in (("face", "haarcascade_frontalface_default.xml"), ("eye", "haarcascade_eye.xml"),
                                   ("smile", "haarcascade_smile.xml")):
                    c = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, fname))
                    if not c.empty():
                        self.haar[key] = c
            except ImportError:
                self.error, self.error_at = "opencv-python is not installed (pip install opencv-python)", time.time()
                print(f"⚠️ {self.error}")
                return False
            try:
                import mediapipe as mp
                from mediapipe.tasks.python import BaseOptions
                from mediapipe.tasks.python import vision as mpv
                self.face = mpv.FaceLandmarker.create_from_options(mpv.FaceLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=_ensure_model_file("face_landmarker.task")),
                    output_face_blendshapes=True, num_faces=4, running_mode=mpv.RunningMode.IMAGE))
                try:
                    self.pose = mpv.PoseLandmarker.create_from_options(mpv.PoseLandmarkerOptions(
                        base_options=BaseOptions(model_asset_path=_ensure_model_file("pose_landmarker_lite.task")),
                        num_poses=1, running_mode=mpv.RunningMode.IMAGE))
                except Exception as e:
                    self.notes.append(f"Pose landmarker unavailable: {str(e)[:120]}")
                self.mp, self.backend = mp, "mediapipe"
            except Exception as e:
                self.notes.append(f"MediaPipe unavailable ({type(e).__name__}: {str(e)[:120]}) — using OpenCV Haar cascades")
                if "face" in self.haar:
                    self.backend = "opencv-haar"
                else:
                    self.error, self.error_at = "Neither MediaPipe nor OpenCV face cascades could be loaded", time.time()
            if self.backend:
                print(f"👁️  Vision engine: {self.describe()}")
            for n in self.notes:
                print(f"   ℹ️ {n}")
        return self.backend is not None

    def analyze(self, jpeg):
        """One frame → faces (normalised boxes), body/posture, expression signals, tiny gray image."""
        if not self.ensure():
            return None, self.error
        cv2 = self.cv2
        img = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return None, "Frame is not a decodable JPEG/PNG image"
        h, w = img.shape[:2]
        if w > 960:
            img = cv2.resize(img, (960, int(h * 960 / w)))
            h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        out = {"width": w, "height": h, "faces": [], "body_present": False, "posture": None, "expression": None,
               "backend": self.backend, "small": cv2.GaussianBlur(cv2.resize(gray, (96, 72)), (5, 5), 0),
               "brightness": round(float(np.mean(gray)), 1)}
        with self.lock:
            if self.backend == "mediapipe":
                self._mediapipe(img, out)
            else:
                self._haar(gray, out)
        out["face_count"] = len(out["faces"])
        return out, None

    def _mediapipe(self, img, out):
        rgb = np.ascontiguousarray(self.cv2.cvtColor(img, self.cv2.COLOR_BGR2RGB))
        mpimg = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        res = self.face.detect(mpimg)
        faces = []
        for i, lms in enumerate(res.face_landmarks or []):
            xs, ys = [p.x for p in lms], [p.y for p in lms]
            x0, y0, x1, y1 = max(0.0, min(xs)), max(0.0, min(ys)), min(1.0, max(xs)), min(1.0, max(ys))
            face = {"x": round(x0, 4), "y": round(y0, 4), "w": round(x1 - x0, 4), "h": round(y1 - y0, 4)}
            if res.face_blendshapes and i < len(res.face_blendshapes):
                face["_blend"] = {c.category_name: c.score for c in res.face_blendshapes[i]}
            faces.append(face)
        faces.sort(key=lambda f: f["w"] * f["h"], reverse=True)
        if faces and faces[0].get("_blend"):
            b = faces[0]["_blend"]
            avg = lambda *names: sum(b.get(n, 0.0) for n in names) / len(names)
            e = {"smile": avg("mouthSmileLeft", "mouthSmileRight"), "eyes_closed": avg("eyeBlinkLeft", "eyeBlinkRight"),
                 "frown": avg("mouthFrownLeft", "mouthFrownRight"), "jaw_open": b.get("jawOpen", 0.0),
                 "squint": avg("eyeSquintLeft", "eyeSquintRight"), "brow_inner_up": b.get("browInnerUp", 0.0),
                 "brow_down": avg("browDownLeft", "browDownRight"), "mouth_stretch": avg("mouthStretchLeft", "mouthStretchRight")}
            # Crying face ≈ open/stretched mouth + squeezed eyes + raised inner brows — and no smile.
            raw = (0.35 * e["jaw_open"] + 0.25 * e["squint"]
                   + 0.2 * max(e["frown"], e["mouth_stretch"], e["brow_down"]) + 0.2 * e["brow_inner_up"])
            e["distress_face"] = raw * (1 - e["smile"])
            out["expression"] = {k: round(_clamp01(v), 3) for k, v in e.items()}
        for f in faces:
            f.pop("_blend", None)
        out["faces"] = faces
        if self.pose is not None:
            pres = self.pose.detect(mpimg)
            if pres.pose_landmarks:
                lm = pres.pose_landmarks[0]
                vis = lambda i: getattr(lm[i], "visibility", 1.0) or 0.0
                out["body_present"] = True
                if min(vis(11), vis(12), vis(23), vis(24)) > 0.5:
                    sx, sy = (lm[11].x + lm[12].x) / 2, (lm[11].y + lm[12].y) / 2
                    hx, hy = (lm[23].x + lm[24].x) / 2, (lm[23].y + lm[24].y) / 2
                    out["posture"] = "lying" if abs(sx - hx) > abs(sy - hy) else "upright"

    def _haar(self, gray, out):
        h, w = gray.shape[:2]
        side = max(40, w // 16)
        found = self.haar["face"].detectMultiScale(gray, 1.1, 5, minSize=(side, side))
        faces = sorted(([int(a) for a in f] for f in found), key=lambda f: f[2] * f[3], reverse=True)
        out["faces"] = [{"x": round(x / w, 4), "y": round(y / h, 4), "w": round(fw / w, 4), "h": round(fh / h, 4)}
                        for x, y, fw, fh in faces]
        if faces:
            x, y, fw, fh = faces[0]
            roi = gray[y:y + fh, x:x + fw]
            eyes = len(self.haar["eye"].detectMultiScale(roi[: fh // 2], 1.1, 6)) if "eye" in self.haar else None
            smiles = len(self.haar["smile"].detectMultiScale(roi[fh // 2:], 1.7, 20)) if "smile" in self.haar else None
            out["expression"] = {"smile": 1.0 if smiles else 0.0,
                                 "eyes_closed": None if eyes is None else (1.0 if eyes == 0 else 0.0)}


vision_engine = VisionEngine()


# ════════════════════════════════════════════════════════════════
#  Screening: stress index, behaviour label, alerts
# ════════════════════════════════════════════════════════════════
# Stress index = 100 × Σ weight·component. A component that was not measured (no audio, no
# face) contributes 0 — the index never extrapolates from partial data; `coverage` says how much
# of it was actually measured.
STRESS_WEIGHTS = {"cry": 0.40, "loudness": 0.20, "motion": 0.20, "face": 0.10, "events": 0.10}
SCREEN_ALERT_COOLDOWN_SEC = 120
VISION_AUTO_MIN_GAP_SEC = 30   # one vision frame ≈ 1.3k of the free tier's 7k input tokens/min
VISION_MANUAL_MIN_GAP_SEC = 3
VISION_EMOTIONS = {"happy", "calm", "focused", "tired", "sad", "scared", "crying", "upset", "unclear", "no_child"}
VISION_ALERT_TYPE = {"crying": "cry", "scared": "scared", "sad": "sad", "upset": "high_stress"}
VISION_SYSTEM = ("You are the camera assistant of Mitra, a child-companion device. A parent is checking on their "
                 "young child (about 3-8 years old). Describe only what is clearly visible, kindly and factually. "
                 "Never identify people, never guess medical conditions, never invent details you cannot see.")
_screen_lock = threading.Lock()
_screen_state = {}                     # device → rolling per-device state
_screen_history = deque(maxlen=1200)   # ≈1 h at one analysis every 3 s
_screen_latest = {}                    # device → latest public result
_screen_alert_at = {}                  # alert type → last time a screening alert of that type fired
_refs_lock = threading.Lock()


def stress_level(v):
    if v is None:
        return None
    return "calm" if v <= 30 else "elevated" if v <= 65 else "high"


def loudness_parts(level_dbfs, baseline_dbfs):
    """(cry gate 0..1, loudness component 0..1, dB over baseline or None)."""
    if level_dbfs is None:
        return None, None, None
    if baseline_dbfs is not None:
        over = level_dbfs - baseline_dbfs
        return _clamp01((over - 10) / 20), _clamp01((over - 6) / 24), round(over, 1)
    # Uncalibrated room: absolute level (−45 dBFS ≈ normal talk → 0, −25 dBFS ≈ loud cry → 1).
    return _clamp01((level_dbfs + 45) / 20), _clamp01((level_dbfs + 50) / 30), None


def behaviour_label(present, motion, expr, eyes_closed_streak, posture, cry_c, audio_stress):
    if not present:
        return "away"
    expr = expr or {}
    if (cry_c or 0) >= 0.5 or (expr.get("distress_face") or 0) >= 0.55 or (audio_stress or 0) >= 0.7:
        return "distressed"
    if motion is not None and motion >= 0.6:
        return "restless"
    if (eyes_closed_streak >= 2 or posture == "lying") and (motion is None or motion < 0.2):
        return "sleepy"
    if motion is not None and motion >= 0.3:
        return "active"
    return "calm"


def vision_observe(jpeg, context):
    """Groq vision pass → {observation, emotion, distress, confidence, model, latency_ms, error}."""
    prompt = ("Look at this frame from the child's room camera.\n"
              f"Other sensors (they can be wrong): {context or 'none'}\n"
              "Reply with ONLY one JSON object, no markdown:\n"
              '{"observation": "one or two short sentences: what the child is doing and how they look", '
              '"emotion": "happy|calm|focused|tired|sad|scared|crying|upset|unclear|no_child", '
              '"distress": true or false (true only if the child clearly looks distressed, crying, scared or hurt), '
              '"confidence": 0.0 to 1.0}')
    t0 = time.time()
    raw, model, err = vision_complete(prompt, jpeg, system=VISION_SYSTEM, max_tokens=400)
    out = {"model": model, "latency_ms": int((time.time() - t0) * 1000), "at": time.time(), "error": err,
           "observation": None, "emotion": None, "distress": False, "confidence": None}
    if raw:
        m = re.search(r"\{.*\}", raw, re.S)
        parsed = None
        if m:
            try:
                parsed = json.loads(m.group(0))
            except ValueError:
                parsed = None
        if isinstance(parsed, dict):
            out["observation"] = clean_for_speech(str(parsed.get("observation") or ""))[:400] or None
            emo = str(parsed.get("emotion") or "").strip().lower()
            out["emotion"] = emo if emo in VISION_EMOTIONS else "unclear"
            out["distress"] = parsed.get("distress") is True or str(parsed.get("distress")).lower() == "true"
            out["confidence"] = _float(parsed.get("confidence"), None, 0, 1)
        else:
            out["observation"], out["emotion"] = clean_for_speech(raw)[:400], "unclear"
    return out


def _sensor_context(cry_prob, over_db, motion, expr, child_text):
    bits = []
    if cry_prob is not None:
        bits.append(f"cry-sound model P(crying)={cry_prob:.2f} (it also scores loud speech high)")
    if over_db is not None:
        bits.append(f"sound {over_db:+.0f} dB over the quiet-room baseline")
    if motion is not None:
        bits.append(f"movement {motion:.2f} on a 0-1 scale")
    if expr:
        bits.append("face signals: " + ", ".join(f"{k} {v:.2f}" for k, v in expr.items()
                                                 if v is not None and k in ("smile", "eyes_closed", "distress_face")))
    if child_text:
        bits.append(f"the child just said: \"{child_text[:160]}\"")
    return "; ".join(bits) or "none"


def _screen_cooldown_ok(kind, now):
    with _screen_lock:
        if now - _screen_alert_at.get(kind, 0) < SCREEN_ALERT_COOLDOWN_SEC:
            return False
        _screen_alert_at[kind] = now
        return True


def load_references():
    items = _read_json(REFS_FILE, list)
    return items if isinstance(items, list) else []


def save_reference(frame, result):
    ref_id = int(time.time() * 1000)
    fname = f"ref_{ref_id}.jpg"
    _write_bytes_atomic(os.path.join(CAPTURES_DIR, fname), frame)
    vision = result.get("vision") or {}
    entry = {"id": ref_id, "url": f"/captures/{fname}", "at": _now_iso(), "label": result.get("pose_or_expression"),
             "faces": result.get("face_count"), "stress_index": result.get("stress_index"),
             "observation": vision.get("observation"), "emotion": vision.get("emotion"), "model": vision.get("model"),
             "vision_error": vision.get("error"), "device": result.get("device")}
    with _refs_lock:
        items = load_references() + [entry]
        for old in items[:-24]:
            path = safe_join(CAPTURES_DIR, old.get("url", "").rsplit("/", 1)[-1])
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass
        _write_json(REFS_FILE, items[-24:])
    return entry


def screening_analyze(device, frame=None, wav=None, client_motion=None, want_vision=False,
                      context_text=None, keep_reference=False):
    """One screening cycle on real inputs. Returns the public result (and creates alerts)."""
    t_start = now = time.time()
    s = load_settings()
    with _screen_lock:
        st = _screen_state.get(device)
        if not st or now - st["last_seen"] > 30:   # new session after a gap
            st = {"started_at": now, "last_seen": now, "last_present_at": None, "prev_small": None, "prev_t": None,
                  "eyes_closed_streak": 0, "cry_streak": 0, "stress_streak": 0, "absent_alerted": False,
                  "prev_audio": None, "last_vision_at": 0.0, "last_vision": None}
            _screen_state[device] = st
        st["last_seen"] = now
        _screen_devices[device] = now
    timings, errors = {}, {}

    # ── local vision ──
    va = None
    if frame:
        t0 = time.time()
        va, err = vision_engine.analyze(frame)
        timings["vision_local_ms"] = int((time.time() - t0) * 1000)
        if err:
            errors["vision"] = err
    present = bool(va and (va["face_count"] or va["body_present"])) if va else None
    motion, motion_source = None, None
    if client_motion is not None:
        motion, motion_source = _clamp01(client_motion), "browser frame difference (5 fps)"
    if va is not None:
        if st["prev_small"] is not None and st["prev_small"].shape == va["small"].shape:
            diff = vision_engine.cv2.absdiff(va["small"], st["prev_small"])
            server_motion = _clamp01(float(np.mean(diff > 25)) / 0.15)
            if motion is None:
                motion, motion_source = server_motion, "hub frame difference between analyses"
        st["prev_small"] = va["small"]
    expr = va.get("expression") if va else None
    if expr and (expr.get("eyes_closed") or 0) >= 0.55:
        st["eyes_closed_streak"] += 1
    elif va is not None:
        st["eyes_closed_streak"] = 0
    if present:
        st["last_present_at"], st["absent_alerted"] = now, False
    absence_sec = 0 if present or present is None else round(now - (st["last_present_at"] or st["started_at"]), 1)

    # ── audio ML ──
    y = lv = cry_prob = None
    if wav:
        t0 = time.time()
        y, err = decode_wav_16k(wav)
        if err:
            errors["audio"] = err
        else:
            lv = audio_levels(y)
            if len(y) >= CRY_SR // 2:
                cry_prob, err = cry_engine.predict(y)
                if err:
                    errors["cry_model"] = err
            else:
                errors["cry_model"] = "Audio clip shorter than 0.5 s"
        timings["audio_ml_ms"] = int((time.time() - t0) * 1000)
    baseline = s.get("room_baseline_dbfs")
    level = lv["p90_dbfs"] if lv else None
    cry_gate, loud_c, over_db = loudness_parts(level, baseline)
    smile_damp = 1 - 0.6 * (expr.get("smile") or 0) if expr else 1.0   # laughing is loud and "cry-like"
    cry_c = cry_prob * cry_gate * smile_damp if cry_prob is not None and cry_gate is not None else None
    audio_stress = round(_clamp01(0.67 * cry_c + 0.33 * loud_c), 3) if cry_c is not None else (
        round(loud_c, 3) if loud_c is not None else None)

    # ── stress index ──
    events = min(1.0, recent_distress_count(600) / 3)
    parts = {"cry": cry_c, "loudness": loud_c, "motion": motion,
             "face": expr.get("distress_face") if expr else None, "events": events}
    stress = int(round(100 * sum(STRESS_WEIGHTS[k] * _clamp01(v) for k, v in parts.items() if v is not None)))
    coverage = round(sum(STRESS_WEIGHTS[k] for k, v in parts.items() if v is not None), 2)
    label = behaviour_label(present, motion, expr, st["eyes_closed_streak"], va.get("posture") if va else None,
                            cry_c, audio_stress) if va else None

    # ── streaks for alerts ──
    cry_hit = cry_prob is not None and cry_prob >= float(s.get("cry_threshold") or 0.5) and (cry_gate or 0) >= 0.5
    st["cry_streak"] = st["cry_streak"] + 1 if cry_hit else 0
    st["stress_streak"] = st["stress_streak"] + 1 if stress >= int(s.get("stress_threshold") or 66) else 0

    # ── Groq vision pass (manual, chat sync, or automatically on a spike) ──
    vision = None
    child_msg = context_text or ((last_child_message(20) or {}).get("text"))
    spike = cry_hit or st["stress_streak"] >= 1
    gap = now - st["last_vision_at"]
    if frame and ((want_vision and gap >= VISION_MANUAL_MIN_GAP_SEC) or (spike and gap >= VISION_AUTO_MIN_GAP_SEC)):
        st["last_vision_at"] = now
        vision = vision_observe(frame, _sensor_context(cry_prob, over_db, motion, expr, child_msg))
        vision["trigger"] = "manual" if want_vision else "stress spike"
        timings["vision_llm_ms"] = vision["latency_ms"]
        st["last_vision"] = vision

    # ── alerts (one per cycle; real evidence attached) ──
    alert = None
    s_trig = s.get("triggers", {})
    clip = None
    if y is not None and len(y):
        joined = np.concatenate([st["prev_audio"], y]) if st["prev_audio"] is not None else y
        clip = {"bytes": pcm16_wav(joined), "duration": len(joined) / CRY_SR}
    recent_chat = last_child_message(10)
    if st["cry_streak"] >= 2 and not recent_chat and _screen_cooldown_ok("cry", now):
        alert = create_alert("cry", f"Cry model P(crying)={cry_prob:.2f} for 2 windows in a row"
                             + (f", {over_db:+.0f} dB over the room baseline" if over_db is not None else "")
                             + " — Live Screening", round(clip["duration"], 1) if clip else 0,
                             {"source": "screening", "cry_probability": round(cry_prob, 3), "stress_index": stress},
                             capture=clip, frame=frame)
    elif vision and vision.get("distress") and (vision.get("confidence") or 0) >= 0.6 \
            and vision.get("emotion") in VISION_ALERT_TYPE and _screen_cooldown_ok("vision", now):
        kind = VISION_ALERT_TYPE[vision["emotion"]]
        alert = create_alert(kind, f"Seen on camera: {vision['observation']}", 0,
                             {"source": "vision", "observation": vision["observation"], "vision_model": vision["model"],
                              "stress_index": stress}, capture=clip, frame=frame)
    elif st["stress_streak"] >= 2 and _screen_cooldown_ok("high_stress", now):
        alert = create_alert("high_stress", f"Stress index {stress}/100 for 2 analyses in a row — Live Screening", 0,
                             {"source": "screening", "stress_index": stress, "cry_probability": cry_prob,
                              "observation": (st["last_vision"] or {}).get("observation") if st["last_vision"]
                              and now - st["last_vision"]["at"] < 60 else None}, capture=clip, frame=frame)
    elif s_trig.get("absent") and absence_sec >= int(s.get("absent_after_sec") or 60) and not st["absent_alerted"]:
        st["absent_alerted"] = True
        alert = create_alert("absent", f"No child detected on camera for {int(absence_sec)} s — Live Screening", 0,
                             {"source": "screening"}, frame=frame)
    if y is not None:
        st["prev_audio"] = y[-CRY_SR * 3:]

    result = {
        "ok": True, "device": device, "t": now,
        "child_present": present, "face_count": va["face_count"] if va else None,
        "faces": va["faces"] if va else [], "body_present": va["body_present"] if va else None,
        "posture": va["posture"] if va else None, "brightness": va["brightness"] if va else None,
        "motion_agitation_score": round(motion, 3) if motion is not None else None, "motion_source": motion_source,
        "pose_or_expression": label, "expression": expr, "absence_sec": absence_sec,
        "cry_probability": round(cry_prob, 3) if cry_prob is not None else None,
        "audio_dbfs": level, "audio_db": round(max(0.0, level + float(s.get("monitor_calibration_db") or 90)), 1) if level is not None else None,
        "db_over_baseline": over_db, "baseline_dbfs": baseline, "audio_stress_score": audio_stress,
        "audio_seconds": lv["seconds"] if lv else None,
        "stress_index": stress, "stress_level": stress_level(stress), "coverage": coverage,
        "components": {k: (round(_clamp01(v), 3) if v is not None else None) for k, v in parts.items()},
        "cry_streak": st["cry_streak"], "stress_streak": st["stress_streak"],
        "vision": vision, "last_vision": st["last_vision"], "alert": alert,
        "engines": {"face": vision_engine.describe(), "cry": (cry_engine.info.get("model") if cry_engine.ready else None),
                    "vision_llm": _models.get("vision")},
        "errors": errors, "timings_ms": timings,
    }
    result["timings_ms"]["total_ms"] = int((time.time() - t_start) * 1000)
    if keep_reference and frame:
        result["reference"] = save_reference(frame, result)
    public = {k: v for k, v in result.items() if k not in ("alert",)}
    with _screen_lock:
        _screen_latest[device] = public
        _screen_history.append({"t": now, "device": device, "stress": stress, "cry": result["cry_probability"],
                                "db": result["audio_db"], "motion": result["motion_agitation_score"],
                                "present": present, "label": label})
    return result


# ════════════════════════════════════════════════════════════════
#  Parent Language Digest + Whisper voice input
# ════════════════════════════════════════════════════════════════
def _digest_worker(lang, child_id, child_text, mitra_id, mitra_text, alert_id=None):
    name = LANGUAGES.get(lang, {}).get("name", lang)
    messages = [
        {"role": "system", "content": "You translate a young child's conversation into natural, simple English for "
                                      "their parent. Reply with ONLY a JSON object and nothing else."},
        {"role": "user", "content": f"Language: {name}\nchild: {json.dumps(child_text or '', ensure_ascii=False)}\n"
                                    f"mitra: {json.dumps(mitra_text or '', ensure_ascii=False)}\n"
                                    'Return {"child": "<English translation>", "mitra": "<English translation>"}'},
    ]
    text, err = llm_complete(messages, max_tokens=900, temperature=0.2, model=digest_model())
    if not text:
        print(f"⚠️ Language digest failed: {err}")
        return
    m = re.search(r"\{.*\}", text, re.S)
    try:
        parsed = json.loads(m.group(0)) if m else {}
    except ValueError:
        parsed = {}
    updates = {}
    if child_id and parsed.get("child"):
        updates[child_id] = {"en": str(parsed["child"])[:600]}
    if mitra_id and parsed.get("mitra"):
        updates[mitra_id] = {"en": str(parsed["mitra"])[:900]}
    if updates:
        update_transcript(updates)
    if alert_id and parsed.get("child"):
        update_alert(alert_id, {"note_en": f"Child said (English): \"{str(parsed['child'])[:200]}\""})


def language_digest(lang, child_entry=None, mitra_entry=None, alert_id=None):
    """English translation for the parent, in the background, for non-English exchanges."""
    if lang == "en" or lang not in LANGUAGES or not groq_configured():
        return
    threading.Thread(target=_digest_worker, daemon=True, name="digest", args=(
        lang, child_entry and child_entry["id"], child_entry and child_entry["text"],
        mitra_entry and mitra_entry["id"], mitra_entry and mitra_entry["text"], alert_id)).start()


WHISPER_LANG_NAMES = {"english": "en", "hindi": "hi", "telugu": "te", "tamil": "ta", "kannada": "kn",
                      "spanish": "es", "french": "fr"}


def transcribe_audio(data, filename, language):
    """Groq Whisper → (dict, error). dict: text, language, duration, model, no_speech."""
    client = _get_groq()
    if client is None:
        return None, "GROQ_API_KEY is not set in .env"
    model = refresh_models()["whisper"]
    if not model:
        return None, "No Whisper model is available to this Groq API key"
    # The prompt only biases spelling of the companion's name ("Mitra", not "Maetra").
    kwargs = dict(model=model, file=(filename, data), response_format="verbose_json", temperature=0, prompt="Mitra")
    if language in LANGUAGES:
        kwargs["language"] = language
    t0 = time.time()
    try:
        resp = client.with_options(max_retries=1, timeout=30).audio.transcriptions.create(**kwargs)
    except Exception as e:
        return None, _rate_limit_note(e) or f"Whisper error: {_scrub(e)}"
    d = resp.model_dump() if hasattr(resp, "model_dump") else dict(resp)
    segments = d.get("segments") or []
    no_speech = bool(segments) and all((sg.get("no_speech_prob") or 0) > 0.6 for sg in segments)
    det = str(d.get("language") or "").lower()
    return {"text": "" if no_speech else (d.get("text") or "").strip(), "model": model,
            "language": language if language in LANGUAGES else WHISPER_LANG_NAMES.get(det, LANGUAGES.get(det) and det),
            "detected": det or None, "duration": d.get("duration"), "no_speech": no_speech,
            "latency_ms": int((time.time() - t0) * 1000)}, None


# ════════════════════════════════════════════════════════════════
#  Flask app
# ════════════════════════════════════════════════════════════════
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_AUDIO_BYTES + 1024 * 1024
# Open CORS for the legacy/companion API (file:// dashboards, LAN devices), but NOT for
# settings/notification routes — those carry phone numbers and can trigger paid calls.
CORS(app, resources={r"^/(?!api/(settings|notify|notifications|telegram|ntfy|qr)).*": {"origins": "*"}})


def _read_image_upload(field="frame"):
    """JPEG/PNG bytes from a multipart field → (bytes, error); (None, None) when absent."""
    f = request.files.get(field)
    if not f:
        return None, None
    data = f.read(MAX_IMAGE_BYTES + 1)
    if not data:
        return None, None
    if len(data) > MAX_IMAGE_BYTES:
        return None, "Image is larger than 4 MB"
    if not (data[:3] == b"\xff\xd8\xff" or data[:8] == b"\x89PNG\r\n\x1a\n"):
        return None, "Frame must be a JPEG or PNG image"
    return data, None


def _decode_data_url_image(value):
    """'data:image/jpeg;base64,…' → bytes (or None)."""
    if not isinstance(value, str) or not value.startswith("data:image/"):
        return None
    try:
        data = base64.b64decode(value.split(",", 1)[1], validate=False)
    except (ValueError, IndexError):
        return None
    if len(data) > MAX_IMAGE_BYTES or not (data[:3] == b"\xff\xd8\xff" or data[:8] == b"\x89PNG\r\n\x1a\n"):
        return None
    return data


def _parent_clip_path(phrase_id):
    for directory in (PARENT_VOICE_DIR, LEGACY_PARENT_VOICE_DIR):
        path = safe_join(directory, f"parent_voice_{phrase_id}.wav")
        if path and os.path.exists(path):
            return path
    return None


def _send_from(directory, filename, mimetype):
    path = safe_join(directory, filename)
    if path and os.path.isfile(path):
        return send_file(path, mimetype=mimetype)
    return None


# ── Core ────────────────────────────────────────────────────────
@app.route("/")
def index():
    dashboard_path = _p("parent_dashboard.html")
    if os.path.exists(dashboard_path):
        resp = send_file(dashboard_path)
        # Always revalidate (cheap 304) so an updated dashboard is never served stale from cache.
        resp.headers["Cache-Control"] = "no-cache"
        return resp
    return "<h1>Mitra Parent Server Running</h1><p>parent_dashboard.html not found</p>"


@app.route("/favicon.ico")
def favicon():
    return "", 204   # the page ships an inline SVG icon; answer stray requests without a 404


@app.route("/ping")
def ping():
    return jsonify({"status": "ok", "time": time.time()})


@app.route("/alerts", methods=["GET"])
def get_alerts():
    data = load_data()
    by_alert = {}
    for n in load_notifications():
        if n.get("alert_id") is not None:
            by_alert.setdefault(n["alert_id"], []).append(
                {k: n.get(k) for k in ("channel", "status", "detail", "target", "at")})
    for a in data["alerts"]:
        a["notifications"] = by_alert.get(a.get("id"), [])
    return jsonify(data)


@app.route("/alert", methods=["POST"])
def post_alert():
    """Body: JSON {type, duration_sec, note} — or multipart with the same fields plus an
    `audio` WAV capture and optional source / peak_db / threshold_db / baseline_db."""
    is_multipart = (request.content_type or "").startswith("multipart/")
    fields = request.form if is_multipart else (request.get_json(silent=True) or {})
    alert_type = str(fields.get("type") or "cry").strip().lower()
    if not ALERT_TYPE_RE.match(alert_type):
        alert_type = "cry"
    extra = {}
    for key in ("peak_db", "threshold_db", "baseline_db"):
        v = _float(fields.get(key), None, 0, 200)
        if v is not None:
            extra[key] = round(v, 1)
    source = str(fields.get("source") or "").strip().lower()
    if source and ALERT_TYPE_RE.match(source):
        extra["source"] = source
    capture = frame = None
    if is_multipart and request.files.get("audio"):
        data, err = _read_upload()
        if err:
            return jsonify({"ok": False, "error": err}), 400
        info = _wav_info(data)
        if not info:
            return jsonify({"ok": False, "error": "Capture must be a PCM WAV file"}), 400
        capture = {"bytes": data, "duration": info["duration"]}
    if is_multipart:
        frame, err = _read_image_upload("frame")
        if err:
            return jsonify({"ok": False, "error": err}), 400
    alert = create_alert(alert_type, str(fields.get("note") or "")[:300],
                         _float(fields.get("duration_sec"), 0.0, 0, 3600), extra, capture, frame=frame)
    return jsonify({"ok": True, "alert": alert})


@app.route("/alert/<int:alert_id>/read", methods=["POST"])
def mark_read(alert_id):
    with _alerts_lock:
        data = load_data()
        for a in data["alerts"]:
            if a["id"] == alert_id:
                a["read"] = True
        save_data(data)
    return jsonify({"ok": True})


@app.route("/alerts/clear", methods=["POST"])
def clear_alerts():
    with _alerts_lock:
        data = load_data()
        for a in data["alerts"]:
            _delete_capture(a)
        data["alerts"] = []
        recalc_stats(data)
        save_data(data)
    return jsonify({"ok": True})


@app.route("/status", methods=["POST"])
def post_status():
    body = request.get_json(silent=True) or {}
    set_status(body.get("status", "unknown"))
    return jsonify({"ok": True})


@app.route("/status", methods=["GET"])
def get_status():
    """The hub is online whenever this answers. `status` reflects what is really active:
    the Mitra device heartbeat (if running), Child Companion screens, Live Screening, or idle."""
    return jsonify(hub_status())


# ── Media ───────────────────────────────────────────────────────
@app.route("/audio/tts/<filename>")
def get_tts_audio(filename):
    return _send_from(TTS_CACHE_DIR, filename, "audio/mpeg") or ("TTS audio not found", 404)


@app.route("/audio/parent-msg/<filename>")
def get_parent_message(filename):
    return _send_from(PARENT_MSG_DIR, filename, "audio/wav") or ("Message not found", 404)


@app.route("/audio/<filename>")
def get_audio(filename):
    return (_send_from(PARENT_VOICE_DIR, filename, "audio/wav")
            or _send_from(LEGACY_PARENT_VOICE_DIR, filename, "audio/wav")
            or ("Audio file not found", 404))


@app.route("/captures/<filename>")
def get_capture(filename):
    mimetype = {"wav": "audio/wav", "jpg": "image/jpeg"}.get(filename.rsplit(".", 1)[-1].lower())
    if not mimetype:
        return "Capture not found", 404
    return _send_from(CAPTURES_DIR, filename, mimetype) or ("Capture not found", 404)


# ── Companion ───────────────────────────────────────────────────
def _soft_tale(lang, queue):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT + "\n\n" + language_rule(lang) + "\n\n"
         "Right now, tell ONE gentle, calming bedtime story of 4 to 6 short sentences about cozy things "
         "(stars, moonlight, sleepy animals). Slow and soothing, with a sleepy happy ending and no question at the end."},
        {"role": "user", "content": "Please tell me a calming bedtime story."},
    ]
    mark_activity("thinking", 20)
    text, err = llm_complete(messages, max_tokens=500, temperature=0.8)
    if not text:
        return jsonify({"ok": False, "error": err}), 503
    audio_url, tts_err = synthesize(text, lang)
    mark_activity("speaking", 15)
    entry = log_transcript("mitra", text, lang, kind="story", url=audio_url,
                           source="parent_soft_tale" if queue else "companion")
    language_digest(lang, mitra_entry=entry)
    item = enqueue_audio("story", text, audio_url, lang, error=tts_err) if queue else None
    return jsonify({"ok": True, "emotion": "calm", "intent": "soft_tale", "language": lang, "reply": text,
                    "audio_url": audio_url, "tts_error": tts_err, "queued": item})


def _frame_context(frame, user_msg, device):
    """Camera frame sent with a chat message → (local analysis, Groq vision observation)."""
    local = None
    if frame:
        va, _ = vision_engine.analyze(frame)
        if va:
            local = {"face_count": va["face_count"], "child_present": bool(va["face_count"] or va["body_present"]),
                     "expression": va.get("expression"), "posture": va.get("posture")}
            _screen_devices[device or "companion"] = time.time()
    vision = vision_observe(frame, _sensor_context(None, None, None, (local or {}).get("expression"), user_msg)) if frame else None
    return local, vision


@app.route("/api/interact", methods=["POST"])
def interact_bot():
    """Child ↔ Mitra. Body: {message, action: chat|cry_simulate|soft_tale, language: en|hi|…|auto, queue,
    frame: optional 'data:image/jpeg;base64,…' camera frame (Live Screening "Sync with Child Chat")}."""
    body = request.get_json(silent=True) or {}
    user_msg = str(body.get("message") or "").strip()[:500]
    action = str(body.get("action") or "chat").strip()
    lang = resolve_language(body.get("language"), user_msg)
    frame = _decode_data_url_image(body.get("frame"))
    device = str(body.get("device") or "")[:40] or None
    mark_activity("thinking", 20)

    if action == "soft_tale":
        return _soft_tale(lang, bool(body.get("queue")))

    intent, matched = ("cry", "cry_simulate") if action == "cry_simulate" else classify_intent(user_msg)
    child_text = user_msg or ("(crying)" if intent == "cry" else "")
    child_entry = log_transcript("child", child_text, lang) if child_text else None

    # Multimodal: what Mitra sees right now (only when the companion attached a camera frame).
    frame_local, vision = (_frame_context(frame, user_msg, device) if frame and action == "chat" else (None, None))
    if vision and vision.get("observation") and child_entry:
        update_transcript({child_entry["id"]: {"seen": vision["observation"], "seen_emotion": vision.get("emotion")}})

    # 1. Distress emotions → alert + parent voice + localized comfort
    if intent in EMOTION_CLIPS:
        if action == "cry_simulate":
            note = "Cry test triggered from Child Companion"
        else:
            note = {"cry": "Child said they are crying", "scared": "Child expressed fear",
                    "sad": "Child expressed sadness/loneliness"}[intent] + f": '{user_msg[:60]}'"
        extra = {"source": "companion", "language": lang, "matched": matched}
        if vision and vision.get("observation"):
            extra["observation"] = vision["observation"]
        alert = create_alert(intent, note, 0.0, extra, frame=frame)
        reply = reply_for(intent, lang)
        clip_id = EMOTION_CLIPS[intent]
        has_clip = _parent_clip_path(clip_id) is not None
        parent_url = f"/audio/parent_voice_{clip_id}.wav" if has_clip else None
        parent_text = PHRASES[clip_id] if has_clip else None
        audio_url, tts_err = synthesize(reply, lang)
        mark_activity("speaking", 12)
        if has_clip:
            log_transcript("parent", parent_text, None, kind="parent_clip", url=parent_url)
        mitra_entry = log_transcript("mitra", reply, lang)
        language_digest(lang, child_entry if action != "cry_simulate" else None, mitra_entry, alert["id"])
        return jsonify({
            "ok": True, "emotion": intent, "intent": intent, "matched": matched, "language": lang,
            "parent_voice_url": parent_url, "parent_voice_text": parent_text,
            "parent_voice_missing": None if has_clip else clip_id,
            "reply": reply, "audio_url": audio_url, "tts_error": tts_err, "alert": alert,
            "vision": vision, "frame_analysis": frame_local,
        })

    # 2. Stop / interruption
    if intent == "stop":
        reply = reply_for("stop", lang)
        audio_url, tts_err = synthesize(reply, lang)
        mitra_entry = log_transcript("mitra", reply, lang)
        language_digest(lang, child_entry, mitra_entry)
        return jsonify({"ok": True, "emotion": "stopped", "intent": "stop", "matched": matched, "language": lang,
                        "reply": reply, "audio_url": audio_url, "tts_error": tts_err})

    # 3. Visual distress seen during chat → alert the parent even before words say so.
    alert = None
    if vision and vision.get("distress") and (vision.get("confidence") or 0) >= 0.6 \
            and vision.get("emotion") in VISION_ALERT_TYPE and _screen_cooldown_ok("vision", time.time()):
        alert = create_alert(VISION_ALERT_TYPE[vision["emotion"]], f"Seen on camera during chat: {vision['observation']}",
                             0.0, {"source": "vision", "language": lang, "observation": vision["observation"],
                                   "vision_model": vision.get("model")}, frame=frame)

    # 4. Conversation via Groq with short memory (+ what the camera sees)
    llm_err = None
    if not user_msg:
        reply = reply_for("greeting", lang)
    else:
        with _history_lock:
            history = list(_chat_history)
        system = SYSTEM_PROMPT + "\n\n" + language_rule(lang)
        if vision and vision.get("observation"):
            system += ("\n\nMitra can see the child through its camera right now. A vision model reports: \""
                       + vision["observation"] + "\". Use this naturally only when it helps (for example noticing a "
                       "smile, a toy, or that they look tired or upset and gently comforting them). Never say anything "
                       "frightening and never describe the camera itself.")
        messages = [{"role": "system", "content": system}] + history + [{"role": "user", "content": user_msg}]
        reply, llm_err = llm_complete(messages)
        if reply:
            with _history_lock:
                _chat_history.append({"role": "user", "content": user_msg})
                _chat_history.append({"role": "assistant", "content": reply})
        else:
            print(f"⚠️ {llm_err}")
            reply = reply_for("fallback", lang)
    audio_url, tts_err = synthesize(reply, lang)
    mark_activity("speaking", 12)
    mitra_entry = log_transcript("mitra", reply, lang)
    language_digest(lang, child_entry, mitra_entry, alert["id"] if alert else None)
    return jsonify({"ok": True, "emotion": "happy", "intent": "chat", "language": lang, "reply": reply,
                    "audio_url": audio_url, "tts_error": tts_err, "llm_error": llm_err,
                    "vision": vision, "frame_analysis": frame_local, "alert": alert})


@app.route("/api/transcript", methods=["GET"])
def get_transcript():
    limit = int(_float(request.args.get("limit"), 60, 1, TRANSCRIPT_LIMIT))
    items = _read_json(TRANSCRIPT_FILE, list)
    return jsonify({"items": items[-limit:] if isinstance(items, list) else []})


@app.route("/api/transcript/clear", methods=["POST"])
def clear_transcript():
    with _transcript_lock:
        _write_json(TRANSCRIPT_FILE, [])
    with _history_lock:
        _chat_history.clear()
    return jsonify({"ok": True})


@app.route("/api/parent-speak", methods=["POST"])
def parent_speak():
    """Queue a parent message for the Child Companion.
    multipart `audio` (WAV push-to-talk) | JSON {text, language} (Edge-TTS) | JSON {phrase_id} (existing clip)."""
    label = load_settings().get("parent_label") or "Mama/Papa"
    lang, duration = None, None
    if request.files.get("audio"):
        data, err = _read_upload()
        if err:
            return jsonify({"ok": False, "error": err}), 400
        info = _wav_info(data)
        if not info or info["duration"] < 0.3:
            return jsonify({"ok": False, "error": "Recording must be a PCM WAV of at least 0.3 s"}), 400
        fname = f"msg_{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}.wav"
        _write_bytes_atomic(os.path.join(PARENT_MSG_DIR, fname), data)
        _prune_dir(PARENT_MSG_DIR, keep=60, prefix="msg_")
        url, text, kind, duration = f"/audio/parent-msg/{fname}", "", "parent_recording", round(info["duration"], 1)
    else:
        body = request.get_json(silent=True) or {}
        phrase_id = str(body.get("phrase_id") or "").strip()
        if phrase_id:
            if phrase_id not in PHRASES:
                return jsonify({"ok": False, "error": "Unknown phrase id"}), 400
            if not _parent_clip_path(phrase_id):
                return jsonify({"ok": False, "error": f"parent_voice_{phrase_id}.wav is not on Mitra yet — record it in Voice Vault"}), 404
            url, text, kind = f"/audio/parent_voice_{phrase_id}.wav", PHRASES[phrase_id], "parent_clip"
        else:
            text = str(body.get("text") or "").strip()[:300]
            if not text:
                return jsonify({"ok": False, "error": "Send an audio recording, text or phrase_id"}), 400
            lang = resolve_language(body.get("language"), text)
            url, err = synthesize(text, lang)
            if not url:
                return jsonify({"ok": False, "error": err}), 502
            kind = "parent_tts"
    item = enqueue_audio(kind, text, url, lang, label=label, duration_sec=duration)
    log_transcript("parent", text or "(voice message)", lang, kind=kind, url=url, source="parent")
    return jsonify({"ok": True, "item": item})


@app.route("/api/pending-audio", methods=["GET"])
def pending_audio():
    now = time.time()
    with _pending_lock:
        items = [dict(i) for i in reversed(_pending)][:20]
        companions = [{"device": d, "state": v["state"], "last_seen": v["t"]}
                      for d, v in _companion_state.items() if now - v["t"] <= 30]
    return jsonify({"items": items, "companions": companions, "server_time": now, "ttl_sec": PENDING_TTL_SEC})


COMPANION_STATES = {"idle", "listening", "thinking", "speaking", "parent", "transcribing"}


@app.route("/api/pending-audio/claim", methods=["POST"])
def claim_pending_audio():
    """Called by Child Companion devices every 2 s: heartbeat (+ current state) and claim of queued items."""
    body = request.get_json(silent=True) or {}
    device = str(body.get("device") or "companion")[:40]
    state = str(body.get("state") or "idle")
    state = state if state in COMPANION_STATES else "idle"
    now, out = time.time(), []
    with _pending_lock:
        _companion_state[device] = {"t": now, "state": state}
        for item in _pending:
            if item["claimed_at"] is None and now - item["created_at"] <= PENDING_TTL_SEC:
                item["claimed_at"], item["claimed_by"] = now, device
                out.append(dict(item))
    return jsonify({"items": out})


@app.route("/api/parent-voice", methods=["GET"])
def list_parent_voice():
    items = []
    for pid, text in PHRASES.items():
        path = _parent_clip_path(pid)
        if not path:
            continue
        info = None
        try:
            with open(path, "rb") as f:
                info = _wav_info(f.read(64 * 1024 * 1024))
        except OSError:
            pass
        items.append({"id": pid, "text": text, "url": f"/audio/parent_voice_{pid}.wav", "size": os.path.getsize(path),
                      "updated": datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M:%S"),
                      "duration_sec": round(info["duration"], 1) if info else None})
    return jsonify({"items": items})


@app.route("/api/parent-voice/<phrase_id>", methods=["POST"])
def upload_parent_voice(phrase_id):
    if phrase_id not in PHRASES:
        return jsonify({"ok": False, "error": "Unknown phrase id"}), 404
    data, err = _read_upload()
    if err:
        return jsonify({"ok": False, "error": err}), 400
    info = _wav_info(data)
    if not info:
        return jsonify({"ok": False, "error": "Not a PCM WAV file"}), 400
    if (info["rate"], info["channels"], info["sampwidth"]) != (16000, 1, 2):
        return jsonify({"ok": False, "error": f"Expected 16 kHz mono 16-bit WAV, got {info['rate']} Hz, "
                                              f"{info['channels']} ch, {info['sampwidth'] * 8}-bit"}), 400
    if info["duration"] < 0.3:
        return jsonify({"ok": False, "error": "Recording is too short"}), 400
    path = os.path.join(PARENT_VOICE_DIR, f"parent_voice_{phrase_id}.wav")
    _write_bytes_atomic(path, data)
    print(f"🎙️ Saved parent voice: {os.path.basename(path)} ({info['duration']:.1f}s)")
    return jsonify({"ok": True, "id": phrase_id, "url": f"/audio/parent_voice_{phrase_id}.wav",
                    "duration_sec": round(info["duration"], 1), "size": len(data)})


@app.route("/api/telemetry", methods=["POST"])
def post_telemetry():
    """Mitra hardware may post sound levels: {db} or {samples: [{t, db}]} (t = unix seconds)."""
    body = request.get_json(silent=True) or {}
    samples = body.get("samples") if isinstance(body.get("samples"), list) else [body]
    source = "screening" if body.get("source") == "screening" else "device"
    if source == "screening":
        _screen_devices[str(body.get("device") or "screening")[:40]] = time.time()
    accepted = 0
    with _telemetry_lock:
        for s in samples[:500]:
            if not isinstance(s, dict):
                continue
            db = _float(s.get("db"), None, 0, 160)
            if db is None:
                continue
            _telemetry.append({"t": _float(s.get("t"), time.time()), "db": round(db, 1), "source": source})
            accepted += 1
    return jsonify({"ok": accepted > 0, "accepted": accepted})


@app.route("/api/telemetry", methods=["GET"])
def get_telemetry():
    """Sound levels posted by the Mitra device and by Live Screening (`source` per sample)."""
    seconds = _float(request.args.get("seconds"), 60, 1, 600)
    cutoff = time.time() - seconds
    with _telemetry_lock:
        samples = sorted((s for s in _telemetry if s["t"] >= cutoff), key=lambda s: s["t"])
        last_at = _telemetry[-1]["t"] if _telemetry else None
    sources = sorted({s.get("source", "device") for s in samples})
    return jsonify({"source": sources[0] if len(sources) == 1 else ("mixed" if sources else "device"),
                    "samples": samples, "last_at": last_at})


# ── Voice input (Groq Whisper) ─────────────────────────────────
_AUDIO_EXT = {"audio/webm": "webm", "audio/ogg": "ogg", "audio/mp4": "mp4", "audio/mpeg": "mp3", "audio/wav": "wav",
              "audio/x-wav": "wav", "audio/wave": "wav", "audio/flac": "flac", "audio/x-m4a": "m4a"}


@app.route("/api/transcribe", methods=["POST"])
def transcribe_route():
    """Child Companion voice input: multipart `audio` (webm/ogg/mp4/wav) + `language` → Whisper text."""
    f = request.files.get("audio")
    if not f:
        return jsonify({"ok": False, "error": "No audio received"}), 400
    data = f.read(MAX_AUDIO_BYTES + 1)
    if not data or len(data) > MAX_AUDIO_BYTES:
        return jsonify({"ok": False, "error": "Audio missing or larger than 10 MB"}), 400
    mime = (f.mimetype or "").split(";")[0].lower()
    ext = _AUDIO_EXT.get(mime) or (f.filename.rsplit(".", 1)[-1].lower() if f.filename and "." in f.filename else "webm")
    lang = str(request.form.get("language") or "").lower()
    result, err = transcribe_audio(data, f"speech.{ext}", lang)
    if err:
        return jsonify({"ok": False, "error": err}), 502
    return jsonify({"ok": True, **result})


# ── Live screening ──────────────────────────────────────────────
def engines_status():
    return {
        "numpy": np is not None,
        "vision_engine": {"ready": vision_engine.ready, "backend": vision_engine.describe(),
                          "error": vision_engine.error, "notes": vision_engine.notes},
        "cry_model": {"ready": cry_engine.ready, "info": cry_engine.info, "error": cry_engine.error},
        "vision_llm": {"model": _models.get("vision"), "checked": bool(_models.get("checked_at")),
                       "probes": _models.get("vision_probes"), "error": _models.get("error")},
    }


@app.route("/api/screening/analyze", methods=["POST"])
def screening_analyze_route():
    """multipart: frame (JPEG), audio (WAV, ~3 s), device, motion (0-1 measured in the browser),
    vision=1 (run Groq vision now), context (recent child words), reference=1 (keep as reference frame)."""
    if np is None:
        return jsonify({"ok": False, "error": "numpy is not installed on the hub"}), 503
    frame, err = _read_image_upload("frame")
    if err:
        return jsonify({"ok": False, "error": err}), 400
    wav = None
    if request.files.get("audio"):
        wav, err = _read_upload()
        if err:
            return jsonify({"ok": False, "error": err}), 400
    if not frame and not wav:
        return jsonify({"ok": False, "error": "Send a camera `frame` (JPEG) and/or an `audio` WAV clip"}), 400
    form = request.form
    yes = lambda k: str(form.get(k) or "").lower() in ("1", "true", "yes", "on")
    result = screening_analyze(str(form.get("device") or "screening")[:40], frame, wav,
                               _float(form.get("motion"), None, 0, 1), want_vision=yes("vision"),
                               context_text=(form.get("context") or "")[:300] or None, keep_reference=yes("reference"))
    return jsonify(result)


@app.route("/api/screening/state", methods=["GET"])
def screening_state():
    seconds = _float(request.args.get("seconds"), 300, 10, 3600)
    now = time.time()
    with _screen_lock:
        recent = [r for r in _screen_latest.values() if now - r["t"] <= 600]
        history = [h for h in _screen_history if h["t"] >= now - seconds]
    latest = max(recent, key=lambda r: r["t"]) if recent else None
    active = [d for d, t in list(_screen_devices.items()) if now - t <= 15]
    return jsonify({"latest": latest, "history": history, "active_devices": active, "engines": engines_status(),
                    "weights": STRESS_WEIGHTS, "server_time": now})


@app.route("/api/screening/references", methods=["GET"])
def screening_references():
    return jsonify({"items": list(reversed(load_references()))})


@app.route("/api/models", methods=["GET"])
def models_route():
    force = request.args.get("refresh") == "1" and _same_origin()
    if groq_configured():
        refresh_models(force=force)
    return jsonify(models_public())


# ── Parent alerting ─────────────────────────────────────────────
def lan_ipv4_addresses():
    addrs = []
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))   # UDP connect sends nothing; it only picks the outgoing interface
        addrs.append(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    try:
        for a in socket.gethostbyname_ex(socket.gethostname())[2]:
            if a not in addrs:
                addrs.append(a)
    except OSError:
        pass
    return [a for a in addrs if not a.startswith(("127.", "169.254.", "0."))]


def qr_svg(data):
    if segno is None:
        return None
    try:
        return segno.make(data, error="m").svg_inline(scale=4, border=2, dark="#1A1410", light="#FFFFFF")
    except Exception:
        return None


@app.route("/api/capabilities", methods=["GET"])
def capabilities():
    m = refresh_models() if groq_configured() else models_public()
    host = request.host
    port = host.rsplit(":", 1)[1] if re.search(r":\d+$", host) else ("443" if request.scheme == "https" else "80")
    return jsonify({
        "twilio_configured": twilio_configured(),
        "telegram_configured": telegram_configured(),
        "email_configured": email_configured(),
        "groq_configured": groq_configured(),
        "ntfy_env_topic": bool(env("NTFY_TOPIC")),
        "ntfy_server": ntfy_server(),
        "edge_tts_available": edge_tts is not None,
        "stt_available": bool(m.get("whisper")),
        "vision_llm_available": bool(m.get("vision")),
        "models": {k: m.get(k) for k in ("text", "text_source", "text_note", "vision", "whisper", "vision_probes",
                                         "error", "checked_at")},
        "engines": engines_status(),
        "lan_urls": [f"http://{a}:{port}" for a in lan_ipv4_addresses()],
        "credentials_writable": _from_hub_machine(),
        "qr_available": segno is not None,
        "languages": [{"code": c, "name": i["name"], "native": i["native"], "voice": i["voice"]} for c, i in LANGUAGES.items()],
    })


@app.route("/api/qr", methods=["GET"])
def qr_route():
    if not _same_origin():
        return _forbidden()
    data = str(request.args.get("data") or "")
    if not re.match(r"^https?://[^\s]{3,300}$", data):
        return jsonify({"ok": False, "error": "Only http(s) links"}), 400
    svg = qr_svg(data)
    if not svg:
        return jsonify({"ok": False, "error": "QR generator unavailable (pip install segno)"}), 503
    if "xmlns=" not in svg[:200]:   # inline SVG omits the namespace; a standalone image needs it
        svg = svg.replace("<svg ", '<svg xmlns="http://www.w3.org/2000/svg" ', 1)
    return app.response_class(svg, mimetype="image/svg+xml")


@app.route("/api/settings/credentials", methods=["GET"])
def get_credentials():
    if not _same_origin():
        return _forbidden()
    return jsonify({"credentials": credentials_public(), "writable": _from_hub_machine(),
                    "env_file": os.path.exists(ENV_PATH), "gitignored": _gitignored_env()})


@app.route("/api/settings/credentials", methods=["POST"])
def post_credentials():
    """Save provider keys to .env (masked on read, applied immediately — no restart)."""
    if not _same_origin():
        return _forbidden()
    if not _from_hub_machine():
        return jsonify({"ok": False, "error": "For safety, provider keys can only be changed on the hub computer "
                                              "itself (open http://localhost on that PC)."}), 403
    clean, errors = validate_credentials(request.get_json(silent=True))
    if errors:
        return jsonify({"ok": False, "errors": errors, "error": " · ".join(errors.values())}), 400
    if not clean:
        return jsonify({"ok": False, "error": "Nothing to save"}), 400
    write_env_values(clean)
    print(f"🔑 Updated in .env: {', '.join(sorted(clean))}")
    return jsonify({"ok": True, "saved": sorted(clean), "credentials": credentials_public(),
                    "twilio_configured": twilio_configured(), "telegram_configured": telegram_configured(),
                    "email_configured": email_configured()})


@app.route("/api/telegram/detect-chat", methods=["POST"])
def telegram_detect_chat():
    """getMe + getUpdates: lists chats that messaged the bot; with save=true stores the chat id."""
    if not _same_origin():
        return _forbidden()
    body = request.get_json(silent=True) or {}
    token = str(body.get("token") or "").strip() or env("TELEGRAM_BOT_TOKEN")
    if not token:
        return jsonify({"ok": False, "error": "Enter your bot token from @BotFather first"}), 400
    if not CREDENTIAL_SPECS["TELEGRAM_BOT_TOKEN"][0].match(token):
        return jsonify({"ok": False, "error": "That bot token doesn't look right — it looks like 123456789:AA…"}), 400
    hide = lambda e: str(e).replace(token, "***")[:300]
    try:
        me = requests.get(f"https://api.telegram.org/bot{token}/getMe", timeout=10)
        mej = me.json() if me.content else {}
    except (requests.RequestException, ValueError) as e:
        return jsonify({"ok": False, "error": f"Couldn't reach Telegram: {hide(e)}"}), 502
    if not mej.get("ok"):
        # A provider verdict, not a malformed request: 200 + ok:false with Telegram's own reason.
        return jsonify({"ok": False, "error": f"Telegram rejected the bot token (HTTP {me.status_code}: "
                                              f"{hide(mej.get('description', 'unknown error'))})"})
    bot = mej.get("result") or {}
    try:
        up = requests.get(f"https://api.telegram.org/bot{token}/getUpdates", params={"limit": 100, "timeout": 0},
                          timeout=15)
        upj = up.json() if up.content else {}
    except (requests.RequestException, ValueError) as e:
        return jsonify({"ok": False, "error": f"Couldn't read updates: {hide(e)}"}), 502
    if not upj.get("ok"):
        hint = " — this bot has a webhook set; getUpdates only works without one (deleteWebhook)" if up.status_code == 409 else ""
        return jsonify({"ok": False, "error": f"Telegram HTTP {up.status_code}: {hide(upj.get('description', ''))}{hint}"})
    chats = {}
    for u in upj.get("result", []):
        for key in ("message", "edited_message", "channel_post", "my_chat_member"):
            m = u.get(key)
            c = (m or {}).get("chat") or {}
            if "id" not in c:
                continue
            name = c.get("title") or " ".join(x for x in (c.get("first_name"), c.get("last_name")) if x) \
                or c.get("username") or str(c["id"])
            chats[c["id"]] = {"id": str(c["id"]), "type": c.get("type"), "name": name,
                              "username": c.get("username"), "date": m.get("date")}
    found = sorted(chats.values(), key=lambda c: c.get("date") or 0, reverse=True)
    saved = None
    chosen = str(body.get("chat_id") or "").strip() or (found[0]["id"] if len(found) == 1 else "")
    if body.get("save") and chosen:
        if not _from_hub_machine():
            return jsonify({"ok": False, "error": "Keys can only be saved on the hub computer itself"}), 403
        if not CREDENTIAL_SPECS["TELEGRAM_CHAT_ID"][0].match(chosen):
            return jsonify({"ok": False, "error": "Invalid chat id"}), 400
        updates = {"TELEGRAM_CHAT_ID": chosen}
        if body.get("token"):
            updates["TELEGRAM_BOT_TOKEN"] = token
        write_env_values(updates)
        saved = chosen
    username = bot.get("username")
    return jsonify({"ok": True, "bot": {"username": username, "name": bot.get("first_name"),
                                        "link": f"https://t.me/{username}" if username else None},
                    "chats": found, "saved_chat_id": saved, "telegram_configured": telegram_configured(),
                    "hint": None if found else "No messages yet — open the bot link, press Start, send any "
                                               "message, then press Auto-detect again."})


def _ntfy_payload(topic):
    url = f"{ntfy_server()}/{topic}" if topic else None
    return {"topic": topic or None, "url": url, "qr_svg": qr_svg(url) if url else None, "server": ntfy_server()}


@app.route("/api/ntfy/connect", methods=["GET", "POST"])
def ntfy_connect():
    """GET: current topic + QR. POST {regenerate?}: create a private random topic, enable ntfy, return QR."""
    if not _same_origin():
        return _forbidden()
    if request.method == "GET":
        return jsonify({"ok": True, **_ntfy_payload(ntfy_topic(load_settings()))})
    body = request.get_json(silent=True) or {}
    with _settings_lock:
        current = load_settings()
        topic = current.get("ntfy_topic") or ""
        if body.get("regenerate") or not topic:
            topic = f"mitra-care-{secrets.token_hex(6)}"
        current.update(ntfy_topic=topic, enable_ntfy=True)
        _write_json(SETTINGS_FILE, current)
    return jsonify({"ok": True, **_ntfy_payload(topic), "settings": current})


@app.route("/api/settings", methods=["GET"])
def get_settings():
    if not _same_origin():
        return _forbidden()
    s = load_settings()
    return jsonify({"settings": s, "ntfy_topic_effective": ntfy_topic(s),
                    "ntfy_topic_source": "settings" if s.get("ntfy_topic") else ("env" if env("NTFY_TOPIC") else None)})


@app.route("/api/settings", methods=["POST"])
def post_settings():
    if not _same_origin():
        return _forbidden()
    clean, errors = validate_settings_patch(request.get_json(silent=True))
    if errors:
        return jsonify({"ok": False, "errors": errors}), 400
    with _settings_lock:
        current = load_settings()
        for k, v in clean.items():
            if k == "triggers":
                current["triggers"].update(v)
            else:
                current[k] = v
        _write_json(SETTINGS_FILE, current)
    return jsonify({"ok": True, "settings": current, "ntfy_topic_effective": ntfy_topic(current)})


@app.route("/api/notifications", methods=["GET"])
def get_notifications():
    if not _same_origin():
        return _forbidden()
    limit = int(_float(request.args.get("limit"), 30, 1, NOTIF_LIMIT))
    return jsonify({"items": list(reversed(load_notifications()))[:limit]})


@app.route("/api/notify/test", methods=["POST"])
def notify_test():
    if not _same_origin():
        return _forbidden()
    channel = str((request.get_json(silent=True) or {}).get("channel") or "")
    if channel not in ("sms", "call", "ntfy", "telegram", "email"):
        return jsonify({"ok": False, "error": "channel must be sms, call, ntfy, telegram or email"}), 400
    with _cooldown_lock:
        wait = 10 - (time.time() - _last_test_at.get(channel, 0))
        if wait > 0:
            return jsonify({"ok": False, "error": f"Wait {int(wait) + 1}s before testing {channel} again"}), 429
        _last_test_at[channel] = time.time()
    s = load_settings()
    text = "This is a test alert from Mitra. If you received this, parent alerts are working."
    if channel == "sms":
        attempt = send_sms(s["parent_phone"], text, test=True)
    elif channel == "call":
        attempt = place_call(s["parent_phone"], text, test=True)
    elif channel == "ntfy":
        attempt = send_ntfy(ntfy_topic(s), "Mitra test alert", text, test=True)
    elif channel == "email":
        attempt = send_email("Mitra test alert", text, test=True)
    else:
        attempt = send_telegram("🧪 " + text, test=True)
    record_attempts([attempt])
    return jsonify({"ok": attempt["status"] == "sent", "attempt": attempt})


def _make_hub_server(port):
    """Threaded Werkzeug server that answers on IPv4 *and* IPv6.

    - Binds "::" with IPV6_V6ONLY off, so http://localhost (which Windows and
      browsers resolve to ::1 first) answers instantly instead of waiting ~2 s
      for the refused-IPv6 → IPv4 fallback on every new connection.
    - On Windows, uses SO_EXCLUSIVEADDRUSE instead of SO_REUSEADDR: with the
      latter a second `python mitra_server.py` silently binds the same port and
      the browser randomly talks to the stale process.
    """
    from werkzeug.serving import ThreadedWSGIServer

    class HubServer(ThreadedWSGIServer):
        allow_reuse_address = sys.platform != "win32"

        def server_bind(self):
            if sys.platform == "win32" and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            if self.address_family == socket.AF_INET6:
                self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
            super().server_bind()

    # has_dualstack_ipv6() is False when the IPv6 stack is disabled → plain IPv4.
    host = env("MITRA_HOST") or ("::" if socket.has_ipv6 and socket.has_dualstack_ipv6() else "0.0.0.0")
    return HubServer(host, port, app)


if __name__ == "__main__":
    port = int(env("MITRA_PORT") or 5000)
    server = _make_hub_server(port)   # exits with a clear message if the port is taken
    print("=" * 56)
    print("🌐 Mitra hub starting…")
    print(f"   Dashboard:  http://127.0.0.1:{port}  (or http://localhost:{port})")
    print(f"   Listening:  {'IPv4 + IPv6' if server.address_family == socket.AF_INET6 else 'IPv4 only'} on port {port}")
    for _a in lan_ipv4_addresses():
        print(f"   On Wi-Fi:   http://{_a}:{port}  (alerts & dashboard; camera/mic need the hub PC or https)")
    print(f"   Groq:       {'key set — verifying models…' if groq_configured() else 'not configured (GROQ_API_KEY)'}")
    print(f"   Edge-TTS:   {'available' if edge_tts else 'missing (pip install edge-tts)'}")
    print(f"   Twilio:     {'ready' if twilio_configured() else 'not configured'}")
    print(f"   Telegram:   {'ready' if telegram_configured() else 'not configured'}")
    print(f"   Email:      {'ready' if email_configured() else 'not configured'}")
    print(f"   ntfy:       {ntfy_server()} (topic set in settings or NTFY_TOPIC)")
    print("=" * 56)

    def _warm_up():
        # Real readiness checks in the background so the dashboard is available immediately.
        if groq_configured():
            refresh_models()
        vision_engine.ensure()
        cry_engine.ensure()
    threading.Thread(target=_warm_up, daemon=True, name="warm-up").start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n👋 Mitra hub stopped.")
    finally:
        server.server_close()
