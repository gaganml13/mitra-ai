"""
mitra_server.py — Local bridge between Mitra and the Parent Dashboard

Run this FIRST before running mitra.py:
    python mitra_server.py

It starts a tiny local web server on http://localhost:5000
The dashboard (parent_dashboard.html) connects to this server.
Mitra posts cry alerts here, and the dashboard polls for new ones.

No internet needed. Everything is local on your machine.
"""

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import json
import os
import time
import asyncio
import edge_tts
from datetime import datetime
from groq import Groq

# Load .env file automatically
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(_env_path):
    with open(_env_path, "r", encoding="utf-8") as _f:
        for _line in _f:
            if "=" in _line and not _line.startswith("#"):
                _k, _v = _line.strip().split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

_GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "your_groq_api_key_here")
groq_client = None
try:
    groq_client = Groq(api_key=_GROQ_API_KEY)
except Exception as _e:
    print(f"⚠️ Groq client init: {_e}")

TTS_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tts_cache")
os.makedirs(TTS_CACHE_DIR, exist_ok=True)

async def _synth_async(text: str, out_path: str):
    comm = edge_tts.Communicate(text.strip(), voice="en-US-AriaNeural")
    await comm.save(out_path)

def generate_tts(text: str):
    """Generate crisp Edge-TTS audio file on the fly and return relative URL."""
    if not text or not text.strip():
        return None
    try:
        clean_text = text.replace("*", "").replace("#", "").strip()
        fname = f"mitra_{int(time.time()*1000)}.mp3"
        fpath = os.path.join(TTS_CACHE_DIR, fname)
        asyncio.run(_synth_async(clean_text, fpath))
        if os.path.exists(fpath) and os.path.getsize(fpath) > 100:
            return f"/audio/tts/{fname}"
    except Exception as e:
        print(f"⚠️ Edge-TTS generation error: {e}")
    return None

_SYSTEM_PROMPT = (
    "You are Mitra, a smart, caring, and cheerful AI companion for young children (aged 3 to 8). "
    "Rules for your responses:\n"
    "1. Always address what the child or parent is actually asking or saying with precision.\n"
    "2. If asked for a story, tell an engaging, well-crafted, imaginative story in 3-4 sentences.\n"
    "3. If asked an educational or factual question (e.g. why the sky is blue, what animals eat), explain it simply, accurately, and clearly.\n"
    "4. If the user mentions any issue (like overlapping voices or audio), acknowledge it politely and clearly without confusion (e.g. 'I understand! I've stopped the other voice so now only I am speaking. What can I do for you?').\n"
    "5. Keep responses concise (2 to 4 sentences), kind, helpful, and grammatically complete without trailing off or repeating random sounds."
)

conversation_history = [{"role": "system", "content": _SYSTEM_PROMPT}]

app = Flask(__name__)
CORS(app)   # allow dashboard (file://) to talk to this server

# ── Storage: simple JSON file so alerts survive restarts ──
DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mitra_alerts.json")

def load_data():
    if not os.path.exists(DATA_FILE):
        return {"alerts": [], "stats": {"total_cries": 0, "today_cries": 0, "last_cry": None}}
    try:
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {"alerts": [], "stats": {"total_cries": 0, "today_cries": 0, "last_cry": None}}

def save_data(data):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=2)

def recalc_stats(data):
    today = datetime.now().strftime("%Y-%m-%d")
    data["stats"]["total_cries"] = len(data["alerts"])
    data["stats"]["today_cries"] = sum(
        1 for a in data["alerts"] if a.get("date", "") == today
    )
    if data["alerts"]:
        data["stats"]["last_cry"] = data["alerts"][-1]["timestamp"]
    else:
        data["stats"]["last_cry"] = None

# ── ROUTES ────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the parent dashboard directly when visiting http://localhost:5000"""
    dashboard_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "parent_dashboard.html")
    if os.path.exists(dashboard_path):
        return send_file(dashboard_path)
    return "<h1>Mitra Parent Server Running</h1><p>parent_dashboard.html not found</p>"

@app.route("/ping")
def ping():
    """Dashboard uses this to check if server is running."""
    return jsonify({"status": "ok", "time": time.time()})

@app.route("/alerts", methods=["GET"])
def get_alerts():
    """Dashboard polls this for the full alert list."""
    data = load_data()
    return jsonify(data)

@app.route("/alert", methods=["POST"])
def post_alert():
    """
    Mitra calls this when it detects crying.
    Body JSON: { "type": "cry"|"sad"|"scared", "duration_sec": float, "note": str }
    """
    body = request.get_json(silent=True) or {}
    now  = datetime.now()

    alert = {
        "id":           int(time.time() * 1000),
        "type":         body.get("type", "cry"),
        "duration_sec": round(float(body.get("duration_sec", 0)), 1),
        "note":         body.get("note", ""),
        "timestamp":    now.strftime("%H:%M:%S"),
        "date":         now.strftime("%Y-%m-%d"),
        "datetime":     now.strftime("%Y-%m-%d %H:%M:%S"),
        "read":         False,
    }

    data = load_data()
    data["alerts"].append(alert)
    # Keep only last 200 alerts
    data["alerts"] = data["alerts"][-200:]
    recalc_stats(data)
    save_data(data)

    print(f"🚨 Alert received: {alert['type']} at {alert['timestamp']}")
    return jsonify({"ok": True, "alert": alert})

@app.route("/alert/<int:alert_id>/read", methods=["POST"])
def mark_read(alert_id):
    data = load_data()
    for a in data["alerts"]:
        if a["id"] == alert_id:
            a["read"] = True
    save_data(data)
    return jsonify({"ok": True})

@app.route("/alerts/clear", methods=["POST"])
def clear_alerts():
    data = load_data()
    data["alerts"] = []
    recalc_stats(data)
    save_data(data)
    return jsonify({"ok": True})

@app.route("/status", methods=["POST"])
def post_status():
    """Mitra posts its current status here (listening, speaking, sleeping)."""
    body = request.get_json(silent=True) or {}
    status_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mitra_status.json")
    with open(status_file, "w") as f:
        json.dump({"status": body.get("status", "unknown"), "time": time.time()}, f)
    return jsonify({"ok": True})

@app.route("/status", methods=["GET"])
def get_status():
    status_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mitra_status.json")
    if not os.path.exists(status_file):
        return jsonify({"status": "offline", "time": 0})
    try:
        with open(status_file, "r") as f:
            data = json.load(f)
        # If last update was > 10 seconds ago, consider offline
        if time.time() - data.get("time", 0) > 10:
            data["status"] = "offline"
        return jsonify(data)
    except Exception:
        return jsonify({"status": "offline", "time": 0})

@app.route("/audio/tts/<filename>")
def get_tts_audio(filename):
    """Serve synthesized Edge-TTS neural audio files."""
    path = os.path.join(TTS_CACHE_DIR, filename)
    if os.path.exists(path):
        return send_file(path, mimetype="audio/mpeg")
    return "TTS audio not found", 404

@app.route("/audio/<filename>")
def get_audio(filename):
    """Serve parent voice audio files for playback in the web dashboard."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "parent_voice", filename)
    if os.path.exists(path):
        return send_file(path, mimetype="audio/wav")
    alt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mitra trail", "parent_voice", filename)
    if os.path.exists(alt_path):
        return send_file(alt_path, mimetype="audio/wav")
    return "Audio file not found", 404

@app.route("/api/interact", methods=["POST"])
def interact_bot():
    """
    Direct Child-Bot interaction endpoint for the web interface.
    Features: Contextual memory, Edge-TTS audio generation, emotion recognition.
    """
    global conversation_history
    body = request.get_json(silent=True) or {}
    user_msg = (body.get("message") or "").strip()
    action = body.get("action", "chat")

    now = datetime.now()
    timestamp_str = now.strftime("%H:%M:%S")

    # Update status to active
    status_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mitra_status.json")
    try:
        with open(status_file, "w") as f:
            json.dump({"status": "speaking", "time": time.time()}, f)
    except Exception:
        pass

    # 1. Simulate Baby Crying Action
    if action == "cry_simulate" or any(w in user_msg.lower() for w in ["waaah", "wah wah", "crying"]):
        alert = {
            "id": int(time.time() * 1000),
            "type": "cry",
            "duration_sec": 4.8,
            "note": "Crying distress detected — Parent voice comfort activated",
            "timestamp": timestamp_str,
            "date": now.strftime("%Y-%m-%d"),
            "datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
            "read": False,
        }
        data = load_data()
        data["alerts"].append(alert)
        recalc_stats(data)
        save_data(data)

        comfort_reply = "Mama and Papa love you so much! Does that feel better? Let me tell you a little bedtime story."
        audio_url = generate_tts(comfort_reply)

        return jsonify({
            "ok": True,
            "emotion": "cry",
            "parent_voice_url": "/audio/parent_voice_comfort_2.wav",
            "parent_voice_text": "Hey sweetheart, it's okay. You are so loved. More than all the stars in the sky.",
            "reply": comfort_reply,
            "audio_url": audio_url,
            "alert": alert
        })

    # 2. Stop / Interruption Command
    if any(w in user_msg.lower() for w in ["stop", "enough", "hush", "be quiet"]):
        stop_reply = "Okay! I'll stop right now. What would you like to do instead?"
        audio_url = generate_tts(stop_reply)
        return jsonify({
            "ok": True,
            "emotion": "stopped",
            "reply": stop_reply,
            "audio_url": audio_url
        })

    # 3. Scared / Fear Emotion
    if any(w in user_msg.lower() for w in ["scared", "dark", "monster", "afraid", "nightmare", "fear"]):
        alert = {
            "id": int(time.time() * 1000),
            "type": "scared",
            "duration_sec": 3.2,
            "note": f"Child expressed fear: '{user_msg[:40]}'",
            "timestamp": timestamp_str,
            "date": now.strftime("%Y-%m-%d"),
            "datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
            "read": False,
        }
        data = load_data()
        data["alerts"].append(alert)
        recalc_stats(data)
        save_data(data)

        scared_reply = "You are super brave! The dark is just a blanket the stars sleep under. Shall we tell a funny story to make you smile?"
        audio_url = generate_tts(scared_reply)

        return jsonify({
            "ok": True,
            "emotion": "scared",
            "parent_voice_url": "/audio/parent_voice_scared_1.wav",
            "parent_voice_text": "There's nothing to be scared of. I'm always watching over you.",
            "reply": scared_reply,
            "audio_url": audio_url,
            "alert": alert
        })

    # 4. Sad / Lonely Emotion
    if any(w in user_msg.lower() for w in ["sad", "lonely", "alone", "miss mama", "miss mommy", "miss papa", "miss daddy"]):
        alert = {
            "id": int(time.time() * 1000),
            "type": "sad",
            "duration_sec": 3.5,
            "note": f"Child expressed sadness/loneliness: '{user_msg[:40]}'",
            "timestamp": timestamp_str,
            "date": now.strftime("%Y-%m-%d"),
            "datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
            "read": False,
        }
        data = load_data()
        data["alerts"].append(alert)
        recalc_stats(data)
        save_data(data)

        sad_reply = "Mama and Papa are thinking of you right now! Let's play a fun game or sing a song together!"
        audio_url = generate_tts(sad_reply)

        return jsonify({
            "ok": True,
            "emotion": "sad",
            "parent_voice_url": "/audio/parent_voice_comfort_5.wav",
            "parent_voice_text": "I'll be back before you know it. And I'll give you the biggest hug!",
            "reply": sad_reply,
            "audio_url": audio_url,
            "alert": alert
        })

    # 5. Normal Precise Conversation via Groq AI with Memory
    reply_text = "Hi there! I'm Mitra, your best friend! What would you like to play or talk about?"
    if groq_client and user_msg:
        try:
            conversation_history.append({"role": "user", "content": user_msg})
            # Keep history to last 10 messages (plus system prompt)
            if len(conversation_history) > 11:
                conversation_history = [conversation_history[0]] + conversation_history[-10:]

            resp = groq_client.chat.completions.create(
                model="qwen/qwen3.8-27b",
                messages=conversation_history,
                max_tokens=150,
                temperature=0.7
            )
            reply_text = resp.choices[0].message.content.strip()
            conversation_history.append({"role": "assistant", "content": reply_text})
        except Exception as e:
            print(f"⚠️ Groq interaction error: {e}")
            reply_text = "I am listening closely! Tell me more about what you want to do!"

    audio_url = generate_tts(reply_text)

    return jsonify({
        "ok": True,
        "emotion": "happy",
        "reply": reply_text,
        "audio_url": audio_url
    })

if __name__ == "__main__":
    print("=" * 50)
    print("🌐 Mitra Parent Server starting...")
    print("   Dashboard: open parent_dashboard.html in browser")
    print("   API:       http://localhost:5000")
    print("=" * 50)
    # Install flask-cors if missing
    try:
        from flask_cors import CORS
    except ImportError:
        print("Installing flask-cors...")
        os.system("pip install flask-cors --break-system-packages -q")
        from flask_cors import CORS
        CORS(app)
    app.run(host="0.0.0.0", port=5000, debug=False)
