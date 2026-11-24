"""
mitra_server.py  —  Parent Dashboard Server
=============================================
Run this ON THE RASPBERRY PI alongside mitra.py.

    python mitra_server.py

Parents open http://<raspberry-pi-ip>:5050 on any phone/laptop/tablet
connected to the SAME Wi-Fi network.

No internet needed — everything runs locally on the Pi.

Install once:
    pip install flask
"""

from flask import Flask, request, jsonify, Response, send_from_directory
import json, os, time, threading, queue
from datetime import datetime

# ── Paths ─────────────────────────────────────────────────────────────────
BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
PARENT_VOICE_DIR = os.path.join(BASE_DIR, "parent_voice")
DATA_FILE        = os.path.join(BASE_DIR, "dashboard_data.json")

os.makedirs(PARENT_VOICE_DIR, exist_ok=True)

# ── In-memory event store (also persisted to JSON) ────────────────────────
_data_lock = threading.Lock()

def _load_data():
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "events":    [],   # {time, type, detail}
        "rhymes":    [],   # {time, name}
        "topics":    [],   # {time, topic}
        "session_start": None,
    }

def _save_data(data):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=2)

_db = _load_data()

# ── SSE event queue (for real-time push to browser) ───────────────────────
_sse_clients = []
_sse_lock    = threading.Lock()

def _push_sse(event_type: str, payload: dict):
    """Push a real-time event to all connected browser tabs."""
    msg = f"event: {event_type}\ndata: {json.dumps(payload)}\n\n"
    with _sse_lock:
        dead = []
        for q in _sse_clients:
            try:
                q.put_nowait(msg)
            except Exception:
                dead.append(q)
        for q in dead:
            _sse_clients.remove(q)

# ── Flask app ─────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=BASE_DIR)

# ─── API: Mitra → Server ─────────────────────────────────────────────────

@app.route("/api/event", methods=["POST"])
def api_event():
    """Mitra posts every emotion/cry/rhyme event here."""
    body = request.get_json(silent=True) or {}
    ts   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    event_type = body.get("type", "unknown")   # sad / scared / lonely / cry / rhyme / topic
    detail     = body.get("detail", "")

    entry = {"time": ts, "type": event_type, "detail": detail}

    with _data_lock:
        if event_type == "rhyme":
            _db["rhymes"].append({"time": ts, "name": detail})
        elif event_type == "topic":
            _db["topics"].append({"time": ts, "topic": detail})
        elif event_type == "session_start":
            _db["session_start"] = ts
        else:
            _db["events"].append(entry)
        # Keep last 200 of each list
        _db["events"] = _db["events"][-200:]
        _db["rhymes"]  = _db["rhymes"][-100:]
        _db["topics"]  = _db["topics"][-100:]
        _save_data(_db)

    _push_sse("new_event", entry)
    return jsonify({"ok": True})

@app.route("/api/data")
def api_data():
    """Return all stored data for initial dashboard load."""
    with _data_lock:
        return jsonify(_db)

# ─── API: Voice recordings ────────────────────────────────────────────────

@app.route("/api/recordings")
def api_recordings():
    """List all parent voice recordings."""
    files = []
    if os.path.exists(PARENT_VOICE_DIR):
        for f in sorted(os.listdir(PARENT_VOICE_DIR)):
            if f.endswith(".wav"):
                path  = os.path.join(PARENT_VOICE_DIR, f)
                label = f.replace("parent_voice_", "").replace(".wav", "").replace("_", " ")
                files.append({
                    "filename": f,
                    "label":    label,
                    "size_kb":  round(os.path.getsize(path) / 1024, 1),
                    "modified": datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M"),
                })
    return jsonify(files)

@app.route("/api/upload_voice", methods=["POST"])
def api_upload_voice():
    """
    Parent records their voice in browser → browser sends WAV blob here →
    saved to parent_voice/<emotion>_<n>.wav → Mitra picks it up automatically.
    """
    emotion = request.form.get("emotion", "comfort")
    # Sanitize
    emotion = emotion.strip().lower().replace(" ", "_")[:30]

    if "audio" not in request.files:
        return jsonify({"ok": False, "error": "No audio file"}), 400

    audio_file = request.files["audio"]

    # Find next available index  e.g. comfort_1, comfort_2 …
    existing = [f for f in os.listdir(PARENT_VOICE_DIR)
                if f.startswith(f"parent_voice_{emotion}_") and f.endswith(".wav")]
    indices = []
    for f in existing:
        try:
            indices.append(int(f.replace(f"parent_voice_{emotion}_", "").replace(".wav", "")))
        except ValueError:
            pass
    idx = max(indices, default=0) + 1

    filename = f"parent_voice_{emotion}_{idx}.wav"
    save_path = os.path.join(PARENT_VOICE_DIR, filename)
    audio_file.save(save_path)

    _push_sse("recording_added", {"filename": filename, "emotion": emotion})
    return jsonify({"ok": True, "saved_as": filename})

@app.route("/api/delete_voice/<filename>", methods=["DELETE"])
def api_delete_voice(filename):
    """Delete a parent voice recording."""
    # Safety: only allow deleting files in parent_voice dir
    safe_name = os.path.basename(filename)
    path      = os.path.join(PARENT_VOICE_DIR, safe_name)
    if os.path.exists(path):
        os.remove(path)
        _push_sse("recording_deleted", {"filename": safe_name})
        return jsonify({"ok": True})
    return jsonify({"ok": False, "error": "Not found"}), 404

@app.route("/api/play_voice/<filename>", methods=["GET"])
def api_play_voice(filename):
    """Stream a WAV file back to the browser for preview playback."""
    safe_name = os.path.basename(filename)
    return send_from_directory(PARENT_VOICE_DIR, safe_name)

# ─── SSE endpoint ─────────────────────────────────────────────────────────

@app.route("/api/stream")
def api_stream():
    """Server-Sent Events — browser connects once and gets live updates."""
    q = queue.Queue()
    with _sse_lock:
        _sse_clients.append(q)

    def generate():
        # Send a heartbeat first so browser knows we're alive
        yield "event: connected\ndata: {}\n\n"
        try:
            while True:
                try:
                    msg = q.get(timeout=25)
                    yield msg
                except queue.Empty:
                    yield ": heartbeat\n\n"   # keep connection alive
        except GeneratorExit:
            with _sse_lock:
                if q in _sse_clients:
                    _sse_clients.remove(q)

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache",
                             "X-Accel-Buffering": "no"})

# ─── Serve dashboard HTML ─────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "dashboard.html")

# ─── Main ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import socket as _sock
    try:
        s = _sock.socket(_sock.AF_INET, _sock.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except Exception:
        local_ip = "localhost"

    print("=" * 55)
    print("  🧸  Mitra Parent Dashboard Server")
    print("=" * 55)
    print(f"  Open on this device : http://localhost:5050")
    print(f"  Open on phone/tablet: http://{local_ip}:5050")
    print(f"  (Both devices must be on the same Wi-Fi)")
    print("=" * 55)

    app.run(host="0.0.0.0", port=5050, debug=False, threaded=True)
