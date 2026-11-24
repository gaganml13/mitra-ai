# 🌿 MITRA: Smart AI Companion & Intelligent Child Care Monitor

**MITRA** is an intelligent, voice-first AI companion designed for children aged 3–8, combined with an acoustic safety monitor and parental reassurance system. 

It provides an engaging, educational companion for the child while offering parents real-time monitoring and peace of mind without intruding on the child's privacy.

---

## ✨ Key Features

### 🧒 1. Child Perspective: The Playful Companion
* **Conversational AI Brain:** Powered by Groq's high-speed inference engine (`qwen/qwen3.8-27b`) tuned for kid-friendly, playful, safe, and short interactions.
* **OpenAI Whisper Speech-to-Text:** Tuned with custom prompts for young voices, names, and cultural vocabulary.
* **Studio-Quality Neural Voice:** Uses Microsoft Edge-TTS (`en-US-AriaNeural`) with an offline speech fallback (`pyttsx3`).
* **Instant Voice Interruption:** Fast-path interrupt listener allows the child to say *"Mitra stop"* or *"Mitra enough"* to immediately cut off speech in under 320ms.
* **Interactive Web Avatar:** Animated cartoon robot face in the browser that blinks, reacts, and moves its mouth when speaking.

### 🛡️ 2. Acoustic Safety & Distress Detection
* **Real-Time Cry Detection:** Dual-pipeline acoustic monitor runs continuously in the background:
  * **Machine Learning Model (`cry_model.pkl`):** Trained classifier using `librosa` acoustic feature extraction (40 MFCCs, delta MFCCs, chroma, and zero-crossing rate).
  * **Heuristic Engine:** Measures RMS energy, amplitude peaks, and coefficient of variation (CV) for crying cadence.
* **Sentiment & Emotion Analysis:** Understands vocal cues and emotional language (fear, sadness, loneliness, bedtime fatigue, achievements).

### 💖 3. Parental Reassurance Layer
* **Personalized Parent Voice Comfort:** When crying or distress is detected, Mitra immediately stops talking and plays pre-recorded audio clips in the **actual parents' voices** (`parent_voice/`) to soothe the child before bridging into a comforting bedtime story.
* **Voice Recording Studio:** Built-in web tool for parents to record bedtime, comfort, praise, and morning voice notes.

### 📊 4. Parent Monitoring Dashboard (`http://localhost:5000`)
* **Zero-Cloud Local Server:** Runs locally via a lightweight Flask bridge (`mitra_server.py`).
* **Real-Time Distress Alerts:** Instant notifications with timestamps and duration for crying, fear, or loneliness.
* **Activity Analytics:** 24-hour timeline and breakdown of distress events.

---

## 🏛️ System Architecture

```text
                             [ Microphone ]
                                    │
       ┌────────────────────────────┴───────────────────────────┐
       ▼                                                        ▼
[ Real-Time Cry Detector ]                            [ Conversation Thread ]
 • Librosa Feature Extraction                          • Wake Word: "Mitra"
 • Scikit-Learn Classifier                             • Whisper STT Engine
 • Cadence Heuristics                                  • Groq LLM (Qwen 27B)
       │                                               • Edge-TTS Synthesis
       ├─ If Distress / Cry Detected:                  • Fast Interrupt ("Stop")
       │   1. Stop active playback (sd.stop)                    │
       │   2. Play Parent Voice (.wav)                          │
       │   3. Mitra soft follow-up comfort                      │
       │   4. HTTP POST Alert ────────────────────────┐         │
       │                                              ▼         ▼
       └────────────────────────────────────► [ Parent Dashboard ]
                                              (Flask on port 5000)
```

---

## 📂 Project Structure

```
mitra/
├── parent_dashboard.html       # Unified Web App (Child Companion Bot + Parent Dashboard)
├── mitra_server.py             # Flask API server & Edge-TTS neural speech generator
├── mitra_ultimate.py           # Standalone Python console runtime (Microphone + Whisper)
├── mitra_calibrate.py          # Microphone calibration utility (room noise & crying thresholds)
├── cry_detector_ml.py          # Real-time ML cry classification module
├── train_cry_model.py          # Audio feature extraction and ML model training script
├── cry_model.pkl               # Trained crying detection classifier
├── cry_scaler.pkl              # Feature normalizer / scaler
├── parent_voice/               # Soothing audio clips recorded by parents (.wav)
├── run_mitra.bat               # 1-Click launcher for Windows
├── run_mitra.ps1               # PowerShell runner script
├── .gitignore                  # Git ignore rules for virtualenvs, caches, and large archives
└── README.md                   # Project documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites
* **Operating System:** Windows 10/11, macOS, or Linux
* **Python:** Python 3.10 or 3.11
* **FFmpeg:** Required by OpenAI Whisper for audio decoding
  ```powershell
  # On Windows (via winget):
  winget install Gyan.FFmpeg
  ```

### 2. Installation
Open your terminal in the project directory:

```powershell
# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install sounddevice soundfile numpy openai-whisper edge-tts groq flask flask-cors pyttsx3 librosa joblib scikit-learn
```

---

## 🎮 How to Run

### Mode A: Web Companion & Parent Dashboard (Recommended)

1. Start the server:
   ```powershell
   python mitra_server.py
   ```
2. Open your web browser to:
   👉 **`http://localhost:5000`**

3. **Explore both views:**
   * **`🧒 Child Companion Bot`:** Interactive animated avatar, text chat, voice input (click the 🎙️ mic), and one-click story/joke buttons.
   * **`📊 Parent Monitor`:** Real-time alert log, distress statistics, and parent voice recording studio.

---

### Mode B: Standalone Voice Companion (Console)

If you prefer talking hands-free directly through your laptop's microphone:

1. Check your audio devices:
   ```powershell
   python -c "import sounddevice as sd; print(sd.query_devices())"
   ```
   *(Ensure `INPUT_DEVICE` and `OUTPUT_DEVICE` in `mitra_ultimate.py` match your microphone and speakers).*

2. Start the standalone runtime:
   ```powershell
   python mitra_ultimate.py
   ```

3. **Voice Interactions:**
   * Say **`"Mitra"`** to wake the companion up.
   * Ask questions or request stories.
   * Say **`"Mitra stop"`** mid-sentence to interrupt.
   * Simulate a crying sound to trigger the parent voice soothe.

---

## 🧪 Testing & Verification

| Test Scenario | Action | Expected Output |
| :--- | :--- | :--- |
| **Q&A / Conversation** | Ask *"Why is the sky blue?"* | Crisp, accurate, kid-friendly 2-sentence explanation via studio neural voice. |
| **Storytelling** | Click `🦖 Dino Story` or say *"Tell a dinosaur story"* | Imaginative 3-sentence children's story. |
| **Stop Interruption** | Click `⏹️ 'Mitra Stop'` or say *"Mitra stop"* | Audio halts immediately (< 320ms) and Mitra asks what you want to do instead. |
| **Parent Comfort** | Click `😨 'I am Scared'` | Plays soothing parent voice recording (`scared_1.wav`) followed by Mitra's reassurance. |
| **Cry Distress Alert** | Click `😭 Simulate Baby Cry` | Triggers parent voice comfort, logs a **`🚨 Cry Alert`** on the Parent Dashboard, and increments alert count. |

---

## 🛠️ Tech Stack

* **AI Reasoning:** Groq API (`qwen/qwen3.8-27b`)
* **Speech Recognition:** OpenAI Whisper (`base` model)
* **Voice Synthesis:** Microsoft Edge Neural TTS (`en-US-AriaNeural`) with `pyttsx3` offline fallback
* **Machine Learning:** Scikit-Learn (`SVC`), Librosa, Joblib, NumPy
* **Audio I/O:** SoundDevice, SoundFile, Web Audio API
* **Web & Dashboard:** Python Flask, Flask-CORS, Vanilla HTML5/CSS3/JavaScript (No external web frameworks required)

---

## 🔒 Safety & Privacy

* **Local Audio Processing:** Continuous wake-word monitoring and cry analysis happen locally on-device. Audio streams are discarded from memory immediately after processing.
* **Zero Cloud Storage for Audio:** Microphone audio is never stored or uploaded to external cloud storage.
* **Child Guardrails:** AI system prompts enforce kind, safe, positive, and age-appropriate responses for children.
