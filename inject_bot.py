import re

with open("parent_dashboard.html", "r", encoding="utf-8") as f:
    html = f.read()

# 1. Add CSS styles before </style>
child_css = """
/* ── VIEW SWITCHER TABS ── */
.nav-switcher {
  display: flex; gap: 8px; background: var(--bg);
  padding: 4px; border-radius: 30px; border: 1px solid var(--border);
}
.nav-tab {
  border: none; background: transparent; padding: 8px 18px;
  border-radius: 20px; font-family: 'DM Sans', sans-serif;
  font-size: 0.82rem; font-weight: 600; cursor: pointer;
  color: var(--ink2); transition: all 0.2s;
  display: flex; align-items: center; gap: 6px;
}
.nav-tab:hover { color: var(--ink); }
.nav-tab.active {
  background: var(--card); color: var(--teal);
  box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.tab-badge {
  background: var(--rose); color: #fff; font-size: 0.65rem;
  padding: 1px 6px; border-radius: 10px; font-family: 'DM Mono', monospace;
}

/* ── CHILD COMPANION VIEW ── */
#childView {
  max-width: 900px; margin: 24px auto 0; padding: 0 20px;
}
.child-hero {
  background: linear-gradient(145deg, #ffffff 0%, #f7fafc 100%);
  border: 1px solid var(--border); border-radius: 24px;
  padding: 32px 24px; box-shadow: 0 10px 30px rgba(42,157,143,0.08);
  text-align: center; margin-bottom: 24px;
}

/* ── AVATAR ROBOT FACE ── */
.avatar-container {
  width: 140px; height: 140px; margin: 0 auto 16px;
  background: linear-gradient(135deg, #2a9d8f, #7c5cbf);
  border-radius: 50%; display: flex; align-items: center;
  justify-content: center; position: relative;
  box-shadow: 0 8px 24px rgba(124,92,191,0.25);
  transition: transform 0.3s;
}
.avatar-face {
  width: 110px; height: 110px; background: #fff;
  border-radius: 50%; display: flex; flex-direction: column;
  align-items: center; justify-content: center; position: relative;
}
.avatar-eyes {
  display: flex; gap: 26px; margin-bottom: 10px;
}
.avatar-eye {
  width: 16px; height: 16px; background: #1a1410;
  border-radius: 50%; position: relative;
  animation: blinkEye 4s infinite;
}
.avatar-eye::after {
  content: ''; position: absolute; top: 2px; right: 2px;
  width: 5px; height: 5px; background: #fff; border-radius: 50%;
}
@keyframes blinkEye {
  0%, 96%, 100% { transform: scaleY(1); }
  98% { transform: scaleY(0.1); }
}
.avatar-mouth {
  width: 28px; height: 14px; border-bottom: 4px solid #1a1410;
  border-radius: 0 0 20px 20px; transition: all 0.2s;
}
.avatar-container.speaking .avatar-mouth {
  animation: talkMouth 0.35s infinite alternate;
}
@keyframes talkMouth {
  0% { height: 6px; width: 20px; }
  100% { height: 18px; width: 28px; background: #e85d5d; border-radius: 10px; }
}
.bot-status-pill {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 6px 16px; border-radius: 20px; background: var(--surface);
  border: 1px solid var(--border); font-size: 0.8rem;
  font-weight: 600; color: var(--teal); margin-bottom: 8px;
}

/* ── CHAT BOX ── */
.chat-window {
  background: var(--card); border: 1px solid var(--border);
  border-radius: 18px; padding: 20px; height: 320px;
  overflow-y: auto; display: flex; flex-direction: column; gap: 12px;
  margin-bottom: 18px; scroll-behavior: smooth;
}
.chat-msg {
  max-width: 80%; padding: 12px 18px; border-radius: 18px;
  font-size: 0.95rem; line-height: 1.5; animation: popMsg 0.25s ease;
}
@keyframes popMsg {
  from { opacity: 0; transform: scale(0.95) translateY(6px); }
  to { opacity: 1; transform: scale(1) translateY(0); }
}
.chat-msg.mitra {
  align-self: flex-start; background: #f0f7f6;
  border: 1px solid #cce8e4; color: #164e46;
  border-bottom-left-radius: 4px;
}
.chat-msg.child {
  align-self: flex-end; background: #7c5cbf;
  color: #fff; border-bottom-right-radius: 4px;
}
.chat-msg.parent-voice {
  align-self: flex-start; background: #fff0f0;
  border: 1px solid #ffd4d4; color: #b91c1c;
  border-bottom-left-radius: 4px; font-weight: 500;
}

/* ── CONTROLS & PILLS ── */
.chat-input-bar {
  display: flex; gap: 10px; margin-bottom: 16px;
}
.chat-input {
  flex: 1; padding: 14px 20px; border-radius: 30px;
  border: 1px solid var(--border); background: var(--card);
  font-family: inherit; font-size: 0.95rem; outline: none;
}
.chat-input:focus { border-color: var(--teal); }
.btn-send {
  padding: 12px 24px; border-radius: 30px; border: none;
  background: var(--teal); color: #fff; font-weight: 600;
  font-family: inherit; cursor: pointer; transition: background 0.2s;
}
.btn-send:hover { background: #238276; }
.btn-mic {
  width: 48px; height: 48px; border-radius: 50%; border: none;
  background: var(--surface); border: 1px solid var(--border);
  font-size: 1.2rem; cursor: pointer; display: flex;
  align-items: center; justify-content: center; transition: all 0.2s;
}
.btn-mic:hover { background: #e2f0ed; }
.btn-mic.recording {
  background: #fee2e2; border-color: #ef4444; color: #ef4444;
  animation: pulseMic 1s infinite;
}
@keyframes pulseMic {
  0%, 100% { transform: scale(1); }
  50% { transform: scale(1.1); }
}

.quick-prompts {
  display: flex; flex-wrap: wrap; gap: 8px; justify-content: center;
}
.prompt-btn {
  background: var(--card); border: 1px solid var(--border);
  padding: 8px 14px; border-radius: 20px; font-size: 0.8rem;
  font-family: inherit; cursor: pointer; transition: all 0.15s;
  color: var(--ink); display: flex; align-items: center; gap: 6px;
}
.prompt-btn:hover {
  background: var(--teal); color: #fff; border-color: var(--teal);
  transform: translateY(-1px);
}
.prompt-btn.danger {
  border-color: #fca5a5; background: #fff5f5; color: #b91c1c;
}
.prompt-btn.danger:hover {
  background: #ef4444; color: #fff; border-color: #ef4444;
}
"""

html = html.replace("</style>", child_css + "\n</style>", 1)

# 2. Add Switcher inside <header>
old_header = """<header>
  <div class="brand">
    <div class="brand-mark">🌿</div>
    <div>
      <div class="brand-name">Mitra</div>
      <div class="brand-sub">Parent Dashboard</div>
    </div>
  </div>"""

new_header = """<header>
  <div class="brand">
    <div class="brand-mark">🌿</div>
    <div>
      <div class="brand-name">Mitra</div>
      <div class="brand-sub">Smart AI Companion</div>
    </div>
  </div>
  
  <div class="nav-switcher">
    <button class="nav-tab active" id="btnChildTab" onclick="switchView('child')">🧒 Child Companion Bot</button>
    <button class="nav-tab" id="btnParentTab" onclick="switchView('parent')">📊 Parent Monitor <span class="tab-badge" id="tabAlertCount">0</span></button>
  </div>"""

html = html.replace(old_header, new_header, 1)

# 3. Add Child View markup and wrap .main inside parentView
child_markup = """
<!-- 🧒 CHILD COMPANION INTERACTIVE VIEW -->
<div id="childView">
  <div class="child-hero">
    <div class="avatar-container" id="avatarBox">
      <div class="avatar-face">
        <div class="avatar-eyes">
          <div class="avatar-eye"></div>
          <div class="avatar-eye"></div>
        </div>
        <div class="avatar-mouth" id="avatarMouth"></div>
      </div>
    </div>
    <div class="bot-status-pill" id="botMoodPill">✨ Mitra is Ready to Play!</div>
    <div style="font-size:0.85rem;color:var(--ink2);">
      Talk to Mitra using your mic or click any story below!
    </div>
  </div>

  <div class="chat-window" id="childChatBox">
    <div class="chat-msg mitra">
      👋 Hi there! I'm <strong>Mitra</strong>, your fun AI best friend! Ask me anything, or tap a button below to hear an exciting story! 🎈
    </div>
  </div>

  <div class="chat-input-bar">
    <button class="btn-mic" id="micBtn" title="Press to speak" onclick="toggleMic()">🎙️</button>
    <input type="text" class="chat-input" id="chatInput" placeholder="Talk to Mitra... (e.g., 'Tell me a dinosaur story!')" onkeydown="if(event.key==='Enter') sendChat()"/>
    <button class="btn-send" onclick="sendChat()">Send 🚀</button>
  </div>

  <div class="quick-prompts">
    <button class="prompt-btn" onclick="quickSend('Tell me a story about a dancing dinosaur!')">🦖 Dino Story</button>
    <button class="prompt-btn" onclick="quickSend('Tell me a fun space adventure!')">🚀 Space Adventure</button>
    <button class="prompt-btn" onclick="quickSend('Tell me a super funny kid joke!')">🤣 Tell a Joke</button>
    <button class="prompt-btn" onclick="quickSend('Sing Twinkle Twinkle Little Star!')">🎵 Sing a Rhyme</button>
    <button class="prompt-btn danger" onclick="quickSend('Mitra, I am scared of the dark and monster under my bed')">😨 'I am Scared'</button>
    <button class="prompt-btn danger" onclick="quickSend('', 'cry_simulate')">😭 Simulate Baby Cry</button>
    <button class="prompt-btn" onclick="quickSend('Mitra stop please!')">⏹️ 'Mitra Stop'</button>
  </div>
</div>

<div id="parentView" style="display:none;">
"""

html = html.replace('<div class="main">', child_markup + '\n<div class="main">', 1)

# Close parentView before </script>
html = html.replace('</div>\n\n<div id="toast"></div>', '</div>\n</div>\n\n<div id="toast"></div>', 1)

# 4. Add JavaScript handlers before </script>
child_js = """
// ── VIEW SWITCHER ─────────────────────────────────────────
function switchView(view){
  const childV = document.getElementById('childView');
  const parentV = document.getElementById('parentView');
  const btnC = document.getElementById('btnChildTab');
  const btnP = document.getElementById('btnParentTab');
  if(view === 'child'){
    childV.style.display = 'block';
    parentV.style.display = 'none';
    btnC.classList.add('active');
    btnP.classList.remove('active');
  } else {
    childV.style.display = 'none';
    parentV.style.display = 'block';
    btnP.classList.add('active');
    btnC.classList.remove('active');
  }
}

// ── CHILD BOT INTERACTION ─────────────────────────────────
let isBotSpeaking = false;
let botAudio = null;

function appendChat(text, sender, note){
  const box = document.getElementById('childChatBox');
  const msg = document.createElement('div');
  msg.className = `chat-msg ${sender}`;
  if(note){
    msg.innerHTML = `<strong>${note}</strong><br/>${text}`;
  } else {
    msg.textContent = text;
  }
  box.appendChild(msg);
  box.scrollTop = box.scrollHeight;
}

function setAvatarMood(status, moodText){
  const pill = document.getElementById('botMoodPill');
  const avatar = document.getElementById('avatarBox');
  pill.textContent = moodText;
  if(status === 'speaking'){
    avatar.classList.add('speaking');
  } else {
    avatar.classList.remove('speaking');
  }
}

function speakInBrowser(text, onDone){
  if(!('speechSynthesis' in window)){
    if(onDone) onDone();
    return;
  }
  window.speechSynthesis.cancel();
  const utter = new SpeechSynthesisUtterance(text);
  utter.rate = 1.0;
  utter.pitch = 1.2; // Kid-friendly cheerful pitch
  utter.onstart = () => setAvatarMood('speaking', '🗣️ Mitra is Talking...');
  utter.onend = () => {
    setAvatarMood('idle', '✨ Mitra is Ready to Play!');
    if(onDone) onDone();
  };
  utter.onerror = () => {
    setAvatarMood('idle', '✨ Mitra is Ready to Play!');
    if(onDone) onDone();
  };
  window.speechSynthesis.speak(utter);
}

async function sendToBot(userText, action){
  if(botAudio){
    try{ botAudio.pause(); }catch(e){}
  }
  window.speechSynthesis.cancel();

  if(userText){
    appendChat(userText, 'child');
  } else if(action === 'cry_simulate'){
    appendChat('😭 [Child cries: waaah waaah!]', 'child');
  }

  setAvatarMood('speaking', '🤔 Mitra is Thinking...');

  try{
    const res = await fetch(`${SERVER}/api/interact`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ message: userText, action: action || 'chat' })
    });
    const data = await res.json();

    if(data.alert){
      toast(`🚨 Alert sent to Parent Dashboard (${data.alert.type.toUpperCase()})`);
      pollServer(); // Immediately update alert count
    }

    if(data.parent_voice_url){
      // 1. Play comforting Parent Voice first
      appendChat(`💖 Mama/Papa: "${data.parent_voice_text}"`, 'parent-voice');
      setAvatarMood('speaking', '💖 Playing Parent Voice...');
      botAudio = new Audio(data.parent_voice_url);
      botAudio.onended = () => {
        // 2. Mitra follows up gently
        appendChat(data.reply, 'mitra');
        speakInBrowser(data.reply);
      };
      botAudio.play().catch(e => {
        appendChat(data.reply, 'mitra');
        speakInBrowser(data.reply);
      });
    } else {
      appendChat(data.reply, 'mitra');
      speakInBrowser(data.reply);
    }
  } catch(e){
    appendChat("Oops! I couldn't reach my brain server right now, but I'm still your friend!", 'mitra');
    setAvatarMood('idle', '✨ Mitra is Ready to Play!');
  }
}

function sendChat(){
  const input = document.getElementById('chatInput');
  const val = input.value.trim();
  if(!val) return;
  input.value = '';
  sendToBot(val, 'chat');
}

function quickSend(text, action){
  sendToBot(text, action);
}

// ── SPEECH RECOGNITION (MIC) ──────────────────────────────
let recognition = null;
let isRecording = false;

if('webkitSpeechRecognition' in window || 'SpeechRecognition' in window){
  const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
  recognition = new SpeechRec();
  recognition.continuous = false;
  recognition.interimResults = false;
  recognition.lang = 'en-US';

  recognition.onstart = () => {
    isRecording = true;
    document.getElementById('micBtn').classList.add('recording');
    setAvatarMood('listening', '👂 Listening to you...');
  };

  recognition.onresult = (evt) => {
    const transcript = evt.results[0][0].transcript;
    sendToBot(transcript, 'chat');
  };

  recognition.onerror = (err) => {
    isRecording = false;
    document.getElementById('micBtn').classList.remove('recording');
    setAvatarMood('idle', '✨ Mitra is Ready to Play!');
  };

  recognition.onend = () => {
    isRecording = false;
    document.getElementById('micBtn').classList.remove('recording');
  };
}

function toggleMic(){
  if(!recognition){
    alert('Voice input is best supported in Google Chrome or Edge.');
    return;
  }
  if(isRecording){
    recognition.stop();
  } else {
    recognition.start();
  }
}
"""

html = html.replace('</script>\n</body>', child_js + '\n</script>\n</body>', 1)

with open("parent_dashboard.html", "w", encoding="utf-8") as f:
    f.write(html)

print("Updated parent_dashboard.html successfully!")
