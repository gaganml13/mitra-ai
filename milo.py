import speech_recognition as sr
import pyttsx3
from groq import Groq

# -----------------------------
# GROQ API KEY
# -----------------------------
client = Groq(api_key=os.environ.get("GROQ_API_KEY", "your_groq_api_key_here"))

# -----------------------------
# TEXT TO SPEECH SETUP
# -----------------------------
engine = pyttsx3.init()
engine.setProperty('rate', 170)   # speed
engine.setProperty('volume', 1)

voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)   # female voice if available

# -----------------------------
# AI MEMORY
# -----------------------------
messages = [
    {
        "role": "system",
        "content": """
You are Milo, a fun AI companion for kids aged 3 to 8.
Be funny, kind, safe, playful and cheerful.
Keep answers short.
"""
    }
]

# -----------------------------
# SPEAK FUNCTION
# -----------------------------
def speak(text):
    print("AI:", text)
    engine.say(text)
    engine.runAndWait()

# -----------------------------
# LISTEN FUNCTION
# -----------------------------
def listen():
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("\n🎤 Speak now...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio)
        print("You:", text)
        return text

    except:
        return None

# -----------------------------
# MAIN LOOP
# -----------------------------
speak("Hello! I am Milo. Talk to me!")

while True:
    user_input = listen()

    if user_input is None:
        speak("Sorry, I didn't hear you.")
        continue

    if user_input.lower() in ["exit", "stop", "bye"]:
        speak("Goodbye my friend!")
        break

    messages.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        temperature=0.8
    )

    ai_reply = response.choices[0].message.content

    messages.append({"role": "assistant", "content": ai_reply})

    speak(ai_reply)