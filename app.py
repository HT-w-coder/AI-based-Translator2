import streamlit as st
import whisper
import tempfile
from deep_translator import GoogleTranslator
from elevenlabs import generate, set_api_key
from pathlib import Path
import base64

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Real-Time Bilingual Assistant 🌍", layout="wide")

set_api_key("YOUR_API_KEY")  # Replace with your ElevenLabs API key
VOICE_NAME = "Rachel"       # ElevenLabs voice
model = whisper.load_model("base")

# ---------------- FUNCTIONS ----------------
def translate_audio(audio_bytes, input_language):
    # Save uploaded audio to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(audio_bytes)
        temp_audio_path = f.name

    # Transcribe audio
    text = model.transcribe(temp_audio_path, language=input_language)["text"]

    # Determine output language
    output_language = "en" if input_language == "es" else "es"

    # Translate text
    translation = GoogleTranslator(source=input_language, target=output_language).translate(text)

    # Generate TTS audio
    audio_out = generate(
        text=translation,
        voice=VOICE_NAME,
        model="eleven_multilingual_v2"
    )

    # Save TTS to temporary file
    tts_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts_file.write(audio_out)
    tts_file.close()

    return text, translation, tts_file.name

def audio_player(file_path):
    """Return a Streamlit audio player for a local file."""
    audio_bytes = Path(file_path).read_bytes()
    st.audio(audio_bytes, format="audio/mp3")

# ---------------- STREAMLIT UI ----------------
st.title("Real-Time Bilingual Assistant 🌍")
st.write("Speak in Spanish or English, and listen to the translation with realistic voice instantly.")

# Audio input
audio_input = st.file_uploader("🎙️ Upload your audio (WAV or MP3)", type=["wav", "mp3"])
input_lang = st.radio("Language you are speaking", ["es", "en"], index=0)

if audio_input is not None:
    st.info("Processing audio... Please wait.")
    audio_bytes = audio_input.read()
    text, translation, tts_file_path = translate_audio(audio_bytes, input_lang)

    st.subheader("📝 Transcription")
    st.write(text)

    st.subheader("🌐 Translation")
    st.write(translation)

    st.subheader("🔊 Spoken Translation")
    audio_player(tts_file_path)
