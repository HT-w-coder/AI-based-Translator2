import streamlit as st
import whisper
import tempfile
from deep_translator import GoogleTranslator
from elevenlabs import generate, set_api_key
from pathlib import Path
from pydub import AudioSegment
from pydub.utils import which
import io
import subprocess
import sys
import os

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Multilingual Voice Assistant 🌍", layout="wide")

set_api_key("YOUR_API_KEY")  # Replace with your ElevenLabs API key
VOICE_NAME = "Rachel"       # ElevenLabs voice
model = whisper.load_model("base")

# Supported languages
LANGUAGES = {
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Italian": "it",
    "Portuguese": "pt",
    "Russian": "ru",
    "Chinese": "zh",
    "Japanese": "ja",
    "Korean": "ko",
    "Hindi": "hi"
}

# ---------------- FFMPEG SETUP ----------------
def ensure_ffmpeg():
    """Check if ffmpeg/ffprobe exist, else download portable binaries."""
    ffmpeg_path = which("ffmpeg")
    ffprobe_path = which("ffprobe")

    if ffmpeg_path and ffprobe_path:
        AudioSegment.converter = ffmpeg_path
        AudioSegment.ffprobe = ffprobe_path
        return

    st.warning("ffmpeg/ffprobe not found. Downloading portable ffmpeg for your platform...")

    import platform
    import urllib.request
    import zipfile
    import tarfile

    # Temporary folder for ffmpeg
    temp_dir = os.path.join(tempfile.gettempdir(), "ffmpeg")
    os.makedirs(temp_dir, exist_ok=True)

    system = platform.system()
    ffmpeg_exe = os.path.join(temp_dir, "ffmpeg")
    ffprobe_exe = os.path.join(temp_dir, "ffprobe")

    if system == "Windows":
        url = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
        zip_path = os.path.join(temp_dir, "ffmpeg.zip")
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(temp_dir)
        # Assume binaries are in first folder
        bin_folder = next(os.path.join(temp_dir, d, "bin") for d in os.listdir(temp_dir) if os.path.isdir(os.path.join(temp_dir, d, "bin")))
        AudioSegment.converter = os.path.join(bin_folder, "ffmpeg.exe")
        AudioSegment.ffprobe = os.path.join(bin_folder, "ffprobe.exe")

    else:  # Linux/MacOS
        url = "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-i686-static.tar.xz"
        tar_path = os.path.join(temp_dir, "ffmpeg.tar.xz")
        urllib.request.urlretrieve(url, tar_path)
        with tarfile.open(tar_path) as tar_ref:
            tar_ref.extractall(temp_dir)
        # Find ffmpeg/ffprobe in extracted folder
        for root, dirs, files in os.walk(temp_dir):
            if "ffmpeg" in files and "ffprobe" in files:
                AudioSegment.converter = os.path.join(root, "ffmpeg")
                AudioSegment.ffprobe = os.path.join(root, "ffprobe")
                break

ensure_ffmpeg()

# ---------------- FUNCTIONS ----------------
def convert_to_wav(audio_bytes):
    """Convert uploaded audio to WAV format compatible with Whisper"""
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        audio.export(f.name, format="wav")
        return f.name

def translate_audio(audio_bytes, input_lang_code, output_lang_code):
    # Convert audio to WAV
    temp_audio_path = convert_to_wav(audio_bytes)

    # Transcribe audio
    text = model.transcribe(temp_audio_path, language=input_lang_code)["text"]

    # Translate text
    translation = GoogleTranslator(source=input_lang_code, target=output_lang_code).translate(text)

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
    audio_bytes = Path(file_path).read_bytes()
    st.audio(audio_bytes, format="audio/mp3")

# ---------------- STREAMLIT UI ----------------
st.title("Multilingual Voice Assistant 🌍")
st.write("Upload audio, select the input and output languages, and hear the translation.")

# Audio input
audio_input = st.file_uploader("🎙️ Upload your audio (WAV or MP3)", type=["wav", "mp3"])
input_lang = st.selectbox("Select the language you are speaking", list(LANGUAGES.keys()), index=0)
output_lang = st.selectbox("Select the language for translation", list(LANGUAGES.keys()), index=1)

if audio_input is not None:
    st.info("Processing audio... Please wait.")
    audio_bytes = audio_input.read()
    text, translation, tts_file_path = translate_audio(audio_bytes, LANGUAGES[input_lang], LANGUAGES[output_lang])

    st.subheader("📝 Transcription")
    st.write(text)

    st.subheader("🌐 Translation")
    st.write(translation)

    st.subheader("🔊 Spoken Translation")
    audio_player(tts_file_path)
