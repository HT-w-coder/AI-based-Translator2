import gradio as gr
import whisper
import tempfile
from deep_translator import GoogleTranslator
from elevenlabs import generate, set_api_key

# CONFIGURATION
set_api_key("YOUR_API_KEY")  # Replace with your ElevenLabs API key
VOICE_NAME = "Rachel"  # Test voice from ElevenLabs
model = whisper.load_model("base")  # Load the Whisper speech-to-text model

# FUNCTIONS
def translate_audio(audio, input_language):
    # Save uploaded audio to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(audio)
        temp_audio_path = f.name

    # Transcribe audio using Whisper
    text = model.transcribe(temp_audio_path, language=input_language)["text"]

    # Determine output language: if input is Spanish, translate to English, else to Spanish
    output_language = "en" if input_language == "es" else "es"
    
    # Translate text using Google Translator
    translation = GoogleTranslator(source=input_language, target=output_language).translate(text)

    # Generate spoken audio of the translation using ElevenLabs
    audio_out = generate(
        text=translation,
        voice=VOICE_NAME,
        model="eleven_multilingual_v2"
    )

    # Save generated audio to a temporary file and return it
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as out_file:
        out_file.write(audio_out)
        return text, translation, out_file.name

# INTERFACE
interface = gr.Interface(
    fn=translate_audio,
    inputs=[
        gr.Audio(source="microphone", type="binary", label="🎙️ Speak here"),
        gr.Radio(["es", "en"], label="Language you are speaking", value="es")
    ],
    outputs=[
        gr.Textbox(label="📝 Transcription"),
        gr.Textbox(label="🌐 Translation"),
        gr.Audio(label="🔊 Spoken Translation")
    ],
    title="Real-Time Bilingual Assistant 🌍",
    description="Speak in Spanish or English. Listen to the translation with realistic voice instantly."
)

# Launch the Gradio interface
interface.launch()
