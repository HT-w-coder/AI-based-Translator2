import streamlit as st
from google.cloud import texttospeech
from google.cloud import translate_v2 as translate
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Set Google credentials from .env
GOOGLE_CREDS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
if not GOOGLE_CREDS or not os.path.exists(GOOGLE_CREDS):
    st.error("Service account JSON file not found. Check .env!")
else:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_CREDS

st.set_page_config(page_title="AI Translator & TTS", page_icon="🌐")
st.title("🌐 AI-Based Translator & Text-to-Speech")

# Map language codes to Google TTS codes
language_codes = {
    "en": "en-US",
    "es": "es-ES",
    "fr": "fr-FR",
    "de": "de-DE",
    "hi": "hi-IN"
}

def translate_text_auto(text, target_language):
    """Detects the source language and translates text."""
    translate_client = translate.Client()
    result = translate_client.translate(text, target_language=target_language)
    detected_lang = result['detectedSourceLanguage']
    translated_text = result['translatedText']
    return detected_lang, translated_text

def convert_text_to_speech(text, language_code):
    """Convert text to speech using Google Cloud TTS."""
    client = texttospeech.TextToSpeechClient()
    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code=language_code,
        ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
    )
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
    response = client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )
    return response.audio_content

# Streamlit form
with st.form("translator_form"):
    input_text = st.text_area("Enter text to translate:")
    target_language = st.selectbox(
        "Select target language:",
        options=list(language_codes.keys()),
        format_func=lambda x: x.upper()
    )
    submit = st.form_submit_button("Translate & Listen")

if submit and input_text.strip():
    with st.spinner("Translating..."):
        detected_lang, translated_text = translate_text_auto(input_text, target_language)
    st.success(f"✅ Translation complete! Detected source language: {detected_lang.upper()}")
    st.write(f"**Translated Text ({target_language.upper()}):** {translated_text}")

    with st.spinner("Converting to speech..."):
        tts_code = language_codes[target_language]
        audio_content = convert_text_to_speech(translated_text, tts_code)

    st.audio(audio_content, format="audio/mp3")
