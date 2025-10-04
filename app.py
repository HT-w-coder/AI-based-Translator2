import streamlit as st
from google.cloud import texttospeech
from google.cloud import translate_v2 as translate
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Set Google credentials from .env
GOOGLE_CREDS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
if not GOOGLE_CREDS:
    st.error("Please set GOOGLE_APPLICATION_CREDENTIALS in your .env file")
else:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_CREDS

st.set_page_config(page_title="Translator & TTS", page_icon="🌐")
st.title("🌐 Multilingual Translator & Text-to-Speech")

# Map language codes to Google TTS codes
language_codes = {
    "en": "en-US",
    "es": "es-ES",
    "fr": "fr-FR",
    "de": "de-DE",
    "hi": "hi-IN"
}

# Translate text
def translate_text(text, target_language):
    translate_client = translate.Client()
    result = translate_client.translate(text, target_language=target_language)
    return result['translatedText']

# Convert text to speech
def convert_text_to_speech_google_cloud(text, language_code):
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
    text = st.text_area("Enter text to translate:")
    target_language = st.selectbox(
        "Select target language:",
        options=list(language_codes.keys()),
        format_func=lambda x: x.upper()
    )
    submitted = st.form_submit_button("Translate & Listen")

if submitted and text.strip():
    with st.spinner("Translating..."):
        translated_text = translate_text(text, target_language)
    st.success("✅ Translation complete!")
    st.write(f"**Translated Text ({target_language.upper()}):** {translated_text}")

    with st.spinner("Converting to speech..."):
        language_code = language_codes[target_language]
        audio_content = convert_text_to_speech_google_cloud(translated_text, language_code)

    st.audio(audio_content, format="audio/mp3")
