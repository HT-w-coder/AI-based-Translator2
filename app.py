import streamlit as st

# Configure Streamlit page (MUST be the very first Streamlit command)
st.set_page_config(
    page_title="AI Translator & TTS", 
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

from dotenv import load_dotenv
import os
import io

# Load environment variables
load_dotenv()

# Check Google credentials from .env
GOOGLE_CREDS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
credentials_available = False

if GOOGLE_CREDS and os.path.exists(GOOGLE_CREDS):
    try:
        # Try to validate it's a proper JSON file
        with open(GOOGLE_CREDS, 'r') as f:
            content = f.read()
        if '"type": "service_account"' in content:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_CREDS
            credentials_available = True
            # Import Google Cloud libraries only if credentials are available
            from google.cloud import texttospeech
            from google.cloud import translate_v2 as translate
    except Exception:
        pass

# App header
st.title("🌐 AI-Based Translator & Text-to-Speech")
st.markdown("---")

if not credentials_available:
    st.error("⚠️ **Google Cloud credentials not properly configured**")
    st.info("""
    **To enable full functionality:**
    
    1. **Get Google Cloud Service Account JSON** from https://console.cloud.google.com/
    2. **Enable APIs**: Cloud Translation API + Text-to-Speech API
    3. **Upload the JSON file** to your workspace
    4. **Update .env file** with: `GOOGLE_APPLICATION_CREDENTIALS=/workspace/your-file.json`
    
    **For now, you can use DEMO MODE below** ⬇️
    """)
    st.markdown("---")
    st.markdown("### 🎭 **DEMO MODE** - UI Preview")
else:
    st.markdown("### ✅ Translate text between languages and convert to speech using Google Cloud AI")

# Map language codes to Google TTS codes and display names
language_options = {
    "English": {"code": "en", "tts": "en-US"},
    "Spanish": {"code": "es", "tts": "es-ES"},
    "French": {"code": "fr", "tts": "fr-FR"},
    "German": {"code": "de", "tts": "de-DE"},
    "Hindi": {"code": "hi", "tts": "hi-IN"},
    "Italian": {"code": "it", "tts": "it-IT"},
    "Portuguese": {"code": "pt", "tts": "pt-BR"},
    "Japanese": {"code": "ja", "tts": "ja-JP"},
    "Korean": {"code": "ko", "tts": "ko-KR"},
    "Chinese (Simplified)": {"code": "zh", "tts": "zh-CN"}
}

def translate_text_auto(text, target_language_code):
    """Detects the source language and translates text."""
    if not credentials_available:
        return "en", f"[DEMO] {text} → {target_language_code.upper()}"
    
    try:
        translate_client = translate.Client()
        result = translate_client.translate(text, target_language=target_language_code)
        detected_lang = result['detectedSourceLanguage']
        translated_text = result['translatedText']
        return detected_lang, translated_text
    except Exception as e:
        st.error(f"Translation error: {str(e)}")
        return None, None

def convert_text_to_speech(text, language_code):
    """Convert text to speech using Google Cloud TTS."""
    if not credentials_available:
        return None
        
    try:
        client = texttospeech.TextToSpeechClient()
        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(
            language_code=language_code,
            ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3
        )
        response = client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )
        return response.audio_content
    except Exception as e:
        st.error(f"Text-to-speech error: {str(e)}")
        return None

# Demo translations for when credentials aren't available
demo_translations = {
    "hello": {"Spanish": "hola", "French": "bonjour", "German": "hallo", "Hindi": "नमस्ते"},
    "how are you": {"Spanish": "¿cómo estás?", "French": "comment allez-vous?", "German": "wie geht es dir?", "Hindi": "आप कैसे हैं?"},
    "thank you": {"Spanish": "gracias", "French": "merci", "German": "danke", "Hindi": "धन्यवाद"},
    "good morning": {"Spanish": "buenos días", "French": "bonjour", "German": "guten Morgen", "Hindi": "सुप्रभात"}
}

def demo_translate(text, target_language):
    """Demo translation for testing UI"""
    text_lower = text.lower().strip()
    if text_lower in demo_translations and target_language in demo_translations[text_lower]:
        return "en", demo_translations[text_lower][target_language]
    else:
        return "en", f"[DEMO Translation to {target_language}]: {text}"

# Create two columns for better layout
col1, col2 = st.columns([2, 1])

with col1:
    # Streamlit form
    with st.form("translator_form"):
        st.subheader("📝 Enter Your Text")
        
        if not credentials_available:
            st.info("💡 Try: 'hello', 'how are you', 'thank you', or 'good morning'")
            
        input_text = st.text_area(
            "Text to translate:",
            height=150,
            placeholder="Enter the text you want to translate..." if credentials_available else "Try: hello, how are you, thank you..."
        )
        
        target_language = st.selectbox(
            "🎯 Select target language:",
            options=list(language_options.keys()),
            index=1  # Default to Spanish
        )
        
        button_text = "🚀 Translate & Generate Speech" if credentials_available else "🎭 Demo Translation"
        submit = st.form_submit_button(button_text, use_container_width=True)

with col2:
    st.subheader("ℹ️ About")
    if credentials_available:
        st.success("""
        ✅ **Google Cloud Connected**
        
        🔍 Auto-detect source language
        🌍 Real translation via Google AI
        🔊 Generate natural speech audio
        """)
    else:
        st.warning("""
        ⚠️ **Demo Mode Active**
        
        🔍 Simulated language detection
        🌍 Sample translations only  
        🔊 Audio requires Google Cloud setup
        """)

# Process the translation
if submit and input_text.strip():
    target_lang_info = language_options[target_language]
    
    spinner_text = "🔄 Translating text..." if credentials_available else "🎭 Demo translating..."
    
    with st.spinner(spinner_text):
        if credentials_available:
            detected_lang, translated_text = translate_text_auto(input_text, target_lang_info["code"])
        else:
            detected_lang, translated_text = demo_translate(input_text, target_language)
    
    if translated_text:
        # Display results
        st.markdown("---")
        mode_text = "📋 Translation Results" if credentials_available else "📋 Demo Translation Results"
        st.subheader(mode_text)
        
        col_result1, col_result2 = st.columns(2)
        
        with col_result1:
            st.success(f"✅ **Source Language Detected:** {detected_lang.upper()}")
            st.markdown("**Original Text:**")
            st.info(input_text)
        
        with col_result2:
            st.success(f"✅ **Translated to:** {target_language}")
            st.markdown("**Translated Text:**")
            st.info(translated_text)
        
        # Generate speech (only if credentials available)
        if credentials_available:
            with st.spinner("🎵 Converting to speech..."):
                tts_code = target_lang_info["tts"]
                audio_content = convert_text_to_speech(translated_text, tts_code)
            
            if audio_content:
                st.subheader("🔊 Audio Output")
                st.audio(audio_content, format="audio/mp3")
                
                # Download button for audio
                st.download_button(
                    label="📥 Download Audio",
                    data=audio_content,
                    file_name=f"translation_{target_lang_info['code']}.mp3",
                    mime="audio/mp3"
                )
        else:
            st.info("🔊 **Audio generation requires Google Cloud Text-to-Speech API setup**")
    
elif submit:
    st.warning("⚠️ Please enter some text to translate!")

# Footer
st.markdown("---")
if credentials_available:
    st.markdown("**✅ Powered by Google Cloud Translate & Text-to-Speech APIs**")
else:
    st.markdown("**🎭 Demo Mode - Setup Google Cloud for full functionality**")