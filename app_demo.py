import streamlit as st

# Configure Streamlit page (MUST be the very first Streamlit command)
st.set_page_config(
    page_title="AI Translator & TTS", 
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Demo mode - no Google Cloud credentials needed
DEMO_MODE = True

if DEMO_MODE:
    st.warning("🔧 **DEMO MODE** - Google Cloud credentials not configured. This is a UI preview only.")

# App header
st.title("🌐 AI-Based Translator & Text-to-Speech")
st.markdown("---")
st.markdown("### Translate text between languages and convert to speech using Google Cloud AI")

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

def demo_translate(text, target_language):
    """Demo translation function"""
    demo_translations = {
        "Hello, how are you?": {
            "Spanish": "Hola, ¿cómo estás?",
            "French": "Bonjour, comment allez-vous?",
            "German": "Hallo, wie geht es dir?",
            "Hindi": "नमस्ते, आप कैसे हैं?",
        }
    }
    
    if text in demo_translations and target_language in demo_translations[text]:
        return "en", demo_translations[text][target_language]
    else:
        return "en", f"[DEMO] Translation to {target_language}: {text}"

# Create two columns for better layout
col1, col2 = st.columns([2, 1])

with col1:
    # Streamlit form
    with st.form("translator_form"):
        st.subheader("📝 Enter Your Text")
        input_text = st.text_area(
            "Text to translate:",
            height=150,
            placeholder="Try: Hello, how are you?"
        )
        
        target_language = st.selectbox(
            "🎯 Select target language:",
            options=list(language_options.keys()),
            index=1  # Default to Spanish
        )
        
        submit = st.form_submit_button("🚀 Translate & Generate Speech", use_container_width=True)

with col2:
    st.subheader("ℹ️ About")
    st.info("""
    This app uses Google Cloud AI to:
    
    🔍 **Auto-detect** source language
    
    🌍 **Translate** to your target language
    
    🔊 **Generate** natural speech audio
    """)
    
    if DEMO_MODE:
        st.error("""
        **Setup Required:**
        
        1. Get Google Cloud service account JSON
        2. Update .env file with correct path
        3. Enable Translation & TTS APIs
        """)

# Process the translation
if submit and input_text.strip():
    if DEMO_MODE:
        # Demo functionality
        with st.spinner("🔄 Demo translating..."):
            detected_lang, translated_text = demo_translate(input_text, target_language)
        
        # Display results
        st.markdown("---")
        st.subheader("📋 Translation Results (DEMO)")
        
        col_result1, col_result2 = st.columns(2)
        
        with col_result1:
            st.success(f"✅ **Source Language Detected:** {detected_lang.upper()}")
            st.markdown(f"**Original Text:**")
            st.info(input_text)
        
        with col_result2:
            st.success(f"✅ **Translated to:** {target_language}")
            st.markdown(f"**Translated Text:**")
            st.info(translated_text)
        
        st.warning("🔊 **Audio generation requires Google Cloud TTS API setup**")
        
elif submit:
    st.warning("⚠️ Please enter some text to translate!")

# Footer
st.markdown("---")
st.markdown("**Powered by Google Cloud Translate & Text-to-Speech APIs**")