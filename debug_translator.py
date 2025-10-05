#!/usr/bin/env python3
"""
Debug version of the translator app with comprehensive error logging
"""

import streamlit as st
import traceback
import sys
import os

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(
    page_title="AI Translator - Debug Mode", 
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 AI Translator - Debug Mode")
st.warning("This is the debug version. All errors will be displayed with full details.")

# Global error handler
def handle_error(error, context="Unknown"):
    """Handle and display errors with full details."""
    st.error(f"❌ Error in {context}")
    
    with st.expander("🔍 Full Error Details"):
        st.write(f"**Error Type:** {type(error).__name__}")
        st.write(f"**Error Message:** {str(error)}")
        st.write(f"**Context:** {context}")
        
        # Full traceback
        st.code(traceback.format_exc())
        
        # System info
        st.write("**System Information:**")
        st.write(f"- Python Version: {sys.version}")
        st.write(f"- Streamlit Version: {st.__version__}")
        
        # Session state
        st.write("**Session State Keys:**")
        st.write(list(st.session_state.keys()))

# Test basic functionality
st.header("🧪 Basic Functionality Tests")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Import Tests")
    
    # Test imports one by one
    imports_to_test = [
        ('streamlit', 'st'),
        ('transformers', 'transformers'),
        ('torch', 'torch'),
        ('langdetect', 'langdetect'),
        ('numpy', 'np'),
        ('time', 'time'),
        ('tempfile', 'tempfile'),
        ('pathlib', 'Path'),
        ('io', 'io'),
        ('base64', 'base64'),
        ('json', 'json')
    ]
    
    for module_name, import_alias in imports_to_test:
        try:
            exec(f"import {module_name}")
            st.success(f"✅ {module_name}")
        except ImportError as e:
            st.error(f"❌ {module_name}: {e}")

with col2:
    st.subheader("Session State Test")
    
    # Test session state operations
    try:
        # Initialize test values
        if 'test_counter' not in st.session_state:
            st.session_state.test_counter = 0
            
        if st.button("Increment Counter"):
            st.session_state.test_counter += 1
            
        st.write(f"Counter: {st.session_state.test_counter}")
        st.success("✅ Session state working")
        
    except Exception as e:
        handle_error(e, "Session State Test")

# Test the main translator app components
st.header("🔧 Translator Components Test")

try:
    # Try to import the main translator
    from translator_app import MultilingualTranslator, TextToSpeech, detect_language, TRANSFORMERS_AVAILABLE
    
    st.success("✅ Main app components imported successfully")
    
    # Test translator initialization
    if st.button("Test Translator Initialization"):
        try:
            if TRANSFORMERS_AVAILABLE:
                translator = MultilingualTranslator()
                st.success("✅ Translator initialized")
                
                # Show supported languages
                st.write("**Supported Languages:**")
                st.write(translator.language_names)
            else:
                st.error("❌ Transformers not available")
                
        except Exception as e:
            handle_error(e, "Translator Initialization")
    
    # Test TTS initialization
    if st.button("Test TTS Initialization"):
        try:
            tts = TextToSpeech()
            st.success("✅ TTS initialized")
        except Exception as e:
            handle_error(e, "TTS Initialization")
    
    # Test language detection
    test_text = st.text_input("Test Language Detection:", "Hello world")
    if test_text:
        try:
            detected = detect_language(test_text)
            st.success(f"✅ Detected language: {detected}")
        except Exception as e:
            handle_error(e, "Language Detection")

except ImportError as e:
    handle_error(e, "Main App Import")

# Manual translation test
st.header("🌍 Manual Translation Test")

if TRANSFORMERS_AVAILABLE:
    text_to_translate = st.text_area("Enter text to translate:", "Hello, how are you?")
    
    col1, col2 = st.columns(2)
    with col1:
        source_lang = st.selectbox("From:", ['en', 'es', 'fr', 'de', 'it'])
    with col2:
        target_lang = st.selectbox("To:", ['en', 'es', 'fr', 'de', 'it'])
    
    if st.button("🔄 Test Translation"):
        try:
            translator = MultilingualTranslator()
            
            with st.spinner("Translating..."):
                result = translator.translate(text_to_translate, source_lang, target_lang)
                
            if result.startswith("❌"):
                st.error(result)
            else:
                st.success("✅ Translation successful!")
                st.write(f"**Result:** {result}")
                
        except Exception as e:
            handle_error(e, "Manual Translation")

# Browser TTS Test
st.header("🔊 Browser TTS Test")

tts_text = st.text_input("Text to speak:", "Hello, this is a test")

if st.button("🔊 Test Browser TTS"):
    try:
        clean_text = tts_text.replace("'", "\\'").replace('"', '\\"')
        
        tts_script = f"""
        <script>
        console.log('Testing TTS...');
        if ('speechSynthesis' in window) {{
            const utterance = new SpeechSynthesisUtterance('{clean_text}');
            utterance.onstart = () => console.log('Speech started');
            utterance.onend = () => console.log('Speech ended');
            utterance.onerror = (e) => console.error('Speech error:', e);
            window.speechSynthesis.speak(utterance);
        }} else {{
            alert('TTS not supported');
        }}
        </script>
        """
        
        st.components.v1.html(tts_script, height=0)
        st.success("✅ TTS command sent")
        
    except Exception as e:
        handle_error(e, "Browser TTS Test")

# Error simulation
st.header("⚠️ Error Simulation")

if st.button("Simulate KeyError"):
    try:
        test_dict = {'a': 1}
        _ = test_dict['nonexistent_key']
    except Exception as e:
        handle_error(e, "Simulated KeyError")

if st.button("Simulate AttributeError"):
    try:
        test_obj = None
        _ = test_obj.some_attribute
    except Exception as e:
        handle_error(e, "Simulated AttributeError")

st.markdown("---")
st.info("💡 Use this debug mode to identify exactly what's causing errors after translation.")