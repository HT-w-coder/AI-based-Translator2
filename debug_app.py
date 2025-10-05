#!/usr/bin/env python3
"""
Simple translator app with basic functionality if transformers fail
"""

import streamlit as st
from langdetect import detect
import sys

st.set_page_config(
    page_title="AI Multilingual Translator - Debug Mode",
    page_icon="🌐",
    layout="wide"
)

def main():
    st.title("🌐 AI Multilingual Translator - Debug Mode")
    
    st.warning("Running in debug mode due to transformers import issues")
    
    # Test imports
    st.header("🔍 Import Status Check")
    
    imports_status = {}
    
    # Test basic imports
    try:
        import transformers
        imports_status['transformers'] = f"✅ {transformers.__version__}"
    except ImportError as e:
        imports_status['transformers'] = f"❌ {str(e)}"
    
    try:
        import torch
        imports_status['torch'] = f"✅ {torch.__version__}"
    except ImportError as e:
        imports_status['torch'] = f"❌ {str(e)}"
    
    try:
        from transformers import MarianMTModel, MarianTokenizer
        imports_status['MarianMT'] = "✅ Available"
    except ImportError as e:
        imports_status['MarianMT'] = f"❌ {str(e)}"
    
    try:
        from transformers import AutoModel, AutoTokenizer
        imports_status['AutoModel'] = "✅ Available"
    except ImportError as e:
        imports_status['AutoModel'] = f"❌ {str(e)}"
    
    # Display status
    for package, status in imports_status.items():
        st.write(f"**{package}**: {status}")
    
    st.write(f"**Python Version**: {sys.version}")
    
    # Basic language detection test
    st.header("🧪 Basic Language Detection Test")
    
    test_text = st.text_input("Enter text to detect language:", "Hello, how are you?")
    
    if test_text:
        try:
            detected_lang = detect(test_text)
            st.success(f"Detected language: **{detected_lang}**")
        except Exception as e:
            st.error(f"Language detection failed: {e}")
    
    # Installation help
    st.header("🔧 Installation Help")
    
    st.markdown("""
    ### Try these fixes in order:
    
    **1. Update transformers:**
    ```bash
    pip uninstall transformers tokenizers -y
    pip install transformers==4.21.3 tokenizers==0.13.3
    ```
    
    **2. Use compatible Python version:**
    ```bash
    # Python 3.13 has issues, use 3.11
    conda create -n translator python=3.11
    conda activate translator
    pip install transformers torch streamlit langdetect
    ```
    
    **3. Clean install:**
    ```bash
    pip cache purge
    pip install --no-cache-dir transformers torch
    ```
    
    **4. Minimal requirements:**
    ```bash
    pip install streamlit==1.28.0 transformers==4.21.3 torch langdetect numpy
    ```
    """)
    
    # Web Speech TTS test
    st.header("🎵 Text-to-Speech Test")
    
    tts_text = st.text_input("Text to speak:", "Hello world")
    
    if tts_text:
        # JavaScript for browser TTS
        tts_js = f"""
        <button onclick="
            const utterance = new SpeechSynthesisUtterance('{tts_text}');
            speechSynthesis.speak(utterance);
        " style="
            background: #667eea;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
        ">
            🔊 Speak with Browser TTS
        </button>
        """
        
        st.markdown(tts_js, unsafe_allow_html=True)
        st.info("Browser TTS should work even if transformers fails")

if __name__ == "__main__":
    main()