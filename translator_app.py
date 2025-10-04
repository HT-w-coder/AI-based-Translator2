#!/usr/bin/env python3
"""
Multilingual Text-to-Speech Translator - Streamlit App
A Streamlit-based web application for translating text between multiple languages
and converting the translated text to speech using offline AI models.
"""

import os
import tempfile
import time
from pathlib import Path
import io

import streamlit as st
from transformers import MarianMTModel, MarianTokenizer
import pyttsx3
from langdetect import detect, DetectorFactory
import torch
import plotly.graph_objects as go
import plotly.express as px

# Set seed for consistent language detection
DetectorFactory.seed = 0

# Page configuration
st.set_page_config(
    page_title="AI Multilingual Translator",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin: -1rem -1rem 2rem -1rem;
        border-radius: 0 0 10px 10px;
    }
    
    .stSelectbox > label {
        font-weight: 600 !important;
        color: #1f2937 !important;
    }
    
    .stTextArea > label {
        font-weight: 600 !important;
        color: #1f2937 !important;
    }
    
    .translation-box {
        background: #f8fafc;
        border: 2px dashed #cbd5e1;
        border-radius: 8px;
        padding: 1rem;
        min-height: 150px;
        font-size: 16px;
        line-height: 1.6;
    }
    
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .success-message {
        background: #dcfce7;
        border: 1px solid #16a34a;
        color: #15803d;
        padding: 0.75rem;
        border-radius: 6px;
        margin: 0.5rem 0;
    }
    
    .error-message {
        background: #fef2f2;
        border: 1px solid #ef4444;
        color: #dc2626;
        padding: 0.75rem;
        border-radius: 6px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

class MultilingualTranslator:
    """Handles text translation between multiple languages using Hugging Face models."""
    
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Supported language pairs for translation
        self.supported_pairs = {
            'en': ['es', 'fr', 'de', 'it', 'pt', 'ru', 'zh', 'ja', 'ko', 'ar', 'hi'],
            'es': ['en', 'fr', 'de', 'it', 'pt'],
            'fr': ['en', 'es', 'de', 'it', 'pt'],
            'de': ['en', 'es', 'fr', 'it', 'pt'],
            'it': ['en', 'es', 'fr', 'de', 'pt'],
            'pt': ['en', 'es', 'fr', 'de', 'it'],
            'ru': ['en'],
            'zh': ['en'],
            'ja': ['en'],
            'ko': ['en'],
            'ar': ['en'],
            'hi': ['en']
        }
        
        # Language names
        self.language_names = {
            'en': 'English',
            'es': 'Spanish',
            'fr': 'French',
            'de': 'German',
            'it': 'Italian',
            'pt': 'Portuguese',
            'ru': 'Russian',
            'zh': 'Chinese',
            'ja': 'Japanese',
            'ko': 'Korean',
            'ar': 'Arabic',
            'hi': 'Hindi'
        }
        
    def get_model_name(self, source_lang, target_lang):
        """Get the appropriate model name for translation."""
        if source_lang == 'zh':
            source_lang = 'zh_cn'
        
        model_name = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
        return model_name
    
    @st.cache_resource
    def load_model(_self, source_lang, target_lang):
        """Load translation model and tokenizer for given language pair."""
        model_key = f"{source_lang}_{target_lang}"
        
        if model_key not in _self.models:
            try:
                model_name = _self.get_model_name(source_lang, target_lang)
                
                with st.spinner(f"Loading model for {source_lang} → {target_lang}..."):
                    tokenizer = MarianTokenizer.from_pretrained(model_name)
                    model = MarianMTModel.from_pretrained(model_name).to(_self.device)
                
                _self.tokenizers[model_key] = tokenizer
                _self.models[model_key] = model
                
                return True
            except Exception as e:
                st.error(f"Error loading model for {source_lang} → {target_lang}: {str(e)}")
                return False
        
        return True
    
    def translate(self, text, source_lang, target_lang):
        """Translate text from source language to target language."""
        if source_lang == target_lang:
            return text
        
        # Check if translation pair is supported
        if source_lang not in self.supported_pairs or target_lang not in self.supported_pairs[source_lang]:
            # Try reverse translation through English
            if source_lang != 'en' and target_lang != 'en':
                try:
                    # Translate to English first
                    english_text = self.translate(text, source_lang, 'en')
                    # Then translate from English to target
                    return self.translate(english_text, 'en', target_lang)
                except:
                    return f"Translation not supported for {source_lang} → {target_lang}"
            else:
                return f"Translation not supported for {source_lang} → {target_lang}"
        
        # Load model if not already loaded
        if not self.load_model(source_lang, target_lang):
            return "Error: Could not load translation model"
        
        model_key = f"{source_lang}_{target_lang}"
        
        try:
            # Tokenize input text
            tokenizer = self.tokenizers[model_key]
            model = self.models[model_key]
            
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate translation
            with torch.no_grad():
                translated = model.generate(**inputs, max_new_tokens=512, do_sample=True, temperature=0.7)
            
            # Decode the translation
            translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
            return translated_text
            
        except Exception as e:
            return f"Translation error: {str(e)}"

class TextToSpeech:
    """Handles text-to-speech conversion using pyttsx3."""
    
    def __init__(self):
        try:
            self.engine = pyttsx3.init()
            self.setup_engine()
            self.available = True
        except Exception as e:
            st.warning(f"TTS engine initialization failed: {e}")
            self.available = False
        
    def setup_engine(self):
        """Configure the TTS engine with default settings."""
        if not self.available:
            return
            
        # Set properties
        self.engine.setProperty('rate', 150)  # Speed of speech
        self.engine.setProperty('volume', 0.9)  # Volume level (0.0 to 1.0)
        
        # Get available voices
        voices = self.engine.getProperty('voices')
        if voices:
            self.engine.setProperty('voice', voices[0].id)
    
    def get_available_voices(self):
        """Get list of available voices."""
        if not self.available:
            return []
            
        voices = self.engine.getProperty('voices')
        voice_list = []
        
        for voice in voices:
            voice_info = {
                'id': voice.id,
                'name': voice.name,
                'languages': getattr(voice, 'languages', [])
            }
            voice_list.append(voice_info)
        
        return voice_list
    
    def set_voice_for_language(self, language_code):
        """Set appropriate voice for the given language."""
        if not self.available:
            return False
            
        voices = self.engine.getProperty('voices')
        
        # Language mapping for voice selection
        language_voice_mapping = {
            'en': ['english', 'en_'],
            'es': ['spanish', 'es_'],
            'fr': ['french', 'fr_'],
            'de': ['german', 'de_'],
            'it': ['italian', 'it_'],
            'pt': ['portuguese', 'pt_'],
            'ru': ['russian', 'ru_'],
            'zh': ['chinese', 'zh_', 'mandarin'],
            'ja': ['japanese', 'ja_'],
            'ko': ['korean', 'ko_'],
            'ar': ['arabic', 'ar_'],
            'hi': ['hindi', 'hi_']
        }
        
        if language_code in language_voice_mapping:
            search_terms = language_voice_mapping[language_code]
            
            for voice in voices:
                voice_name_lower = voice.name.lower()
                if any(term in voice_name_lower for term in search_terms):
                    self.engine.setProperty('voice', voice.id)
                    return True
        
        # Default to first available voice if no match found
        if voices:
            self.engine.setProperty('voice', voices[0].id)
        
        return False
    
    def speak_to_file(self, text, filename, language_code='en'):
        """Convert text to speech and save to file."""
        if not self.available:
            return False
            
        try:
            # Set appropriate voice for language
            self.set_voice_for_language(language_code)
            
            # Save to file
            self.engine.save_to_file(text, filename)
            self.engine.runAndWait()
            
            return True
        except Exception as e:
            st.error(f"TTS Error: {str(e)}")
            return False

# Initialize session state
if 'translator' not in st.session_state:
    st.session_state.translator = MultilingualTranslator()
    
if 'tts_engine' not in st.session_state:
    st.session_state.tts_engine = TextToSpeech()
    
if 'translation_history' not in st.session_state:
    st.session_state.translation_history = []

def detect_language(text):
    """Detect language of input text."""
    try:
        detected_lang = detect(text)
        return detected_lang
    except:
        return 'en'  # Default to English if detection fails

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🌐 AI Multilingual Translator</h1>
        <p>Translate text and convert to speech in multiple languages - Completely Offline!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Settings")
        
        # Device info
        device = "🖥️ CPU" if st.session_state.translator.device == "cpu" else "🚀 GPU"
        st.info(f"**Device:** {device}")
        
        # TTS status
        tts_status = "✅ Available" if st.session_state.tts_engine.available else "❌ Unavailable"
        st.info(f"**Text-to-Speech:** {tts_status}")
        
        # Language statistics
        st.subheader("📊 Supported Languages")
        languages = st.session_state.translator.language_names
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Languages", len(languages))
        with col2:
            st.metric("Translation Models", "Helsinki-NLP")
        
        # Language list
        with st.expander("View All Languages"):
            for code, name in languages.items():
                st.write(f"🗣️ **{name}** ({code})")
        
        # Clear history
        if st.button("🗑️ Clear History"):
            st.session_state.translation_history = []
            st.success("History cleared!")
    
    # Main content
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.header("🔄 Translation")
        
        # Language selection
        languages = st.session_state.translator.language_names
        lang_options = ["Auto-detect"] + list(languages.values())
        lang_codes = ["auto"] + list(languages.keys())
        
        lcol1, lcol2, lcol3 = st.columns([2, 1, 2])
        
        with lcol1:
            source_lang_idx = st.selectbox(
                "From Language",
                range(len(lang_options)),
                format_func=lambda x: lang_options[x],
                key="source_lang"
            )
            source_lang = lang_codes[source_lang_idx]
        
        with lcol2:
            if st.button("⇄", help="Swap languages", key="swap_btn"):
                if source_lang != "auto":
                    # Swap logic
                    current_target = st.session_state.get("target_lang", 1)
                    current_source = st.session_state.get("source_lang", 0)
                    
                    if current_source != 0:  # Not auto-detect
                        st.session_state["source_lang"] = current_target
                        st.session_state["target_lang"] = current_source
                        st.rerun()
                else:
                    st.warning("Cannot swap when auto-detect is selected")
        
        with lcol2:
            target_lang_idx = st.selectbox(
                "To Language",
                range(1, len(lang_options)),  # Exclude auto-detect for target
                format_func=lambda x: lang_options[x],
                index=0,  # Default to Spanish
                key="target_lang"
            )
            target_lang = lang_codes[target_lang_idx]
        
        # Text input
        input_text = st.text_area(
            "Enter text to translate",
            height=200,
            max_chars=5000,
            placeholder="Type your text here...",
            key="input_text"
        )
        
        # Character count and detected language
        char_count = len(input_text)
        col_info1, col_info2 = st.columns([1, 1])
        
        with col_info1:
            st.caption(f"Characters: {char_count}/5000")
        
        with col_info2:
            if input_text and source_lang == "auto":
                detected = detect_language(input_text)
                detected_name = languages.get(detected, detected)
                st.caption(f"🔍 Detected: {detected_name}")
        
        # Translation button
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
        
        with col_btn1:
            translate_btn = st.button(
                "🔄 Translate",
                type="primary",
                disabled=not input_text.strip(),
                use_container_width=True
            )
        
        with col_btn2:
            speak_original_btn = st.button(
                "🎵 Speak Original",
                disabled=not input_text.strip() or not st.session_state.tts_engine.available,
                use_container_width=True
            )
        
        # Translation logic
        if translate_btn and input_text.strip():
            with st.spinner("Translating..."):
                # Detect source language if auto
                actual_source_lang = source_lang
                if source_lang == "auto":
                    actual_source_lang = detect_language(input_text)
                
                # Perform translation
                translated_text = st.session_state.translator.translate(
                    input_text, actual_source_lang, target_lang
                )
                
                # Display translation
                st.subheader("📝 Translation Result")
                st.markdown(f'<div class="translation-box">{translated_text}</div>', unsafe_allow_html=True)
                
                # Add to history
                history_item = {
                    'timestamp': time.time(),
                    'original': input_text,
                    'translated': translated_text,
                    'source_lang': actual_source_lang,
                    'target_lang': target_lang,
                    'source_name': languages.get(actual_source_lang, actual_source_lang),
                    'target_name': languages.get(target_lang, target_lang)
                }
                st.session_state.translation_history.insert(0, history_item)
                
                # Keep only last 10 translations
                if len(st.session_state.translation_history) > 10:
                    st.session_state.translation_history = st.session_state.translation_history[:10]
                
                # Copy and speak buttons
                col_action1, col_action2 = st.columns([1, 1])
                
                with col_action1:
                    st.code(translated_text, language=None)
                
                with col_action2:
                    if st.button("🎵 Speak Translation", key="speak_translation"):
                        if st.session_state.tts_engine.available:
                            with st.spinner("Generating speech..."):
                                # Create temporary file
                                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                                    success = st.session_state.tts_engine.speak_to_file(
                                        translated_text, tmp_file.name, target_lang
                                    )
                                    
                                    if success:
                                        # Read the audio file
                                        with open(tmp_file.name, 'rb') as audio_file:
                                            audio_bytes = audio_file.read()
                                        
                                        st.audio(audio_bytes, format='audio/wav')
                                        
                                        # Clean up
                                        os.unlink(tmp_file.name)
                                    else:
                                        st.error("Failed to generate speech")
                        else:
                            st.error("Text-to-speech not available")
        
        # Handle speak original
        if speak_original_btn and input_text.strip():
            if st.session_state.tts_engine.available:
                with st.spinner("Generating speech..."):
                    actual_source_lang = source_lang
                    if source_lang == "auto":
                        actual_source_lang = detect_language(input_text)
                    
                    # Create temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                        success = st.session_state.tts_engine.speak_to_file(
                            input_text, tmp_file.name, actual_source_lang
                        )
                        
                        if success:
                            # Read the audio file
                            with open(tmp_file.name, 'rb') as audio_file:
                                audio_bytes = audio_file.read()
                            
                            st.audio(audio_bytes, format='audio/wav')
                            
                            # Clean up
                            os.unlink(tmp_file.name)
                        else:
                            st.error("Failed to generate speech")
            else:
                st.error("Text-to-speech not available")
    
    with col2:
        st.header("📋 Translation History")
        
        if st.session_state.translation_history:
            for i, item in enumerate(st.session_state.translation_history):
                with st.expander(f"{item['source_name']} → {item['target_name']}", expanded=i==0):
                    st.write(f"**Original:** {item['original'][:100]}...")
                    st.write(f"**Translation:** {item['translated'][:100]}...")
                    
                    timestamp = time.strftime('%H:%M:%S', time.localtime(item['timestamp']))
                    st.caption(f"🕐 {timestamp}")
                    
                    # Quick actions
                    col_hist1, col_hist2 = st.columns([1, 1])
                    with col_hist1:
                        if st.button("📋 Copy", key=f"copy_{i}"):
                            st.code(item['translated'])
                    
                    with col_hist2:
                        if st.button("🎵 Speak", key=f"speak_hist_{i}"):
                            if st.session_state.tts_engine.available:
                                with st.spinner("Generating speech..."):
                                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                                        success = st.session_state.tts_engine.speak_to_file(
                                            item['translated'], tmp_file.name, item['target_lang']
                                        )
                                        
                                        if success:
                                            with open(tmp_file.name, 'rb') as audio_file:
                                                audio_bytes = audio_file.read()
                                            
                                            st.audio(audio_bytes, format='audio/wav')
                                            os.unlink(tmp_file.name)
        else:
            st.info("No translations yet. Start translating to see history here!")
    
    # Features section
    st.markdown("---")
    st.header("🌟 Features")
    
    feat_col1, feat_col2, feat_col3, feat_col4 = st.columns(4)
    
    with feat_col1:
        st.markdown("""
        <div class="feature-card">
            <h4>🌍 12+ Languages</h4>
            <p>Support for major world languages including English, Spanish, French, German, Italian, Portuguese, Russian, Chinese, Japanese, Korean, Arabic, and Hindi</p>
        </div>
        """, unsafe_allow_html=True)
    
    with feat_col2:
        st.markdown("""
        <div class="feature-card">
            <h4>🤖 AI-Powered</h4>
            <p>Uses advanced Transformer models from Hugging Face for accurate translations without requiring internet connection</p>
        </div>
        """, unsafe_allow_html=True)
    
    with feat_col3:
        st.markdown("""
        <div class="feature-card">
            <h4>🎵 Text-to-Speech</h4>
            <p>Convert translated text to natural-sounding speech with voice selection based on target language</p>
        </div>
        """, unsafe_allow_html=True)
    
    with feat_col4:
        st.markdown("""
        <div class="feature-card">
            <h4>🔒 Privacy First</h4>
            <p>All processing happens locally on your machine. No data is sent to external services or stored remotely</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()