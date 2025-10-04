#!/usr/bin/env python3
"""
Multilingual Text-to-Speech Translator
A Flask-based web application for translating text between multiple languages
and converting the translated text to speech using offline AI models.
"""

import os
import tempfile
import threading
from pathlib import Path

from flask import Flask, render_template, request, jsonify, send_file
from transformers import MarianMTModel, MarianTokenizer, pipeline
import pyttsx3
from langdetect import detect, DetectorFactory
import torch

# Set seed for consistent language detection
DetectorFactory.seed = 0

app = Flask(__name__)

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
        
    def get_model_name(self, source_lang, target_lang):
        """Get the appropriate model name for translation."""
        # Special cases for some language pairs
        if source_lang == 'zh':
            source_lang = 'zh_cn'
        
        model_name = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
        return model_name
    
    def load_model(self, source_lang, target_lang):
        """Load translation model and tokenizer for given language pair."""
        model_key = f"{source_lang}_{target_lang}"
        
        if model_key not in self.models:
            try:
                model_name = self.get_model_name(source_lang, target_lang)
                print(f"Loading model: {model_name}")
                
                tokenizer = MarianTokenizer.from_pretrained(model_name)
                model = MarianMTModel.from_pretrained(model_name).to(self.device)
                
                self.tokenizers[model_key] = tokenizer
                self.models[model_key] = model
                
                print(f"Successfully loaded model for {source_lang} -> {target_lang}")
            except Exception as e:
                print(f"Error loading model for {source_lang} -> {target_lang}: {str(e)}")
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
                    return f"Translation not supported for {source_lang} -> {target_lang}"
            else:
                return f"Translation not supported for {source_lang} -> {target_lang}"
        
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
            print(f"Translation error: {str(e)}")
            return f"Translation error: {str(e)}"

class TextToSpeech:
    """Handles text-to-speech conversion using pyttsx3."""
    
    def __init__(self):
        self.engine = pyttsx3.init()
        self.setup_engine()
        
    def setup_engine(self):
        """Configure the TTS engine with default settings."""
        # Set properties
        self.engine.setProperty('rate', 150)  # Speed of speech
        self.engine.setProperty('volume', 0.9)  # Volume level (0.0 to 1.0)
        
        # Get available voices
        voices = self.engine.getProperty('voices')
        if voices:
            self.engine.setProperty('voice', voices[0].id)
    
    def get_available_voices(self):
        """Get list of available voices."""
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
        try:
            # Set appropriate voice for language
            self.set_voice_for_language(language_code)
            
            # Save to file
            self.engine.save_to_file(text, filename)
            self.engine.runAndWait()
            
            return True
        except Exception as e:
            print(f"TTS Error: {str(e)}")
            return False

# Initialize global instances
translator = MultilingualTranslator()
tts_engine = TextToSpeech()

@app.route('/')
def index():
    """Main page route."""
    return render_template('index.html')

@app.route('/api/detect-language', methods=['POST'])
def detect_language():
    """API endpoint to detect language of input text."""
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text.strip():
            return jsonify({'error': 'No text provided'}), 400
        
        # Detect language
        detected_lang = detect(text)
        
        return jsonify({
            'detected_language': detected_lang,
            'confidence': 'high'  # langdetect doesn't provide confidence scores
        })
        
    except Exception as e:
        return jsonify({'error': f'Language detection failed: {str(e)}'}), 500

@app.route('/api/translate', methods=['POST'])
def translate_text():
    """API endpoint to translate text."""
    try:
        data = request.get_json()
        text = data.get('text', '')
        source_lang = data.get('source_lang', 'auto')
        target_lang = data.get('target_lang', 'en')
        
        if not text.strip():
            return jsonify({'error': 'No text provided'}), 400
        
        # Auto-detect source language if needed
        if source_lang == 'auto':
            try:
                source_lang = detect(text)
            except:
                source_lang = 'en'  # Default to English if detection fails
        
        # Perform translation
        translated_text = translator.translate(text, source_lang, target_lang)
        
        return jsonify({
            'original_text': text,
            'translated_text': translated_text,
            'source_language': source_lang,
            'target_language': target_lang
        })
        
    except Exception as e:
        return jsonify({'error': f'Translation failed: {str(e)}'}), 500

@app.route('/api/speak', methods=['POST'])
def text_to_speech():
    """API endpoint to convert text to speech."""
    try:
        data = request.get_json()
        text = data.get('text', '')
        language = data.get('language', 'en')
        
        if not text.strip():
            return jsonify({'error': 'No text provided'}), 400
        
        # Create temporary file for audio
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            temp_filename = temp_file.name
        
        # Convert text to speech
        success = tts_engine.speak_to_file(text, temp_filename, language)
        
        if success and os.path.exists(temp_filename):
            def remove_file():
                """Remove temporary file after some delay."""
                import time
                time.sleep(10)  # Wait 10 seconds before cleanup
                try:
                    os.unlink(temp_filename)
                except:
                    pass
            
            # Start cleanup thread
            threading.Thread(target=remove_file, daemon=True).start()
            
            return send_file(
                temp_filename,
                as_attachment=True,
                download_name='speech.wav',
                mimetype='audio/wav'
            )
        else:
            return jsonify({'error': 'Failed to generate speech'}), 500
            
    except Exception as e:
        return jsonify({'error': f'Text-to-speech failed: {str(e)}'}), 500

@app.route('/api/supported-languages')
def get_supported_languages():
    """API endpoint to get list of supported languages."""
    languages = {
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
    
    return jsonify(languages)

@app.route('/api/voices')
def get_voices():
    """API endpoint to get available TTS voices."""
    voices = tts_engine.get_available_voices()
    return jsonify(voices)

if __name__ == '__main__':
    print("Starting Multilingual Text-to-Speech Translator...")
    print(f"Using device: {translator.device}")
    print("Available at: http://localhost:5000")
    
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    app.run(debug=True, host='0.0.0.0', port=5000)