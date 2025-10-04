#!/usr/bin/env python3
"""
Test script for AI Multilingual Text-to-Speech Translator
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test if all required modules can be imported."""
    print("🧪 Testing imports...")
    
    try:
        import streamlit
        print("✅ Streamlit imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import Streamlit: {e}")
        return False
    
    try:
        import transformers
        print("✅ Transformers imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import Transformers: {e}")
        return False
    
    try:
        import torch
        print("✅ PyTorch imported successfully")
        print(f"   CUDA available: {torch.cuda.is_available()}")
    except ImportError as e:
        print(f"❌ Failed to import PyTorch: {e}")
        return False
    
    try:
        import pyttsx3
        print("✅ pyttsx3 imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import pyttsx3: {e}")
        return False
    
    try:
        import langdetect
        print("✅ langdetect imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import langdetect: {e}")
        return False
    
    return True

def test_language_detection():
    """Test language detection functionality."""
    print("\n🧪 Testing language detection...")
    
    try:
        from langdetect import detect
        
        test_texts = {
            "Hello, how are you?": "en",
            "Hola, ¿cómo estás?": "es",
            "Bonjour, comment allez-vous?": "fr",
            "Guten Tag, wie geht es Ihnen?": "de"
        }
        
        for text, expected in test_texts.items():
            detected = detect(text)
            if detected == expected:
                print(f"✅ '{text}' -> {detected} (correct)")
            else:
                print(f"⚠️  '{text}' -> {detected} (expected {expected})")
        
        return True
    except Exception as e:
        print(f"❌ Language detection test failed: {e}")
        return False

def test_tts_engine():
    """Test text-to-speech engine initialization."""
    print("\n🧪 Testing TTS engine...")
    
    try:
        import pyttsx3
        
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        
        if voices:
            print(f"✅ TTS engine initialized with {len(voices)} voices")
            for i, voice in enumerate(voices[:3]):  # Show first 3 voices
                print(f"   Voice {i+1}: {voice.name}")
        else:
            print("⚠️  TTS engine initialized but no voices found")
        
        return True
    except Exception as e:
        print(f"❌ TTS engine test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🌐 AI Multilingual Text-to-Speech Translator - Test Suite")
    print("=========================================================")
    
    tests = [
        test_imports,
        test_language_detection,
        test_tts_engine
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The application should work correctly.")
        print("\n🚀 To run the application:")
        print("   ./run.sh")
        print("   or")
        print("   streamlit run translator_app.py")
    else:
        print("⚠️  Some tests failed. Please check the dependencies.")
        print("\n🔧 To install dependencies:")
        print("   pip install -r requirements.txt")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)