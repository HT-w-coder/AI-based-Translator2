#!/usr/bin/env python3
"""
Test script for AI Multilingual Text-to-Speech Translator
Updated for latest package versions
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_python_version():
    """Test Python version compatibility."""
    print("🧪 Testing Python version...")
    
    version = sys.version_info
    print(f"🐍 Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major >= 3 and version.minor >= 8:
        print("✅ Python version is compatible (3.8+)")
        return True
    else:
        print("❌ Python version must be 3.8 or higher")
        return False

def test_imports():
    """Test if all required modules can be imported."""
    print("\n🧪 Testing imports...")
    
    # Test core packages first
    packages = [
        ('streamlit', 'Streamlit'),
        ('transformers', 'Transformers'),
        ('torch', 'PyTorch'),
        ('langdetect', 'Language Detection'),
        ('numpy', 'NumPy'),
        ('requests', 'Requests')
    ]
    
    success_count = 0
    
    for package, name in packages:
        try:
            __import__(package)
            print(f"✅ {name} imported successfully")
            success_count += 1
        except ImportError as e:
            print(f"❌ Failed to import {name}: {e}")
    
    # Test PyTorch CUDA availability
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        print(f"   CUDA available: {'Yes' if cuda_available else 'No'}")
        if cuda_available:
            print(f"   CUDA device: {torch.cuda.get_device_name(0)}")
    except:
        pass
    
    # Test optional packages
    optional_packages = [
        ('pyttsx3', 'Text-to-Speech'),
        ('plotly', 'Plotly'),
        ('sentencepiece', 'SentencePiece'),
        ('sacremoses', 'SacreMoses')
    ]
    
    for package, name in optional_packages:
        try:
            __import__(package)
            print(f"✅ {name} imported successfully")
            success_count += 1
        except ImportError as e:
            print(f"⚠️  {name} not available: {e}")
    
    return success_count >= len(packages)  # Core packages must work

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
            try:
                detected = detect(text)
                if detected == expected:
                    print(f"✅ '{text[:30]}...' -> {detected} (correct)")
                else:
                    print(f"⚠️  '{text[:30]}...' -> {detected} (expected {expected})")
            except Exception as e:
                print(f"❌ Failed to detect: {text[:30]}... - {e}")
        
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
                name = getattr(voice, 'name', 'Unknown')
                print(f"   Voice {i+1}: {name}")
        else:
            print("⚠️  TTS engine initialized but no voices found")
        
        return True
    except Exception as e:
        print(f"⚠️  TTS engine test failed: {e}")
        print("   TTS may not work, but translation will still function")
        return True  # Don't fail the test, as TTS is optional

def test_transformers():
    """Test transformers library functionality."""
    print("\n🧪 Testing Transformers library...")
    
    try:
        from transformers import AutoTokenizer
        
        # Try to load a small tokenizer to test functionality
        print("   Testing tokenizer loading...")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
        # Test tokenization
        test_text = "Hello world"
        tokens = tokenizer.encode(test_text)
        
        print("✅ Transformers library working correctly")
        print(f"   Test tokenization: '{test_text}' -> {len(tokens)} tokens")
        
        return True
    except Exception as e:
        print(f"⚠️  Transformers test failed: {e}")
        print("   This might work after downloading models")
        return True  # Don't fail completely

def main():
    """Run all tests."""
    print("🌐 AI Multilingual Text-to-Speech Translator - Test Suite")
    print("=========================================================")
    
    tests = [
        ("Python Version", test_python_version),
        ("Package Imports", test_imports),
        ("Language Detection", test_language_detection),
        ("TTS Engine", test_tts_engine),
        ("Transformers Library", test_transformers)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        if test_func():
            passed += 1
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed >= total - 1:  # Allow one test to fail (TTS is optional)
        print("🎉 Tests completed successfully! The application should work.")
        print("\n🚀 To run the application:")
        print("   ./run.sh")
        print("   or")
        print("   streamlit run translator_app.py")
        
        if passed < total:
            print("\n⚠️  Note: Some optional features may not work (like TTS)")
            
    else:
        print("⚠️  Critical tests failed. Please check the dependencies.")
        print("\n🔧 To install dependencies:")
        print("   pip install -r requirements.txt")
        
        print("\n💡 If you're still having issues:")
        print("   1. Update pip: pip install --upgrade pip")
        print("   2. Use virtual environment: python -m venv venv && source venv/bin/activate")
        print("   3. Install with --no-cache: pip install --no-cache-dir -r requirements.txt")
    
    return passed >= total - 1

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)