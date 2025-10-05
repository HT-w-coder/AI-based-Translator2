#!/usr/bin/env python3
"""
Test Web Speech API in Streamlit
"""

import streamlit as st

st.set_page_config(page_title="TTS Test", page_icon="🔊")

st.title("🔊 Browser Text-to-Speech Test")

# Enhanced JavaScript for TTS
st.markdown("""
<script>
function testTTS(text, lang) {
    console.log('Testing TTS with:', text, lang);
    
    if ('speechSynthesis' in window) {
        // Stop any ongoing speech
        window.speechSynthesis.cancel();
        
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.lang = lang || 'en-US';
        utterance.rate = 1;
        utterance.pitch = 1;
        utterance.volume = 1;
        
        // Wait for voices to load
        function speak() {
            const voices = window.speechSynthesis.getVoices();
            console.log('Available voices:', voices.length);
            
            if (voices.length > 0) {
                // Find best voice for language
                let voice = voices.find(v => v.lang === utterance.lang) || 
                           voices.find(v => v.lang.startsWith(utterance.lang.split('-')[0])) ||
                           voices[0];
                
                utterance.voice = voice;
                console.log('Using voice:', voice ? voice.name : 'default');
            }
            
            utterance.onstart = () => console.log('Speech started');
            utterance.onend = () => console.log('Speech ended');
            utterance.onerror = (e) => console.error('Speech error:', e);
            
            window.speechSynthesis.speak(utterance);
        }
        
        if (window.speechSynthesis.getVoices().length === 0) {
            window.speechSynthesis.addEventListener('voiceschanged', speak, {once: true});
        } else {
            speak();
        }
        
        return true;
    } else {
        alert('Speech synthesis not supported');
        return false;
    }
}
</script>
""", unsafe_allow_html=True)

# Test texts
test_texts = {
    'English': ('Hello! This is a test of the browser text-to-speech system.', 'en-US'),
    'Spanish': ('¡Hola! Esta es una prueba del sistema de texto a voz.', 'es-ES'),
    'French': ('Bonjour! Ceci est un test du système de synthèse vocale.', 'fr-FR'),
    'German': ('Hallo! Dies ist ein Test des Text-zu-Sprache-Systems.', 'de-DE'),
}

st.write("Click the buttons below to test browser TTS:")

for language, (text, lang_code) in test_texts.items():
    st.write(f"**{language}**: {text}")
    
    button_html = f"""
    <button onclick="testTTS('{text}', '{lang_code}')" 
            style="
                background: #667eea;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                cursor: pointer;
                margin: 5px;
            ">
        🔊 Speak {language}
    </button>
    """
    st.markdown(button_html, unsafe_allow_html=True)

st.markdown("---")

# Custom text test
st.subheader("Custom Text Test")
custom_text = st.text_area("Enter your own text:", "Type something here to test")
language_select = st.selectbox("Select language:", [
    ('English', 'en-US'),
    ('Spanish', 'es-ES'), 
    ('French', 'fr-FR'),
    ('German', 'de-DE'),
    ('Italian', 'it-IT'),
    ('Portuguese', 'pt-PT')
], format_func=lambda x: x[0])

if custom_text:
    clean_text = custom_text.replace("'", "\\'").replace('"', '\\"')
    lang_code = language_select[1]
    
    button_html = f"""
    <button onclick="testTTS('{clean_text}', '{lang_code}')" 
            style="
                background: #10b981;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 8px;
                cursor: pointer;
                font-size: 16px;
                font-weight: 500;
            ">
        🔊 Speak Custom Text
    </button>
    """
    st.markdown(button_html, unsafe_allow_html=True)

# Browser info
st.markdown("---")
st.subheader("Browser Compatibility")
st.markdown("""
<script>
document.write('<p><strong>Browser:</strong> ' + navigator.userAgent + '</p>');
document.write('<p><strong>Speech Synthesis Support:</strong> ' + ('speechSynthesis' in window ? '✅ Yes' : '❌ No') + '</p>');

if ('speechSynthesis' in window) {
    window.speechSynthesis.addEventListener('voiceschanged', function() {
        const voices = window.speechSynthesis.getVoices();
        document.write('<p><strong>Available Voices:</strong> ' + voices.length + '</p>');
        voices.slice(0, 5).forEach(voice => {
            document.write('<p>• ' + voice.name + ' (' + voice.lang + ')</p>');
        });
    });
    
    // Trigger voice loading
    window.speechSynthesis.getVoices();
}
</script>
""", unsafe_allow_html=True)

st.info("""
**Browser TTS should work in:**
- ✅ Chrome/Chromium (all platforms)
- ✅ Firefox (all platforms)  
- ✅ Safari (macOS/iOS)
- ✅ Edge (Windows)
- ✅ Mobile browsers (iOS Safari, Android Chrome)
""")

if st.button("🔄 Test Browser Compatibility"):
    st.markdown("""
    <script>
    if ('speechSynthesis' in window) {
        testTTS('Browser compatibility test successful!', 'en-US');
    } else {
        alert('Speech synthesis is not supported in this browser');
    }
    </script>
    """, unsafe_allow_html=True)