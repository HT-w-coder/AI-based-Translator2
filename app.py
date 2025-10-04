(cd "$(git rev-parse --show-toplevel)" && git apply --3way <<'EOF'
diff --git a/app.py b/app.py
--- a/app.py
+++ b/app.py
@@ -1,72 +1,161 @@
-import streamlit as st
-from google.cloud import texttospeech
-from google.cloud import translate_v2 as translate
-from dotenv import load_dotenv
-import os
-
-# Load environment variables
-load_dotenv()
-
-# Set Google credentials from .env
-GOOGLE_CREDS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
-if not GOOGLE_CREDS or not os.path.exists(GOOGLE_CREDS):
-    st.error("Service account JSON file not found. Check .env!")
-else:
-    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_CREDS
-
-st.set_page_config(page_title="AI Translator & TTS", page_icon="🌐")
-st.title("🌐 AI-Based Translator & Text-to-Speech")
-
-# Map language codes to Google TTS codes
-language_codes = {
-    "en": "en-US",
-    "es": "es-ES",
-    "fr": "fr-FR",
-    "de": "de-DE",
-    "hi": "hi-IN"
-}
-
-def translate_text_auto(text, target_language):
-    """Detects the source language and translates text."""
-    translate_client = translate.Client()
-    result = translate_client.translate(text, target_language=target_language)
-    detected_lang = result['detectedSourceLanguage']
-    translated_text = result['translatedText']
-    return detected_lang, translated_text
-
-def convert_text_to_speech(text, language_code):
-    """Convert text to speech using Google Cloud TTS."""
-    client = texttospeech.TextToSpeechClient()
-    synthesis_input = texttospeech.SynthesisInput(text=text)
-    voice = texttospeech.VoiceSelectionParams(
-        language_code=language_code,
-        ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
-    )
-    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
-    response = client.synthesize_speech(
-        input=synthesis_input, voice=voice, audio_config=audio_config
-    )
-    return response.audio_content
-
-# Streamlit form
-with st.form("translator_form"):
-    input_text = st.text_area("Enter text to translate:")
-    target_language = st.selectbox(
-        "Select target language:",
-        options=list(language_codes.keys()),
-        format_func=lambda x: x.upper()
-    )
-    submit = st.form_submit_button("Translate & Listen")
-
-if submit and input_text.strip():
-    with st.spinner("Translating..."):
-        detected_lang, translated_text = translate_text_auto(input_text, target_language)
-    st.success(f"✅ Translation complete! Detected source language: {detected_lang.upper()}")
-    st.write(f"**Translated Text ({target_language.upper()}):** {translated_text}")
-
-    with st.spinner("Converting to speech..."):
-        tts_code = language_codes[target_language]
-        audio_content = convert_text_to_speech(translated_text, tts_code)
-
-    st.audio(audio_content, format="audio/mp3")
-
+import streamlit as st
+from google.cloud import texttospeech
+from google.cloud import translate_v2 as translate
+from dotenv import load_dotenv
+import os
+import io
+
+# Load environment variables
+load_dotenv()
+
+# Set Google credentials from .env
+GOOGLE_CREDS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
+if not GOOGLE_CREDS or not os.path.exists(GOOGLE_CREDS):
+    st.error("⚠️ Service account JSON file not found. Please check your .env file configuration!")
+    st.info("Make sure your .env file contains: GOOGLE_APPLICATION_CREDENTIALS=path/to/your/service-account.json")
+    st.stop()
+else:
+    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_CREDS
+
+# Configure Streamlit page
+st.set_page_config(
+    page_title="AI Translator & TTS", 
+    page_icon="🌐",
+    layout="wide",
+    initial_sidebar_state="expanded"
+)
+
+# App header
+st.title("🌐 AI-Based Translator & Text-to-Speech")
+st.markdown("---")
+st.markdown("### Translate text between languages and convert to speech using Google Cloud AI")
+
+# Map language codes to Google TTS codes and display names
+language_options = {
+    "English": {"code": "en", "tts": "en-US"},
+    "Spanish": {"code": "es", "tts": "es-ES"},
+    "French": {"code": "fr", "tts": "fr-FR"},
+    "German": {"code": "de", "tts": "de-DE"},
+    "Hindi": {"code": "hi", "tts": "hi-IN"},
+    "Italian": {"code": "it", "tts": "it-IT"},
+    "Portuguese": {"code": "pt", "tts": "pt-BR"},
+    "Japanese": {"code": "ja", "tts": "ja-JP"},
+    "Korean": {"code": "ko", "tts": "ko-KR"},
+    "Chinese (Simplified)": {"code": "zh", "tts": "zh-CN"}
+}
+
+def translate_text_auto(text, target_language_code):
+    """Detects the source language and translates text."""
+    try:
+        translate_client = translate.Client()
+        result = translate_client.translate(text, target_language=target_language_code)
+        detected_lang = result['detectedSourceLanguage']
+        translated_text = result['translatedText']
+        return detected_lang, translated_text
+    except Exception as e:
+        st.error(f"Translation error: {str(e)}")
+        return None, None
+
+def convert_text_to_speech(text, language_code):
+    """Convert text to speech using Google Cloud TTS."""
+    try:
+        client = texttospeech.TextToSpeechClient()
+        synthesis_input = texttospeech.SynthesisInput(text=text)
+        voice = texttospeech.VoiceSelectionParams(
+            language_code=language_code,
+            ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
+        )
+        audio_config = texttospeech.AudioConfig(
+            audio_encoding=texttospeech.AudioEncoding.MP3
+        )
+        response = client.synthesize_speech(
+            input=synthesis_input, voice=voice, audio_config=audio_config
+        )
+        return response.audio_content
+    except Exception as e:
+        st.error(f"Text-to-speech error: {str(e)}")
+        return None
+
+# Create two columns for better layout
+col1, col2 = st.columns([2, 1])
+
+with col1:
+    # Streamlit form
+    with st.form("translator_form"):
+        st.subheader("📝 Enter Your Text")
+        input_text = st.text_area(
+            "Text to translate:",
+            height=150,
+            placeholder="Enter the text you want to translate..."
+        )
+        
+        target_language = st.selectbox(
+            "🎯 Select target language:",
+            options=list(language_options.keys()),
+            index=1  # Default to Spanish
+        )
+        
+        submit = st.form_submit_button("🚀 Translate & Generate Speech", use_container_width=True)
+
+with col2:
+    st.subheader("ℹ️ About")
+    st.info("""
+    This app uses Google Cloud AI to:
+    
+    🔍 **Auto-detect** source language
+    
+    🌍 **Translate** to your target language
+    
+    🔊 **Generate** natural speech audio
+    """)
+
+# Process the translation
+if submit and input_text.strip():
+    target_lang_info = language_options[target_language]
+    
+    with st.spinner("🔄 Translating text..."):
+        detected_lang, translated_text = translate_text_auto(
+            input_text, 
+            target_lang_info["code"]
+        )
+    
+    if translated_text:
+        # Display results
+        st.markdown("---")
+        st.subheader("📋 Translation Results")
+        
+        col_result1, col_result2 = st.columns(2)
+        
+        with col_result1:
+            st.success(f"✅ **Source Language Detected:** {detected_lang.upper()}")
+            st.markdown(f"**Original Text:**")
+            st.info(input_text)
+        
+        with col_result2:
+            st.success(f"✅ **Translated to:** {target_language}")
+            st.markdown(f"**Translated Text:**")
+            st.info(translated_text)
+        
+        # Generate speech
+        with st.spinner("🎵 Converting to speech..."):
+            tts_code = target_lang_info["tts"]
+            audio_content = convert_text_to_speech(translated_text, tts_code)
+        
+        if audio_content:
+            st.subheader("🔊 Audio Output")
+            st.audio(audio_content, format="audio/mp3")
+            
+            # Download button for audio
+            st.download_button(
+                label="📥 Download Audio",
+                data=audio_content,
+                file_name=f"translation_{target_lang_info['code']}.mp3",
+                mime="audio/mp3"
+            )
+    
+elif submit:
+    st.warning("⚠️ Please enter some text to translate!")
+
+# Footer
+st.markdown("---")
+st.markdown("**Powered by Google Cloud Translate & Text-to-Speech APIs**")
EOF
)
