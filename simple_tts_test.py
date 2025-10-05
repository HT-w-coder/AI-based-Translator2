import streamlit as st

st.title("🔊 Simple TTS Test for Streamlit")

# Test if the simple approach works
text_input = st.text_input("Enter text to speak:", "Hello, this is a test")

if st.button("🔊 Speak Now"):
    # Clean the text
    clean_text = text_input.replace("'", "\\'").replace('"', '\\"').replace('\n', ' ')
    
    # Use st.components.v1.html to execute JavaScript
    speech_html = f"""
    <script>
    console.log('Button clicked, starting speech...');
    
    if ('speechSynthesis' in window) {{
        // Stop any existing speech
        window.speechSynthesis.cancel();
        
        // Create and configure utterance
        const utterance = new SpeechSynthesisUtterance('{clean_text}');
        utterance.rate = 1;
        utterance.pitch = 1;
        utterance.volume = 1;
        
        // Event listeners
        utterance.onstart = () => console.log('Speech started');
        utterance.onend = () => console.log('Speech finished');
        utterance.onerror = (e) => console.error('Speech error:', e);
        
        // Speak immediately
        window.speechSynthesis.speak(utterance);
        
        // Also try with a delay
        setTimeout(() => {{
            if (window.speechSynthesis.paused) {{
                window.speechSynthesis.resume();
            }}
        }}, 100);
        
    }} else {{
        alert('Speech synthesis not supported');
    }}
    </script>
    """
    
    # Execute the JavaScript
    st.components.v1.html(speech_html, height=0)
    st.success("Speech command sent to browser!")

# Alternative method - direct HTML button
st.markdown("---")
st.subheader("Alternative: Direct HTML Button")

direct_html = f"""
<button onclick="
    const text = '{text_input.replace("'", "\\'")}';
    if ('speechSynthesis' in window) {{
        window.speechSynthesis.cancel();
        const utterance = new SpeechSynthesisUtterance(text);
        window.speechSynthesis.speak(utterance);
    }} else {{
        alert('Speech not supported');
    }}
" style="
    background: #667eea;
    color: white;
    border: none;
    padding: 10px 20px;
    border-radius: 5px;
    cursor: pointer;
">
🔊 Direct HTML Button
</button>
"""

st.markdown(direct_html, unsafe_allow_html=True)

# Debug info
st.markdown("---")
st.subheader("Debug Information")

debug_js = """
<script>
document.write('<p>Browser: ' + navigator.userAgent + '</p>');
document.write('<p>Speech Synthesis: ' + ('speechSynthesis' in window ? 'Supported' : 'Not Supported') + '</p>');

if ('speechSynthesis' in window) {
    const voices = speechSynthesis.getVoices();
    document.write('<p>Voices loaded: ' + voices.length + '</p>');
    
    if (voices.length === 0) {
        speechSynthesis.addEventListener('voiceschanged', function() {
            const newVoices = speechSynthesis.getVoices();
            document.write('<p>Voices after event: ' + newVoices.length + '</p>');
        });
    }
}
</script>
"""

st.markdown(debug_js, unsafe_allow_html=True)

# Test different approaches
st.markdown("---")
st.subheader("Multiple Test Approaches")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Method 1: Direct"):
        st.components.v1.html(f"""
        <script>
        speechSynthesis.speak(new SpeechSynthesisUtterance('{text_input}'));
        </script>
        """, height=0)

with col2:
    if st.button("Method 2: Delayed"):
        st.components.v1.html(f"""
        <script>
        setTimeout(() => {{
            speechSynthesis.speak(new SpeechSynthesisUtterance('{text_input}'));
        }}, 500);
        </script>
        """, height=0)

with col3:
    if st.button("Method 3: Event"):
        st.components.v1.html(f"""
        <script>
        window.addEventListener('load', () => {{
            speechSynthesis.speak(new SpeechSynthesisUtterance('{text_input}'));
        }});
        </script>
        """, height=0)