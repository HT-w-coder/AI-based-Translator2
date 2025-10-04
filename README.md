# AI Multilingual Text-to-Speech Translator

A powerful, privacy-focused multilingual translator with text-to-speech capabilities built using Streamlit. This application works completely offline using state-of-the-art AI models from Hugging Face.

## 🌟 Features

- **🌍 Multilingual Support**: Translate between 12+ languages including English, Spanish, French, German, Italian, Portuguese, Russian, Chinese, Japanese, Korean, Arabic, and Hindi
- **🤖 AI-Powered**: Uses Helsinki-NLP MarianMT models for accurate neural machine translation
- **🎵 Text-to-Speech**: Convert translated text to speech using pyttsx3 engine
- **🔒 Privacy-First**: All processing happens locally - no data sent to external services
- **📱 Modern Interface**: Beautiful and responsive Streamlit web interface
- **📋 Translation History**: Keep track of your recent translations
- **🔄 Language Auto-Detection**: Automatically detect source language
- **⚡ Fast Processing**: Optimized for both CPU and GPU usage

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone or download this repository**
   ```bash
   git clone <repository-url>
   cd multilingual-translator
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run translator_app.py
   ```

4. **Access the application**
   - Open your web browser and go to `http://localhost:8501`
   - The application will automatically download required AI models on first use

## 📋 Requirements

The application requires the following Python packages:

- `streamlit==1.28.0` - Web framework
- `transformers==4.35.0` - Hugging Face transformers for translation models
- `torch==2.1.0` - PyTorch for neural networks
- `pyttsx3==2.90` - Text-to-speech engine
- `langdetect==1.0.9` - Language detection
- `sentencepiece==0.1.99` - Tokenization
- `sacremoses==0.0.53` - Text preprocessing
- `plotly==5.17.0` - Interactive visualizations

## 🎯 Usage

### Basic Translation

1. **Select Languages**:
   - Choose source language (or use "Auto-detect")
   - Choose target language

2. **Enter Text**:
   - Type or paste text in the input area (up to 5000 characters)
   - The app will automatically detect the language if auto-detect is enabled

3. **Translate**:
   - Click "🔄 Translate" button
   - View the translation result below

4. **Text-to-Speech**:
   - Click "🎵 Speak Original" to hear the original text
   - Click "🎵 Speak Translation" to hear the translated text

### Advanced Features

- **Language Swapping**: Use the ⇄ button to swap source and target languages
- **Translation History**: View recent translations in the sidebar
- **Copy Results**: Copy translations to clipboard
- **Audio Playback**: Listen to translations with built-in audio player

## 🌐 Supported Languages

| Language | Code | Native Name |
|----------|------|-------------|
| English | en | English |
| Spanish | es | Español |
| French | fr | Français |
| German | de | Deutsch |
| Italian | it | Italiano |
| Portuguese | pt | Português |
| Russian | ru | Русский |
| Chinese | zh | 中文 |
| Japanese | ja | 日本語 |
| Korean | ko | 한국어 |
| Arabic | ar | العربية |
| Hindi | hi | हिन्दी |

## 🔧 Technical Details

### Translation Models

The application uses Helsinki-NLP's MarianMT models from Hugging Face:
- Models are downloaded automatically on first use
- Support for both direct translation and pivot translation through English
- Models are cached locally for faster subsequent use

### Text-to-Speech

- Uses `pyttsx3` for offline text-to-speech conversion
- Automatically selects appropriate voices based on target language
- Generates audio files that can be played directly in the browser

### Performance Optimization

- **GPU Support**: Automatically uses CUDA if available for faster translation
- **Model Caching**: Translation models are cached in memory
- **Streamlit Caching**: Uses Streamlit's caching mechanisms for optimal performance

## 🛠️ Troubleshooting

### Common Issues

1. **TTS Not Working**:
   - Ensure your system has audio output capabilities
   - On Linux, you may need to install espeak: `sudo apt-get install espeak`

2. **Model Download Fails**:
   - Check your internet connection for first-time setup
   - Models are downloaded from Hugging Face Hub

3. **Out of Memory**:
   - Try reducing input text length
   - Close other applications to free up RAM
   - Use CPU instead of GPU if memory is limited

### System Requirements

- **RAM**: Minimum 4GB, recommended 8GB+
- **Storage**: 2-3GB for models (downloaded automatically)
- **Internet**: Required for initial model download only

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

### Development Setup

1. Fork the repository
2. Create a virtual environment: `python -m venv venv`
3. Activate it: `source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)
4. Install dependencies: `pip install -r requirements.txt`
5. Make your changes and test
6. Submit a pull request

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Helsinki-NLP** for the excellent MarianMT translation models
- **Hugging Face** for the transformers library and model hosting
- **Streamlit** for the amazing web framework
- **pyttsx3** for offline text-to-speech capabilities

## 📞 Support

If you encounter any issues or have questions:
1. Check the troubleshooting section above
2. Search existing issues in the repository
3. Create a new issue with detailed information about your problem

---

**Enjoy translating! 🌍✨**