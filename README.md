# AI-Based Translator & Text-to-Speech App

A Streamlit web application that translates text between multiple languages and converts the translated text to natural-sounding speech using Google Cloud AI services.

## Features

- 🔍 **Auto Language Detection**: Automatically detects the source language of input text
- 🌍 **Multi-Language Translation**: Supports 10 languages including English, Spanish, French, German, Hindi, Italian, Portuguese, Japanese, Korean, and Chinese (Simplified)  
- 🔊 **Text-to-Speech**: Converts translated text to natural speech audio
- 📥 **Audio Download**: Download the generated audio as MP3 files
- 🎨 **Modern UI**: Clean and responsive Streamlit interface

## Prerequisites

1. **Google Cloud Project**: You need a Google Cloud project with the following APIs enabled:
   - Cloud Translation API
   - Cloud Text-to-Speech API

2. **Service Account**: Create a service account with appropriate permissions and download the JSON key file.

## Installation & Setup

1. **Install dependencies**:
   ```bash
   pip install streamlit google-cloud-translate google-cloud-texttospeech python-dotenv loguru
   ```

2. **Configure Google Cloud credentials**:
   - Place your service account JSON file in the project directory
   - Create a `.env` file with:
     ```
     GOOGLE_APPLICATION_CREDENTIALS=your-service-account-file.json
     ```

## Usage

1. **Start the application**:
   ```bash
   streamlit run app.py
   ```

2. **Use the web interface**:
   - Enter text you want to translate
   - Select target language from the dropdown
   - Click "Translate & Generate Speech"
   - Listen to the audio and optionally download it

## Supported Languages

| Language | Code | TTS Support |
|----------|------|-------------|
| English | en | ✅ |
| Spanish | es | ✅ |  
| French | fr | ✅ |
| German | de | ✅ |
| Hindi | hi | ✅ |
| Italian | it | ✅ |
| Portuguese | pt | ✅ |
| Japanese | ja | ✅ |
| Korean | ko | ✅ |
| Chinese (Simplified) | zh | ✅ |

## Architecture

The app uses:
- **Streamlit** for the web interface
- **Google Cloud Translate API** for translation
- **Google Cloud Text-to-Speech API** for audio generation
- **python-dotenv** for environment variable management

## Error Handling

The app includes comprehensive error handling for:
- Missing or invalid Google Cloud credentials
- Translation API errors  
- Text-to-Speech API errors
- Network connectivity issues

## Notes

- The app requires active internet connection for Google Cloud API calls
- Make sure your Google Cloud project has sufficient quota for the APIs
- Audio files are generated in MP3 format for broad compatibility