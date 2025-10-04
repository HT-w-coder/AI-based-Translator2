# 🔧 Installation Troubleshooting Guide

If you're encountering package installation issues, try these solutions:

## 🚀 Quick Fix for PyTorch Version Issues

### Option 1: Use Basic Requirements First
```bash
# Install basic packages first
pip install -r requirements-basic.txt

# Then install additional packages
pip install langdetect pyttsx3 plotly sentencepiece sacremoses
```

### Option 2: Manual Installation
```bash
# Update pip first
pip install --upgrade pip

# Install PyTorch (latest version)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other packages
pip install streamlit transformers langdetect pyttsx3 plotly numpy requests
```

### Option 3: Use Conda (Recommended for complex environments)
```bash
# Create conda environment
conda create -n translator python=3.9
conda activate translator

# Install PyTorch
conda install pytorch torchaudio cpuonly -c pytorch

# Install other packages
pip install streamlit transformers langdetect pyttsx3 plotly sacremoses sentencepiece
```

## 🐛 Common Issues & Solutions

### 1. PyTorch Version Not Found
**Error**: `No matching distribution found for torch==2.1.0`

**Solution**: 
- Use `torch>=2.5.0` instead of exact version
- Or install latest: `pip install torch --upgrade`

### 2. Python Version Issues
**Error**: `Requires-Python <3.5` or similar

**Solution**:
- Ensure you're using Python 3.8+
- Check version: `python --version`
- Use virtual environment: `python -m venv venv`

### 3. TTS (pyttsx3) Installation Issues
**Error**: Various pyttsx3 errors

**Solution**:
- On Linux: `sudo apt-get install espeak espeak-data libespeak1 libespeak-dev`
- On macOS: Should work by default
- On Windows: Should work by default
- Skip if not needed: The app works without TTS

### 4. Transformers/SentencePiece Issues
**Error**: Various compilation errors

**Solution**:
- Install pre-compiled wheels: `pip install --only-binary=all transformers sentencepiece`
- Or skip problematic packages initially and install them later

## 🎯 Minimal Installation (Translation Only)

If you just want basic translation functionality:

```bash
pip install streamlit torch transformers langdetect numpy
```

Then run with basic features:
```bash
streamlit run translator_app.py
```

## 🔍 Testing Your Installation

Run the test script to check what's working:
```bash
python test_app.py
```

This will show which components are working and which need attention.

## 💡 Platform-Specific Notes

### Linux (Ubuntu/Debian)
```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install python3-pip python3-venv espeak

# Then install Python packages
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### macOS
```bash
# Using Homebrew
brew install python3

# Then install packages
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Windows
```bash
# In Command Prompt or PowerShell
python -m venv venv
venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 🚀 Alternative: Docker Installation

If you continue having issues, you can run it with Docker:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "translator_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
# Build and run
docker build -t translator .
docker run -p 8501:8501 translator
```

## 📞 Still Having Issues?

1. Check your Python version: `python --version` (need 3.8+)
2. Update pip: `pip install --upgrade pip`
3. Try virtual environment
4. Run test script: `python test_app.py`
5. Install packages one by one to isolate issues

The app can run with minimal dependencies - you don't need everything to work!