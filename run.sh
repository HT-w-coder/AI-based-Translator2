#!/bin/bash

# AI Multilingual Translator Setup Script
# This script sets up the environment and runs the Streamlit application

echo "🌐 AI Multilingual Text-to-Speech Translator"
echo "============================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check Python version
python_version=$(python3 -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
echo "🐍 Python version: $python_version"

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not installed. Please install pip."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Check if installation was successful
if [ $? -eq 0 ]; then
    echo "✅ Dependencies installed successfully!"
else
    echo "❌ Failed to install dependencies. Please check the error messages above."
    exit 1
fi

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p models cache

# Display system information
echo ""
echo "📊 System Information:"
echo "====================="
echo "🖥️  OS: $(uname -s)"
echo "💾 Architecture: $(uname -m)"

# Check for GPU support
if command -v nvidia-smi &> /dev/null; then
    echo "🚀 NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1
else
    echo "🖥️  Using CPU (GPU not detected or NVIDIA drivers not installed)"
fi

echo ""
echo "🎉 Setup complete! Starting the application..."
echo ""
echo "📖 Usage Instructions:"
echo "- The app will open in your default browser"
echo "- First run may take longer as AI models are downloaded"
echo "- Use Ctrl+C to stop the application"
echo ""

# Run the Streamlit application
echo "🚀 Starting Streamlit application..."
streamlit run translator_app.py --server.port 8501 --server.address 0.0.0.0

echo ""
echo "👋 Application stopped. Thank you for using AI Multilingual Translator!"