#!/bin/bash

# AI Multilingual Translator - Quick Fix Script
# This script fixes common installation issues

echo "🔧 AI Multilingual Translator - Quick Fix"
echo "========================================="

# Check Python version
python_version=$(python3 -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
echo "🐍 Python version: $python_version"

# Function to fix transformers installation
fix_transformers() {
    echo "🔄 Fixing transformers installation..."
    
    # Uninstall existing transformers
    pip uninstall transformers -y
    pip uninstall tokenizers -y
    
    # Clear pip cache
    pip cache purge
    
    # Install specific versions that work well together
    pip install torch>=2.0.0 --upgrade
    pip install tokenizers>=0.13.0
    pip install transformers>=4.21.0
    
    echo "✅ Transformers reinstalled"
}

# Function to test imports
test_imports() {
    echo "🧪 Testing imports..."
    
    python3 -c "
import sys
try:
    import transformers
    print('✅ transformers:', transformers.__version__)
except ImportError as e:
    print('❌ transformers import failed:', e)
    sys.exit(1)

try:
    from transformers import MarianMTModel, MarianTokenizer
    print('✅ MarianMT models available')
except ImportError:
    try:
        from transformers import AutoModel, AutoTokenizer
        print('⚠️  Using AutoModel as fallback')
    except ImportError as e:
        print('❌ Model import failed:', e)
        sys.exit(1)

try:
    import torch
    print('✅ torch:', torch.__version__)
except ImportError as e:
    print('❌ torch import failed:', e)
    sys.exit(1)

try:
    import streamlit
    print('✅ streamlit:', streamlit.__version__)
except ImportError as e:
    print('❌ streamlit import failed:', e)
    sys.exit(1)

print('🎉 All core imports successful!')
"
}

# Main execution
echo ""
echo "Choose an option:"
echo "1. Quick fix - reinstall transformers"
echo "2. Full reinstall - all packages"
echo "3. Test imports only"
echo "4. Install minimal requirements"

read -p "Enter choice (1-4): " choice

case $choice in
    1)
        fix_transformers
        test_imports
        ;;
    2)
        echo "🔄 Full reinstall..."
        pip uninstall -r requirements.txt -y
        pip install -r requirements.txt --upgrade
        test_imports
        ;;
    3)
        test_imports
        ;;
    4)
        echo "🔄 Installing minimal requirements..."
        pip install streamlit torch transformers>=4.21.0 langdetect numpy requests
        test_imports
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "🚀 If tests pass, try running:"
echo "   streamlit run translator_app.py"
echo ""
echo "💡 If you still have issues:"
echo "   - Try: pip install transformers==4.21.3 torch tokenizers"
echo "   - Or use Python 3.9-3.11 instead of 3.13"