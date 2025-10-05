#!/bin/bash

# EMERGENCY FIX for MarianMTModel Import Error
# Use this if you're getting "cannot import name 'MarianMTModel'"

echo "🚨 EMERGENCY FIX: MarianMTModel Import Error"
echo "============================================"

echo "Current Python version:"
python3 --version

echo ""
echo "🔧 Applying emergency fix..."

# Step 1: Complete cleanup
echo "1️⃣ Cleaning up existing installations..."
pip uninstall transformers tokenizers torch torchaudio -y 2>/dev/null || true

# Step 2: Clear cache
echo "2️⃣ Clearing pip cache..."
pip cache purge

# Step 3: Install very specific versions that work
echo "3️⃣ Installing compatible versions..."
pip install torch==2.0.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cpu
pip install tokenizers==0.13.3
pip install transformers==4.21.3

# Step 4: Install other requirements
echo "4️⃣ Installing other dependencies..."
pip install streamlit==1.28.0 langdetect==1.0.9 numpy==1.24.3 requests==2.31.0

# Step 5: Test the fix
echo "5️⃣ Testing imports..."
python3 -c "
print('Testing imports...')
try:
    import transformers
    print('✅ transformers:', transformers.__version__)
    
    from transformers import MarianMTModel, MarianTokenizer
    print('✅ MarianMT models imported successfully!')
    
    import torch
    print('✅ torch:', torch.__version__)
    
    import streamlit
    print('✅ streamlit:', streamlit.__version__)
    
    print()
    print('🎉 SUCCESS! All imports working.')
    print('You can now run: streamlit run translator_app.py')
    
except Exception as e:
    print('❌ Import failed:', e)
    print()
    print('🔄 Try the debug app instead:')
    print('   streamlit run debug_app.py')
    print()
    print('💡 Or consider using Python 3.11:')
    print('   conda create -n translator python=3.11')
    print('   conda activate translator')
    print('   pip install -r requirements-fixed.txt')
"

echo ""
echo "🚀 Fix complete! Try running the app now:"
echo "   streamlit run translator_app.py"
echo ""
echo "🐛 If still having issues, use debug mode:"
echo "   streamlit run debug_app.py"