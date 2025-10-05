#!/usr/bin/env python3
"""
Quick test script to verify transformers installation
"""

def test_transformers_import():
    """Test different ways to import transformers"""
    print("🧪 Testing Transformers Import Methods...")
    print("=" * 50)
    
    # Test 1: Direct import
    try:
        from transformers import MarianMTModel, MarianTokenizer
        print("✅ Method 1: Direct MarianMT import - SUCCESS")
        return True, "Direct import works"
    except ImportError as e1:
        print(f"❌ Method 1: Direct import failed - {e1}")
        
        # Test 2: Alternative path
        try:
            from transformers.models.marian import MarianMTModel, MarianTokenizer
            print("✅ Method 2: Alternative path import - SUCCESS")
            return True, "Alternative path works"
        except ImportError as e2:
            print(f"❌ Method 2: Alternative path failed - {e2}")
            
            # Test 3: AutoModel fallback
            try:
                from transformers import AutoModel, AutoTokenizer
                print("✅ Method 3: AutoModel fallback - SUCCESS")
                return True, "AutoModel fallback works"
            except ImportError as e3:
                print(f"❌ Method 3: AutoModel fallback failed - {e3}")
                
                # Test 4: Basic transformers
                try:
                    import transformers
                    print(f"⚠️  Basic transformers import works (version: {transformers.__version__})")
                    print("   But model classes are not available")
                    return False, f"Transformers available but models not importable"
                except ImportError as e4:
                    print(f"❌ Method 4: Basic transformers failed - {e4}")
                    return False, "Transformers not installed"

def main():
    import sys
    
    print(f"🐍 Python Version: {sys.version}")
    print(f"🔧 Platform: {sys.platform}")
    print()
    
    success, message = test_transformers_import()
    
    print()
    print("=" * 50)
    if success:
        print("🎉 RESULT: Transformers is working!")
        print(f"   Status: {message}")
        print()
        print("✅ You can run: streamlit run translator_app.py")
    else:
        print("❌ RESULT: Transformers has issues")
        print(f"   Issue: {message}")
        print()
        print("🔧 Try these fixes:")
        print("   1. ./emergency_fix.sh")
        print("   2. pip install transformers==4.21.3 tokenizers==0.13.3")
        print("   3. streamlit run debug_app.py (works without transformers)")

if __name__ == "__main__":
    main()