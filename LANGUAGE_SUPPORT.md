# 🚫 Unavailable Language Pairs

Some translation models are not available on Hugging Face. Here's what's supported:

## ✅ **Fully Supported Languages**

### **From English to:**
- Spanish (es) ✅
- French (fr) ✅  
- German (de) ✅
- Italian (it) ✅
- Portuguese (pt) ✅
- Russian (ru) ✅
- Chinese (zh) ✅
- Japanese (ja) ✅
- Arabic (ar) ✅
- Hindi (hi) ✅

### **To English from:**
- Spanish ✅
- French ✅
- German ✅
- Italian ✅
- Portuguese ✅
- Russian ✅
- Chinese ✅
- Japanese ✅
- Arabic ✅

### **European Language Pairs:**
- Spanish ↔ French, German, Italian, Portuguese ✅
- French ↔ German, Italian, Portuguese ✅
- German ↔ Italian, Portuguese ✅
- Italian ↔ Portuguese ✅

## ❌ **Not Available**

### **Korean (ko)**
- **Issue**: `Helsinki-NLP/opus-mt-en-ko` model does not exist
- **Workaround**: Use Google Translate API or other services for Korean
- **Alternative**: Train your own model or use multilingual models

### **Other Limited Pairs:**
- Most Asian language pairs (except through English)
- African languages
- Less common European languages

## 💡 **Workarounds**

### **For Unavailable Pairs:**
1. **Pivot Translation**: Many pairs work through English
   - Korean → English → Spanish (works)
   - Arabic → English → French (works)

2. **Alternative Services:**
   ```python
   # You can integrate other translation services
   # Google Translate, DeepL, etc.
   ```

3. **Multilingual Models:**
   - Use `facebook/m2m100` models for more language pairs
   - Use `google/mt5` for multilingual translation

## 🔄 **How the App Handles Missing Models**

1. **Detection**: App checks for unavailable models
2. **Error Message**: Shows clear "Model not available" message
3. **Fallback**: Tries pivot translation through English when possible
4. **User Feedback**: Clear indication of what's supported

## 📋 **Supported Language Codes**

```
en - English     ✅ (hub language)
es - Spanish     ✅ (full support)
fr - French      ✅ (full support)
de - German      ✅ (full support)
it - Italian     ✅ (full support)
pt - Portuguese  ✅ (full support)
ru - Russian     ✅ (to/from English)
zh - Chinese     ✅ (to/from English)
ja - Japanese    ✅ (to/from English)
ar - Arabic      ✅ (to/from English)
hi - Hindi       ✅ (to/from English)
ko - Korean      ❌ (not available)
```

The app now properly handles these limitations and provides clear feedback to users about what's available.