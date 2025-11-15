# Multilingual Chatterbox Verification Guide

This guide helps you verify that the project is using the latest multilingual version of Chatterbox (v2) and that it works correctly with Open WebUI.

## What's Included in Multilingual v2

The current implementation is based on the multilingual version from resemble-ai/chatterbox commit [a9742ee](https://github.com/resemble-ai/chatterbox/commit/a9742ee281c4cb8a47a0e88d4576c11869de2f2c) and subsequent v2 updates, which includes:

### Supported Languages (23 Total)

- **ar**: Arabic
- **da**: Danish  
- **de**: German
- **el**: Greek
- **en**: English
- **es**: Spanish
- **fi**: Finnish
- **fr**: French
- **he**: Hebrew
- **hi**: Hindi
- **it**: Italian
- **ja**: Japanese (with kanji-to-hiragana conversion)
- **ko**: Korean (with Hangul decomposition)
- **ms**: Malay
- **nl**: Dutch
- **no**: Norwegian
- **pl**: Polish
- **pt**: Portuguese
- **ru**: Russian (with stress marking)
- **sv**: Swedish
- **sw**: Swahili
- **tr**: Turkish
- **zh**: Chinese (with Cangjie encoding)

### Key Features

1. **Advanced Text Normalization**:
   - Japanese: Kanji to hiragana conversion using pykakasi
   - Korean: Hangul syllable decomposition to Jamo
   - Chinese: Cangjie encoding with pkuseg segmentation
   - Hebrew: Diacritics addition (requires dicta_onnx)
   - Russian: Stress marking (requires russian-text-stresser)

2. **Model Configuration**:
   - Model file: `t3_mtl23ls_v2.safetensors`
   - Tokenizer: `grapheme_mtl_merged_expanded_v1.json`
   - Vocabulary size: 2454 tokens (vs 704 for English-only)

3. **API Compatibility**:
   - OpenAI-compatible endpoint: `/v1/audio/speech`
   - Language ID parameter support
   - Automatic language detection from voice codes

## Verifying the Installation

### 1. Check Version

```bash
cat .latest-version.generated.txt
```

Should show: `0.2.1` (or higher)

### 2. Verify Model Files

Check that the multilingual model files exist:

```bash
# In your huggingface cache or local directory
ls -la ~/.cache/huggingface/hub/models--ResembleAI--chatterbox/
```

Look for:
- `t3_mtl23ls_v2.safetensors` (multilingual model weights)
- `grapheme_mtl_merged_expanded_v1.json` (multilingual tokenizer)
- `Cangjie5_TC.json` (Chinese character mapping)

### 3. Test Multilingual Model Loading

```python
from chatterbox_vllm.tts import ChatterboxTTS

# Load multilingual model
model = ChatterboxTTS.from_pretrained_multilingual(
    max_batch_size=1,
    max_model_len=800,
)

# Check supported languages
languages = model.get_supported_languages()
print(f"Supported languages: {len(languages)}")
print(f"Languages: {', '.join(languages.keys())}")

# Should print: 23 languages including ar, da, de, el, en, es, fi, fr, he, hi, it, ja, ko, ms, nl, no, pl, pt, ru, sv, sw, tr, zh
```

### 4. Test Language-Specific Generation

```python
import torchaudio as ta

# Test French
audio_fr = model.generate(
    ["Bonjour! Comment allez-vous?"],
    language_id="fr",
    exaggeration=0.5
)
ta.save("test_fr.mp3", audio_fr[0], model.sr)

# Test Japanese (with kanji normalization)
audio_ja = model.generate(
    ["こんにちは、お元気ですか？"],
    language_id="ja",
    exaggeration=0.5
)
ta.save("test_ja.mp3", audio_ja[0], model.sr)

# Test Chinese (with Cangjie encoding)
audio_zh = model.generate(
    ["你好，很高兴见到你。"],
    language_id="zh",
    exaggeration=0.5
)
ta.save("test_zh.mp3", audio_zh[0], model.sr)
```

## API Server Verification

### 1. Start the Multilingual API Server

```bash
# Using environment variable
export CHATTERBOX_MODEL=multilingual
python api_server.py

# Or using the convenience script
./start-api-server.sh --low-vram  # Adjust based on your GPU
```

The server should start and display:
```
[INFO] Multilingual model loaded. Supported languages: ar, da, de, el, en, es, fi, fr, he, hi, it, ja, ko, ms, nl, no, pl, pt, ru, sv, sw, tr, zh
```

### 2. Test API Endpoints

Run the test suite:

```bash
python test_api.py
```

This will test:
- Health check endpoint
- Models listing
- English generation
- Multilingual generation (French, German, Spanish)

### 3. Manual API Testing

Test different languages:

```bash
# French
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "input": "Bonjour! Comment ça va?",
    "voice": "alloy",
    "language_id": "fr"
  }' \
  --output test_french.mp3

# Japanese
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "input": "こんにちは、お元気ですか？",
    "voice": "alloy",
    "language_id": "ja"
  }' \
  --output test_japanese.mp3

# Spanish
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "input": "¡Hola! ¿Cómo estás?",
    "voice": "alloy",
    "language_id": "es"
  }' \
  --output test_spanish.mp3

# Chinese
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "input": "你好，很高兴见到你。",
    "voice": "alloy",
    "language_id": "zh"
  }' \
  --output test_chinese.mp3
```

## Open WebUI Integration

### Configuration Steps

1. **Start the Chatterbox API Server**:
   ```bash
   export CHATTERBOX_MODEL=multilingual
   python api_server.py
   ```

2. **Configure Open WebUI**:
   
   Go to Open WebUI Settings → Audio:
   
   - **TTS Engine**: Select "OpenAI"
   - **API Base URL**: `http://localhost:8000/v1`
   - **API Key**: (leave empty or use any value)
   - **Model**: `tts-1` or `tts-1-hd`
   - **Voice**: Select from:
     - Standard voices: `alloy`, `echo`, `fable`, `onyx`, `nova`, `shimmer`
     - Language-specific: `en`, `fr`, `de`, `es`, `ja`, `zh`, etc.

3. **Test in Open WebUI**:
   
   In a chat, enable TTS and type messages in different languages:
   - English: "Hello, how are you?"
   - French: "Bonjour, comment allez-vous?"
   - Spanish: "Hola, ¿cómo estás?"
   - Japanese: "こんにちは、お元気ですか？"
   - Chinese: "你好，很高兴见到你。"

   The system will automatically detect the language or you can specify it using the `language_id` parameter if your Open WebUI client supports it.

### Voice Selection for Different Languages

For optimal results with specific languages, you can use language-specific voice references:

- **French**: Use voice code `fr`
- **German**: Use voice code `de`
- **Spanish**: Use voice code `es`
- **Chinese**: Use voice code `zh`
- **Japanese**: Use voice code `ja`
- etc.

Example API request with language-specific voice:

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "input": "Bonjour tout le monde!",
    "voice": "fr",
    "language_id": "fr"
  }' \
  --output french_native.mp3
```

## Troubleshooting

### Missing Language-Specific Dependencies

Some languages require optional dependencies:

- **Japanese** (kanji processing): Already installed (`pykakasi`)
- **Chinese** (segmentation): Already installed (`spacy-pkuseg`)
- **Hebrew** (diacritics): Optional - install with `pip install dicta-onnx`
- **Russian** (stress): Optional - install with `pip install git+https://github.com/Vuizur/add-stress-to-epub`

Without these dependencies, the system will fall back to basic text processing with a warning.

### Language Not Recognized

Make sure to:
1. Use the correct 2-letter language code (e.g., `fr` not `french`)
2. Set the `language_id` parameter explicitly in the API request
3. Check that the model type is `multilingual` (not `english`)

### Model Loading Issues

If you get errors about missing model files:

1. Delete the cached model:
   ```bash
   rm -rf ~/.cache/huggingface/hub/models--ResembleAI--chatterbox
   ```

2. Restart the server to re-download the latest model:
   ```bash
   export CHATTERBOX_MODEL=multilingual
   python api_server.py
   ```

## Verification Checklist

Use this checklist to ensure everything is working:

- [ ] Version is 0.2.1 or higher
- [ ] Multilingual model files are present (t3_mtl23ls_v2.safetensors)
- [ ] API server starts with "Multilingual model loaded" message
- [ ] Server reports 23 supported languages
- [ ] Health check endpoint returns model_type="multilingual"
- [ ] English text generation works
- [ ] French text generation works
- [ ] At least one Asian language works (Japanese, Chinese, or Korean)
- [ ] Open WebUI can connect to the API
- [ ] Open WebUI can generate speech in multiple languages

## Reference

- Original Chatterbox repo: https://github.com/resemble-ai/chatterbox
- Multilingual commit: [a9742ee](https://github.com/resemble-ai/chatterbox/commit/a9742ee281c4cb8a47a0e88d4576c11869de2f2c)
- Multilingual v2 update: [bf169fe](https://github.com/resemble-ai/chatterbox/commit/bf169fe5f518760cb0b6c6a6eba3f885e10fa86f)
- This project: https://github.com/groxaxo/chatterbox-vllm2
