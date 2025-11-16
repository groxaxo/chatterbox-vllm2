# Chatterbox TTS on vLLM - Spanish TTS Ready\! 🇪🇸🎵

A high-performance port of [Chatterbox TTS](https://github.com/resemble-ai/chatterbox) to vLLM, optimized for low VRAM GPUs with OpenAI-compatible API support and full Spanish language capabilities.

## 🙏 Acknowledgments

This project builds upon the excellent work of:
- **[Resemble AI](https://github.com/resemble-ai)** - Original [Chatterbox TTS](https://github.com/resemble-ai/chatterbox) model and implementation  
- **[randombk](https://github.com/randombk)** - Initial [vLLM port](https://github.com/randombk/chatterbox-vllm) that made efficient inference possible

Special thanks to these pioneers for making such advanced TTS technology openly available\!

## 🚀 Key Features

- ✅ **OpenAI-Compatible API** - Drop-in replacement for OpenAI TTS API, works with Open WebUI and other clients
- ✅ **Ultra-Low VRAM Support** - Runs on 4-6GB GPUs with BnB/AWQ quantization (RTX 2060, GTX 1660 Ti)  
- ✅ **Optimized for 8GB GPUs** - Runs efficiently on RTX 3060, RTX 2070, etc.
- ✅ **Full Spanish Support** - Native Spanish language processing with OpenAI-compatible endpoints
- ✅ **23 Languages Total** - Multilingual support with automatic language detection
- ✅ **Production Ready** - Complete Docker setup, health checks, and monitoring
- ✅ **Multiple Audio Formats** - MP3, WAV, FLAC, Opus, AAC, PCM

## 🎯 Current Working Setup (Spanish TTS)

This setup has been tested and verified to work with **Spanish text-to-speech** using only **5GB VRAM**:

### Quick Start for Spanish TTS

```bash
# Clone and setup
git clone https://github.com/groxaxo/chatterbox-vllm2.git
cd chatterbox-vllm2
uv venv
source .venv/bin/activate  
uv sync

# Start the Spanish-optimized server
CUDA_VISIBLE_DEVICES=2 \
CHATTERBOX_MODEL=multilingual \
CHATTERBOX_MAX_BATCH_SIZE=1 \
CHATTERBOX_MAX_MODEL_LEN=400 \
CHATTERBOX_GPU_MEMORY_UTILIZATION=0.15 \
python api_server.py
```

### Test Spanish TTS

```bash
# Generate Spanish speech
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "input": "Hola\! Bienvenido al sistema de texto a voz en español.",
    "voice": "es",
    "language_id": "es"
  }' \
  --output spanish_speech.mp3
```

### Open WebUI Configuration

Configure Open WebUI for Spanish TTS:
- **TTS Engine:** OpenAI
- **API Base URL:** `http://localhost:8000/v1` 
- **API Key:** (leave empty)
- **Model:** `tts-1`
- **Voice:** `es`

---

**Note**: This is a community project and is not officially affiliated with Resemble AI or any corporate entity.

## 📦 Installation

### System Requirements
- **OS**: Linux or WSL2
- **GPU**: NVIDIA GPU with 4GB+ VRAM
  - Ultra-Low VRAM (4-6GB): RTX 2060, GTX 1660 Ti, GTX 1650
  - Low VRAM (8GB): RTX 3060, RTX 2070, RTX 2060 Super
  - Medium/High VRAM (12GB+): RTX 3080, RTX 3090, RTX 4090
- **Software**: Python 3.10+, CUDA toolkit

### Quick Installation

```bash
# Clone the repository
git clone https://github.com/groxaxo/chatterbox-vllm2.git
cd chatterbox-vllm2

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate
uv sync
```

The package will automatically download model weights from Hugging Face Hub (~1-2GB).

## 🎯 Usage Options

### Option 1: API Server (Recommended)

```bash
# Start with Spanish optimization (5GB VRAM usage)
CUDA_VISIBLE_DEVICES=2 \
CHATTERBOX_MODEL=multilingual \
CHATTERBOX_MAX_BATCH_SIZE=1 \
CHATTERBOX_MAX_MODEL_LEN=400 \
CHATTERBOX_GPU_MEMORY_UTILIZATION=0.15 \
python api_server.py
```

### Option 2: Using the Startup Script

```bash
# For low VRAM GPUs (8GB)
./start-api-server.sh --low-vram

# For ultra-low VRAM GPUs (4-6GB with quantization) 
./start-api-server.sh --ultra-low-vram
```

### Option 3: Docker Deployment

```bash
# Using Docker Compose
docker-compose up -d

# Or build and run manually
docker build -t chatterbox-tts-api .
docker run --gpus all -p 8000:8000 chatterbox-tts-api
```

## 🌍 Spanish TTS Examples

### Via API (OpenAI Compatible)

```bash
# Basic Spanish TTS
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1", 
    "input": "¡Hola\! ¿Cómo estás hoy?",
    "voice": "es"
  }' \
  --output greeting.mp3

# Advanced Spanish with emotion
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "input": "¡Excelente trabajo\! Me encanta el resultado.",
    "voice": "es",
    "exaggeration": 0.8,
    "response_format": "mp3"
  }' \
  --output praise.mp3
```

### Via Python Library

```python
import torchaudio as ta
from chatterbox_vllm.tts import ChatterboxTTS

# Initialize for Spanish
model = ChatterboxTTS.from_pretrained_multilingual(
    max_batch_size=1,
    max_model_len=400,
    gpu_memory_utilization=0.15,
    enforce_eager=True,
)

# Generate Spanish speech
spanish_texts = [
    "Hola, soy una voz generada por IA.",
    "¡Bienvenido al futuro de la síntesis de voz\!",
    "Esta tecnología es increíblemente avanzada."
]

audios = model.generate(spanish_texts, language_id='es', exaggeration=0.6)

for idx, audio in enumerate(audios):
    ta.save(f"spanish_output_{idx}.mp3", audio, model.sr)

model.shutdown()
```

## 🌍 Multilingual Support

Chatterbox TTS supports **23 languages** including Spanish with automatic language detection.

**Supported Languages:**
Arabic (ar), Danish (da), German (de), Greek (el), English (en), **Spanish (es)**, Finnish (fi), French (fr), Hebrew (he), Hindi (hi), Italian (it), Japanese (ja), Korean (ko), Malay (ms), Dutch (nl), Norwegian (no), Polish (pl), Portuguese (pt), Russian (ru), Swedish (sv), Swahili (sw), Turkish (tr), Chinese (zh)

### Language-Specific Usage

```bash
# Spanish
curl -X POST http://localhost:8000/v1/audio/speech \
  -d '{"model": "tts-1", "input": "¡Hola mundo\!", "voice": "es"}' \
  http://localhost:8000/v1/audio/speech --output spanish.mp3

# French  
curl -X POST http://localhost:8000/v1/audio/speech \
  -d '{"model": "tts-1", "input": "Bonjour le monde\!", "voice": "fr"}' \
  http://localhost:8000/v1/audio/speech --output french.mp3

# German
curl -X POST http://localhost:8000/v1/audio/speech \
  -d '{"model": "tts-1", "input": "Hallo Welt\!", "voice": "de"}' \
  http://localhost:8000/v1/audio/speech --output german.mp3
```

## 🔧 API Reference

### Health Check
```bash
curl http://localhost:8000/health
```

### List Models  
```bash
curl http://localhost:8000/v1/models
```

### TTS Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | string | `"tts-1"` | Model to use |
| `input` | string | required | Text to synthesize (max 4096 chars) |
| `voice` | string | `"alloy"` | Voice (alloy, echo, fable, onyx, nova, shimmer, or language codes) |
| `response_format` | string | `"mp3"` | Audio format (mp3, wav, flac, opus, aac, pcm) |
| `speed` | float | `1.0` | Speech speed (0.25 to 4.0) |
| `exaggeration` | float | `0.5` | Emotion level (0.0 to 2.0) |
| `language_id` | string | auto | Language code (es, fr, de, etc.) |

## ⚡ Performance

### Current Working Configuration
- **VRAM Usage:** ~5GB (70% less than previous 16.5GB)
- **Generation Speed:** ~2-3 seconds per request
- **Quality:** Excellent Spanish pronunciation and natural speech
- **GPU:** RTX 3090 GPU 2 (clean, no conflicts)

### Benchmark Results
- **Speech Token Generation:** ~2.25s
- **Waveform Generation:** ~0.97s  
- **Total Generation Time:** ~3.2s per request
- **Throughput:** ~180 tokens/second

## 🐛 Known Limitations

- Uses internal vLLM APIs (may need updates for future vLLM versions)
- CFG scale must be set globally, not per-request
- Some advanced features from original Chatterbox not yet implemented

## 📄 License

This project is licensed under the same terms as the original Chatterbox TTS project.

## 🤝 Contributing

Contributions are welcome\! Please feel free to submit issues and pull requests.

---

**Ready to generate high-quality Spanish speech with minimal VRAM usage\! 🇪🇸🎵**
