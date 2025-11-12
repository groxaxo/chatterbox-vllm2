# Quick Start Guide - Chatterbox TTS on vLLM

**Get the Chatterbox TTS API server running on your GPU in minutes!**

This guide helps you set up an OpenAI-compatible TTS API optimized for GPUs with as little as 8GB VRAM.

## About This Project

This is a community-enhanced fork that builds upon:
- **[Resemble AI's Chatterbox](https://github.com/resemble-ai/chatterbox)** - The original high-quality TTS model
- **[randombk's vLLM port](https://github.com/randombk/chatterbox-vllm)** - High-performance vLLM implementation

Our additions: OpenAI-compatible API, low VRAM optimization, enhanced multilingual support, and production-ready deployment tools.

## Prerequisites

- Linux or WSL2
- NVIDIA GPU with 8GB+ VRAM (tested on RTX 3060)
- CUDA installed
- Python 3.10+
- `git` and [`uv`](https://pypi.org/project/uv/) package manager

## Installation

### 1. Clone and Install

```bash
git clone https://github.com/groxaxo/chatterbox-vllm2.git
cd chatterbox-vllm2
```

### 2. Set up Virtual Environment

```bash
# Install uv if not already installed
pip install uv

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate
uv sync
```

This will automatically download the required model weights from Hugging Face Hub.

## Running the API Server

### Option 1: Using the Start Script (Recommended)

The easiest way to start the server with optimized settings:

```bash
# For 4-6GB VRAM GPUs (RTX 2060, GTX 1660 Ti) - with quantization
./start-api-server.sh --ultra-low-vram

# For 8GB VRAM GPUs (RTX 3060, RTX 2070)
./start-api-server.sh --low-vram

# For 12GB VRAM GPUs (RTX 3080, RTX 2080Ti)
./start-api-server.sh --medium-vram

# For 24GB+ VRAM GPUs (RTX 3090, RTX 4090)
./start-api-server.sh --high-vram

# Use English-only model for even lower VRAM usage
./start-api-server.sh --ultra-low-vram --english
```

**Note:** Ultra-low VRAM mode requires the `bitsandbytes` library:
```bash
pip install bitsandbytes
# or with uv:
uv pip install ".[ultra-low-vram]"
```

### Option 2: Direct Python Command

```bash
# Multilingual model (default)
CHATTERBOX_MODEL=multilingual python api_server.py

# English-only model (uses less VRAM)
CHATTERBOX_MODEL=english python api_server.py
```

The server will start on `http://localhost:8000`.

## Testing the API

### Quick Test with curl

Once the server is running, test it in a new terminal:

```bash
# Generate English speech
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "input": "Hello! This is a test of the Chatterbox TTS API.",
    "voice": "alloy"
  }' \
  --output test_speech.mp3

# Play the audio (Linux)
mpv test_speech.mp3
# or
ffplay test_speech.mp3
```

### Run the Test Suite

```bash
# In a new terminal (while server is running)
python test_api.py
```

### Try Different Languages

```bash
# French
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "input": "Bonjour! Comment allez-vous?",
    "voice": "fr",
    "language_id": "fr"
  }' \
  --output french.mp3

# German
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "input": "Guten Tag! Wie geht es Ihnen?",
    "voice": "de",
    "language_id": "de"
  }' \
  --output german.mp3

# Spanish
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "input": "¡Hola! ¿Cómo estás?",
    "voice": "alloy",
    "language_id": "es"
  }' \
  --output spanish.mp3
```

## Integrating with Open WebUI

1. Make sure the Chatterbox TTS API server is running
2. Open your Open WebUI instance
3. Go to **Settings** → **Audio**
4. Configure TTS:
   - **TTS Engine**: OpenAI
   - **API Base URL**: `http://localhost:8000/v1`
   - **API Key**: (any value or leave empty)
   - **Model**: `tts-1`
   - **Voice**: Choose from `alloy`, `echo`, `fable`, `onyx`, `nova`, `shimmer`

### For Multilingual in Open WebUI

To use different languages in Open WebUI, you can:
- Use language codes as voice names: `fr`, `de`, `es`, `zh`, etc.
- The system will automatically detect and use the appropriate language

## Memory Usage

### Ultra-Low VRAM Configuration (4-6GB GPUs)
Using `--ultra-low-vram` with BnB 4-bit quantization:
- **Model Loading**: ~3-4 GB
- **During Generation**: ~4-6 GB peak
- **Recommended**: 4GB+ VRAM
- **Target GPUs**: RTX 2060 (6GB), GTX 1660 Ti (6GB), GTX 1650 (4GB)
- **Trade-offs**: Slight quality reduction, ~10-20% slower inference

### Low VRAM Configuration (8GB GPUs)
Default configuration without quantization:
- **Model Loading**: ~5-6 GB
- **During Generation**: ~7-8 GB peak
- **Recommended**: 8GB+ VRAM
- **Target GPUs**: RTX 3060, RTX 2070, RTX 2060 Super

### If You Get Out of Memory Errors

1. **Enable ultra-low VRAM mode**:
   ```bash
   ./start-api-server.sh --ultra-low-vram
   ```

2. **Reduce max_model_len**:
   ```bash
   CHATTERBOX_MAX_MODEL_LEN=600 python api_server.py
   ```

3. **Use English-only model**:
   ```bash
   CHATTERBOX_MODEL=english python api_server.py
   ```

4. **Enable quantization manually**:
   ```bash
   CHATTERBOX_USE_QUANTIZATION=true \
   CHATTERBOX_QUANTIZATION_METHOD=bnb-4bit \
   python api_server.py
   ```

5. **Restart the service** (to clear any cached data):
   ```bash
   # Stop the server (Ctrl+C)
   # Start it again
   python api_server.py
   ```

## Advanced Usage

### Environment Variables

```bash
# Model selection
export CHATTERBOX_MODEL=multilingual  # or 'english'

# Memory settings
export CHATTERBOX_MAX_BATCH_SIZE=1    # Keep at 1 for low VRAM
export CHATTERBOX_MAX_MODEL_LEN=800   # Lower = less VRAM (600 for ultra-low)

# Ultra-low VRAM quantization settings
export CHATTERBOX_USE_QUANTIZATION=true          # Enable quantization
export CHATTERBOX_QUANTIZATION_METHOD=bnb-4bit   # bnb-4bit, bnb-8bit, or awq
export CHATTERBOX_QUANTIZE_S3GEN=true            # Quantize S3Gen model
export CHATTERBOX_QUANTIZE_VOICE_ENCODER=true    # Quantize Voice Encoder

# Server settings
export CHATTERBOX_HOST=0.0.0.0        # Listen on all interfaces
export CHATTERBOX_PORT=8000           # Server port

# Then start the server
python api_server.py
```

### Using with Python Client

```python
import requests

response = requests.post(
    "http://localhost:8000/v1/audio/speech",
    json={
        "model": "tts-1",
        "input": "Hello, world!",
        "voice": "alloy",
    }
)

with open("output.mp3", "wb") as f:
    f.write(response.content)
```

### Using with OpenAI Python Client

```python
from openai import OpenAI

client = OpenAI(
    api_key="not-needed",
    base_url="http://localhost:8000/v1"
)

response = client.audio.speech.create(
    model="tts-1",
    voice="alloy",
    input="Hello, world!",
)

response.stream_to_file("output.mp3")
```

## Docker Deployment

### Using Docker Compose (Recommended)

```bash
# Build and start
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

### Using Docker Directly

```bash
# Build
docker build -t chatterbox-tts-api .

# Run
docker run --gpus all -p 8000:8000 \
  -e CHATTERBOX_MODEL=multilingual \
  -e CHATTERBOX_MAX_BATCH_SIZE=1 \
  -e CHATTERBOX_MAX_MODEL_LEN=800 \
  chatterbox-tts-api
```

## Troubleshooting

### Server Won't Start

1. **Check GPU**:
   ```bash
   nvidia-smi
   ```
   Make sure your GPU is detected and has free memory.

2. **Check CUDA**:
   ```python
   python3 -c "import torch; print(torch.cuda.is_available())"
   ```

3. **Check Dependencies**:
   ```bash
   uv sync
   ```

### Out of Memory Errors

- Reduce `CHATTERBOX_MAX_MODEL_LEN` to 600 or lower
- Use `CHATTERBOX_MODEL=english` instead of multilingual
- Ensure no other GPU-intensive processes are running

### Audio Quality Issues

- Adjust `temperature` (0.7-0.9 is recommended)
- Adjust `exaggeration` (0.4-0.6 for neutral voice)
- Try different voice references

### Slow Generation

- Generation time scales with text length
- Expected: 2-4s for short texts, 5-10s for paragraphs
- Make sure CUDA graphs are not causing issues (they're disabled by default in low VRAM mode)

## Performance Tips

1. **Reuse Voice References**: Cache audio conditionals for frequently used voices
2. **Batch Requests**: If you have more VRAM, increase `max_batch_size`
3. **Shorter Texts**: Break long texts into paragraphs for faster processing
4. **Use English Model**: If you only need English, use the English model for better performance

## Next Steps

- Read [API_USAGE.md](API_USAGE.md) for detailed API documentation
- Check [README.md](README.md) for more information about the project
- Try the example scripts:
  - `python example-api-client.py` - Various API usage examples
  - `python example-tts-min-vram.py` - Direct library usage
  - `python example-tts-multilingual.py` - Multilingual examples

## Getting Help

If you encounter issues:
1. Check the server logs for error messages
2. Review the troubleshooting section above
3. Check existing issues on GitHub
4. Create a new issue with:
   - Your GPU model and VRAM
   - Error messages
   - Configuration used

## Supported Languages

The multilingual model supports 23 languages:

Arabic (ar), Danish (da), German (de), Greek (el), English (en), Spanish (es), 
Finnish (fi), French (fr), Hebrew (he), Hindi (hi), Italian (it), Japanese (ja), 
Korean (ko), Malay (ms), Dutch (nl), Norwegian (no), Polish (pl), Portuguese (pt), 
Russian (ru), Swedish (sv), Swahili (sw), Turkish (tr), Chinese (zh)
