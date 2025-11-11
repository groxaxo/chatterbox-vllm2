# Chatterbox TTS API Server

OpenAI-compatible TTS API server for Chatterbox TTS on vLLM, optimized for GPUs with less than 10GB VRAM.

## Features

- ✅ **OpenAI-compatible API** - Drop-in replacement for OpenAI TTS API
- ✅ **Open WebUI compatible** - Works seamlessly with Open WebUI
- ✅ **Low VRAM optimization** - Runs on GPUs with as little as 8GB VRAM (e.g., RTX 3060)
- ✅ **Multilingual support** - 23 languages supported
- ✅ **Multiple voice options** - Compatible with OpenAI voice names
- ✅ **Multiple audio formats** - MP3, WAV, FLAC, Opus, AAC, PCM

## Quick Start

### 1. Install Dependencies

```bash
# Install the package with API dependencies
uv sync
```

### 2. Start the Server

#### For English model (lower VRAM usage):
```bash
CHATTERBOX_MODEL=english python api_server.py
```

#### For Multilingual model (recommended):
```bash
CHATTERBOX_MODEL=multilingual python api_server.py
```

The server will start on `http://0.0.0.0:8000` by default.

### 3. Test the API

```bash
# Simple test with curl
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "input": "Hello! This is a test of the Chatterbox TTS API.",
    "voice": "alloy"
  }' \
  --output speech.mp3

# Test with different language (multilingual model only)
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "input": "Bonjour! Ceci est un test.",
    "voice": "fr",
    "language_id": "fr"
  }' \
  --output speech_fr.mp3
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CHATTERBOX_MODEL` | `multilingual` | Model variant: `english` or `multilingual` |
| `CHATTERBOX_HOST` | `0.0.0.0` | Server host address |
| `CHATTERBOX_PORT` | `8000` | Server port |
| `CHATTERBOX_MAX_BATCH_SIZE` | `1` | Maximum batch size (keep at 1 for low VRAM) |
| `CHATTERBOX_MAX_MODEL_LEN` | `800` | Maximum model length in tokens |
| `CHATTERBOX_CFG_SCALE` | - | CFG scale (if supported) |

### Low VRAM Configuration (RTX 3060 / 8GB)

For GPUs with 8GB VRAM, use these settings:

```bash
export CHATTERBOX_MODEL=multilingual
export CHATTERBOX_MAX_BATCH_SIZE=1
export CHATTERBOX_MAX_MODEL_LEN=800
python api_server.py
```

These settings should keep GPU memory usage under 7GB, leaving headroom for system processes.

### Higher VRAM Configuration (RTX 3090 / 24GB)

For GPUs with more VRAM, you can increase performance:

```bash
export CHATTERBOX_MODEL=multilingual
export CHATTERBOX_MAX_BATCH_SIZE=3
export CHATTERBOX_MAX_MODEL_LEN=1000
python api_server.py
```

## API Endpoints

### POST /v1/audio/speech

Generate speech from text (OpenAI-compatible).

**Request Body:**

```json
{
  "model": "tts-1",
  "input": "Text to synthesize",
  "voice": "alloy",
  "response_format": "mp3",
  "speed": 1.0,
  "exaggeration": 0.5,
  "temperature": 0.8,
  "language_id": "en"
}
```

**Parameters:**

- `model` (string, required): Model to use. Options: `tts-1`, `tts-1-hd`
- `input` (string, required): Text to synthesize (max 4096 characters)
- `voice` (string, optional): Voice name. Options:
  - OpenAI voices: `alloy`, `echo`, `fable`, `onyx`, `nova`, `shimmer`
  - Language codes (multilingual): `en`, `fr`, `de`, `es`, `zh`, etc.
- `response_format` (string, optional): Audio format. Options: `mp3` (default), `wav`, `flac`, `opus`, `aac`, `pcm`
- `speed` (float, optional): Speech speed (0.25 to 4.0, default: 1.0)
- `exaggeration` (float, optional): Emotion exaggeration (0.0 to 2.0, default: 0.5)
- `temperature` (float, optional): Sampling temperature (0.0 to 2.0, default: 0.8)
- `language_id` (string, optional): Language code for multilingual model

**Response:** Audio file in the requested format

### GET /v1/models

List available models (OpenAI-compatible).

**Response:**

```json
{
  "object": "list",
  "data": [
    {
      "id": "tts-1",
      "object": "model",
      "created": 1700000000,
      "owned_by": "chatterbox-vllm"
    },
    {
      "id": "tts-1-hd",
      "object": "model",
      "created": 1700000000,
      "owned_by": "chatterbox-vllm"
    }
  ]
}
```

### GET /health

Health check endpoint.

**Response:**

```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_type": "multilingual",
  "gpu_memory_mb": 6543.21
}
```

### GET /

API information endpoint.

**Response:**

```json
{
  "message": "Chatterbox TTS API Server",
  "model_type": "multilingual",
  "supported_languages": ["ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi", "it", "ja", "ko", "ms", "nl", "no", "pl", "pt", "ru", "sv", "sw", "tr", "zh"],
  "endpoints": {
    "tts": "/v1/audio/speech",
    "health": "/health",
    "models": "/v1/models"
  }
}
```

## Open WebUI Integration

To use this API with Open WebUI:

1. Start the Chatterbox TTS API server:
   ```bash
   python api_server.py
   ```

2. In Open WebUI, go to Settings → Audio

3. Configure TTS settings:
   - **TTS Engine**: OpenAI
   - **API Base URL**: `http://localhost:8000/v1`
   - **API Key**: (leave empty or use any value)
   - **Model**: `tts-1` or `tts-1-hd`
   - **Voice**: Choose from `alloy`, `echo`, `fable`, `onyx`, `nova`, `shimmer`

4. For multilingual support, you can use language codes as voice names:
   - Voice: `fr` for French
   - Voice: `de` for German
   - Voice: `zh` for Chinese
   - etc.

## Supported Languages (Multilingual Model)

The multilingual model supports the following 23 languages:

| Code | Language | Code | Language | Code | Language |
|------|----------|------|----------|------|----------|
| `ar` | Arabic | `he` | Hebrew | `pt` | Portuguese |
| `da` | Danish | `hi` | Hindi | `ru` | Russian |
| `de` | German | `it` | Italian | `sv` | Swedish |
| `el` | Greek | `ja` | Japanese | `sw` | Swahili |
| `en` | English | `ko` | Korean | `tr` | Turkish |
| `es` | Spanish | `ms` | Malay | `zh` | Chinese |
| `fi` | Finnish | `nl` | Dutch | | |
| `fr` | French | `no` | Norwegian | | |

## Python Client Example

```python
import requests

# Generate speech
response = requests.post(
    "http://localhost:8000/v1/audio/speech",
    json={
        "model": "tts-1",
        "input": "Hello! This is a test of the Chatterbox TTS API.",
        "voice": "alloy",
        "response_format": "mp3",
    }
)

# Save to file
with open("output.mp3", "wb") as f:
    f.write(response.content)
```

## Using with OpenAI Python Client

The API is compatible with the OpenAI Python client:

```python
from openai import OpenAI

client = OpenAI(
    api_key="not-needed",
    base_url="http://localhost:8000/v1"
)

response = client.audio.speech.create(
    model="tts-1",
    voice="alloy",
    input="Hello! This is a test of the Chatterbox TTS API.",
)

response.stream_to_file("output.mp3")
```

## Performance Notes

### Memory Usage

With the default low-VRAM configuration:
- **Model Load**: ~5-6 GB VRAM
- **Per Request**: ~1-2 GB additional VRAM
- **Total**: ~7-8 GB peak usage

This fits comfortably on an RTX 3060 with 8GB VRAM.

### Generation Speed

On an RTX 3060:
- Short texts (1-2 sentences): ~2-4 seconds
- Medium texts (paragraph): ~5-10 seconds
- Long texts: ~15-30 seconds

Speed scales roughly linearly with text length.

## Troubleshooting

### Out of Memory Errors

If you encounter OOM errors:

1. Reduce `CHATTERBOX_MAX_MODEL_LEN`:
   ```bash
   export CHATTERBOX_MAX_MODEL_LEN=600
   ```

2. Ensure `CHATTERBOX_MAX_BATCH_SIZE=1`

3. Try the English model instead of multilingual:
   ```bash
   export CHATTERBOX_MODEL=english
   ```

### Model Not Loading

If the model fails to load:

1. Check GPU availability:
   ```python
   import torch
   print(torch.cuda.is_available())
   ```

2. Verify dependencies are installed:
   ```bash
   uv sync
   ```

3. Check logs for specific error messages

### Quality Issues

For better quality:

1. Increase diffusion steps (slower but higher quality):
   - This is currently not exposed via API but can be added if needed

2. Adjust temperature (0.7-0.9 works well):
   ```json
   {"temperature": 0.8}
   ```

3. Use appropriate exaggeration (0.4-0.6 for neutral, higher for more emotion):
   ```json
   {"exaggeration": 0.5}
   ```

## Docker Deployment

Create a `Dockerfile`:

```dockerfile
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Install Python and dependencies
RUN apt-get update && apt-get install -y python3.10 python3-pip git

# Install uv
RUN pip3 install uv

# Copy project
WORKDIR /app
COPY . .

# Install dependencies
RUN uv sync

# Expose port
EXPOSE 8000

# Run server
ENV CHATTERBOX_MODEL=multilingual
ENV CHATTERBOX_MAX_BATCH_SIZE=1
ENV CHATTERBOX_MAX_MODEL_LEN=800

CMD ["python3", "api_server.py"]
```

Build and run:

```bash
docker build -t chatterbox-tts-api .
docker run --gpus all -p 8000:8000 chatterbox-tts-api
```

## License

Same as the main Chatterbox vLLM project.
