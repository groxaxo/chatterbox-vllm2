# Chatterbox TTS on vLLM

A high-performance port of [Chatterbox TTS](https://github.com/resemble-ai/chatterbox) to vLLM, optimized for low VRAM GPUs with OpenAI-compatible API support.

## 🙏 Acknowledgments

This project builds upon the excellent work of:
- **[Resemble AI](https://github.com/resemble-ai)** - Original [Chatterbox TTS](https://github.com/resemble-ai/chatterbox) model and implementation
- **[randombk](https://github.com/randombk)** - Initial [vLLM port](https://github.com/randombk/chatterbox-vllm) that made efficient inference possible

Special thanks to these pioneers for making such advanced TTS technology openly available!

## 🚀 Why This Fork?

This fork extends the original vLLM port with:
- ✅ **OpenAI-Compatible API** - Drop-in replacement for OpenAI TTS API, works with Open WebUI and other clients
- ✅ **Low VRAM Optimization** - Runs efficiently on 8GB GPUs (RTX 3060, RTX 2070, etc.)
- ✅ **Full Multilingual Support** - 23 languages with automatic language detection
- ✅ **Production Ready** - Complete Docker setup, systemd service templates, health checks
- ✅ **Comprehensive Documentation** - Easy-to-follow guides for all use cases

### Performance Benefits (from original vLLM port)
- **~4x speedup** in generation tokens/s without batching
- **Over 10x speedup** with batching enabled
- Significantly improved GPU memory efficiency
- Eliminates CPU-GPU sync bottlenecks from HF transformers

👉 **Quick Start**: See [QUICKSTART.md](QUICKSTART.md) for setup on low VRAM GPUs  
👉 **API Documentation**: See [API_USAGE.md](API_USAGE.md) for complete API reference

---

**Note**: This is a community project and is not officially affiliated with Resemble AI or any corporate entity.

## Generation Samples

![Sample 1](docs/audio-sample-01.mp3)
<audio controls>
  <source src="docs/audio-sample-01.mp3" type="audio/mp3">
</audio>

![Sample 2](docs/audio-sample-02.mp3)
<audio controls>
  <source src="docs/audio-sample-02.mp3" type="audio/mp3">
</audio>

![Sample 3](docs/audio-sample-03.mp3)
<audio controls>
  <source src="docs/audio-sample-03.mp3" type="audio/mp3">
</audio>


# ✨ Features & Status

## What Works
* ✅ **Speech Cloning** - Audio and text conditioning for voice matching
* ✅ **High-Quality Output** - Matches original Chatterbox quality
* ✅ **Context Free Guidance (CFG)** - Configurable via `CHATTERBOX_CFG_SCALE` environment variable
* ✅ **Exaggeration Control** - Adjust emotion and expressiveness (0.0 to 2.0)
* ✅ **vLLM Batching** - Significant speedup for multiple requests
* ✅ **OpenAI API** - Full compatibility with OpenAI TTS clients
* ✅ **Low VRAM Support** - Optimized for 8GB GPUs (RTX 3060, RTX 2070)
* ✅ **23 Languages** - Multilingual support with automatic language detection
* ✅ **Production Ready** - Docker, systemd templates, health checks
* ✅ **Multiple Audio Formats** - MP3, WAV, FLAC, Opus, AAC, PCM

## Known Limitations
*Note: Some limitations inherited from the original vLLM port*

* ⚠️ **vLLM Internal APIs** - Uses internal vLLM APIs with workarounds
  * Currently compatible with vLLM 0.10.0
  * Future refactoring may be needed for newer vLLM versions
  * Track progress: [vLLM Issue #21989](https://github.com/vllm-project/vllm/issues/21989)
* ⚠️ **Learned Speech Positional Embeddings** - Not yet supported in vLLM
  * Minor quality impact, mostly imperceptible
* ⚠️ **CFG Per-Request** - CFG scale must be set globally, not per-request
* ⚠️ **API Stability** - APIs may evolve; stability expected at v1.0.0

## Roadmap
* 🔄 Code cleanup and refactoring
* 🔄 Improved vLLM integration patterns
* 🔄 Enhanced benchmarking tools
* 🔄 Streaming audio support
* 🔄 Voice cloning from user uploads

# 📦 Installation

## System Requirements
- **OS**: Linux or WSL2 (Windows Subsystem for Linux)
- **GPU**: NVIDIA GPU with 8GB+ VRAM
  - Tested on: RTX 3060 (8GB), RTX 3080 (12GB), RTX 3090 (24GB)
  - AMD GPUs may work with minor modifications (untested)
- **Software**: Python 3.10+, CUDA toolkit

## Quick Installation

Prerequisites: Install `git` and [`uv`](https://pypi.org/project/uv/) package manager

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

### Troubleshooting Installation

If you encounter CUDA issues, try:
```bash
# Reset the environment and use alternative install method
rm -rf .venv
uv venv
source .venv/bin/activate
uv pip install -e .
```

## Updating

To update from a previous version:
```bash
cd chatterbox-vllm2
git pull
source .venv/bin/activate
uv sync
```

Model weights will be automatically updated if needed.

# 🎯 Quick Start Examples

## Option 1: OpenAI-Compatible API Server (Recommended)

The easiest way to use Chatterbox TTS - compatible with Open WebUI and OpenAI clients:

```bash
# Start the API server (automatically optimized for your GPU)
./start-api-server.sh --low-vram    # For 8GB GPUs
./start-api-server.sh --medium-vram # For 12GB GPUs
./start-api-server.sh --high-vram   # For 24GB+ GPUs

# Or start directly with Python
CHATTERBOX_MODEL=multilingual python api_server.py
```

Then test it:
```bash
# Generate speech with curl
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "input": "Hello! This is a test.",
    "voice": "alloy"
  }' \
  --output speech.mp3

# Or use the Python test suite
python test_api.py
```

See [API_USAGE.md](API_USAGE.md) for complete API documentation.

## Option 2: Python Library

Use Chatterbox TTS directly in your Python code ([example-tts.py](example-tts.py)):

```python
import torchaudio as ta
from chatterbox_vllm.tts import ChatterboxTTS

# Initialize model (optimized for low VRAM)
model = ChatterboxTTS.from_pretrained(
    gpu_memory_utilization=0.4,
    max_model_len=1000,
    enforce_eager=True,  # Reduces startup time
)

# Generate speech
prompts = [
    "You are listening to a demo of the Chatterbox TTS model running on vLLM.",
    "This is a separate prompt to test batching.",
    "And here is a third prompt, slightly longer than the first.",
]

audios = model.generate(prompts, exaggeration=0.8)
for idx, audio in enumerate(audios):
    ta.save(f"output_{idx}.mp3", audio, model.sr)
```

## Option 3: Docker Deployment

Production-ready Docker setup with health checks:

```bash
# Using Docker Compose (recommended)
docker-compose up -d

# Or build and run manually
docker build -t chatterbox-tts-api .
docker run --gpus all -p 8000:8000 chatterbox-tts-api
```

See [docker-compose.yml](docker-compose.yml) for configuration options.

## Option 4: Gradio Web UI

Interactive web interface for testing and demos:

```bash
python gradio_tts_app.py
```

Then open your browser to the URL shown (typically `http://localhost:7860`).

# 🌍 Multilingual Support

Chatterbox TTS supports **23 languages** with automatic language detection:

**Supported Languages:**
Arabic (ar), Danish (da), German (de), Greek (el), English (en), Spanish (es), Finnish (fi), French (fr), Hebrew (he), Hindi (hi), Italian (it), Japanese (ja), Korean (ko), Malay (ms), Dutch (nl), Norwegian (no), Polish (pl), Portuguese (pt), Russian (ru), Swedish (sv), Swahili (sw), Turkish (tr), Chinese (zh)

## Usage

### Via API Server
```bash
# French
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model": "tts-1", "input": "Bonjour!", "voice": "fr", "language_id": "fr"}' \
  --output french.mp3

# Spanish
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model": "tts-1", "input": "¡Hola!", "voice": "alloy", "language_id": "es"}' \
  --output spanish.mp3
```

### Via Python Library
See [example-tts-multilingual.py](example-tts-multilingual.py) for complete examples.

## Known Limitations
*Note: These are inherited from the original vLLM port*
- Alignment Stream Analyzer not implemented (may cause occasional repetitions or end-of-audio noise)
- Learned speech positional encodings not yet supported in vLLM
- Russian text stress markers not implemented

For more details on multilingual features, see the [original Chatterbox documentation](https://github.com/resemble-ai/chatterbox#supported-languages).

# 🔌 API Server Features

## OpenAI-Compatible REST API

A production-ready API server that's fully compatible with OpenAI's TTS API specification:

### Core Features
- ✅ **Drop-in OpenAI Replacement** - Works with any OpenAI TTS client
- ✅ **Open WebUI Integration** - Seamless integration with Open WebUI
- ✅ **23 Languages** - Full multilingual support with auto-detection
- ✅ **6 Audio Formats** - MP3, WAV, FLAC, Opus, AAC, PCM
- ✅ **Multiple Voices** - OpenAI-compatible voices plus language-specific options
- ✅ **Low VRAM Optimized** - Runs on 8GB GPUs (RTX 3060, RTX 2070, etc.)
- ✅ **Health Checks** - Built-in health monitoring and status endpoints

### Starting the Server

```bash
# Using the convenience script (recommended)
./start-api-server.sh --low-vram     # For 8GB GPUs
./start-api-server.sh --medium-vram  # For 12GB GPUs  
./start-api-server.sh --high-vram    # For 24GB+ GPUs

# Or set environment variables manually
export CHATTERBOX_MODEL=multilingual
export CHATTERBOX_MAX_BATCH_SIZE=1
export CHATTERBOX_MAX_MODEL_LEN=800
python api_server.py
```

The server starts on `http://localhost:8000` by default.

### Testing the API

```bash
# Quick test with curl
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "input": "Hello! This is Chatterbox TTS.",
    "voice": "alloy"
  }' \
  --output speech.mp3

# Comprehensive test suite
python test_api.py
```

### Open WebUI Integration

Configure Open WebUI to use Chatterbox TTS:

1. Start the API server: `./start-api-server.sh --low-vram`
2. In Open WebUI → Settings → Audio:
   - **TTS Engine**: OpenAI
   - **API Base URL**: `http://localhost:8000/v1`
   - **API Key**: (leave empty or any value)
   - **Model**: `tts-1` or `tts-1-hd`
   - **Voice**: `alloy`, `echo`, `fable`, `onyx`, `nova`, `shimmer` or language codes (`en`, `fr`, `de`, etc.)

### API Documentation

For complete API reference, configuration options, and integration guides, see:
- **[API_USAGE.md](API_USAGE.md)** - Complete API reference
- **[QUICKSTART.md](QUICKSTART.md)** - Step-by-step setup guide

# Benchmarks

To run a benchmark, tweak and run `benchmark.py`.
The following results were obtained with batching on a 6.6k-word input (`docs/benchmark-text-1.txt`), generating ~40min of audio.

Notes:
 * I'm not _entirely_ sure what the tokens/s figures from vLLM are showing - the figures probably aren't directly comparable to others, but the results speak for themselves.
 * With vLLM, **the T3 model is no longer the bottleneck**
   * Vast majority of time is now spent on the S3Gen model, which is not ported/portable to vLLM. This currently uses the original reference implementation from the Chatterbox repo, so there's potential for integrating some of the other community optimizations here.
   * This also means the vLLM section of the model never fully ramps to its peak throughput in these benchmarks.
 * Benchmarks are done without CUDA graphs, as that is currently causing correctness issues.
 * There are some issues with my very rudimentary chunking logic, which is causing some occasional artifacts in output quality.

## Run 1: RTX 3090

**Results using v0.1.3**

System Specs:
 * RTX 3090: 24GB VRAM
 * AMD Ryzen 9 7900X @ 5.70GHz
 * 128GB DDR5 4800 MT/s

Settings & Results:
* Input text: `docs/benchmark-text-1.txt` (6.6k words)
* Input audio: `docs/audio-sample-03.mp3`
* Exaggeration: 0.5, CFG: 0.5, Temperature: 0.8
* CUDA graphs disabled, vLLM max memory utilization=0.6
* Generated output length: 39m54s
* Wall time: 1m33s (including model load and application init)
* Generation time (without model startup time): 87s
  * Time spent in T3 Llama token generation: 13.3s
  * Time spent in S3Gen waveform generation: 60.8s

Logs:
```
[BENCHMARK] Text chunked into 154 chunks
Giving vLLM 56.48% of GPU memory (13587.20 MB)
[config.py:1472] Using max model len 1200
[default_loader.py:272] Loading weights took 0.14 seconds
[gpu_model_runner.py:1801] Model loading took 1.0179 GiB and 0.198331 seconds
[gpu_model_runner.py:2238] Encoder cache will be initialized with a budget of 8192 tokens, and profiled with 178 conditionals items of the maximum feature size.
[gpu_worker.py:232] Available KV cache memory: 11.78 GiB
[kv_cache_utils.py:716] GPU KV cache size: 102,880 tokens
[kv_cache_utils.py:720] Maximum concurrency for 1,200 tokens per request: 85.73x
[BENCHMARK] Model loaded in 7.499186038970947 seconds
Adding requests: 100%|██████| 154/154 [00:00<00:00, 1105.84it/s]
Processed prompts: 100%|█████| 154/154 [00:13<00:00, 11.75it/s, est. speed input: 2193.47 toks/s, output: 4577.88 toks/s]
[T3] Speech Token Generation time: 13.25s
[S3Gen] Wavform Generation time: 60.85s
[BENCHMARK] Generation completed in 74.89441227912903 seconds
[BENCHMARK] Audio saved to benchmark.mp3
[BENCHMARK] Total time: 87.40947437286377 seconds

real	1m33.458s
user	2m21.452s
sys	0m2.362s
```


## Run 2: RTX 3060ti

**Results outdated; using v0.1.0**

System Specs:
 * RTX 3060ti: 8GB VRAM
 * Intel i7-7700K @ 4.20GHz
 * 32GB DDR4 2133 MT/s

Settings & Results:
* Input text: `docs/benchmark-text-1.txt` (6.6k words)
* Input audio: `docs/audio-sample-03.mp3`
* Exaggeration: 0.5, CFG: 0.5, Temperature: 0.8
* CUDA graphs disabled, vLLM max memory utilization=0.6
* Generated output length: 40m15s
* Wall time: 4m26s
* Generation time (without model startup time): 238s
  * Time spent in T3 Llama token generation: 36.4s
  * Time spent in S3Gen waveform generation: 201s

Logs:
```
[BENCHMARK] Text chunked into 154 chunks.
INFO [config.py:1472] Using max model len 1200
INFO [default_loader.py:272] Loading weights took 0.39 seconds
INFO [gpu_model_runner.py:1801] Model loading took 1.0107 GiB and 0.497231 seconds
INFO [gpu_model_runner.py:2238] Encoder cache will be initialized with a budget of 8192 tokens, and profiled with 241 conditionals items of the maximum feature size.
INFO [gpu_worker.py:232] Available KV cache memory: 3.07 GiB
INFO [kv_cache_utils.py:716] GPU KV cache size: 26,816 tokens
INFO [kv_cache_utils.py:720] Maximum concurrency for 1,200 tokens per request: 22.35x
Adding requests: 100%|████| 40/40 [00:00<00:00, 947.42it/s]
Processed prompts: 100%|████| 40/40 [00:09<00:00,  4.15it/s, est. speed input: 799.18 toks/s, output: 1654.94 toks/s]
[T3] Speech Token Generation time: 9.68s
[S3Gen] Wavform Generation time: 53.66s
Adding requests: 100%|████| 40/40 [00:00<00:00, 858.75it/s]
Processed prompts: 100%|████| 40/40 [00:08<00:00,  4.69it/s, est. speed input: 938.19 toks/s, output: 1874.97 toks/s]
[T3] Speech Token Generation time: 8.58s
[S3Gen] Wavform Generation time: 53.86s
Adding requests: 100%|████| 40/40 [00:00<00:00, 815.60it/s]
Processed prompts: 100%|████| 40/40 [00:09<00:00,  4.19it/s, est. speed input: 726.62 toks/s, output: 1531.24 toks/s]
[T3] Speech Token Generation time: 9.60s
[S3Gen] Wavform Generation time: 49.89s
Adding requests: 100%|████| 34/34 [00:00<00:00, 938.61it/s]
Processed prompts: 100%|████| 34/34 [00:08<00:00,  3.98it/s, est. speed input: 714.68 toks/s, output: 1439.42 toks/s]
[T3] Speech Token Generation time: 8.59s
[S3Gen] Wavform Generation time: 43.58s
[BENCHMARK] Generation completed in 238.42230987548828 seconds
[BENCHMARK] Audio saved to benchmark.mp3
[BENCHMARK] Total time: 259.1808190345764 seconds

real    4m26.803s
user    4m42.393s
sys     0m4.285s
```


# Chatterbox Architecture

I could not find an official explanation of the Chatterbox architecture, so below is my best explanation based on the codebase. Chatterbox broadly follows the [CosyVoice](https://funaudiollm.github.io/cosyvoice2/) architecture, applying intermediate fusion multimodal conditioning to a 0.5B parameter Llama model.

<div align="center">
  <img src="https://github.com/randombk/chatterbox-vllm/raw/refs/heads/master/docs/chatterbox-architecture.svg" alt="Chatterbox Architecture" width="100%" />
  <p><em>Chatterbox Architecture Diagram</em></p>
</div>

# Implementation Notes

## CFG Implementation Details

vLLM does not support CFG natively, so substantial hacks were needed to make it work. At a high level, we trick vLLM into thinking the model has double the hidden dimension size as it actually does, then splitting and restacking the states to invoke Llama with double the original batch size. This does pose a risk that vLLM will underestimate the memory requirements of the model - more research is needed into whether vLLM's initial profiling pass will capture this nuance.


<div align="center">
  <img src="https://github.com/randombk/chatterbox-vllm/raw/refs/heads/master/docs/vllm-cfg-impl.svg" alt="vLLM CFG Implementation" width="100%" />
  <p><em>vLLM CFG Implementation</em></p>
</div>

# Changelog

## `0.2.1`
* Updated to multilingual v2

## `0.2.0`
* Initial multilingual support.

## `0.1.5`
* Fix Python packaging missing the tokenizer.json file

## `0.1.4`
* Change default step count back to 10 due to feedback about quality degradation.
* Fixed a bug in the `gradio_tts_app.py` implementation (#13).
* Fixed a bug with how symlinks don't work if the module is installed normally (vs as a dev environment) (#12).
  * The whole structure of how this project should be integrated into downstream repos is something that needs rethinking.

## `0.1.3`
* Added ability to tweak S3Gen diffusion steps, and default it to 5 (originally 10). This improves performance with nearly indistinguishable quality loss.

## `0.1.2`
* Update to `vllm 0.10.0`
* Fixed error where batched requests sometimes get truncated, or otherwise jumbled.
  * This also removes the need to double-apply batching when submitting requests. You can submit as many prompts as you'd like into the `generate` function, and `vllm` should perform the batching internally without issue. See changes to `benchmark.py` for details.
  * There is still a (very rare, theoretical) possibility that this issue can still happen. If it does, submit a ticket with repro steps, and tweak your max batch size or max token count as a workaround.


## `0.1.1`
* Misc minor cleanups
* API changes:
  * Use `max_batch_size` instead of `gpu_memory_utilization`
  * Use `compile=False` (default) instead of `enforce_eager=True`
  * Look at the latest examples to follow API changes. As a reminder, I do not expect the API to become stable until `1.0.0`.

## `0.1.0`
* Initial publication to pypi
* Moved audio conditioning processing out of vLLM to avoid re-computing it for every request.

## `0.0.1`
* Initial release
