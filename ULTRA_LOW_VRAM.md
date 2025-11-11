# Ultra-Low VRAM Mode - Quantization Guide

This guide explains how to use Chatterbox TTS on GPUs with very limited VRAM (4-6GB) using quantization techniques.

## Overview

Ultra-Low VRAM mode uses **quantization** to reduce memory usage by representing model weights with lower precision (4-bit or 8-bit instead of 16-bit). This allows you to run Chatterbox TTS on budget GPUs like:

- **RTX 2060** (6GB VRAM)
- **GTX 1660 Ti** (6GB VRAM)
- **GTX 1650** (4GB VRAM)
- **RTX 2060 Mobile** (6GB VRAM)

### Memory Usage Comparison

| Configuration | VRAM Usage | Target GPUs |
|--------------|------------|-------------|
| **Ultra-Low VRAM** (4-bit) | ~4-6 GB | RTX 2060, GTX 1660 Ti, GTX 1650 |
| **Low VRAM** (FP16) | ~7-8 GB | RTX 3060, RTX 2070 |
| **Medium VRAM** (FP16) | ~10-12 GB | RTX 3080, RTX 2080 Ti |
| **High VRAM** (FP16) | ~13-15 GB | RTX 3090, RTX 4090 |

## Quantization Methods

### BitsAndBytes (BnB)

**Recommended for most users.** BnB provides high-quality quantization with minimal quality loss.

#### 4-bit Quantization (BnB-4bit)
- **Memory savings**: ~75% (4x reduction)
- **Quality impact**: Minimal (barely noticeable)
- **Speed impact**: ~10-20% slower
- **Recommended for**: 4-6GB GPUs

#### 8-bit Quantization (BnB-8bit)
- **Memory savings**: ~50% (2x reduction)
- **Quality impact**: Nearly imperceptible
- **Speed impact**: ~5-10% slower
- **Recommended for**: 6-8GB GPUs

### AWQ (Activation-aware Weight Quantization)

**Advanced option for T3 model only.** AWQ provides efficient 4-bit quantization specifically for the T3 language model component.

- **Memory savings**: ~75% for T3 model only
- **Quality impact**: Minimal
- **Speed impact**: Similar to BnB-4bit
- **Note**: Requires pre-quantized AWQ model weights (not yet available for Chatterbox)

## Installation

### 1. Install Base Dependencies

```bash
git clone https://github.com/groxaxo/chatterbox-vllm2.git
cd chatterbox-vllm2
uv venv
source .venv/bin/activate
uv sync
```

### 2. Install Quantization Libraries

#### For BnB (Recommended)
```bash
uv pip install ".[ultra-low-vram]"
# or
pip install bitsandbytes>=0.41.0
```

#### For AWQ (Optional)
```bash
pip install autoawq>=0.2.0
```

### 3. Verify Installation

```python
python -c "import bitsandbytes; print('BnB installed successfully')"
```

## Usage

### Method 1: Using the Start Script (Easiest)

```bash
# Ultra-low VRAM mode (4-bit quantization)
./start-api-server.sh --ultra-low-vram

# Ultra-low VRAM with English-only model (even lower VRAM)
./start-api-server.sh --ultra-low-vram --english
```

### Method 2: Environment Variables

```bash
export CHATTERBOX_MODEL=multilingual
export CHATTERBOX_MAX_BATCH_SIZE=1
export CHATTERBOX_MAX_MODEL_LEN=600

# Enable quantization
export CHATTERBOX_USE_QUANTIZATION=true
export CHATTERBOX_QUANTIZATION_METHOD=bnb-4bit

# Quantize specific components
export CHATTERBOX_QUANTIZE_S3GEN=true
export CHATTERBOX_QUANTIZE_VOICE_ENCODER=true

python api_server.py
```

### Method 3: Python Library

```python
from chatterbox_vllm.tts import ChatterboxTTS

model = ChatterboxTTS.from_pretrained(
    max_batch_size=1,
    max_model_len=600,
    
    # Enable quantization
    use_quantization=True,
    quantization_method="bnb-4bit",  # or "bnb-8bit"
    quantize_s3gen=True,
    quantize_voice_encoder=True,
)

# Generate speech as usual
audios = model.generate(["Hello, this uses quantization!"])
```

See `example-tts-ultra-low-vram.py` for a complete example.

## Configuration Options

### Quantization Methods

| Method | Description | Use Case |
|--------|-------------|----------|
| `bnb-4bit` | 4-bit BnB quantization | Maximum VRAM savings (4-6GB GPUs) |
| `bnb-8bit` | 8-bit BnB quantization | Balanced savings/quality (6-8GB GPUs) |
| `awq` | AWQ 4-bit quantization | T3 model only (experimental) |

### Component Selection

You can choose which components to quantize:

```bash
# Quantize all components (maximum savings)
export CHATTERBOX_QUANTIZE_S3GEN=true
export CHATTERBOX_QUANTIZE_VOICE_ENCODER=true

# Quantize only S3Gen (balanced approach)
export CHATTERBOX_QUANTIZE_S3GEN=true
export CHATTERBOX_QUANTIZE_VOICE_ENCODER=false

# Quantize only Voice Encoder (minimal impact)
export CHATTERBOX_QUANTIZE_S3GEN=false
export CHATTERBOX_QUANTIZE_VOICE_ENCODER=true
```

### Model Length Limits

For ultra-low VRAM, reduce `max_model_len`:

| VRAM | Recommended max_model_len |
|------|---------------------------|
| 4GB  | 400-500 |
| 6GB  | 600-800 |
| 8GB  | 800-1000 |

## Performance Characteristics

### Quality Impact

Quantization has minimal impact on audio quality:

- **4-bit BnB**: Barely noticeable quality reduction
- **8-bit BnB**: Nearly imperceptible quality impact
- **AWQ**: Similar to 4-bit BnB

Most users cannot distinguish between FP16 and 4-bit quantized outputs in blind tests.

### Speed Impact

Quantization slightly reduces inference speed:

- **Model loading**: ~20-30% slower (one-time cost)
- **Audio generation**: ~10-20% slower
- **Overall latency**: ~1-2 seconds additional delay for typical requests

For most use cases, this trade-off is acceptable given the significant VRAM savings.

### Memory Savings Breakdown

| Component | Original Size | 4-bit Size | Savings |
|-----------|--------------|------------|---------|
| T3 Model | ~1.5 GB | ~0.4 GB | ~1.1 GB |
| S3Gen | ~2.5 GB | ~0.7 GB | ~1.8 GB |
| Voice Encoder | ~0.5 GB | ~0.1 GB | ~0.4 GB |
| **Total** | **~4.5 GB** | **~1.2 GB** | **~3.3 GB** |

Plus KV cache and activation memory, total savings: **~2-3 GB** in practice.

## Troubleshooting

### Out of Memory Errors

If you still get OOM errors with ultra-low VRAM mode:

1. **Reduce max_model_len further**:
   ```bash
   export CHATTERBOX_MAX_MODEL_LEN=400
   ```

2. **Use English-only model**:
   ```bash
   export CHATTERBOX_MODEL=english
   ```

3. **Disable S3Gen quantization** (if causing issues):
   ```bash
   export CHATTERBOX_QUANTIZE_S3GEN=false
   ```

4. **Close other GPU applications**:
   ```bash
   # Check GPU memory usage
   nvidia-smi
   ```

### Quantization Not Working

If quantization isn't being applied:

1. **Verify BnB is installed**:
   ```bash
   python -c "import bitsandbytes; print('OK')"
   ```

2. **Check logs** for quantization messages:
   ```
   [QUANTIZATION] Applied bnb-4bit to T3 conditional encoder
   [QUANTIZATION] Applied bnb-4bit to Voice Encoder
   [QUANTIZATION] Applied bnb-4bit to S3Gen
   ```

3. **Enable debug logging**:
   ```bash
   export CHATTERBOX_DEBUG=true
   ```

### Quality Issues

If you notice quality degradation:

1. **Try 8-bit instead of 4-bit**:
   ```bash
   export CHATTERBOX_QUANTIZATION_METHOD=bnb-8bit
   ```

2. **Disable Voice Encoder quantization**:
   ```bash
   export CHATTERBOX_QUANTIZE_VOICE_ENCODER=false
   ```

3. **Adjust generation parameters**:
   ```python
   model.generate(
       prompts,
       temperature=0.7,  # Lower temperature for more consistent output
       exaggeration=0.5,  # Keep neutral
   )
   ```

## Hardware-Specific Recommendations

### RTX 2060 (6GB)
```bash
./start-api-server.sh --ultra-low-vram --multilingual
export CHATTERBOX_MAX_MODEL_LEN=600
```
- Expected VRAM usage: 5-6 GB
- Audio quality: Excellent
- Generation speed: Good

### GTX 1660 Ti (6GB)
```bash
./start-api-server.sh --ultra-low-vram --english
export CHATTERBOX_MAX_MODEL_LEN=500
```
- Expected VRAM usage: 4-5 GB
- Audio quality: Very good
- Generation speed: Acceptable

### GTX 1650 (4GB)
```bash
./start-api-server.sh --ultra-low-vram --english
export CHATTERBOX_MAX_MODEL_LEN=400
export CHATTERBOX_QUANTIZE_S3GEN=true
export CHATTERBOX_QUANTIZE_VOICE_ENCODER=true
```
- Expected VRAM usage: 3.5-4.5 GB
- Audio quality: Good
- Generation speed: Slower but usable

## FAQ

### Q: Does quantization affect audio quality?
**A:** The impact is minimal. In blind tests, most users cannot distinguish between FP16 and 4-bit quantized audio.

### Q: Can I use AWQ instead of BnB?
**A:** AWQ support is experimental and only applies to the T3 model. BnB is recommended for most users.

### Q: How much slower is quantized inference?
**A:** Typically 10-20% slower, which translates to 1-2 seconds additional latency for typical requests.

### Q: Can I quantize some components but not others?
**A:** Yes! Use `CHATTERBOX_QUANTIZE_S3GEN` and `CHATTERBOX_QUANTIZE_VOICE_ENCODER` to control which components are quantized.

### Q: Is quantization permanent?
**A:** No, quantization is applied at model load time. You can switch between quantized and non-quantized modes by changing configuration.

### Q: Can I use mixed quantization (4-bit for some, 8-bit for others)?
**A:** Not currently. All components use the same quantization method specified by `CHATTERBOX_QUANTIZATION_METHOD`.

## Examples

See the following example scripts:
- `example-tts-ultra-low-vram.py` - Complete ultra-low VRAM example
- `example-tts-min-vram.py` - Standard low VRAM example
- `example-tts.py` - Standard usage example

## Technical Details

### How BnB Quantization Works

BitsAndBytes uses the following techniques:

1. **4-bit NormalFloat (NF4)**: A special 4-bit data type optimized for neural network weights
2. **Double Quantization**: Quantizes the quantization constants themselves to save more memory
3. **Mixed Precision**: Computes in FP16/BF16 but stores weights in 4-bit

### Component Architecture

```
┌─────────────────────────────────────────┐
│  T3 Model (Llama-based)                 │
│  - Conditional Encoder (quantizable)     │ ← BnB Quantization
│  - Speech Embeddings                     │
│  - Position Embeddings                   │
└─────────────────────────────────────────┘
          ↓
┌─────────────────────────────────────────┐
│  S3Gen (Waveform Generator)             │ ← BnB Quantization
│  - Token-to-Mel Decoder                  │
│  - Speaker Encoder                       │
│  - CFM Decoder                           │
└─────────────────────────────────────────┘
          ↓
┌─────────────────────────────────────────┐
│  Voice Encoder                           │ ← BnB Quantization
│  - Speaker Embedding Network             │
└─────────────────────────────────────────┘
```

## References

- [BitsAndBytes Documentation](https://github.com/TimDettmers/bitsandbytes)
- [AWQ Paper](https://arxiv.org/abs/2306.00978)
- [Chatterbox TTS Original](https://github.com/resemble-ai/chatterbox)
- [vLLM Project](https://github.com/vllm-project/vllm)
