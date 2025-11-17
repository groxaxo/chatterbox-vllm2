#!/usr/bin/env python3
"""
OpenAI-compatible TTS API server for Chatterbox TTS on vLLM.
Compatible with Open WebUI and other OpenAI TTS API clients.
"""

import io
import os
from pathlib import Path
from typing import Optional, Literal
from contextlib import asynccontextmanager

import torch
import torchaudio as ta
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

from chatterbox_vllm.tts import ChatterboxTTS


# Global model instance - always use multilingual
model = None
model_type = "multilingual"  # Always multilingual


class TTSRequest(BaseModel):
    """OpenAI-compatible TTS request model.
    
    Compatible with Open WebUI and other OpenAI TTS API clients.
    Supports both OpenAI preset voices and language codes.
    """
    model: str = Field(default="tts-1", description="Model to use (tts-1 or tts-1-hd)")
    input: str = Field(..., description="Text to synthesize", max_length=4096)
    voice: str = Field(
        default="alloy",
        description="Voice to use. OpenAI presets: alloy, echo, fable, onyx, nova, shimmer (work with any language). "
                    "Language codes: en (English), es (Spanish), fr (French), de (German), etc."
    )
    response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = Field(default="mp3", description="Audio format")
    speed: float = Field(default=1.0, ge=0.25, le=4.0, description="Speed of speech (0.25 to 4.0)")
    
    # Chatterbox-specific extensions (optional)
    exaggeration: Optional[float] = Field(default=0.5, ge=0.0, le=2.0, description="Emotion exaggeration (0.5 is neutral)")
    temperature: Optional[float] = Field(default=0.8, ge=0.0, le=2.0, description="Sampling temperature")
    language_id: Optional[str] = Field(
        default=None,
        description="Explicit language code (en, es, fr, de, etc.). If not provided, language is auto-detected from input or voice parameter."
    )


# Mapping of OpenAI voice names to reference audio files
# OpenAI voices are language-agnostic and work with any language
VOICE_REFERENCES = {
    "alloy": None,  # Use default voice (neutral, works for all languages)
    "echo": "docs/audio-sample-01.mp3",  # Works for all languages
    "fable": "docs/audio-sample-02.mp3",  # Works for all languages
    "onyx": "docs/audio-sample-03.mp3",  # Works for all languages
    "nova": None,  # Use default voice (works for all languages)
    "shimmer": None,  # Use default voice (works for all languages)
}

# Mapping of language codes to voice files (for multilingual)
# These are used when voice is specified as a language code directly
MULTILINGUAL_VOICE_REFERENCES = {
    "en": None,  # English - use default voice
    "es": None,  # Spanish - use default voice
    "fr": "docs/fr_f1.flac",
    "de": "docs/de_f1.flac",
    "zh": "docs/zh_m1.mp3",
}


def get_voice_reference(voice: str) -> Optional[str]:
    """Get the audio reference file path for a given voice name.
    
    OpenAI voice names (alloy, echo, fable, onyx, nova, shimmer) work with any language.
    Language codes (en, es, fr, de, etc.) can also be used as voice identifiers.
    """
    # Check if it's a standard OpenAI voice name
    if voice in VOICE_REFERENCES:
        ref_path = VOICE_REFERENCES[voice]
        if ref_path and Path(ref_path).exists():
            return ref_path
        return None
    
    # Check if it's a language code with a specific voice file
    if voice in MULTILINGUAL_VOICE_REFERENCES:
        ref_path = MULTILINGUAL_VOICE_REFERENCES[voice]
        if ref_path and Path(ref_path).exists():
            return ref_path
    
    return None


def detect_language_from_voice(voice: str) -> Optional[str]:
    """Detect language ID from voice name.
    
    If voice is a language code (en, es, fr, etc.), use it as the language.
    If voice is an OpenAI preset (alloy, echo, etc.), return None for auto-detection.
    """
    # If voice is already a supported language code, use it
    if model and voice in model.get_supported_languages():
        return voice
    
    # If voice is an OpenAI preset name, return None for auto-detection
    if voice in VOICE_REFERENCES:
        return None
    
    # Otherwise return None for auto-detection
    return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize model on startup and cleanup on shutdown."""
    global model, model_type
    
    # Always use multilingual model for Spanish and English support
    model_variant = os.environ.get("CHATTERBOX_MODEL", "multilingual").lower()
    if model_variant != "multilingual":
        print(f"[WARNING] CHATTERBOX_MODEL={model_variant} is not supported. Using 'multilingual' instead.")
        model_variant = "multilingual"
    
    # Settings optimized for <10GB VRAM (e.g., RTX 3060 with 8GB)
    # max_batch_size=1 ensures minimal memory usage
    # max_model_len=800 reduces memory requirements while supporting most TTS use cases
    max_batch_size = int(os.environ.get("CHATTERBOX_MAX_BATCH_SIZE", "1"))
    max_model_len = int(os.environ.get("CHATTERBOX_MAX_MODEL_LEN", "800"))
    
    # Ultra-low VRAM quantization settings (for GPUs with 4-6GB like RTX 2060, GTX 1660 Ti)
    use_quantization = os.environ.get("CHATTERBOX_USE_QUANTIZATION", "false").lower() == "true"
    quantization_method = os.environ.get("CHATTERBOX_QUANTIZATION_METHOD", "bnb-4bit")  # bnb-4bit, bnb-8bit, awq
    quantize_s3gen = os.environ.get("CHATTERBOX_QUANTIZE_S3GEN", "true").lower() == "true"
    quantize_voice_encoder = os.environ.get("CHATTERBOX_QUANTIZE_VOICE_ENCODER", "true").lower() == "true"
    # Optional GPU memory utilization (fraction), defaults to a conservative value
    gpu_memory_utilization = float(os.environ.get("CHATTERBOX_GPU_MEMORY_UTILIZATION", "0.08"))
    
    print(f"[INFO] Loading multilingual model with Spanish and English support")
    print(f"[INFO] Configuration: max_batch_size={max_batch_size}, max_model_len={max_model_len}")
    
    if use_quantization:
        print(f"[INFO] Ultra-low VRAM mode enabled with {quantization_method} quantization")
        print(f"[INFO] Configuration optimized for GPUs with 4-6GB VRAM (RTX 2060, GTX 1660 Ti)")
        print(f"[INFO] Quantization settings: s3gen={quantize_s3gen}, voice_encoder={quantize_voice_encoder}")
    else:
        print(f"[INFO] Configuration optimized for GPUs with <10GB VRAM")
    
    model_kwargs = {
        "max_batch_size": max_batch_size,
        "max_model_len": max_model_len,
        "use_quantization": use_quantization,
        "quantization_method": quantization_method,
        "quantize_s3gen": quantize_s3gen,
        "quantize_voice_encoder": quantize_voice_encoder,
        "gpu_memory_utilization": gpu_memory_utilization,
        "enforce_eager": True,
    }
    
    # Always load multilingual model
    model = ChatterboxTTS.from_pretrained_multilingual(**model_kwargs)
    model_type = "multilingual"
    print(f"[INFO] Multilingual model loaded. Supported languages: {', '.join(model.get_supported_languages().keys())}")
    print(f"[INFO] Model loaded successfully. GPU memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    
    yield
    
    # Cleanup
    if model is not None:
        model.shutdown()
    print("[INFO] Model shutdown complete")


app = FastAPI(
    title="Chatterbox TTS API",
    description="OpenAI-compatible TTS API for Chatterbox TTS on vLLM",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Chatterbox TTS API Server",
        "model_type": model_type,
        "supported_languages": list(model.get_supported_languages().keys()) if model else [],
        "endpoints": {
            "tts": "/v1/audio/speech",
            "health": "/health",
            "models": "/v1/models",
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_type": model_type,
        "gpu_memory_mb": torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0,
    }


@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI-compatible endpoint)."""
    models = [
        {
            "id": "tts-1",
            "object": "model",
            "created": 1700000000,
            "owned_by": "chatterbox-vllm",
        },
        {
            "id": "tts-1-hd",
            "object": "model", 
            "created": 1700000000,
            "owned_by": "chatterbox-vllm",
        }
    ]
    return {"object": "list", "data": models}


@app.post("/v1/audio/speech")
async def create_speech(request: TTSRequest):
    """
    Generate speech from text (OpenAI-compatible endpoint).
    
    This endpoint is compatible with Open WebUI and other OpenAI TTS API clients.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not request.input or len(request.input.strip()) == 0:
        raise HTTPException(status_code=400, detail="Input text is required")
    
    try:
        # Determine language from voice name or explicit language_id
        language_id = request.language_id or detect_language_from_voice(request.voice)
        
        # Default to English if no language specified or detected
        if language_id is None:
            language_id = "en"
        
        # Validate language support (multilingual model always loaded)
        if language_id not in model.get_supported_languages():
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported language '{language_id}'. Supported: {', '.join(model.get_supported_languages().keys())}"
            )
        
        # Get voice reference audio
        audio_prompt_path = get_voice_reference(request.voice)
        
        # Generate audio
        # OpenAI voices (alloy, echo, fable, onyx, nova, shimmer) are language-agnostic
        # and work with any language based on the input text or language_id
        print(f"[TTS] Generating speech for text: '{request.input[:50]}...' (language: {language_id}, voice: {request.voice}, reference: {audio_prompt_path or 'default'})")
        
        audios = model.generate(
            [request.input],
            audio_prompt_path=audio_prompt_path,
            language_id=language_id,
            exaggeration=request.exaggeration,
            temperature=request.temperature,
        )
        
        if not audios or len(audios) == 0:
            raise HTTPException(status_code=500, detail="Failed to generate audio")
        
        audio_tensor = audios[0]
        
        # Convert to requested format
        buffer = io.BytesIO()
        
        # Save audio to buffer in the requested format
        if request.response_format == "mp3":
            ta.save(buffer, audio_tensor, model.sr, format="mp3")
            media_type = "audio/mpeg"
        elif request.response_format == "wav":
            ta.save(buffer, audio_tensor, model.sr, format="wav")
            media_type = "audio/wav"
        elif request.response_format == "flac":
            ta.save(buffer, audio_tensor, model.sr, format="flac")
            media_type = "audio/flac"
        elif request.response_format == "opus":
            # Opus requires oggenc, fall back to wav if not available
            try:
                ta.save(buffer, audio_tensor, model.sr, format="opus")
                media_type = "audio/opus"
            except:
                ta.save(buffer, audio_tensor, model.sr, format="wav")
                media_type = "audio/wav"
        elif request.response_format == "aac":
            # AAC might not be available, fall back to mp3
            try:
                ta.save(buffer, audio_tensor, model.sr, format="mp4", bits_per_sample=16)
                media_type = "audio/aac"
            except:
                ta.save(buffer, audio_tensor, model.sr, format="mp3")
                media_type = "audio/mpeg"
        elif request.response_format == "pcm":
            # PCM is raw audio data
            ta.save(buffer, audio_tensor, model.sr, format="wav")
            media_type = "audio/pcm"
        else:
            # Default to mp3
            ta.save(buffer, audio_tensor, model.sr, format="mp3")
            media_type = "audio/mpeg"
        
        buffer.seek(0)
        audio_data = buffer.read()
        
        print(f"[TTS] Generated {len(audio_data)} bytes of audio ({request.response_format})")
        
        return Response(content=audio_data, media_type=media_type)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"[ERROR] Failed to generate speech: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to generate speech: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    
    # Configuration
    host = os.environ.get("CHATTERBOX_HOST", "0.0.0.0")
    port = int(os.environ.get("CHATTERBOX_PORT", "8000"))
    
    print(f"Starting Chatterbox TTS API server (Multilingual) on {host}:{port}")
    print("=" * 60)
    print("Configuration:")
    print(f"  Model: Multilingual (Spanish & English support)")
    print(f"  Max Batch Size: {os.environ.get('CHATTERBOX_MAX_BATCH_SIZE', '1')}")
    print(f"  Max Model Length: {os.environ.get('CHATTERBOX_MAX_MODEL_LEN', '800')}")
    print(f"  OpenAI Voice Presets: alloy, echo, fable, onyx, nova, shimmer")
    print(f"  Supported Languages: 23 languages including en, es, fr, de, etc.")
    print("=" * 60)
    
    uvicorn.run(
        "api_server:app",
        host=host,
        port=port,
        log_level="info",
    )
