#!/usr/bin/env python3
"""
Example script demonstrating ultra-low VRAM mode with BnB quantization.

This mode is optimized for GPUs with 4-6GB VRAM (RTX 2060, GTX 1660 Ti, etc.)
using 4-bit quantization with BitsAndBytes (BnB) to reduce memory usage.

Requirements:
    pip install bitsandbytes

Expected memory usage: ~4-6GB (compared to ~7-8GB in standard low VRAM mode)
"""

import torch
import torchaudio as ta
from chatterbox_vllm.tts import ChatterboxTTS

AUDIO_PROMPT_PATH = "docs/audio-sample-01.mp3"
MAX_MODEL_LEN = 600  # Reduced from 800 for ultra-low VRAM

if __name__ == "__main__":
    # Print current GPU memory usage
    print(f"[START] Starting GPU memory usage: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"[INFO] Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    print("\n" + "="*70)
    print("ULTRA-LOW VRAM MODE - Using BnB 4-bit Quantization")
    print("="*70)
    print("This mode uses BitsAndBytes (BnB) 4-bit quantization to reduce VRAM usage.")
    print("Expected memory savings: 25-40% compared to standard low VRAM mode")
    print("Target GPUs: RTX 2060 (6GB), GTX 1660 Ti (6GB), GTX 1650 (4GB)")
    print("="*70 + "\n")
    
    # Initialize model with ultra-low VRAM settings
    model = ChatterboxTTS.from_pretrained(
        # Minimal batch and sequence length
        max_batch_size=1,
        max_model_len=MAX_MODEL_LEN,
        
        # Enable quantization
        use_quantization=True,
        quantization_method="bnb-4bit",  # Use 4-bit BnB quantization
        quantize_s3gen=True,             # Quantize S3Gen model
        quantize_voice_encoder=True,     # Quantize Voice Encoder
    )

    print(f"\n[POST-INIT] GPU memory usage after model load: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    
    # Generate audio conditioning
    # The resulting s3gen_ref and cond_emb can be reused for multiple generations
    s3gen_ref, cond_emb = model.get_audio_conditionals(AUDIO_PROMPT_PATH)

    # Generate audio with different prompts
    prompts = [
        "This is a demonstration of ultra-low VRAM mode using BitsAndBytes quantization.",
        "The model is now using 4-bit quantization to reduce memory usage significantly.",
    ]
    
    print(f"\n[INFO] Generating {len(prompts)} audio samples...")
    
    for idx, prompt in enumerate(prompts):
        print(f"\n[PROMPT {idx+1}] {prompt}")
        
        # Generate audio
        cond_emb_adjusted = model.update_exaggeration(cond_emb, exaggeration=0.5)
        audios = model.generate_with_conds(
            [prompt],
            s3gen_ref=s3gen_ref,
            cond_emb=cond_emb_adjusted,
            min_p=0.1,
        )
        
        print(f"[POST-GEN {idx+1}] GPU memory usage: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        
        # Save audio
        output_file = f"test-ultra-low-vram-{idx}.mp3"
        ta.save(output_file, audios[0], model.sr)
        print(f"[SAVED] Audio saved to: {output_file}")
    
    # Print final memory statistics
    print("\n" + "="*70)
    print("MEMORY USAGE SUMMARY")
    print("="*70)
    print(f"Final GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"Peak GPU memory reserved:   {torch.cuda.max_memory_reserved() / 1024**2:.2f} MB")
    print(f"Total GPU memory:           {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print("="*70)
    
    # Cleanup
    model.shutdown()
    print("\n[INFO] Model shutdown complete")
