#!/usr/bin/env python3
"""
Example client for the Chatterbox TTS API server.
Demonstrates various ways to use the OpenAI-compatible API.
"""

import requests
from pathlib import Path

API_BASE_URL = "http://localhost:8000"


def generate_speech(text, voice="alloy", language_id=None, output_file="output.mp3", **kwargs):
    """
    Generate speech using the Chatterbox TTS API.
    
    Args:
        text: Text to synthesize
        voice: Voice name (alloy, echo, fable, onyx, nova, shimmer, or language codes)
        language_id: Explicit language code (for multilingual model)
        output_file: Output file path
        **kwargs: Additional parameters (exaggeration, temperature, response_format, etc.)
    """
    payload = {
        "model": "tts-1",
        "input": text,
        "voice": voice,
        **kwargs
    }
    
    if language_id:
        payload["language_id"] = language_id
    
    print(f"Generating speech: {text[:50]}...")
    print(f"  Voice: {voice}")
    if language_id:
        print(f"  Language: {language_id}")
    
    response = requests.post(
        f"{API_BASE_URL}/v1/audio/speech",
        json=payload,
        timeout=120,
    )
    
    if response.status_code == 200:
        with open(output_file, "wb") as f:
            f.write(response.content)
        print(f"  ✓ Saved to: {output_file}")
        return True
    else:
        print(f"  ✗ Error: {response.status_code} - {response.text}")
        return False


def main():
    """Run example API calls."""
    print("="*60)
    print("Chatterbox TTS API Client Examples")
    print("="*60)
    
    # Example 1: Basic English synthesis
    print("\n1. Basic English synthesis")
    generate_speech(
        "Hello! This is a test of the Chatterbox TTS API running on vLLM.",
        voice="alloy",
        output_file="example_basic.mp3"
    )
    
    # Example 2: Different voice
    print("\n2. Using a different voice")
    generate_speech(
        "This example uses a different voice reference.",
        voice="onyx",
        output_file="example_voice.mp3"
    )
    
    # Example 3: With custom exaggeration
    print("\n3. With custom exaggeration (more emotion)")
    generate_speech(
        "This sentence has more emotional exaggeration!",
        voice="alloy",
        exaggeration=0.8,
        output_file="example_exaggeration.mp3"
    )
    
    # Example 4: WAV format instead of MP3
    print("\n4. Generating WAV format")
    generate_speech(
        "This output will be in WAV format.",
        voice="alloy",
        response_format="wav",
        output_file="example_wav.wav"
    )
    
    # Example 5: Multilingual - French
    print("\n5. Multilingual - French")
    generate_speech(
        "Bonjour! Ceci est un test de l'API de synthèse vocale multilingue.",
        voice="fr",
        language_id="fr",
        output_file="example_french.mp3"
    )
    
    # Example 6: Multilingual - German
    print("\n6. Multilingual - German")
    generate_speech(
        "Hallo! Dies ist ein Test der mehrsprachigen Sprachsynthese-API.",
        voice="de",
        language_id="de",
        output_file="example_german.mp3"
    )
    
    # Example 7: Multilingual - Spanish
    print("\n7. Multilingual - Spanish")
    generate_speech(
        "¡Hola! Esta es una prueba de la API de síntesis de voz multilingüe.",
        voice="alloy",  # Using default voice with explicit language
        language_id="es",
        output_file="example_spanish.mp3"
    )
    
    # Example 8: Multilingual - Chinese
    print("\n8. Multilingual - Chinese")
    generate_speech(
        "你好！这是多语言语音合成API的测试。",
        voice="zh",
        language_id="zh",
        output_file="example_chinese.mp3"
    )
    
    # Example 9: Multilingual - Japanese
    print("\n9. Multilingual - Japanese")
    generate_speech(
        "こんにちは！これは多言語音声合成APIのテストです。",
        voice="alloy",
        language_id="ja",
        output_file="example_japanese.mp3"
    )
    
    print("\n" + "="*60)
    print("Examples complete! Check the generated audio files.")
    print("="*60)


def example_with_openai_client():
    """
    Example using the official OpenAI Python client.
    Requires: pip install openai
    """
    try:
        from openai import OpenAI
        
        print("\n" + "="*60)
        print("Example using OpenAI Python Client")
        print("="*60)
        
        client = OpenAI(
            api_key="not-needed",
            base_url=f"{API_BASE_URL}/v1"
        )
        
        print("\nGenerating speech with OpenAI client...")
        response = client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input="This example uses the official OpenAI Python client!"
        )
        
        response.stream_to_file("example_openai_client.mp3")
        print("  ✓ Saved to: example_openai_client.mp3")
        
    except ImportError:
        print("\n[INFO] OpenAI client not installed. Install with: pip install openai")


if __name__ == "__main__":
    # Check if server is running
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=2)
        if response.status_code != 200:
            print("Error: API server is not responding correctly")
            print("Make sure the server is running: python api_server.py")
            exit(1)
    except requests.exceptions.RequestException:
        print("Error: Cannot connect to API server")
        print("Make sure the server is running: python api_server.py")
        exit(1)
    
    # Run examples
    main()
    
    # Try OpenAI client example
    example_with_openai_client()
