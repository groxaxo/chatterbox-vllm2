#!/usr/bin/env python3
"""
Test script for the Chatterbox TTS API server.
Run this after starting the API server to verify it's working correctly.
"""

import requests
import time
import sys

API_BASE_URL = "http://localhost:8000"


def test_health_check():
    """Test the health check endpoint."""
    print("\n" + "="*60)
    print("Testing health check endpoint...")
    print("="*60)
    
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        response.raise_for_status()
        data = response.json()
        
        print(f"✓ Health check passed")
        print(f"  Status: {data.get('status')}")
        print(f"  Model loaded: {data.get('model_loaded')}")
        print(f"  Model type: {data.get('model_type')}")
        print(f"  GPU memory: {data.get('gpu_memory_mb', 0):.2f} MB")
        return True
    except Exception as e:
        print(f"✗ Health check failed: {e}")
        return False


def test_models_endpoint():
    """Test the models listing endpoint."""
    print("\n" + "="*60)
    print("Testing models endpoint...")
    print("="*60)
    
    try:
        response = requests.get(f"{API_BASE_URL}/v1/models")
        response.raise_for_status()
        data = response.json()
        
        print(f"✓ Models endpoint working")
        print(f"  Available models: {len(data.get('data', []))}")
        for model in data.get('data', []):
            print(f"    - {model.get('id')}")
        return True
    except Exception as e:
        print(f"✗ Models endpoint failed: {e}")
        return False


def test_tts_generation(text, voice="alloy", language_id=None, output_file="test_output.mp3"):
    """Test TTS generation."""
    print("\n" + "="*60)
    print(f"Testing TTS generation...")
    print(f"  Text: {text[:50]}...")
    print(f"  Voice: {voice}")
    if language_id:
        print(f"  Language: {language_id}")
    print("="*60)
    
    try:
        payload = {
            "model": "tts-1",
            "input": text,
            "voice": voice,
            "response_format": "mp3",
        }
        
        if language_id:
            payload["language_id"] = language_id
        
        start_time = time.time()
        response = requests.post(
            f"{API_BASE_URL}/v1/audio/speech",
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        elapsed_time = time.time() - start_time
        
        # Save audio to file
        with open(output_file, "wb") as f:
            f.write(response.content)
        
        file_size = len(response.content) / 1024  # KB
        
        print(f"✓ TTS generation successful")
        print(f"  Generated in: {elapsed_time:.2f}s")
        print(f"  File size: {file_size:.2f} KB")
        print(f"  Saved to: {output_file}")
        return True
    except Exception as e:
        print(f"✗ TTS generation failed: {e}")
        return False


def test_root_endpoint():
    """Test the root endpoint."""
    print("\n" + "="*60)
    print("Testing root endpoint...")
    print("="*60)
    
    try:
        response = requests.get(f"{API_BASE_URL}/")
        response.raise_for_status()
        data = response.json()
        
        print(f"✓ Root endpoint working")
        print(f"  Model type: {data.get('model_type')}")
        print(f"  Supported languages: {len(data.get('supported_languages', []))}")
        if data.get('model_type') == 'multilingual':
            print(f"    Languages: {', '.join(data.get('supported_languages', []))}")
        return True
    except Exception as e:
        print(f"✗ Root endpoint failed: {e}")
        return False


def run_all_tests():
    """Run all tests."""
    print("\n" + "#"*60)
    print("# Chatterbox TTS API Test Suite")
    print("#"*60)
    
    # Wait for server to be ready
    print("\nWaiting for server to be ready...")
    max_retries = 10
    for i in range(max_retries):
        try:
            response = requests.get(f"{API_BASE_URL}/health", timeout=2)
            if response.status_code == 200:
                print("✓ Server is ready!")
                break
        except:
            pass
        
        if i < max_retries - 1:
            print(f"  Attempt {i+1}/{max_retries} - waiting...")
            time.sleep(2)
        else:
            print(f"\n✗ Server not responding after {max_retries} attempts")
            print(f"  Make sure the server is running: python api_server.py")
            return False
    
    results = []
    
    # Test 1: Root endpoint
    results.append(("Root endpoint", test_root_endpoint()))
    
    # Test 2: Health check
    results.append(("Health check", test_health_check()))
    
    # Test 3: Models endpoint
    results.append(("Models endpoint", test_models_endpoint()))
    
    # Test 4: Basic TTS (English)
    results.append((
        "TTS - English",
        test_tts_generation(
            "Hello! This is a test of the Chatterbox TTS API.",
            voice="alloy",
            output_file="test_output_en.mp3"
        )
    ))
    
    # Test 5: TTS with different voice
    results.append((
        "TTS - Different voice",
        test_tts_generation(
            "This is testing a different voice.",
            voice="onyx",
            output_file="test_output_voice.mp3"
        )
    ))
    
    # Test 6: Multilingual (if supported)
    # Get server info first
    try:
        response = requests.get(f"{API_BASE_URL}/")
        data = response.json()
        if data.get('model_type') == 'multilingual':
            results.append((
                "TTS - French",
                test_tts_generation(
                    "Bonjour! Ceci est un test de l'API Chatterbox TTS.",
                    voice="fr",
                    language_id="fr",
                    output_file="test_output_fr.mp3"
                )
            ))
            
            results.append((
                "TTS - German",
                test_tts_generation(
                    "Hallo! Dies ist ein Test der Chatterbox TTS API.",
                    voice="de",
                    language_id="de",
                    output_file="test_output_de.mp3"
                )
            ))
    except:
        pass
    
    # Print summary
    print("\n" + "#"*60)
    print("# Test Summary")
    print("#"*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print("\n" + "="*60)
    print(f"Results: {passed}/{total} tests passed")
    print("="*60)
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return True
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
