#!/usr/bin/env python3
"""
Unit tests for quantization module.

Tests the quantization utilities without requiring actual model loading or GPU.
"""

import sys
import os
import unittest
from unittest.mock import Mock, patch

# Add src directory to path so we can import without installing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


class TestQuantizationModule(unittest.TestCase):
    """Test the quantization module."""
    
    def test_import_quantization_module(self):
        """Test that quantization module can be imported."""
        try:
            from chatterbox_vllm import quantization
            self.assertIsNotNone(quantization)
        except ImportError as e:
            self.fail(f"Failed to import quantization module: {e}")
    
    def test_check_quantization_support(self):
        """Test checking quantization support."""
        from chatterbox_vllm.quantization import check_quantization_support
        
        support = check_quantization_support()
        self.assertIsInstance(support, dict)
        self.assertIn("bitsandbytes", support)
        self.assertIn("awq", support)
        self.assertIsInstance(support["bitsandbytes"], bool)
        self.assertIsInstance(support["awq"], bool)
    
    def test_estimate_memory_savings_4bit(self):
        """Test memory savings estimation for 4-bit quantization."""
        from chatterbox_vllm.quantization import estimate_memory_savings
        
        base_memory = 1000.0  # MB
        result = estimate_memory_savings(base_memory, quantization_bits=4)
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result["base_memory_mb"], base_memory)
        self.assertEqual(result["quantization_bits"], 4)
        self.assertAlmostEqual(result["estimated_memory_mb"], base_memory / 4, places=2)
        self.assertAlmostEqual(result["savings_percent"], 75.0, places=1)
    
    def test_estimate_memory_savings_8bit(self):
        """Test memory savings estimation for 8-bit quantization."""
        from chatterbox_vllm.quantization import estimate_memory_savings
        
        base_memory = 2000.0  # MB
        result = estimate_memory_savings(base_memory, quantization_bits=8)
        
        self.assertEqual(result["quantization_bits"], 8)
        self.assertAlmostEqual(result["estimated_memory_mb"], base_memory / 2, places=2)
        self.assertAlmostEqual(result["savings_percent"], 50.0, places=1)
    
    def test_get_vllm_quantization_config_awq(self):
        """Test vLLM quantization config for AWQ."""
        from chatterbox_vllm.quantization import get_vllm_quantization_config
        
        # Test AWQ
        config = get_vllm_quantization_config("awq")
        # If AWQ is not available, should return empty dict
        self.assertIsInstance(config, dict)
        if config:
            self.assertEqual(config.get("quantization"), "awq")
    
    def test_get_vllm_quantization_config_none(self):
        """Test vLLM quantization config with None."""
        from chatterbox_vllm.quantization import get_vllm_quantization_config
        
        config = get_vllm_quantization_config(None)
        self.assertEqual(config, {})
    
    def test_get_vllm_quantization_config_unknown(self):
        """Test vLLM quantization config with unknown method."""
        from chatterbox_vllm.quantization import get_vllm_quantization_config
        
        config = get_vllm_quantization_config("unknown_method")
        self.assertEqual(config, {})


class TestQuantizationIntegration(unittest.TestCase):
    """Test quantization integration with TTS module."""
    
    def test_tts_module_has_quantization_params(self):
        """Test that TTS module accepts quantization parameters."""
        try:
            from chatterbox_vllm.tts import ChatterboxTTS
            import inspect
            
            # Check from_local method signature
            sig = inspect.signature(ChatterboxTTS.from_local)
            params = sig.parameters
            
            # Verify quantization parameters exist
            self.assertIn("use_quantization", params)
            self.assertIn("quantization_method", params)
            self.assertIn("quantize_s3gen", params)
            self.assertIn("quantize_voice_encoder", params)
            
            # Verify default values
            self.assertEqual(params["use_quantization"].default, False)
            self.assertIsNone(params["quantization_method"].default)
            self.assertEqual(params["quantize_s3gen"].default, False)
            self.assertEqual(params["quantize_voice_encoder"].default, False)
        except ModuleNotFoundError as e:
            self.skipTest(f"Required dependencies not installed: {e}")


def run_tests():
    """Run all tests."""
    print("="*70)
    print("Running Quantization Module Tests")
    print("="*70)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestQuantizationModule))
    suite.addTests(loader.loadTestsFromTestCase(TestQuantizationIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("="*70)
    
    # Return exit code
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_tests())
