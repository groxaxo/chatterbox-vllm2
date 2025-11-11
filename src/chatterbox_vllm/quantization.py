"""
Quantization utilities for ultra-low VRAM mode.

This module provides utilities for quantizing models using BitsAndBytes (BnB)
and AWQ to reduce VRAM usage for users with limited GPU memory.
"""

from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Check availability of quantization libraries
try:
    import bitsandbytes as bnb
    BNB_AVAILABLE = True
except ImportError:
    BNB_AVAILABLE = False
    logger.warning("bitsandbytes not available. Ultra-low VRAM mode will not work. Install with: pip install bitsandbytes")

try:
    import awq
    AWQ_AVAILABLE = True
except ImportError:
    AWQ_AVAILABLE = False
    logger.debug("autoawq not available. AWQ quantization will not be used.")


def check_quantization_support() -> dict[str, bool]:
    """Check which quantization methods are available."""
    return {
        "bitsandbytes": BNB_AVAILABLE,
        "awq": AWQ_AVAILABLE,
    }


def apply_bnb_quantization(model, quantization_config: Optional[dict] = None):
    """
    Apply BitsAndBytes quantization to a PyTorch model.
    
    Args:
        model: PyTorch model to quantize
        quantization_config: Configuration for quantization. Supported keys:
            - load_in_8bit (bool): Use 8-bit quantization (default: False)
            - load_in_4bit (bool): Use 4-bit quantization (default: True)
            - bnb_4bit_compute_dtype: Compute dtype for 4-bit (default: float16)
            - bnb_4bit_use_double_quant (bool): Use double quantization (default: True)
            - bnb_4bit_quant_type (str): Quantization type, 'nf4' or 'fp4' (default: 'nf4')
    
    Returns:
        Quantized model
    """
    if not BNB_AVAILABLE:
        logger.warning("BitsAndBytes not available. Returning unquantized model.")
        return model
    
    import torch
    from transformers import BitsAndBytesConfig
    
    if quantization_config is None:
        quantization_config = {}
    
    # Default to 4-bit NF4 quantization with double quantization
    config = BitsAndBytesConfig(
        load_in_4bit=quantization_config.get("load_in_4bit", True),
        load_in_8bit=quantization_config.get("load_in_8bit", False),
        bnb_4bit_compute_dtype=quantization_config.get("bnb_4bit_compute_dtype", torch.float16),
        bnb_4bit_use_double_quant=quantization_config.get("bnb_4bit_use_double_quant", True),
        bnb_4bit_quant_type=quantization_config.get("bnb_4bit_quant_type", "nf4"),
    )
    
    logger.info(f"Applying BnB quantization with config: {config}")
    
    try:
        # For models that support direct quantization config
        if hasattr(model, 'quantization_config'):
            model.quantization_config = config
        
        # Apply quantization to linear layers
        from bitsandbytes.nn import Linear8bitLt, Linear4bit
        import torch.nn as nn
        
        def replace_linear_with_bnb(module, load_in_8bit=False, load_in_4bit=True):
            """Recursively replace Linear layers with BnB quantized versions."""
            for name, child in module.named_children():
                if isinstance(child, nn.Linear):
                    # Get the linear layer properties
                    in_features = child.in_features
                    out_features = child.out_features
                    bias = child.bias is not None
                    
                    # Create quantized replacement
                    if load_in_8bit:
                        new_layer = Linear8bitLt(
                            in_features, 
                            out_features, 
                            bias=bias,
                            has_fp16_weights=False,
                            threshold=6.0
                        )
                    elif load_in_4bit:
                        new_layer = Linear4bit(
                            in_features,
                            out_features,
                            bias=bias,
                            compute_dtype=config.bnb_4bit_compute_dtype,
                            compress_statistics=config.bnb_4bit_use_double_quant,
                            quant_type=config.bnb_4bit_quant_type,
                        )
                    else:
                        continue
                    
                    # Copy weights if available
                    if hasattr(child, 'weight'):
                        new_layer.weight = child.weight
                    if bias and hasattr(child, 'bias'):
                        new_layer.bias = child.bias
                    
                    # Replace the layer
                    setattr(module, name, new_layer)
                    logger.debug(f"Replaced {name} with BnB quantized layer")
                else:
                    # Recursively apply to child modules
                    replace_linear_with_bnb(child, load_in_8bit, load_in_4bit)
        
        # Apply quantization
        replace_linear_with_bnb(
            model, 
            load_in_8bit=config.load_in_8bit,
            load_in_4bit=config.load_in_4bit
        )
        
        logger.info("Successfully applied BnB quantization to model")
        return model
        
    except Exception as e:
        logger.error(f"Failed to apply BnB quantization: {e}")
        logger.warning("Returning unquantized model")
        return model


def get_vllm_quantization_config(quantization_method: str = "awq") -> dict:
    """
    Get vLLM-compatible quantization configuration.
    
    Args:
        quantization_method: Quantization method to use. Options:
            - "awq": Activation-aware Weight Quantization
            - "gptq": GPTQ quantization (4-bit)
            - "squeezellm": SqueezeLLM quantization
            - None: No quantization (default behavior)
    
    Returns:
        Dictionary with vLLM quantization configuration
    """
    if quantization_method is None:
        return {}
    
    quantization_method = quantization_method.lower()
    
    if quantization_method == "awq":
        if not AWQ_AVAILABLE:
            logger.warning("AWQ not available. Install with: pip install autoawq")
            return {}
        
        logger.info("Using AWQ quantization for vLLM T3 model")
        return {
            "quantization": "awq",
        }
    
    elif quantization_method == "gptq":
        logger.info("Using GPTQ quantization for vLLM T3 model")
        return {
            "quantization": "gptq",
        }
    
    elif quantization_method == "squeezellm":
        logger.info("Using SqueezeLLM quantization for vLLM T3 model")
        return {
            "quantization": "squeezellm",
        }
    
    else:
        logger.warning(f"Unknown quantization method: {quantization_method}")
        return {}


def estimate_memory_savings(base_memory_mb: float, quantization_bits: int = 4) -> dict:
    """
    Estimate memory savings from quantization.
    
    Args:
        base_memory_mb: Base memory usage in MB (FP32 or FP16)
        quantization_bits: Target bits for quantization (4 or 8)
    
    Returns:
        Dictionary with estimated memory usage and savings
    """
    # Assume base is FP16 (2 bytes per parameter)
    base_bytes_per_param = 2
    
    if quantization_bits == 8:
        # 8-bit quantization: 1 byte per parameter
        quant_bytes_per_param = 1
        reduction_factor = 2.0
    elif quantization_bits == 4:
        # 4-bit quantization: 0.5 bytes per parameter
        quant_bytes_per_param = 0.5
        reduction_factor = 4.0
    else:
        reduction_factor = 1.0
        quant_bytes_per_param = base_bytes_per_param
    
    estimated_memory_mb = base_memory_mb / reduction_factor
    savings_mb = base_memory_mb - estimated_memory_mb
    savings_percent = (savings_mb / base_memory_mb) * 100
    
    return {
        "base_memory_mb": base_memory_mb,
        "estimated_memory_mb": estimated_memory_mb,
        "savings_mb": savings_mb,
        "savings_percent": savings_percent,
        "quantization_bits": quantization_bits,
    }
