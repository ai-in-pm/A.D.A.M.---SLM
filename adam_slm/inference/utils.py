"""
Inference utilities for ADAM SLM
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
import os

from ..models import AdamSLM, AdamSLMConfig


def load_model_for_inference(
    checkpoint_path: str,
    device: Optional[torch.device] = None,
    config_path: Optional[str] = None,
) -> tuple:
    """
    Load model from checkpoint for inference
    
    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to load model on
        config_path: Optional path to config file
        
    Returns:
        Tuple of (model, config)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load config
    if config_path is not None:
        config = AdamSLMConfig.from_json(config_path)
    elif "config" in checkpoint:
        config = AdamSLMConfig.from_dict(checkpoint["config"])
    else:
        # Try to find config in same directory
        config_file = os.path.join(os.path.dirname(checkpoint_path), "config.json")
        if os.path.exists(config_file):
            config = AdamSLMConfig.from_json(config_file)
        else:
            raise ValueError("Could not find model configuration")
            
    # Create and load model
    model = AdamSLM(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    return model, config


def optimize_for_inference(model: nn.Module) -> nn.Module:
    """
    Optimize model for inference
    
    Args:
        model: Model to optimize
        
    Returns:
        Optimized model
    """
    # Set to eval mode
    model.eval()
    
    # Disable gradients
    for param in model.parameters():
        param.requires_grad_(False)
        
    # Try to compile model (PyTorch 2.0+)
    if hasattr(torch, 'compile'):
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("Model compiled for faster inference")
        except Exception as e:
            print(f"Could not compile model: {e}")
            
    return model


def estimate_memory_usage(
    model: nn.Module,
    batch_size: int = 1,
    seq_len: int = 1024,
    dtype: torch.dtype = torch.float32,
) -> Dict[str, float]:
    """
    Estimate memory usage for model inference
    
    Args:
        model: Model to analyze
        batch_size: Batch size
        seq_len: Sequence length
        dtype: Data type
        
    Returns:
        Dictionary with memory estimates in MB
    """
    # Model parameters
    param_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
    
    # Activation memory (rough estimate)
    if hasattr(model, 'config'):
        d_model = model.config.d_model
        n_layers = model.config.n_layers
    else:
        # Fallback estimates
        d_model = 768
        n_layers = 12
        
    # Estimate activation memory per layer
    activation_per_layer = batch_size * seq_len * d_model * 4  # 4 bytes for float32
    total_activation = activation_per_layer * n_layers / 1024**2
    
    # KV cache memory
    kv_cache_memory = batch_size * seq_len * d_model * 2 * n_layers * 4 / 1024**2  # K and V
    
    return {
        "parameters_mb": param_memory,
        "activations_mb": total_activation,
        "kv_cache_mb": kv_cache_memory,
        "total_estimated_mb": param_memory + total_activation + kv_cache_memory,
    }


def benchmark_inference(
    model: nn.Module,
    tokenizer,
    prompt: str = "The quick brown fox",
    max_new_tokens: int = 50,
    num_runs: int = 10,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """
    Benchmark inference performance
    
    Args:
        model: Model to benchmark
        tokenizer: Tokenizer
        prompt: Test prompt
        max_new_tokens: Tokens to generate
        num_runs: Number of benchmark runs
        device: Device to run on
        
    Returns:
        Performance metrics
    """
    import time
    
    if device is None:
        device = next(model.parameters()).device
        
    model.eval()
    
    # Encode prompt
    input_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = model.generate(
                input_ids=input_ids,
                max_new_tokens=10,
                do_sample=False,
            )
            
    # Benchmark
    times = []
    
    with torch.no_grad():
        for _ in range(num_runs):
            start_time = time.time()
            
            generated = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
            
            end_time = time.time()
            times.append(end_time - start_time)
            
    # Calculate metrics
    avg_time = sum(times) / len(times)
    tokens_per_second = max_new_tokens / avg_time
    
    return {
        "avg_time_seconds": avg_time,
        "tokens_per_second": tokens_per_second,
        "total_tokens": max_new_tokens,
        "num_runs": num_runs,
    }
