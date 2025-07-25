#!/usr/bin/env python3
"""
Example usage of ADAM SLM
"""

import torch
from adam_slm.models import AdamSLM, get_config
from adam_slm.tokenization import AdamTokenizer
from adam_slm.inference import AdamInference, GenerationConfig


def main():
    """Main example function"""
    # Load model configuration
    config = get_config("adam-slm-base")
    
    # Create model and tokenizer
    model = AdamSLM(config)
    tokenizer = AdamTokenizer("gpt2")
    
    # Setup inference
    inference = AdamInference(
        model=model,
        tokenizer=tokenizer,
        generation_config=GenerationConfig(
            max_new_tokens=100,
            temperature=0.8,
            top_k=50,
        )
    )
    
    # Generate text
    prompts = [
        "The future of artificial intelligence",
        "In a world where technology",
        "Machine learning algorithms"
    ]
    
    for prompt in prompts:
        generated = inference.generate(prompt)
        print(f"Prompt: {prompt}")
        print(f"Generated: {generated}")
        print("-" * 50)


if __name__ == "__main__":
    main()
