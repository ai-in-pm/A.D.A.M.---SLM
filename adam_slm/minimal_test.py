#!/usr/bin/env python3
"""Minimal test to isolate the issue"""

print("Starting minimal test...")

try:
    print("Testing tiktoken import...")
    import tiktoken
    print("✅ tiktoken imported successfully")
    
    print("Testing tiktoken encoding...")
    tokenizer = tiktoken.get_encoding("gpt2")
    print("✅ tiktoken GPT-2 encoding loaded")
    
    print("Testing basic encoding...")
    tokens = tokenizer.encode("Hello world")
    print(f"✅ Encoded 'Hello world' to {len(tokens)} tokens")
    
    print("Testing basic decoding...")
    decoded = tokenizer.decode(tokens)
    print(f"✅ Decoded back to: '{decoded}'")
    
    print("🎉 All basic tests passed!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
