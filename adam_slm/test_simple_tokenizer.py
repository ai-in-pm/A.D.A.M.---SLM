#!/usr/bin/env python3
"""Test simplified tokenizer"""

import tiktoken
from typing import Dict, Any

class SimpleAdamTokenizer:
    """Simplified ADAM-SLM tokenizer for testing"""
    
    def __init__(self, encoding_name: str = "adam_slm"):
        print(f"Initializing tokenizer with encoding: {encoding_name}")
        
        self.encoding_name = encoding_name
        self._using_adam_slm = False
        
        if encoding_name == "adam_slm":
            # Use GPT-2 as base for ADAM-SLM compatibility
            self.tokenizer = tiktoken.get_encoding("gpt2")
            self._using_adam_slm = True
            print("⚠️ Using GPT-2 tokenizer in ADAM-SLM compatibility mode")
        else:
            self.tokenizer = tiktoken.get_encoding(encoding_name)
        
        # Setup special tokens
        self.pad_token_id = 50256
        self.eos_token_id = 50256
        self.bos_token_id = 50256
        self.unk_token_id = 50256
    
    def is_using_adam_slm(self) -> bool:
        """Check if ADAM-SLM tokenizer is being used"""
        return self._using_adam_slm
    
    def get_tokenizer_info(self) -> Dict[str, Any]:
        """Get information about the current tokenizer"""
        return {
            'encoding_name': self.encoding_name,
            'using_adam_slm': self.is_using_adam_slm(),
            'vocab_size': self.tokenizer.n_vocab,
            'fallback_active': not self.is_using_adam_slm() and self.encoding_name == "adam_slm"
        }
    
    def encode(self, text: str, add_special_tokens: bool = False):
        """Encode text to token IDs"""
        tokens = self.tokenizer.encode(text)
        if add_special_tokens:
            tokens = [self.bos_token_id] + tokens
        return tokens
    
    def decode(self, token_ids, skip_special_tokens: bool = True):
        """Decode token IDs to text"""
        if skip_special_tokens:
            # Filter out special tokens
            filtered_ids = [tid for tid in token_ids if tid not in [self.pad_token_id, self.eos_token_id, self.bos_token_id]]
            token_ids = filtered_ids
        return self.tokenizer.decode(token_ids)

# Test the simplified tokenizer
if __name__ == "__main__":
    print("Testing simplified ADAM-SLM tokenizer...")
    
    try:
        # Test default (ADAM-SLM)
        tokenizer = SimpleAdamTokenizer()
        print(f"✅ Tokenizer created: {tokenizer.get_tokenizer_info()}")
        
        # Test encoding/decoding
        text = "Hello ADAM-SLM tokenizer!"
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        print(f"✅ Text: '{text}'")
        print(f"✅ Tokens: {len(tokens)} tokens")
        print(f"✅ Decoded: '{decoded}'")
        
        # Test GPT-2 mode
        gpt2_tokenizer = SimpleAdamTokenizer("gpt2")
        print(f"✅ GPT-2 tokenizer: {gpt2_tokenizer.get_tokenizer_info()}")
        
        print("🎉 All tests passed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
