#!/usr/bin/env python3
"""Basic test for ADAM-SLM tokenizer changes"""

import sys
sys.path.append('.')

try:
    print("Testing basic imports...")
    
    from adam_slm.tokenization.tokenizer import AdamTokenizer
    print('✅ AdamTokenizer import successful')
    
    from adam_slm.tokenization.bpe import BPETokenizer
    print('✅ BPETokenizer import successful')
    
    # Test default tokenizer
    print("Testing tokenizer initialization...")
    tokenizer = AdamTokenizer()
    print(f'✅ Default encoding: {tokenizer.encoding_name}')
    print(f'✅ Using ADAM-SLM: {tokenizer.is_using_adam_slm()}')
    
    # Test encoding
    print("Testing encoding/decoding...")
    text = "Hello ADAM-SLM tokenizer!"
    tokens = tokenizer.encode(text, add_special_tokens=False)
    decoded = tokenizer.decode(tokens)
    print(f'✅ Encoding test: {len(tokens)} tokens')
    print(f'✅ Decoding test: "{decoded}"')
    
    # Test BPE
    print("Testing BPE tokenizer...")
    bpe = BPETokenizer()
    print(f'✅ BPE tokenizer initialized')
    
    print('🎉 All basic tests passed!')
    
except Exception as e:
    print(f'❌ Error: {e}')
    import traceback
    traceback.print_exc()
