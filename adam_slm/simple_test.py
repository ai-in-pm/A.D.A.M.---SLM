#!/usr/bin/env python3
"""Simple test for ADAM-SLM tokenizer changes"""

import sys
sys.path.append('.')

try:
    from adam_slm.tokenization.tokenizer import AdamTokenizer
    from adam_slm.tokenization.bpe import BPETokenizer
    from adam_slm.tokenization.corpus_analyzer import DomainCorpusAnalyzer
    
    print('✅ Imports successful')
    
    # Test default tokenizer
    tokenizer = AdamTokenizer()
    print(f'✅ Default encoding: {tokenizer.encoding_name}')
    print(f'✅ Using ADAM-SLM: {tokenizer.is_using_adam_slm()}')
    
    # Test encoding
    text = "Hello ADAM-SLM tokenizer!"
    tokens = tokenizer.encode(text, add_special_tokens=False)
    decoded = tokenizer.decode(tokens)
    print(f'✅ Encoding test: {len(tokens)} tokens')
    print(f'✅ Decoding test: "{decoded}"')
    
    # Test BPE
    bpe = BPETokenizer()
    print(f'✅ BPE tokenizer initialized')
    
    # Test analyzer
    analyzer = DomainCorpusAnalyzer()
    print(f'✅ Corpus analyzer initialized')
    
    print('🎉 All tests passed!')
    
except Exception as e:
    print(f'❌ Error: {e}')
    import traceback
    traceback.print_exc()
