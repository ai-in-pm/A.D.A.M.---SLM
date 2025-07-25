#!/usr/bin/env python3
"""
Test script to verify ADAM-SLM tokenizer changes
"""

import sys
sys.path.append('.')

def test_adam_slm_tokenizer():
    """Test the updated ADAM-SLM tokenizer system"""
    
    print('🔬 ADAM-SLM Tokenization System Test')
    print('='*50)
    
    try:
        from adam_slm.tokenization.tokenizer import AdamTokenizer
        from adam_slm.tokenization.bpe import BPETokenizer
        from adam_slm.tokenization.corpus_analyzer import DomainCorpusAnalyzer
        
        print('✅ All imports successful')
        
        # Test 1: Default ADAM-SLM tokenizer
        print('\n1. Testing ADAM-SLM tokenizer (default):')
        tokenizer_adam = AdamTokenizer()
        info = tokenizer_adam.get_tokenizer_info()
        print(f'   Encoding: {info["encoding_name"]}')
        print(f'   Using ADAM-SLM: {info["using_adam_slm"]}')
        print(f'   Vocab size: {info["vocab_size"]:,}')
        print(f'   Fallback active: {info["fallback_active"]}')
        
        # Test 2: Explicit GPT-2 tokenizer for comparison
        print('\n2. Testing GPT-2 tokenizer (explicit):')
        tokenizer_gpt2 = AdamTokenizer(encoding_name='gpt2')
        info_gpt2 = tokenizer_gpt2.get_tokenizer_info()
        print(f'   Encoding: {info_gpt2["encoding_name"]}')
        print(f'   Using ADAM-SLM: {info_gpt2["using_adam_slm"]}')
        print(f'   Vocab size: {info_gpt2["vocab_size"]:,}')
        
        # Test 3: Encoding comparison
        test_texts = [
            'Hello world!',
            'Neural networks and deep learning are transforming AI research.',
            'The gradient descent algorithm optimizes the loss function.',
            'import torch; model = torch.nn.Transformer()'
        ]
        
        print('\n3. Encoding comparison:')
        for i, text in enumerate(test_texts, 1):
            tokens_adam = tokenizer_adam.encode(text, add_special_tokens=False)
            tokens_gpt2 = tokenizer_gpt2.encode(text, add_special_tokens=False)
            
            print(f'   Text {i}: "{text}"')
            print(f'     ADAM-SLM: {len(tokens_adam)} tokens')
            print(f'     GPT-2:    {len(tokens_gpt2)} tokens')
            print(f'     Same result: {tokens_adam == tokens_gpt2}')
        
        # Test 4: BPE tokenizer
        print('\n4. Testing BPE tokenizer:')
        bpe = BPETokenizer()
        print(f'   Initialized: {bpe is not None}')
        print(f'   Trained: {bpe.trained}')
        print(f'   Vocab size: {len(bpe.vocab)}')
        
        # Test 5: Corpus analyzer
        print('\n5. Testing corpus analyzer:')
        analyzer = DomainCorpusAnalyzer()
        test_text = """
        Neural networks use attention mechanisms for better performance.
        The transformer architecture revolutionized natural language processing.
        Mathematical symbols like α, β, γ are common in AI research papers.
        """
        domain_vocab = analyzer.extract_domain_vocabulary(test_text)
        print(f'   AI/ML terms found: {len(domain_vocab["ai_ml_terms"])}')
        print(f'   Math symbols found: {len(domain_vocab["mathematical_symbols"])}')
        print(f'   Code constructs found: {len(domain_vocab["code_constructs"])}')
        
        # Test 6: Special token handling
        print('\n6. Testing special tokens:')
        print(f'   PAD token ID: {tokenizer_adam.pad_token_id}')
        print(f'   EOS token ID: {tokenizer_adam.eos_token_id}')
        print(f'   BOS token ID: {tokenizer_adam.bos_token_id}')
        print(f'   UNK token ID: {tokenizer_adam.unk_token_id}')
        
        # Test 7: Backward compatibility
        print('\n7. Testing backward compatibility:')
        old_style_tokenizer = AdamTokenizer(encoding_name='gpt2')
        new_style_tokenizer = AdamTokenizer()  # defaults to adam_slm
        
        test_text = "This is a compatibility test."
        old_tokens = old_style_tokenizer.encode(test_text, add_special_tokens=False)
        new_tokens = new_style_tokenizer.encode(test_text, add_special_tokens=False)
        
        print(f'   Old style (GPT-2): {len(old_tokens)} tokens')
        print(f'   New style (ADAM-SLM): {len(new_tokens)} tokens')
        print(f'   Results compatible: {old_tokens == new_tokens}')
        
        print('\n✅ All tests completed successfully!')
        print('🎉 ADAM-SLM tokenization system is fully functional!')
        
        return True
        
    except Exception as e:
        print(f'❌ Error: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_adam_slm_tokenizer()
    sys.exit(0 if success else 1)
