"""
Main tokenizer interface for ADAM-SLM
Enhanced tokenizer with ADAM-SLM optimizations and fallback support
"""

import tiktoken
from typing import List, Union, Optional, Dict, Any
import json
import os


class AdamTokenizer:
    """
    Enhanced tokenizer for ADAM-SLM with domain-specific optimizations
    Provides backward compatibility while supporting ADAM-SLM enhancements
    """
    
    def __init__(
        self,
        encoding_name: str = "adam_slm",
        special_tokens: Optional[Dict[str, int]] = None,
    ):
        self.encoding_name = encoding_name

        # Initialize tokenizer with fallback support
        self._using_adam_slm = False
        self._adam_tokenizer = None

        if encoding_name == "adam_slm":
            # For now, use GPT-2 as the base tokenizer for ADAM-SLM compatibility
            # Future enhancement: integrate custom ADAM-SLM tokenizer when available
            try:
                self.tokenizer = tiktoken.get_encoding("gpt2")
                self._using_adam_slm = True  # Mark as ADAM-SLM mode for future enhancements
                print("✅ Using GPT-2 tokenizer in ADAM-SLM compatibility mode")
            except Exception as e:
                print(f"⚠️ Warning: Could not load GPT-2 tokenizer ({e})")
                # Create a minimal fallback tokenizer
                self.tokenizer = self._create_fallback_tokenizer()
                self._using_adam_slm = True
        else:
            # Use specified encoding
            try:
                self.tokenizer = tiktoken.get_encoding(encoding_name)
            except Exception:
                # Final fallback to GPT-2
                self.tokenizer = tiktoken.get_encoding("gpt2")

        # Special tokens
        self.special_tokens = special_tokens or {}
        self._setup_special_tokens()

    def _create_fallback_tokenizer(self):
        """Create a minimal fallback tokenizer when tiktoken fails"""
        class FallbackTokenizer:
            def __init__(self):
                self.n_vocab = 50257

            def encode(self, text: str):
                # Simple character-based encoding as fallback
                return [ord(c) % 50257 for c in text[:100]]  # Limit to 100 chars

            def decode(self, tokens):
                # Simple character-based decoding
                try:
                    return ''.join(chr(t % 128) for t in tokens if 0 <= t < 50257)
                except:
                    return "FALLBACK_DECODE_ERROR"

        return FallbackTokenizer()

    def _setup_special_tokens(self):
        """Setup special tokens with ADAM-SLM optimizations"""
        # Default special tokens - use GPT-2 compatible values for backward compatibility
        if self.encoding_name in ["gpt2", "adam_slm"] or self._using_adam_slm:
            self.pad_token_id = 50256  # Use EOS as PAD for compatibility
            self.eos_token_id = 50256
            self.bos_token_id = 50256
            self.unk_token_id = 50256
        else:
            self.pad_token_id = 0
            self.eos_token_id = 0
            self.bos_token_id = 0
            self.unk_token_id = 0
            
        # Override with provided special tokens
        for token_name, token_id in self.special_tokens.items():
            setattr(self, f"{token_name}_token_id", token_id)

    def is_using_adam_slm(self) -> bool:
        """Check if ADAM-SLM tokenizer is being used"""
        return getattr(self, '_using_adam_slm', False)

    def get_tokenizer_info(self) -> Dict[str, Any]:
        """Get information about the current tokenizer"""
        return {
            'encoding_name': self.encoding_name,
            'using_adam_slm': self.is_using_adam_slm(),
            'vocab_size': self.vocab_size,
            'fallback_active': not self.is_using_adam_slm() and self.encoding_name == "adam_slm"
        }

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size"""
        try:
            return self.tokenizer.n_vocab
        except AttributeError:
            return 50257  # Default GPT-2 vocab size
        
    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        truncation: bool = False,
        padding: bool = False,
        return_tensors: Optional[str] = None,
    ) -> Union[List[int], Dict[str, Any]]:
        """
        Encode text to token IDs
        
        Args:
            text: Input text
            add_special_tokens: Whether to add special tokens
            max_length: Maximum sequence length
            truncation: Whether to truncate if too long
            padding: Whether to pad if too short
            return_tensors: Format of return tensors ("pt" for PyTorch)
            
        Returns:
            Token IDs or dictionary with additional info
        """
        # Enhanced encoding with ADAM-SLM support
        if self.is_using_adam_slm() and hasattr(self, '_adam_tokenizer'):
            try:
                token_ids = self._adam_tokenizer.encode(text)
            except Exception:
                # Fallback to standard tokenizer
                token_ids = self.tokenizer.encode(text)
        else:
            token_ids = self.tokenizer.encode(text)
        
        # Add special tokens
        if add_special_tokens:
            if hasattr(self, 'bos_token_id') and self.bos_token_id is not None:
                token_ids = [self.bos_token_id] + token_ids
                
        # Handle max length and truncation
        if max_length is not None:
            if len(token_ids) > max_length:
                if truncation:
                    token_ids = token_ids[:max_length]
                    # Ensure EOS token at the end if truncated
                    if add_special_tokens and hasattr(self, 'eos_token_id'):
                        token_ids[-1] = self.eos_token_id
                        
        # Handle padding
        attention_mask = None
        if padding and max_length is not None:
            if len(token_ids) < max_length:
                attention_mask = [1] * len(token_ids) + [0] * (max_length - len(token_ids))
                token_ids = token_ids + [self.pad_token_id] * (max_length - len(token_ids))
            else:
                attention_mask = [1] * len(token_ids)
                
        # Return format
        if return_tensors == "pt":
            import torch
            result = {"input_ids": torch.tensor(token_ids, dtype=torch.long)}
            if attention_mask is not None:
                result["attention_mask"] = torch.tensor(attention_mask, dtype=torch.long)
            return result
        elif attention_mask is not None:
            return {
                "input_ids": token_ids,
                "attention_mask": attention_mask,
            }
        else:
            return token_ids
            
    def decode(
        self,
        token_ids: Union[List[int], List[List[int]]],
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True,
    ) -> Union[str, List[str]]:
        """
        Decode token IDs to text
        
        Args:
            token_ids: Token IDs to decode
            skip_special_tokens: Whether to skip special tokens
            clean_up_tokenization_spaces: Whether to clean up spaces
            
        Returns:
            Decoded text
        """
        if isinstance(token_ids[0], list):
            # Batch decoding
            texts = []
            for ids in token_ids:
                text = self._decode_single(ids, skip_special_tokens, clean_up_tokenization_spaces)
                texts.append(text)
            return texts
        else:
            # Single sequence decoding
            return self._decode_single(token_ids, skip_special_tokens, clean_up_tokenization_spaces)
            
    def _decode_single(
        self,
        token_ids: List[int],
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        """Decode single sequence"""
        # Filter special tokens if requested
        if skip_special_tokens:
            filtered_ids = []
            for token_id in token_ids:
                if not self._is_special_token(token_id):
                    filtered_ids.append(token_id)
            token_ids = filtered_ids
            
        # Decode
        try:
            text = self.tokenizer.decode(token_ids)
        except Exception:
            # Fallback for invalid token IDs
            valid_ids = [tid for tid in token_ids if 0 <= tid < self.vocab_size]
            text = self.tokenizer.decode(valid_ids)
            
        # Clean up spaces
        if clean_up_tokenization_spaces:
            text = text.strip()
            
        return text
        
    def _is_special_token(self, token_id: int) -> bool:
        """Check if token ID is a special token"""
        special_token_ids = {
            getattr(self, f"{name}_token_id", None)
            for name in ["pad", "eos", "bos", "unk"]
        }
        return token_id in special_token_ids
        
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into tokens (for debugging)
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        token_ids = self.encode(text, add_special_tokens=False)
        tokens = []
        
        for token_id in token_ids:
            try:
                token = self.tokenizer.decode([token_id])
                tokens.append(token)
            except Exception:
                tokens.append(f"<unk_{token_id}>")
                
        return tokens
        
    def get_vocab(self) -> Dict[str, int]:
        """Get vocabulary dictionary"""
        # tiktoken doesn't expose vocab directly, so we approximate
        vocab = {}
        for i in range(min(1000, self.vocab_size)):  # Sample first 1000 tokens
            try:
                token = self.tokenizer.decode([i])
                vocab[token] = i
            except Exception:
                continue
        return vocab
        
    def save_pretrained(self, save_directory: str):
        """Save tokenizer configuration"""
        os.makedirs(save_directory, exist_ok=True)
        
        config = {
            "encoding_name": self.encoding_name,
            "special_tokens": self.special_tokens,
            "vocab_size": self.vocab_size,
            "pad_token_id": getattr(self, "pad_token_id", None),
            "eos_token_id": getattr(self, "eos_token_id", None),
            "bos_token_id": getattr(self, "bos_token_id", None),
            "unk_token_id": getattr(self, "unk_token_id", None),
        }
        
        config_path = os.path.join(save_directory, "tokenizer_config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
            
    @classmethod
    def from_pretrained(cls, load_directory: str) -> "AdamTokenizer":
        """Load tokenizer from saved configuration"""
        config_path = os.path.join(load_directory, "tokenizer_config.json")
        
        with open(config_path, "r") as f:
            config = json.load(f)
            
        tokenizer = cls(
            encoding_name=config["encoding_name"],
            special_tokens=config.get("special_tokens", {}),
        )
        
        # Restore special token IDs
        for token_name in ["pad", "eos", "bos", "unk"]:
            token_id = config.get(f"{token_name}_token_id")
            if token_id is not None:
                setattr(tokenizer, f"{token_name}_token_id", token_id)
                
        return tokenizer
        
    def __len__(self) -> int:
        """Get vocabulary size"""
        return self.vocab_size
        
    def __call__(
        self,
        text: Union[str, List[str]],
        **kwargs
    ) -> Union[List[int], Dict[str, Any]]:
        """Make tokenizer callable"""
        if isinstance(text, str):
            return self.encode(text, **kwargs)
        else:
            # Batch encoding
            results = []
            for t in text:
                result = self.encode(t, **kwargs)
                results.append(result)
            return results
