"""
Byte Pair Encoding (BPE) tokenizer for ADAM-SLM
Enhanced BPE implementation optimized for AI/ML domain content
"""

import re
import json
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict, Counter
import os


class BPETokenizer:
    """
    ADAM-SLM BPE tokenizer implementation
    Optimized for AI/ML research content with domain-specific enhancements
    """
    
    def __init__(self):
        self.vocab = {}
        self.merges = []
        self.special_tokens = {}
        self.trained = False
        
    def train(
        self,
        texts: List[str],
        vocab_size: int = 10000,
        min_frequency: int = 2,
        special_tokens: Optional[List[str]] = None,
    ):
        """
        Train BPE tokenizer on texts
        
        Args:
            texts: Training texts
            vocab_size: Target vocabulary size
            min_frequency: Minimum frequency for merges
            special_tokens: Special tokens to add
        """
        if special_tokens is None:
            special_tokens = ["<pad>", "<unk>", "<s>", "</s>"]
            
        # Initialize vocabulary with byte-level tokens
        self.vocab = {chr(i): i for i in range(256)}
        
        # Add special tokens
        for token in special_tokens:
            self.vocab[token] = len(self.vocab)
            self.special_tokens[token] = self.vocab[token]
            
        # Preprocess texts
        word_freqs = self._get_word_frequencies(texts)
        
        # Split words into characters
        splits = {}
        for word, freq in word_freqs.items():
            splits[word] = list(word)
            
        # Learn merges
        self.merges = []
        
        while len(self.vocab) < vocab_size:
            # Count pairs
            pairs = self._count_pairs(splits, word_freqs)
            
            if not pairs:
                break
                
            # Find most frequent pair
            best_pair = max(pairs, key=pairs.get)
            
            if pairs[best_pair] < min_frequency:
                break
                
            # Merge the pair
            splits = self._merge_vocab(best_pair, splits)
            self.merges.append(best_pair)
            
            # Add merged token to vocabulary
            merged_token = best_pair[0] + best_pair[1]
            self.vocab[merged_token] = len(self.vocab)
            
        self.trained = True
        
    def _get_word_frequencies(self, texts: List[str]) -> Dict[str, int]:
        """Get word frequencies from texts"""
        word_freqs = Counter()
        
        for text in texts:
            # Simple tokenization (split on whitespace and punctuation)
            words = re.findall(r'\w+|[^\w\s]', text.lower())
            word_freqs.update(words)
            
        return dict(word_freqs)
        
    def _count_pairs(self, splits: Dict[str, List[str]], word_freqs: Dict[str, int]) -> Dict[Tuple[str, str], int]:
        """Count adjacent pairs in splits"""
        pairs = defaultdict(int)
        
        for word, freq in word_freqs.items():
            split = splits[word]
            if len(split) == 1:
                continue
                
            for i in range(len(split) - 1):
                pair = (split[i], split[i + 1])
                pairs[pair] += freq
                
        return pairs
        
    def _merge_vocab(self, pair: Tuple[str, str], splits: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Merge a pair in all splits"""
        new_splits = {}
        
        for word in splits:
            split = splits[word]
            if len(split) == 1:
                new_splits[word] = split
                continue
                
            new_split = []
            i = 0
            
            while i < len(split):
                if i < len(split) - 1 and split[i] == pair[0] and split[i + 1] == pair[1]:
                    new_split.append(pair[0] + pair[1])
                    i += 2
                else:
                    new_split.append(split[i])
                    i += 1
                    
            new_splits[word] = new_split
            
        return new_splits
        
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs"""
        if not self.trained:
            raise ValueError("Tokenizer not trained yet")
            
        # Tokenize text
        tokens = self._tokenize(text)
        
        # Convert to IDs
        token_ids = []
        for token in tokens:
            if token in self.vocab:
                token_ids.append(self.vocab[token])
            else:
                # Handle unknown tokens
                unk_id = self.special_tokens.get("<unk>", 0)
                token_ids.append(unk_id)
                
        return token_ids
        
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text"""
        if not self.trained:
            raise ValueError("Tokenizer not trained yet")
            
        # Create reverse vocabulary
        id_to_token = {v: k for k, v in self.vocab.items()}
        
        # Convert IDs to tokens
        tokens = []
        for token_id in token_ids:
            if token_id in id_to_token:
                token = id_to_token[token_id]
                # Skip special tokens in output
                if token not in self.special_tokens:
                    tokens.append(token)
                    
        # Join tokens
        text = "".join(tokens)
        
        # Basic post-processing
        text = text.replace("</w>", " ")
        
        return text.strip()
        
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text using learned merges"""
        # Simple word tokenization
        words = re.findall(r'\w+|[^\w\s]', text.lower())
        
        tokens = []
        for word in words:
            # Split word into characters
            word_tokens = list(word)
            
            # Apply merges
            for merge in self.merges:
                new_word_tokens = []
                i = 0
                
                while i < len(word_tokens):
                    if (i < len(word_tokens) - 1 and 
                        word_tokens[i] == merge[0] and 
                        word_tokens[i + 1] == merge[1]):
                        new_word_tokens.append(merge[0] + merge[1])
                        i += 2
                    else:
                        new_word_tokens.append(word_tokens[i])
                        i += 1
                        
                word_tokens = new_word_tokens
                
            tokens.extend(word_tokens)
            
        return tokens
        
    def save(self, save_dir: str):
        """Save tokenizer to directory"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save vocabulary
        vocab_path = os.path.join(save_dir, "vocab.json")
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)
            
        # Save merges
        merges_path = os.path.join(save_dir, "merges.txt")
        with open(merges_path, "w", encoding="utf-8") as f:
            for merge in self.merges:
                f.write(f"{merge[0]} {merge[1]}\n")
                
        # Save config
        config = {
            "vocab_size": len(self.vocab),
            "special_tokens": self.special_tokens,
            "trained": self.trained,
        }
        
        config_path = os.path.join(save_dir, "tokenizer_config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
            
    @classmethod
    def load(cls, load_dir: str) -> "BPETokenizer":
        """Load tokenizer from directory"""
        tokenizer = cls()
        
        # Load vocabulary
        vocab_path = os.path.join(load_dir, "vocab.json")
        with open(vocab_path, "r", encoding="utf-8") as f:
            tokenizer.vocab = json.load(f)
            
        # Load merges
        merges_path = os.path.join(load_dir, "merges.txt")
        tokenizer.merges = []
        
        with open(merges_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split()
                    if len(parts) == 2:
                        tokenizer.merges.append((parts[0], parts[1]))
                        
        # Load config
        config_path = os.path.join(load_dir, "tokenizer_config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
            
        tokenizer.special_tokens = config.get("special_tokens", {})
        tokenizer.trained = config.get("trained", True)
        
        return tokenizer
        
    @property
    def vocab_size(self) -> int:
        """Get vocabulary size"""
        return len(self.vocab)
        
    def get_vocab(self) -> Dict[str, int]:
        """Get vocabulary dictionary"""
        return self.vocab.copy()
        
    def __len__(self) -> int:
        """Get vocabulary size"""
        return len(self.vocab)
