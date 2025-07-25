"""
A.D.A.M.-SLM BPE Trainer
Enhanced Byte-Pair Encoding with domain-aware vocabulary construction
"""

import re
import json
import math
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple, Any, Optional
from pathlib import Path
import unicodedata

class DomainWeightingSystem:
    """
    Weighting system for domain-specific terms in BPE training
    """
    
    def __init__(self):
        self.domain_weights = {
            'ai_ml_core': 3.0,           # transformer, attention, neural
            'ai_ml_advanced': 2.5,       # self-attention, layer-norm
            'mathematical': 2.0,         # Greek letters, operators
            'technical_code': 1.8,       # Programming constructs
            'academic': 1.5,             # Citation patterns
            'general': 1.0               # Standard language
        }
        
        self.term_categories = self._initialize_term_categories()
    
    def _initialize_term_categories(self) -> Dict[str, Set[str]]:
        """Initialize categorized term sets"""
        
        return {
            'ai_ml_core': {
                'transformer', 'attention', 'embedding', 'neural', 'network',
                'deep', 'learning', 'machine', 'artificial', 'intelligence',
                'gradient', 'backpropagation', 'convolution', 'recurrent'
            },
            'ai_ml_advanced': {
                'multi-head', 'self-attention', 'cross-attention', 'layer-norm',
                'batch-norm', 'dropout', 'activation', 'optimizer', 'adam',
                'learning-rate', 'fine-tuning', 'pre-training'
            },
            'mathematical': {
                'α', 'β', 'γ', 'δ', 'ε', 'θ', 'λ', 'μ', 'σ', 'τ', 'π', 'ω',
                '∇', '∂', '∑', '∏', '∫', '≈', '≤', '≥', '∞', '∈', '∉'
            },
            'technical_code': {
                'torch', 'tensorflow', 'keras', 'numpy', 'pandas', 'sklearn',
                'def', 'class', 'import', 'return', 'forward', 'backward'
            },
            'academic': {
                'et al', 'fig', 'table', 'equation', 'section', 'appendix',
                'references', 'bibliography', 'abstract', 'conclusion'
            }
        }
    
    def get_term_weight(self, term: str) -> float:
        """Get weight for a specific term based on domain classification"""
        
        term_lower = term.lower()
        
        # Check each category
        for category, terms in self.term_categories.items():
            if term_lower in terms or term in terms:
                return self.domain_weights.get(category, 1.0)
        
        # Check for compound terms
        if any(core_term in term_lower for core_term in self.term_categories['ai_ml_core']):
            return self.domain_weights['ai_ml_core']
        
        return self.domain_weights['general']


class EnhancedMergeScorer:
    """
    Advanced merge scoring for A.D.A.M.-BPE training
    """
    
    def __init__(self, domain_weighting: DomainWeightingSystem):
        self.domain_weighting = domain_weighting
        self.scoring_weights = {
            'frequency': 0.35,
            'domain_importance': 0.35,
            'compression_efficiency': 0.20,
            'semantic_coherence': 0.10
        }
    
    def score_merge(self, pair: Tuple[str, str], frequency: int, corpus_stats: Dict[str, Any]) -> float:
        """
        Score a potential merge using multiple criteria
        
        Args:
            pair: Tuple of tokens to potentially merge
            frequency: Frequency of the pair in corpus
            corpus_stats: Corpus statistics for context
            
        Returns:
            Composite score for the merge
        """
        
        # Calculate individual scores
        freq_score = self._frequency_score(frequency, corpus_stats)
        domain_score = self._domain_importance_score(pair)
        compression_score = self._compression_efficiency_score(pair)
        semantic_score = self._semantic_coherence_score(pair)
        
        # Weighted combination
        total_score = (
            freq_score * self.scoring_weights['frequency'] +
            domain_score * self.scoring_weights['domain_importance'] +
            compression_score * self.scoring_weights['compression_efficiency'] +
            semantic_score * self.scoring_weights['semantic_coherence']
        )
        
        return total_score
    
    def _frequency_score(self, frequency: int, corpus_stats: Dict[str, Any]) -> float:
        """Score based on frequency (traditional BPE component)"""
        
        max_frequency = corpus_stats.get('max_pair_frequency', 1000)
        return min(1.0, frequency / max_frequency)
    
    def _domain_importance_score(self, pair: Tuple[str, str]) -> float:
        """Score based on domain importance of the resulting token"""
        
        merged_token = ''.join(pair)
        weight = self.domain_weighting.get_term_weight(merged_token)
        
        # Normalize weight to 0-1 scale
        max_weight = max(self.domain_weighting.domain_weights.values())
        return weight / max_weight
    
    def _compression_efficiency_score(self, pair: Tuple[str, str]) -> float:
        """Score based on compression efficiency"""
        
        # Longer merged tokens generally provide better compression
        merged_length = len(''.join(pair))
        
        # Prefer merges that create meaningful units
        if merged_length >= 4:  # Reasonable subword length
            return 1.0
        elif merged_length >= 2:
            return 0.7
        else:
            return 0.3
    
    def _semantic_coherence_score(self, pair: Tuple[str, str]) -> float:
        """Score based on semantic coherence of the merge"""
        
        merged_token = ''.join(pair)
        
        # Check if merge creates a meaningful technical term
        meaningful_patterns = [
            r'.*ing$',      # -ing endings
            r'.*tion$',     # -tion endings
            r'.*ness$',     # -ness endings
            r'^pre.*',      # pre- prefixes
            r'^multi.*',    # multi- prefixes
            r'^self.*',     # self- prefixes
        ]
        
        for pattern in meaningful_patterns:
            if re.match(pattern, merged_token, re.IGNORECASE):
                return 1.0
        
        # Check for domain-specific compound terms
        if any(term in merged_token.lower() for term in ['neural', 'deep', 'machine', 'artificial']):
            return 0.9
        
        return 0.5  # Default semantic score


class VocabularyOptimizer:
    """
    Optimizes vocabulary composition for A.D.A.M.-SLM
    """
    
    def __init__(self):
        self.target_composition = {
            'base_language': 35000,
            'ai_ml_specialized': 8000,
            'mathematical': 2000,
            'technical_code': 1000,
            'citation_academic': 500,
            'special_tokens': 257
        }
    
    def optimize_vocabulary(self, candidate_vocab: Dict[str, float], target_size: int = 50257) -> Dict[str, float]:
        """
        Optimize vocabulary composition to meet target distribution
        
        Args:
            candidate_vocab: Dictionary of candidate tokens with scores
            target_size: Target vocabulary size
            
        Returns:
            Optimized vocabulary with balanced composition
        """
        
        # Categorize tokens
        categorized_tokens = self._categorize_tokens(candidate_vocab)
        
        # Allocate tokens per category
        optimized_vocab = {}
        
        for category, target_count in self.target_composition.items():
            category_tokens = categorized_tokens.get(category, {})
            
            # Select top tokens for this category
            selected_tokens = dict(
                sorted(category_tokens.items(), key=lambda x: x[1], reverse=True)[:target_count]
            )
            
            optimized_vocab.update(selected_tokens)
        
        return optimized_vocab
    
    def _categorize_tokens(self, vocab: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """Categorize tokens by domain"""
        
        categories = {category: {} for category in self.target_composition.keys()}
        
        for token, score in vocab.items():
            category = self._classify_token(token)
            categories[category][token] = score
        
        return categories
    
    def _classify_token(self, token: str) -> str:
        """Classify token into appropriate category"""
        
        token_lower = token.lower()
        
        # AI/ML specialized terms
        ai_ml_indicators = ['neural', 'deep', 'machine', 'artificial', 'transformer', 'attention']
        if any(indicator in token_lower for indicator in ai_ml_indicators):
            return 'ai_ml_specialized'
        
        # Mathematical symbols
        if any(char in token for char in 'αβγδεθλμσπω∇∂∑∏∫≈≤≥∞'):
            return 'mathematical'
        
        # Technical/code terms
        code_indicators = ['torch', 'tf', 'keras', 'def', 'class', 'import']
        if any(indicator in token_lower for indicator in code_indicators):
            return 'technical_code'
        
        # Academic/citation terms
        academic_indicators = ['et al', 'fig', 'table', 'ref', 'cite']
        if any(indicator in token_lower for indicator in academic_indicators):
            return 'citation_academic'
        
        # Special tokens
        if len(token) == 1 and ord(token) < 256:
            return 'special_tokens'
        
        # Default to base language
        return 'base_language'


class AdamBPETrainer:
    """
    Main A.D.A.M.-SLM BPE trainer with domain-aware enhancements
    """
    
    def __init__(self, target_vocab_size: int = 50257):
        self.target_vocab_size = target_vocab_size
        self.domain_weighting = DomainWeightingSystem()
        self.merge_scorer = EnhancedMergeScorer(self.domain_weighting)
        self.vocab_optimizer = VocabularyOptimizer()
        
        # Training state
        self.vocabulary = {}
        self.merges = []
        self.token_frequencies = Counter()
        self.pair_frequencies = Counter()
    
    def train_adam_bpe(self, corpus_text: str, save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Train A.D.A.M.-SLM BPE tokenizer
        
        Args:
            corpus_text: Training corpus text
            save_path: Optional path to save trained tokenizer
            
        Returns:
            Trained tokenizer components
        """
        
        print("🚀 Starting A.D.A.M.-SLM BPE Training...")
        
        # Phase 1: Initialize vocabulary
        print("📝 Phase 1: Initializing domain-aware vocabulary...")
        self._initialize_vocabulary(corpus_text)
        
        # Phase 2: Iterative merge training
        print("🔄 Phase 2: Training with domain-weighted merges...")
        self._train_merges(corpus_text)
        
        # Phase 3: Vocabulary optimization
        print("⚡ Phase 3: Optimizing vocabulary composition...")
        self._optimize_final_vocabulary()
        
        # Phase 4: Create tokenizer components
        print("🔧 Phase 4: Building tokenizer components...")
        tokenizer_components = self._build_tokenizer_components()
        
        # Save if requested
        if save_path:
            self._save_tokenizer(tokenizer_components, save_path)
            print(f"💾 Tokenizer saved to {save_path}")
        
        print("✅ A.D.A.M.-SLM BPE training completed!")
        return tokenizer_components
    
    def _initialize_vocabulary(self, corpus_text: str):
        """Initialize vocabulary with character-level tokens and domain terms"""
        
        # Start with character-level vocabulary
        chars = set(corpus_text)
        for char in chars:
            if ord(char) < 256:  # ASCII characters
                self.vocabulary[char] = len(self.vocabulary)
        
        # Add special tokens
        special_tokens = ['<pad>', '<eos>', '<bos>', '<unk>']
        for token in special_tokens:
            if token not in self.vocabulary:
                self.vocabulary[token] = len(self.vocabulary)
        
        print(f"   • Initialized with {len(self.vocabulary)} base tokens")
    
    def _train_merges(self, corpus_text: str):
        """Train BPE merges with domain weighting"""
        
        # Tokenize corpus into characters
        tokens = list(corpus_text)
        
        # Calculate corpus statistics
        corpus_stats = {
            'total_tokens': len(tokens),
            'unique_chars': len(set(tokens)),
            'max_pair_frequency': 0
        }
        
        # Iterative merge training
        target_merges = self.target_vocab_size - len(self.vocabulary)
        
        for merge_step in range(target_merges):
            if merge_step % 1000 == 0:
                print(f"   • Merge step {merge_step}/{target_merges}")
            
            # Count pairs
            pairs = self._count_pairs(tokens)
            if not pairs:
                break
            
            # Update corpus stats
            corpus_stats['max_pair_frequency'] = max(pairs.values())
            
            # Score and select best merge
            best_pair = self._select_best_merge(pairs, corpus_stats)
            if not best_pair:
                break
            
            # Apply merge
            tokens = self._apply_merge(tokens, best_pair)
            self.merges.append(best_pair)
            
            # Add to vocabulary
            merged_token = ''.join(best_pair)
            self.vocabulary[merged_token] = len(self.vocabulary)
        
        print(f"   • Completed {len(self.merges)} merges")
    
    def _count_pairs(self, tokens: List[str]) -> Counter:
        """Count adjacent token pairs"""
        
        pairs = Counter()
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            pairs[pair] += 1
        
        return pairs
    
    def _select_best_merge(self, pairs: Counter, corpus_stats: Dict[str, Any]) -> Optional[Tuple[str, str]]:
        """Select best merge using enhanced scoring"""
        
        if not pairs:
            return None
        
        # Score all pairs
        scored_pairs = []
        for pair, frequency in pairs.items():
            score = self.merge_scorer.score_merge(pair, frequency, corpus_stats)
            scored_pairs.append((pair, score))
        
        # Return highest scoring pair
        scored_pairs.sort(key=lambda x: x[1], reverse=True)
        return scored_pairs[0][0] if scored_pairs else None
    
    def _apply_merge(self, tokens: List[str], merge_pair: Tuple[str, str]) -> List[str]:
        """Apply merge to token sequence"""
        
        new_tokens = []
        i = 0
        
        while i < len(tokens):
            if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == merge_pair:
                # Apply merge
                merged_token = ''.join(merge_pair)
                new_tokens.append(merged_token)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        
        return new_tokens
    
    def _optimize_final_vocabulary(self):
        """Optimize final vocabulary composition"""
        
        # Create candidate vocabulary with scores
        candidate_vocab = {}
        for token, token_id in self.vocabulary.items():
            # Score based on domain importance and frequency
            domain_weight = self.domain_weighting.get_term_weight(token)
            frequency_weight = self.token_frequencies.get(token, 1)
            candidate_vocab[token] = domain_weight * math.log(frequency_weight + 1)
        
        # Optimize composition
        self.vocabulary = self.vocab_optimizer.optimize_vocabulary(
            candidate_vocab, self.target_vocab_size
        )
        
        # Reassign token IDs
        sorted_tokens = sorted(self.vocabulary.items(), key=lambda x: x[1], reverse=True)
        self.vocabulary = {token: i for i, (token, _) in enumerate(sorted_tokens)}
        
        print(f"   • Optimized vocabulary to {len(self.vocabulary)} tokens")
    
    def _build_tokenizer_components(self) -> Dict[str, Any]:
        """Build final tokenizer components"""
        
        return {
            'vocabulary': self.vocabulary,
            'merges': self.merges,
            'vocab_size': len(self.vocabulary),
            'special_tokens': {
                'pad_token': '<pad>',
                'eos_token': '<eos>',
                'bos_token': '<bos>',
                'unk_token': '<unk>'
            },
            'domain_weights': self.domain_weighting.domain_weights,
            'training_stats': {
                'total_merges': len(self.merges),
                'vocab_composition': self.vocab_optimizer.target_composition
            }
        }
    
    def _save_tokenizer(self, components: Dict[str, Any], save_path: str):
        """Save tokenizer components to file"""
        
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save vocabulary
        with open(save_path / 'vocab.json', 'w', encoding='utf-8') as f:
            json.dump(components['vocabulary'], f, ensure_ascii=False, indent=2)
        
        # Save merges
        with open(save_path / 'merges.txt', 'w', encoding='utf-8') as f:
            for merge in components['merges']:
                f.write(f"{merge[0]} {merge[1]}\n")
        
        # Save configuration
        config = {
            'vocab_size': components['vocab_size'],
            'special_tokens': components['special_tokens'],
            'domain_weights': components['domain_weights'],
            'training_stats': components['training_stats']
        }
        
        with open(save_path / 'tokenizer_config.json', 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)


def train_adam_bpe_tokenizer():
    """Main function to train A.D.A.M.-SLM BPE tokenizer"""
    
    print("🚀 A.D.A.M.-SLM BPE Tokenizer Training")
    print("="*60)
    
    # Initialize trainer
    trainer = AdamBPETrainer(target_vocab_size=50257)
    
    # Simulate training corpus (in real implementation, would load actual corpus)
    sample_corpus = """
    Transformer architectures have revolutionized natural language processing and machine learning.
    The attention mechanism allows neural networks to focus on relevant parts of the input sequence.
    Deep learning models like BERT and GPT have achieved state-of-the-art performance on various tasks.
    Gradient descent optimization with backpropagation enables efficient training of neural networks.
    Self-attention and multi-head attention are key components of transformer models.
    """
    
    # Train tokenizer
    tokenizer_components = trainer.train_adam_bpe(
        corpus_text=sample_corpus,
        save_path="adam_slm/tokenization/adam_bpe_model"
    )
    
    print("\n📊 Training Results:")
    print(f"   • Vocabulary size: {tokenizer_components['vocab_size']:,}")
    print(f"   • Total merges: {tokenizer_components['training_stats']['total_merges']:,}")
    print(f"   • Special tokens: {list(tokenizer_components['special_tokens'].keys())}")
    
    return tokenizer_components


if __name__ == "__main__":
    results = train_adam_bpe_tokenizer()
