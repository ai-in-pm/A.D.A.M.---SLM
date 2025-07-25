"""
Tokenization utilities for A.D.A.M. SLM
Enhanced with custom A.D.A.M.-SLM BPE tokenizer
"""

# Import original tokenizers
from .tokenizer import AdamTokenizer
from .bpe import BPETokenizer

# Import new A.D.A.M.-SLM custom tokenizer
try:
    from .adam_slm_tokenizer import AdamSLMTokenizer, create_adam_slm_tokenizer
    ADAM_SLM_TOKENIZER_AVAILABLE = True
except ImportError:
    ADAM_SLM_TOKENIZER_AVAILABLE = False

# Import integration system
try:
    from .tokenizer_integration import (
        TokenizerIntegrationManager,
        get_integration_manager,
        get_smart_tokenizer
    )
    INTEGRATION_AVAILABLE = True
except ImportError:
    INTEGRATION_AVAILABLE = False

# Import training components
try:
    from .adam_bpe_trainer import AdamBPETrainer, DomainWeightingSystem, EnhancedMergeScorer
    # Temporarily disable corpus_analyzer import to avoid hanging
    # from .corpus_analyzer import DomainCorpusAnalyzer
    TRAINING_COMPONENTS_AVAILABLE = True
except ImportError:
    TRAINING_COMPONENTS_AVAILABLE = False


class SmartAdamTokenizer:
    """
    Smart tokenizer that automatically selects between GPT-2 and A.D.A.M.-SLM
    Maintains backward compatibility while providing enhanced capabilities
    """

    def __init__(self, encoding_name: str = "adam_slm", fallback_to_gpt2: bool = True):
        """
        Initialize smart tokenizer - now defaults to A.D.A.M.-SLM

        Args:
            encoding_name: 'adam_slm' for custom tokenizer (default), 'gpt2' for original
            fallback_to_gpt2: Whether to fallback to GPT-2 if A.D.A.M. not available
        """
        if INTEGRATION_AVAILABLE:
            try:
                self.integration_manager = get_integration_manager()
                # Set integration mode to prefer A.D.A.M.-SLM
                self.integration_manager.integration_mode = "adam_only" if encoding_name == "adam_slm" else "hybrid"
                self.tokenizer = self.integration_manager.get_tokenizer()
                self.is_smart = True
                print(f"✅ Using A.D.A.M.-SLM tokenizer (mode: {self.integration_manager.integration_mode})")
            except Exception as e:
                if fallback_to_gpt2:
                    print(f"⚠️ A.D.A.M.-SLM tokenizer failed, falling back to GPT-2: {e}")
                    self.tokenizer = AdamTokenizer("gpt2")
                    self.is_smart = False
                else:
                    raise
        else:
            if fallback_to_gpt2:
                print("⚠️ Integration system not available, using GPT-2 tokenizer")
                self.tokenizer = AdamTokenizer("gpt2")
                self.is_smart = False
            else:
                raise ImportError("A.D.A.M.-SLM tokenizer integration not available")

    def encode(self, text: str, **kwargs):
        """Encode text using smart tokenizer selection"""
        return self.tokenizer.encode(text, **kwargs)

    def decode(self, token_ids, **kwargs):
        """Decode tokens using smart tokenizer selection"""
        return self.tokenizer.decode(token_ids, **kwargs)

    def tokenize(self, text: str, **kwargs):
        """Tokenize text using smart tokenizer selection"""
        return self.tokenizer.tokenize(text, **kwargs)

    @property
    def vocab_size(self):
        """Get vocabulary size"""
        if hasattr(self.tokenizer, 'get_vocab_size'):
            return self.tokenizer.get_vocab_size()
        else:
            return getattr(self.tokenizer, 'vocab_size', 50257)

    def get_vocab(self):
        """Get vocabulary"""
        return self.tokenizer.get_vocab()

    # Special token properties for backward compatibility
    @property
    def pad_token_id(self):
        return getattr(self.tokenizer, 'pad_token_id', 50256)

    @property
    def eos_token_id(self):
        return getattr(self.tokenizer, 'eos_token_id', 50256)

    @property
    def bos_token_id(self):
        return getattr(self.tokenizer, 'bos_token_id', 50256)

    @property
    def unk_token_id(self):
        return getattr(self.tokenizer, 'unk_token_id', 50256)


# Build exports list dynamically
__all__ = [
    # Original tokenizers (backward compatibility)
    "AdamTokenizer",
    "BPETokenizer",

    # Smart tokenizer (recommended)
    "SmartAdamTokenizer",
]

# Add A.D.A.M.-SLM tokenizer if available
if ADAM_SLM_TOKENIZER_AVAILABLE:
    __all__.extend([
        "AdamSLMTokenizer",
        "create_adam_slm_tokenizer"
    ])

# Add integration system if available
if INTEGRATION_AVAILABLE:
    __all__.extend([
        "TokenizerIntegrationManager",
        "get_integration_manager",
        "get_smart_tokenizer"
    ])

# Add training components if available
if TRAINING_COMPONENTS_AVAILABLE:
    __all__.extend([
        "AdamBPETrainer",
        "DomainWeightingSystem",
        "EnhancedMergeScorer",
        # "DomainCorpusAnalyzer"  # Temporarily disabled
    ])


def get_tokenizer(encoding_name: str = "adam_slm", **kwargs):
    """
    Get tokenizer instance - now defaults to A.D.A.M.-SLM

    Args:
        encoding_name: 'adam_slm' for custom (default), 'smart' for automatic, 'gpt2' for fallback
        **kwargs: Additional arguments

    Returns:
        Tokenizer instance
    """
    if encoding_name == "adam_slm":
        return SmartAdamTokenizer(encoding_name="adam_slm", **kwargs)
    elif encoding_name == "smart":
        return SmartAdamTokenizer(**kwargs)
    elif encoding_name == "gpt2":
        return AdamTokenizer("gpt2")
    else:
        # Default to A.D.A.M.-SLM tokenizer
        return SmartAdamTokenizer(encoding_name="adam_slm", **kwargs)


# Add factory function to exports
__all__.append("get_tokenizer")
