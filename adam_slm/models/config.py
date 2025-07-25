"""
Configuration classes for A.D.A.M. SLM models
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import json


@dataclass
class AdamSLMConfig:
    """Configuration class for A.D.A.M. SLM model"""
    
    # Model architecture
    vocab_size: int = 50257
    max_seq_len: int = 2048
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    n_kv_heads: Optional[int] = None  # For GQA, defaults to n_heads
    d_ff: int = 3072  # Feed-forward dimension
    
    # Attention settings
    rope_theta: float = 10000.0
    rope_scaling: Optional[Dict[str, Any]] = None
    attention_dropout: float = 0.0
    
    # Regularization
    dropout: float = 0.1
    layer_norm_eps: float = 1e-5
    
    # Training settings
    tie_word_embeddings: bool = True
    use_cache: bool = True
    
    # Advanced features
    use_rope: bool = True
    use_swiglu: bool = True
    use_rms_norm: bool = True
    use_gqa: bool = True

    # Tokenizer settings
    tokenizer_type: str = "adam_slm"  # Default to A.D.A.M.-SLM tokenizer
    tokenizer_fallback: bool = True   # Allow fallback to GPT-2 if needed

    # Initialization
    initializer_range: float = 0.02
    
    def __post_init__(self):
        """Post-initialization validation and setup"""
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
            
        if self.use_gqa and self.n_kv_heads > self.n_heads:
            raise ValueError("n_kv_heads cannot be greater than n_heads for GQA")
            
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
            
        self.head_dim = self.d_model // self.n_heads
        self.kv_head_dim = self.d_model // self.n_kv_heads if self.use_gqa else self.head_dim
        
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "AdamSLMConfig":
        """Create config from dictionary"""
        return cls(**config_dict)
        
    @classmethod
    def from_json(cls, json_path: str) -> "AdamSLMConfig":
        """Load config from JSON file"""
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        
    def to_json(self, json_path: str):
        """Save config to JSON file"""
        with open(json_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


# Predefined configurations
ADAM_SLM_CONFIGS = {
    "adam-slm-small": AdamSLMConfig(
        vocab_size=50257,
        max_seq_len=1024,
        d_model=512,
        n_layers=8,
        n_heads=8,
        n_kv_heads=4,  # GQA
        d_ff=2048,
        dropout=0.1,
    ),
    
    "adam-slm-base": AdamSLMConfig(
        vocab_size=50257,
        max_seq_len=2048,
        d_model=768,
        n_layers=12,
        n_heads=12,
        n_kv_heads=6,  # GQA
        d_ff=3072,
        dropout=0.1,
    ),
    
    "adam-slm-large": AdamSLMConfig(
        vocab_size=50257,
        max_seq_len=4096,
        d_model=1024,
        n_layers=24,
        n_heads=16,
        n_kv_heads=8,  # GQA
        d_ff=4096,
        dropout=0.1,
    ),
}


def get_config(model_name: str) -> AdamSLMConfig:
    """Get predefined configuration by name"""
    if model_name not in ADAM_SLM_CONFIGS:
        raise ValueError(f"Unknown model config: {model_name}. Available: {list(ADAM_SLM_CONFIGS.keys())}")
    return ADAM_SLM_CONFIGS[model_name]
