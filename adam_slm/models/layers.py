"""
Advanced layer components for ADAM SLM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .attention import GroupedQueryAttention, create_causal_mask


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization
    More stable than LayerNorm and used in modern LLMs like LLaMA
    """
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute RMS
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        # Normalize and scale
        return x / rms * self.weight


class SwiGLU(nn.Module):
    """
    SwiGLU activation function
    Combines Swish activation with Gated Linear Unit
    Used in modern LLMs for better performance
    """
    
    def __init__(self, dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = int(2.67 * dim)  # Standard scaling factor
            
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)  # Gate projection
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)  # Down projection
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)  # Up projection
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: swish(W1(x)) * W3(x) -> W2
        gate = F.silu(self.w1(x))  # Swish/SiLU activation
        up = self.w3(x)
        return self.w2(gate * up)


class FeedForward(nn.Module):
    """
    Feed-forward network with choice of activation
    """
    
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
        use_swiglu: bool = True,
    ):
        super().__init__()
        self.use_swiglu = use_swiglu
        
        if use_swiglu:
            self.ffn = SwiGLU(dim, hidden_dim)
        else:
            self.ffn = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, dim),
            )
            
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.ffn(x)
        return self.dropout(output)


class TransformerBlock(nn.Module):
    """
    Advanced Transformer block with modern improvements
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        d_ff: int,
        dropout: float = 0.0,
        layer_norm_eps: float = 1e-6,
        use_rope: bool = True,
        use_swiglu: bool = True,
        use_rms_norm: bool = True,
        rope_theta: float = 10000.0,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        
        # Normalization layers
        norm_class = RMSNorm if use_rms_norm else nn.LayerNorm
        if use_rms_norm:
            self.input_layernorm = norm_class(d_model, layer_norm_eps)
            self.post_attention_layernorm = norm_class(d_model, layer_norm_eps)
        else:
            self.input_layernorm = norm_class(d_model, eps=layer_norm_eps)
            self.post_attention_layernorm = norm_class(d_model, eps=layer_norm_eps)
            
        # Attention layer
        self.self_attn = GroupedQueryAttention(
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            dropout=dropout,
            use_rope=use_rope,
            rope_theta=rope_theta,
            max_seq_len=max_seq_len,
        )
        
        # Feed-forward layer
        self.mlp = FeedForward(
            dim=d_model,
            hidden_dim=d_ff,
            dropout=dropout,
            use_swiglu=use_swiglu,
        )
        
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        
        # Pre-norm attention
        residual = x
        x = self.input_layernorm(x)
        
        # Self-attention
        attn_output, present_key_value = self.self_attn(
            x,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        
        # Residual connection
        x = residual + attn_output
        
        # Pre-norm feed-forward
        residual = x
        x = self.post_attention_layernorm(x)
        
        # Feed-forward
        ffn_output = self.mlp(x)
        
        # Residual connection
        x = residual + ffn_output
        
        return x, present_key_value


class AdamEmbedding(nn.Module):
    """
    Advanced embedding layer with optional scaling
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_seq_len: int,
        dropout: float = 0.0,
        scale_embeddings: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.scale_embeddings = scale_embeddings
        
        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize embeddings
        self._init_embeddings()
        
    def _init_embeddings(self):
        """Initialize embeddings with proper scaling"""
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.token_embedding(input_ids)
        
        if self.scale_embeddings:
            embeddings = embeddings * (self.d_model ** 0.5)
            
        return self.dropout(embeddings)
