"""
Advanced attention mechanisms for ADAM SLM
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) implementation
    Based on "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    """
    
    def __init__(self, dim: int, max_seq_len: int = 2048, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        
        # Precompute frequency tensor
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Precompute cos and sin for efficiency
        self._precompute_freqs_cis(max_seq_len)
        
    def _precompute_freqs_cis(self, seq_len: int):
        """Precompute cos and sin frequencies"""
        t = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)
        
    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None) -> torch.Tensor:
        """Apply rotary position embedding"""
        if seq_len is None:
            seq_len = x.shape[-2]
            
        if seq_len > self.max_seq_len:
            self._precompute_freqs_cis(seq_len)
            
        freqs_cis = self.freqs_cis[:seq_len]
        
        # Reshape x to complex representation
        x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        
        # Apply rotation
        x_rotated = x_complex * freqs_cis.unsqueeze(0).unsqueeze(0)
        
        # Convert back to real representation
        x_out = torch.view_as_real(x_rotated).flatten(-2)
        
        return x_out.type_as(x)


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA) implementation
    Reduces memory usage by sharing key-value heads across query heads
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        dropout: float = 0.0,
        use_rope: bool = True,
        rope_theta: float = 10000.0,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.n_rep = n_heads // n_kv_heads  # Number of repetitions for KV heads
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        assert n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        
        # Linear projections
        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * self.head_dim, d_model, bias=False)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # RoPE
        self.use_rope = use_rope
        if use_rope:
            self.rope = RotaryPositionalEmbedding(
                self.head_dim, max_seq_len, rope_theta
            )
            
        # Scale factor for attention
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """Repeat key-value heads to match query heads"""
        batch_size, seq_len, n_kv_heads, head_dim = x.shape
        if self.n_rep == 1:
            return x
        return x[:, :, :, None, :].expand(batch_size, seq_len, n_kv_heads, self.n_rep, head_dim).reshape(
            batch_size, seq_len, n_kv_heads * self.n_rep, head_dim
        )
        
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        
        # Apply RoPE if enabled
        if self.use_rope:
            q = self.rope(q)
            k = self.rope(k)
            
        # Handle past key-value cache
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=1)
            v = torch.cat([past_v, v], dim=1)
            
        # Update cache
        if use_cache:
            present_key_value = (k, v)
        else:
            present_key_value = None
            
        # Repeat KV heads to match Q heads
        k = self._repeat_kv(k)
        v = self._repeat_kv(v)
        
        # Transpose for attention computation
        q = q.transpose(1, 2)  # (batch_size, n_heads, seq_len, head_dim)
        k = k.transpose(1, 2)  # (batch_size, n_heads, kv_seq_len, head_dim)
        v = v.transpose(1, 2)  # (batch_size, n_heads, kv_seq_len, head_dim)
        
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply attention mask
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask
            
        # Apply softmax
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.n_heads * self.head_dim
        )
        output = self.o_proj(attn_output)
        
        return output, present_key_value


def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Create causal attention mask"""
    mask = torch.full((seq_len, seq_len), float('-inf'), device=device)
    mask = torch.triu(mask, diagonal=1)
    return mask
