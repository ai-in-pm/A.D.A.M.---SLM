"""
Main A.D.A.M. SLM model implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Union
import math

from .config import AdamSLMConfig
from .layers import TransformerBlock, AdamEmbedding, RMSNorm
from .attention import create_causal_mask


class AdamSLM(nn.Module):
    """
    A.D.A.M. SLM - Applied Decision Architecture Matrix - Small Language Model

    A sophisticated transformer-based language model with:
    - Rotary Position Embeddings (RoPE)
    - Grouped Query Attention (GQA)
    - SwiGLU activation
    - RMSNorm normalization
    - KV-Cache support
    """
    
    def __init__(self, config: AdamSLMConfig):
        super().__init__()
        self.config = config

        # Initialize A.D.A.M.-SLM tokenizer
        self._initialize_tokenizer()

        # Embeddings
        self.embed_tokens = AdamEmbedding(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout,
        )
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=config.d_model,
                n_heads=config.n_heads,
                n_kv_heads=config.n_kv_heads,
                d_ff=config.d_ff,
                dropout=config.dropout,
                layer_norm_eps=config.layer_norm_eps,
                use_rope=config.use_rope,
                use_swiglu=config.use_swiglu,
                use_rms_norm=config.use_rms_norm,
                rope_theta=config.rope_theta,
                max_seq_len=config.max_seq_len,
            )
            for _ in range(config.n_layers)
        ])
        
        # Final normalization
        if config.use_rms_norm:
            self.norm = RMSNorm(config.d_model, config.layer_norm_eps)
        else:
            self.norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
            
        # Language modeling head
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Tie embeddings if specified
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.token_embedding.weight
            
        # Initialize weights
        self.apply(self._init_weights)

    def _initialize_tokenizer(self):
        """Initialize A.D.A.M.-SLM tokenizer"""
        try:
            from ..tokenization import get_tokenizer
            self.tokenizer = get_tokenizer(
                encoding_name=self.config.tokenizer_type,
                fallback_to_gpt2=self.config.tokenizer_fallback
            )
            print(f"✅ Model initialized with {self.config.tokenizer_type} tokenizer")
        except Exception as e:
            print(f"⚠️ Failed to initialize tokenizer: {e}")
            # Fallback to basic tokenizer
            from ..tokenization import AdamTokenizer
            self.tokenizer = AdamTokenizer("gpt2")
            print("⚠️ Using GPT-2 fallback tokenizer")

    def get_tokenizer(self):
        """Get the model's tokenizer"""
        return getattr(self, 'tokenizer', None)

    def _init_weights(self, module):
        """Initialize model weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            
    def get_input_embeddings(self):
        return self.embed_tokens.token_embedding
        
    def set_input_embeddings(self, value):
        self.embed_tokens.token_embedding = value
        
    def get_output_embeddings(self):
        return self.lm_head
        
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, dict]:
        
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_hidden_states = output_hidden_states if output_hidden_states is not None else False
        return_dict = return_dict if return_dict is not None else True
        
        batch_size, seq_len = input_ids.shape
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = create_causal_mask(seq_len, input_ids.device)
        elif attention_mask.dim() == 2:
            # Convert padding mask to causal mask
            causal_mask = create_causal_mask(seq_len, input_ids.device)
            # Combine with padding mask
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = attention_mask.expand(batch_size, 1, seq_len, seq_len)
            attention_mask = attention_mask + causal_mask
            
        # Get embeddings
        hidden_states = self.embed_tokens(input_ids)
        
        # Initialize past key values if not provided
        if past_key_values is None:
            past_key_values = [None] * len(self.layers)
            
        # Store all hidden states if requested
        all_hidden_states = () if output_hidden_states else None
        
        # Present key values for caching
        present_key_values = () if use_cache else None
        
        # Forward through transformer layers
        for i, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
                
            hidden_states, present_key_value = layer(
                hidden_states,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                use_cache=use_cache,
            )
            
            if use_cache:
                present_key_values = present_key_values + (present_key_value,)
                
        # Final normalization
        hidden_states = self.norm(hidden_states)
        
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
            
        # Language modeling head
        logits = self.lm_head(hidden_states)
        
        if not return_dict:
            outputs = (logits,)
            if use_cache:
                outputs = outputs + (present_key_values,)
            if output_hidden_states:
                outputs = outputs + (all_hidden_states,)
            return outputs
            
        return {
            "logits": logits,
            "past_key_values": present_key_values if use_cache else None,
            "hidden_states": all_hidden_states if output_hidden_states else None,
        }
        
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate text using the model
        """
        self.eval()
        
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # Initialize generation
        generated = input_ids.clone()
        past_key_values = None
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Forward pass
                outputs = self.forward(
                    input_ids=generated if past_key_values is None else generated[:, -1:],
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                
                logits = outputs["logits"][:, -1, :]  # Get last token logits
                past_key_values = outputs["past_key_values"]
                
                # Apply temperature
                if temperature != 1.0:
                    logits = logits / temperature
                    
                # Apply top-k filtering
                if top_k is not None:
                    top_k = min(top_k, logits.size(-1))
                    top_k_logits, _ = torch.topk(logits, top_k)
                    logits[logits < top_k_logits[:, [-1]]] = float('-inf')
                    
                # Apply top-p (nucleus) filtering
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float('-inf')
                    
                # Sample next token
                if do_sample:
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                    
                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=-1)
                
                # Check for EOS token
                if eos_token_id is not None and (next_token == eos_token_id).all():
                    break
                    
        return generated
        
    def get_num_params(self, non_embedding: bool = True) -> int:
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.embed_tokens.token_embedding.weight.numel()
        return n_params
