"""
Text generation utilities for ADAM SLM
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Optional, Union, Dict, Any
import time


@dataclass
class GenerationConfig:
    """Configuration for text generation"""
    
    # Generation parameters
    max_new_tokens: int = 50
    min_new_tokens: int = 0
    
    # Sampling parameters
    do_sample: bool = True
    temperature: float = 1.0
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    
    # Special tokens
    pad_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    bos_token_id: Optional[int] = None
    
    # Repetition penalty
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    
    # Early stopping
    early_stopping: bool = False
    
    # Beam search (for future implementation)
    num_beams: int = 1
    
    # Performance
    use_cache: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {k: v for k, v in self.__dict__.items()}


def apply_repetition_penalty(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    penalty: float = 1.0,
) -> torch.Tensor:
    """Apply repetition penalty to logits"""
    if penalty == 1.0:
        return logits
        
    # Get unique tokens in input
    unique_tokens = torch.unique(input_ids)
    
    # Apply penalty
    for token in unique_tokens:
        if logits[token] > 0:
            logits[token] = logits[token] / penalty
        else:
            logits[token] = logits[token] * penalty
            
    return logits


def apply_no_repeat_ngram(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    no_repeat_ngram_size: int,
) -> torch.Tensor:
    """Prevent repetition of n-grams"""
    if no_repeat_ngram_size <= 0 or input_ids.size(-1) < no_repeat_ngram_size:
        return logits
        
    # Get the last n-1 tokens
    ngram_prefix = input_ids[-(no_repeat_ngram_size - 1):]
    
    # Find all n-grams in the input that start with this prefix
    for i in range(input_ids.size(-1) - no_repeat_ngram_size + 1):
        ngram = input_ids[i:i + no_repeat_ngram_size]
        if torch.equal(ngram[:-1], ngram_prefix):
            # Set probability of completing this n-gram to zero
            logits[ngram[-1]] = float('-inf')
            
    return logits


def top_k_filtering(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """Apply top-k filtering to logits"""
    if top_k <= 0:
        return logits
        
    top_k = min(top_k, logits.size(-1))
    top_k_logits, _ = torch.topk(logits, top_k)
    logits[logits < top_k_logits[..., [-1]]] = float('-inf')
    
    return logits


def top_p_filtering(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Apply top-p (nucleus) filtering to logits"""
    if top_p >= 1.0:
        return logits
        
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    
    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    
    indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
    logits[indices_to_remove] = float('-inf')
    
    return logits


def generate_text(
    model,
    tokenizer,
    prompt: str,
    config: GenerationConfig,
    device: torch.device = torch.device("cpu"),
) -> str:
    """
    Generate text from a single prompt
    
    Args:
        model: ADAM SLM model
        tokenizer: Tokenizer
        prompt: Input prompt
        config: Generation configuration
        device: Device to run on
        
    Returns:
        Generated text
    """
    model.eval()
    
    # Encode prompt
    input_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0).to(device)
    
    # Generate
    with torch.no_grad():
        generated_ids = _generate_tokens(model, input_ids, config)
        
    # Decode
    generated_text = tokenizer.decode(generated_ids[0].cpu().tolist())
    
    return generated_text


def batch_generate(
    model,
    tokenizer,
    prompts: List[str],
    config: GenerationConfig,
    device: torch.device = torch.device("cpu"),
) -> List[str]:
    """
    Generate text from multiple prompts in batch
    
    Args:
        model: ADAM SLM model
        tokenizer: Tokenizer
        prompts: List of input prompts
        config: Generation configuration
        device: Device to run on
        
    Returns:
        List of generated texts
    """
    model.eval()
    
    # Encode prompts
    input_ids_list = []
    max_len = 0
    
    for prompt in prompts:
        ids = tokenizer.encode(prompt)
        input_ids_list.append(ids)
        max_len = max(max_len, len(ids))
        
    # Pad sequences
    pad_token_id = config.pad_token_id or 0
    padded_input_ids = []
    
    for ids in input_ids_list:
        padded = ids + [pad_token_id] * (max_len - len(ids))
        padded_input_ids.append(padded)
        
    input_ids = torch.tensor(padded_input_ids, dtype=torch.long).to(device)
    
    # Generate
    with torch.no_grad():
        generated_ids = _generate_tokens(model, input_ids, config)
        
    # Decode
    generated_texts = []
    for i in range(generated_ids.size(0)):
        text = tokenizer.decode(generated_ids[i].cpu().tolist())
        generated_texts.append(text)
        
    return generated_texts


def _generate_tokens(
    model,
    input_ids: torch.Tensor,
    config: GenerationConfig,
) -> torch.Tensor:
    """
    Core token generation function
    
    Args:
        model: ADAM SLM model
        input_ids: Input token IDs [batch_size, seq_len]
        config: Generation configuration
        
    Returns:
        Generated token IDs [batch_size, seq_len + new_tokens]
    """
    batch_size, seq_len = input_ids.shape
    device = input_ids.device
    
    # Initialize generation
    generated = input_ids.clone()
    past_key_values = None
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    
    for step in range(config.max_new_tokens):
        # Forward pass
        if config.use_cache and past_key_values is not None:
            # Only pass the last token for efficiency
            model_input = generated[:, -1:]
        else:
            model_input = generated
            
        outputs = model(
            input_ids=model_input,
            past_key_values=past_key_values,
            use_cache=config.use_cache,
        )
        
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs[0]
        past_key_values = outputs.get("past_key_values") if isinstance(outputs, dict) else None
        
        # Get logits for the last token
        next_token_logits = logits[:, -1, :]
        
        # Apply temperature
        if config.temperature != 1.0:
            next_token_logits = next_token_logits / config.temperature
            
        # Apply repetition penalty
        if config.repetition_penalty != 1.0:
            for i in range(batch_size):
                if not finished[i]:
                    next_token_logits[i] = apply_repetition_penalty(
                        next_token_logits[i],
                        generated[i],
                        config.repetition_penalty,
                    )
                    
        # Apply no-repeat n-gram penalty
        if config.no_repeat_ngram_size > 0:
            for i in range(batch_size):
                if not finished[i]:
                    next_token_logits[i] = apply_no_repeat_ngram(
                        next_token_logits[i],
                        generated[i],
                        config.no_repeat_ngram_size,
                    )
                    
        # Apply top-k filtering
        if config.top_k is not None:
            next_token_logits = top_k_filtering(next_token_logits, config.top_k)
            
        # Apply top-p filtering
        if config.top_p is not None:
            next_token_logits = top_p_filtering(next_token_logits, config.top_p)
            
        # Sample next tokens
        if config.do_sample:
            probs = F.softmax(next_token_logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)
        else:
            next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
        # Update finished sequences
        if config.eos_token_id is not None:
            finished = finished | (next_tokens.squeeze(-1) == config.eos_token_id)
            
        # Append to generated sequence
        generated = torch.cat([generated, next_tokens], dim=-1)
        
        # Check if all sequences are finished
        if config.early_stopping and finished.all():
            break
            
        # Check minimum length
        if step < config.min_new_tokens:
            continue
            
    return generated
