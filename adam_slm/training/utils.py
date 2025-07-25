"""
Training utilities for ADAM SLM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, ConstantLR
from typing import Dict, Any, Optional, List
import math
import time
from tqdm import tqdm


def compute_loss(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    """
    Compute cross-entropy loss for language modeling
    
    Args:
        logits: Model predictions [batch_size, seq_len, vocab_size]
        labels: Target labels [batch_size, seq_len]
        ignore_index: Index to ignore in loss computation
        
    Returns:
        Loss tensor
    """
    # Shift logits and labels for next token prediction
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    # Flatten for cross-entropy
    shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    shift_labels = shift_labels.view(-1)
    
    # Compute loss
    loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=ignore_index)
    
    return loss


def compute_perplexity(loss: torch.Tensor) -> float:
    """Compute perplexity from loss"""
    return math.exp(loss.item())


def get_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str,
    num_training_steps: int,
    warmup_steps: int = 0,
    min_lr_ratio: float = 0.1,
    **kwargs
):
    """
    Get learning rate scheduler
    
    Args:
        optimizer: PyTorch optimizer
        scheduler_type: Type of scheduler ("cosine", "linear", "constant")
        num_training_steps: Total number of training steps
        warmup_steps: Number of warmup steps
        min_lr_ratio: Minimum learning rate as ratio of initial LR
        
    Returns:
        Learning rate scheduler
    """
    if scheduler_type == "cosine":
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            
            progress = float(current_step - warmup_steps) / float(max(1, num_training_steps - warmup_steps))
            return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
            
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
    elif scheduler_type == "linear":
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            
            progress = float(current_step - warmup_steps) / float(max(1, num_training_steps - warmup_steps))
            return max(min_lr_ratio, 1.0 - progress)
            
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
    elif scheduler_type == "constant":
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return 1.0
            
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


def evaluate_model(
    model: nn.Module,
    dataloader,
    device: torch.device,
    max_eval_steps: Optional[int] = None,
) -> Dict[str, float]:
    """
    Evaluate model on validation data
    
    Args:
        model: Model to evaluate
        dataloader: Validation data loader
        device: Device to run evaluation on
        max_eval_steps: Maximum number of evaluation steps
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    num_steps = 0
    
    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            if max_eval_steps is not None and step >= max_eval_steps:
                break
                
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass
            outputs = model(input_ids=input_ids)
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs[0]
            
            # Compute loss
            loss = compute_loss(logits, labels)
            
            # Accumulate metrics
            total_loss += loss.item()
            total_tokens += labels.numel()
            num_steps += 1
            
    # Compute averages
    avg_loss = total_loss / num_steps if num_steps > 0 else 0.0
    perplexity = compute_perplexity(torch.tensor(avg_loss))
    
    model.train()
    
    return {
        "eval_loss": avg_loss,
        "eval_perplexity": perplexity,
        "eval_steps": num_steps,
        "eval_tokens": total_tokens,
    }


def generate_sample(
    model: nn.Module,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    device: torch.device = torch.device("cpu"),
) -> str:
    """
    Generate text sample from model
    
    Args:
        model: Model to use for generation
        tokenizer: Tokenizer for encoding/decoding
        prompt: Input prompt
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        device: Device to run generation on
        
    Returns:
        Generated text
    """
    model.eval()
    
    # Encode prompt
    input_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0).to(device)
    
    # Generate
    with torch.no_grad():
        generated = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            do_sample=True,
        )
        
    # Decode
    generated_text = tokenizer.decode(generated[0].cpu().tolist())
    
    model.train()
    
    return generated_text


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    step: int,
    loss: float,
    save_path: str,
    config: Optional[Dict[str, Any]] = None,
):
    """Save training checkpoint"""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "step": step,
        "loss": loss,
        "config": config,
    }
    
    torch.save(checkpoint, save_path)


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Any]:
    """Load training checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # Load optimizer state if provided
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
    # Load scheduler state if provided
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
    return {
        "step": checkpoint.get("step", 0),
        "loss": checkpoint.get("loss", 0.0),
        "config": checkpoint.get("config", {}),
    }


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count model parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "non_trainable_parameters": total_params - trainable_params,
    }


def format_time(seconds: float) -> str:
    """Format time in human readable format"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


class TrainingMetrics:
    """Class to track training metrics"""
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        """Reset all metrics"""
        self.losses = []
        self.learning_rates = []
        self.steps = []
        self.times = []
        self.start_time = time.time()
        
    def update(self, loss: float, lr: float, step: int):
        """Update metrics"""
        self.losses.append(loss)
        self.learning_rates.append(lr)
        self.steps.append(step)
        self.times.append(time.time() - self.start_time)
        
    def get_avg_loss(self, last_n: int = 100) -> float:
        """Get average loss over last n steps"""
        if not self.losses:
            return 0.0
        return sum(self.losses[-last_n:]) / min(len(self.losses), last_n)
        
    def get_current_lr(self) -> float:
        """Get current learning rate"""
        return self.learning_rates[-1] if self.learning_rates else 0.0
        
    def get_steps_per_second(self, last_n: int = 100) -> float:
        """Get training speed in steps per second"""
        if len(self.times) < 2:
            return 0.0
            
        recent_times = self.times[-last_n:]
        if len(recent_times) < 2:
            return 0.0
            
        time_diff = recent_times[-1] - recent_times[0]
        steps_diff = len(recent_times) - 1
        
        return steps_diff / time_diff if time_diff > 0 else 0.0
