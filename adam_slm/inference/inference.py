"""
Main inference class for ADAM SLM
"""

import torch
import torch.nn as nn
from typing import List, Optional, Union, Dict, Any
import time
import os

from .generation import GenerationConfig, generate_text, batch_generate
from ..models import AdamSLM, AdamSLMConfig


class AdamInference:
    """
    High-level inference interface for ADAM SLM
    """
    
    def __init__(
        self,
        model: AdamSLM,
        tokenizer=None,
        device: Optional[torch.device] = None,
        generation_config: Optional[GenerationConfig] = None,
    ):
        self.model = model

        # Use model's tokenizer if available, otherwise get A.D.A.M.-SLM tokenizer
        if tokenizer is None:
            if hasattr(model, 'get_tokenizer') and model.get_tokenizer() is not None:
                self.tokenizer = model.get_tokenizer()
                print("✅ Using model's A.D.A.M.-SLM tokenizer")
            else:
                # Get A.D.A.M.-SLM tokenizer
                from ..tokenization import get_tokenizer
                self.tokenizer = get_tokenizer("adam_slm")
                print("✅ Initialized A.D.A.M.-SLM tokenizer for inference")
        else:
            self.tokenizer = tokenizer
        
        # Setup device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        self.model.to(self.device)
        self.model.eval()
        
        # Default generation config
        if generation_config is None:
            generation_config = GenerationConfig()
        self.generation_config = generation_config
        
        # Performance tracking
        self.generation_stats = {
            "total_generations": 0,
            "total_tokens_generated": 0,
            "total_time": 0.0,
        }
        
    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        tokenizer,
        device: Optional[torch.device] = None,
        generation_config: Optional[GenerationConfig] = None,
    ) -> "AdamInference":
        """
        Load model from checkpoint for inference
        
        Args:
            model_path: Path to model checkpoint
            tokenizer: Tokenizer to use
            device: Device to load model on
            generation_config: Generation configuration
            
        Returns:
            AdamInference instance
        """
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location="cpu")
        
        # Load config
        if "config" in checkpoint:
            model_config = AdamSLMConfig.from_dict(checkpoint["config"])
        else:
            # Try to load from separate config file
            config_path = os.path.join(os.path.dirname(model_path), "config.json")
            if os.path.exists(config_path):
                model_config = AdamSLMConfig.from_json(config_path)
            else:
                raise ValueError("Could not find model configuration")
                
        # Create model
        model = AdamSLM(model_config)
        model.load_state_dict(checkpoint["model_state_dict"])
        
        return cls(
            model=model,
            tokenizer=tokenizer,
            device=device,
            generation_config=generation_config,
        )
        
    def generate(
        self,
        prompt: Union[str, List[str]],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: Optional[bool] = None,
        **kwargs
    ) -> Union[str, List[str]]:
        """
        Generate text from prompt(s)
        
        Args:
            prompt: Input prompt or list of prompts
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p sampling parameter
            do_sample: Whether to use sampling
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text or list of generated texts
        """
        start_time = time.time()
        
        # Create generation config for this call
        config = GenerationConfig(**self.generation_config.to_dict())
        
        # Override with provided parameters
        if max_new_tokens is not None:
            config.max_new_tokens = max_new_tokens
        if temperature is not None:
            config.temperature = temperature
        if top_k is not None:
            config.top_k = top_k
        if top_p is not None:
            config.top_p = top_p
        if do_sample is not None:
            config.do_sample = do_sample
            
        # Update with additional kwargs
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
                
        # Generate
        if isinstance(prompt, str):
            result = generate_text(
                model=self.model,
                tokenizer=self.tokenizer,
                prompt=prompt,
                config=config,
                device=self.device,
            )
            num_prompts = 1
        else:
            result = batch_generate(
                model=self.model,
                tokenizer=self.tokenizer,
                prompts=prompt,
                config=config,
                device=self.device,
            )
            num_prompts = len(prompt)
            
        # Update stats
        generation_time = time.time() - start_time
        self.generation_stats["total_generations"] += num_prompts
        self.generation_stats["total_tokens_generated"] += config.max_new_tokens * num_prompts
        self.generation_stats["total_time"] += generation_time
        
        return result
        
    def chat(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Chat interface for conversational generation
        
        Args:
            message: User message
            history: Conversation history
            system_prompt: System prompt
            **kwargs: Additional generation parameters
            
        Returns:
            Assistant response
        """
        # Build conversation prompt
        prompt_parts = []
        
        if system_prompt:
            prompt_parts.append(f"System: {system_prompt}")
            
        if history:
            for turn in history:
                if turn.get("role") == "user":
                    prompt_parts.append(f"User: {turn['content']}")
                elif turn.get("role") == "assistant":
                    prompt_parts.append(f"Assistant: {turn['content']}")
                    
        prompt_parts.append(f"User: {message}")
        prompt_parts.append("Assistant:")
        
        prompt = "\n".join(prompt_parts)
        
        # Generate response
        response = self.generate(prompt, **kwargs)
        
        # Extract assistant response (remove the prompt part)
        if isinstance(response, str):
            # Find the last "Assistant:" and return everything after it
            assistant_start = response.rfind("Assistant:")
            if assistant_start != -1:
                response = response[assistant_start + len("Assistant:"):].strip()
                
        return response
        
    def complete(
        self,
        text: str,
        max_new_tokens: int = 50,
        **kwargs
    ) -> str:
        """
        Complete a given text
        
        Args:
            text: Text to complete
            max_new_tokens: Maximum tokens to generate
            **kwargs: Additional generation parameters
            
        Returns:
            Completed text
        """
        return self.generate(
            prompt=text,
            max_new_tokens=max_new_tokens,
            **kwargs
        )
        
    def get_perplexity(
        self,
        text: str,
        stride: int = 512,
    ) -> float:
        """
        Calculate perplexity of text
        
        Args:
            text: Text to evaluate
            stride: Stride for sliding window
            
        Returns:
            Perplexity score
        """
        self.model.eval()
        
        # Tokenize text
        token_ids = self.tokenizer.encode(text)
        
        if len(token_ids) < 2:
            return float('inf')
            
        # Calculate perplexity using sliding window
        total_loss = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for i in range(0, len(token_ids) - 1, stride):
                # Get window
                start_idx = max(0, i)
                end_idx = min(len(token_ids), i + stride + 1)
                
                input_ids = torch.tensor(token_ids[start_idx:end_idx], dtype=torch.long).unsqueeze(0).to(self.device)
                
                # Forward pass
                outputs = self.model(input_ids=input_ids)
                logits = outputs["logits"] if isinstance(outputs, dict) else outputs[0]
                
                # Calculate loss
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = input_ids[..., 1:].contiguous()
                
                loss = torch.nn.functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    reduction='sum'
                )
                
                total_loss += loss.item()
                total_tokens += shift_labels.numel()
                
        # Calculate perplexity
        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return perplexity
        
    def get_stats(self) -> Dict[str, Any]:
        """Get generation statistics"""
        stats = self.generation_stats.copy()
        
        if stats["total_time"] > 0:
            stats["tokens_per_second"] = stats["total_tokens_generated"] / stats["total_time"]
            stats["generations_per_second"] = stats["total_generations"] / stats["total_time"]
        else:
            stats["tokens_per_second"] = 0.0
            stats["generations_per_second"] = 0.0
            
        return stats
        
    def reset_stats(self):
        """Reset generation statistics"""
        self.generation_stats = {
            "total_generations": 0,
            "total_tokens_generated": 0,
            "total_time": 0.0,
        }
        
    def optimize_for_inference(self):
        """Optimize model for inference"""
        # Compile model if available (PyTorch 2.0+)
        if hasattr(torch, 'compile'):
            try:
                self.model = torch.compile(self.model)
                print("Model compiled for faster inference")
            except Exception as e:
                print(f"Could not compile model: {e}")
                
        # Set to eval mode and disable gradients
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)
