"""
Main trainer class for A.D.A.M. SLM
"""

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
import os
import json
from typing import Optional, Dict, Any, List
from tqdm import tqdm
import wandb

from .config import TrainingConfig
from .utils import (
    compute_loss, evaluate_model, get_lr_scheduler, save_checkpoint,
    load_checkpoint, count_parameters, format_time, TrainingMetrics,
    generate_sample
)


class AdamTrainer:
    """
    Advanced trainer for A.D.A.M. SLM with modern training features
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        train_dataloader,
        eval_dataloader=None,
        tokenizer=None,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader

        # Initialize A.D.A.M.-SLM tokenizer if not provided
        if tokenizer is None:
            if hasattr(model, 'get_tokenizer') and model.get_tokenizer() is not None:
                self.tokenizer = model.get_tokenizer()
                print("✅ Using model's A.D.A.M.-SLM tokenizer for training")
            else:
                from ..tokenization import get_tokenizer
                self.tokenizer = get_tokenizer("adam_slm")
                print("✅ Initialized A.D.A.M.-SLM tokenizer for training")
        else:
            self.tokenizer = tokenizer
        
        # Setup device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        
        # Setup learning rate scheduler
        self.scheduler = None
        if config.max_steps:
            self.scheduler = get_lr_scheduler(
                optimizer=self.optimizer,
                scheduler_type=config.lr_scheduler_type,
                num_training_steps=config.max_steps,
                warmup_steps=config.warmup_steps,
                min_lr_ratio=config.min_lr_ratio,
            )
            
        # Setup mixed precision
        self.scaler = None
        if config.use_amp and self.device.type == "cuda":
            self.scaler = GradScaler()
            
        # Setup gradient checkpointing
        if config.use_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_eval_loss = float('inf')
        self.metrics = TrainingMetrics()
        
        # Setup logging
        self._setup_logging()
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
        
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer with proper weight decay handling"""
        # Separate parameters that should and shouldn't have weight decay
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Don't apply weight decay to bias terms and layer norm parameters
                if any(nd in name for nd in ["bias", "norm", "ln"]):
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)
                    
        optimizer_grouped_parameters = [
            {
                "params": decay_params,
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": no_decay_params,
                "weight_decay": 0.0,
            },
        ]
        
        return torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
            betas=(self.config.beta1, self.config.beta2),
            eps=self.config.eps,
        )
        
    def _setup_logging(self):
        """Setup logging and monitoring"""
        if "wandb" in self.config.report_to:
            wandb.init(
                project="adam-slm",
                name=self.config.run_name,
                config=self.config.to_dict(),
            )
            
    def train(self) -> Dict[str, Any]:
        """
        Main training loop
        
        Returns:
            Training statistics
        """
        print(f"Starting training on {self.device}")
        print(f"Model parameters: {count_parameters(self.model)}")
        
        self.model.train()
        self.metrics.reset()
        
        # Training loop
        epoch = 0
        while True:
            epoch_loss = 0.0
            num_batches = 0
            
            progress_bar = tqdm(
                self.train_dataloader,
                desc=f"Epoch {epoch + 1}",
                disable=False,
            )
            
            for batch_idx, batch in enumerate(progress_bar):
                # Check if we've reached max steps
                if self.config.max_steps and self.global_step >= self.config.max_steps:
                    break
                    
                # Training step
                loss = self._training_step(batch)
                epoch_loss += loss
                num_batches += 1
                
                # Update metrics
                current_lr = self.optimizer.param_groups[0]['lr']
                self.metrics.update(loss, current_lr, self.global_step)
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f'{loss:.4f}',
                    'lr': f'{current_lr:.2e}',
                    'step': self.global_step,
                })
                
                # Logging
                if self.global_step % self.config.logging_steps == 0:
                    self._log_metrics(loss, current_lr)
                    
                # Evaluation
                if (self.eval_dataloader is not None and 
                    self.global_step % self.config.eval_steps == 0 and 
                    self.global_step > 0):
                    eval_metrics = self._evaluate()
                    self._log_eval_metrics(eval_metrics)
                    
                # Save checkpoint
                if self.global_step % self.config.save_steps == 0 and self.global_step > 0:
                    self._save_checkpoint()
                    
                self.global_step += 1
                
            # Check if we've reached max steps
            if self.config.max_steps and self.global_step >= self.config.max_steps:
                break
                
            epoch += 1
            
        # Final evaluation and save
        if self.eval_dataloader is not None:
            final_eval_metrics = self._evaluate()
            self._log_eval_metrics(final_eval_metrics)
            
        self._save_checkpoint(is_final=True)
        
        return {
            "final_loss": epoch_loss / num_batches if num_batches > 0 else 0.0,
            "total_steps": self.global_step,
            "epochs": epoch,
        }
        
    def _training_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Single training step"""
        # Move batch to device
        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)
        
        # Forward pass with mixed precision
        if self.config.use_amp and self.scaler is not None:
            with autocast():
                outputs = self.model(input_ids=input_ids)
                logits = outputs["logits"] if isinstance(outputs, dict) else outputs[0]
                loss = compute_loss(logits, labels)
                
            # Scale loss for gradient accumulation
            loss = loss / self.config.gradient_accumulation_steps
            
            # Backward pass
            self.scaler.scale(loss).backward()
            
            # Gradient accumulation
            if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.config.max_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.max_grad_norm
                    )
                    
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                
                # Scheduler step
                if self.scheduler is not None:
                    self.scheduler.step()
        else:
            # Regular forward pass
            outputs = self.model(input_ids=input_ids)
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs[0]
            loss = compute_loss(logits, labels)
            
            # Scale loss for gradient accumulation
            loss = loss / self.config.gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Gradient accumulation
            if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.max_grad_norm
                    )
                    
                # Optimizer step
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                # Scheduler step
                if self.scheduler is not None:
                    self.scheduler.step()
                    
        return loss.item() * self.config.gradient_accumulation_steps
        
    def _evaluate(self) -> Dict[str, float]:
        """Evaluate model on validation set"""
        eval_metrics = evaluate_model(
            model=self.model,
            dataloader=self.eval_dataloader,
            device=self.device,
        )
        
        # Update best model tracking
        if eval_metrics["eval_loss"] < self.best_eval_loss:
            self.best_eval_loss = eval_metrics["eval_loss"]
            if self.config.load_best_model_at_end:
                self._save_checkpoint(is_best=True)
                
        return eval_metrics
        
    def _log_metrics(self, loss: float, lr: float):
        """Log training metrics"""
        metrics = {
            "train/loss": loss,
            "train/learning_rate": lr,
            "train/step": self.global_step,
            "train/steps_per_second": self.metrics.get_steps_per_second(),
        }
        
        if "wandb" in self.config.report_to:
            wandb.log(metrics, step=self.global_step)
            
    def _log_eval_metrics(self, eval_metrics: Dict[str, float]):
        """Log evaluation metrics"""
        if "wandb" in self.config.report_to:
            wandb.log(eval_metrics, step=self.global_step)
            
        # Generate sample text if tokenizer is available
        if self.tokenizer is not None:
            sample_text = generate_sample(
                model=self.model,
                tokenizer=self.tokenizer,
                prompt="The future of AI is",
                max_new_tokens=50,
                device=self.device,
            )
            
            if "wandb" in self.config.report_to:
                wandb.log({"sample_text": sample_text}, step=self.global_step)
                
    def _save_checkpoint(self, is_best: bool = False, is_final: bool = False):
        """Save model checkpoint"""
        if is_best:
            save_path = os.path.join(self.config.output_dir, "best_model.pt")
        elif is_final:
            save_path = os.path.join(self.config.output_dir, "final_model.pt")
        else:
            save_path = os.path.join(self.config.output_dir, f"checkpoint-{self.global_step}.pt")
            
        save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            step=self.global_step,
            loss=self.metrics.get_avg_loss(),
            save_path=save_path,
            config=self.config.to_dict(),
        )
        
        # Clean up old checkpoints
        if not is_best and not is_final:
            self._cleanup_checkpoints()
            
    def _cleanup_checkpoints(self):
        """Remove old checkpoints to save space"""
        if self.config.save_total_limit <= 0:
            return
            
        # Get all checkpoint files
        checkpoint_files = []
        for file in os.listdir(self.config.output_dir):
            if file.startswith("checkpoint-") and file.endswith(".pt"):
                step = int(file.split("-")[1].split(".")[0])
                checkpoint_files.append((step, file))
                
        # Sort by step and remove oldest
        checkpoint_files.sort(key=lambda x: x[0])
        while len(checkpoint_files) > self.config.save_total_limit:
            _, file_to_remove = checkpoint_files.pop(0)
            os.remove(os.path.join(self.config.output_dir, file_to_remove))
