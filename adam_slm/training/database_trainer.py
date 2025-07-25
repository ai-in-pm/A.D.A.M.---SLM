"""
Database-Aware Trainer for A.D.A.M. SLM
Extends the base trainer with comprehensive database integration
"""

import os
import time
import torch
from typing import Dict, Any, Optional, List
from pathlib import Path

from .trainer import AdamTrainer
from .config import TrainingConfig
from ..database.training_integration import DatabaseTrainingLogger


class DatabaseAwareTrainer(AdamTrainer):
    """
    Enhanced A.D.A.M. SLM trainer with full database integration

    Features:
    - Automatic training run tracking
    - Real-time metrics logging
    - Model checkpoint registration
    - Experiment management
    - Knowledge base integration
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        config: TrainingConfig,
        run_name: str = None,
        experiment_name: str = None,
        dataset_name: str = None,
        notes: str = None,
        created_by: str = "system",
        **kwargs
    ):
        super().__init__(model, tokenizer, config, **kwargs)
        
        # Database integration
        self.run_name = run_name or f"adam_slm_training_{int(time.time())}"
        self.experiment_name = experiment_name
        self.dataset_name = dataset_name
        self.notes = notes
        self.created_by = created_by
        
        # Initialize database logger
        self.db_logger = None
        self._initialize_database_logger()
        
        # Training state
        self.run_id = None
        self.model_checkpoints = []
        
    def _initialize_database_logger(self):
        """Initialize database logger"""
        try:
            # Convert training config to dict
            model_config = {
                "model_type": "adam-slm",
                "d_model": getattr(self.model.config, 'd_model', 768),
                "n_layers": getattr(self.model.config, 'n_layers', 12),
                "n_heads": getattr(self.model.config, 'n_heads', 12),
                "vocab_size": getattr(self.model.config, 'vocab_size', 50257),
                "max_seq_len": getattr(self.model.config, 'max_seq_len', 2048),
            }
            
            training_config = {
                "learning_rate": self.config.learning_rate,
                "batch_size": self.config.batch_size,
                "max_steps": self.config.max_steps,
                "warmup_steps": self.config.warmup_steps,
                "weight_decay": self.config.weight_decay,
                "gradient_accumulation_steps": getattr(self.config, 'gradient_accumulation_steps', 1),
                "eval_steps": getattr(self.config, 'eval_steps', 500),
                "save_steps": getattr(self.config, 'save_steps', 1000),
            }
            
            self.db_logger = DatabaseTrainingLogger(
                run_name=self.run_name,
                model_config=model_config,
                training_config=training_config,
                dataset_name=self.dataset_name,
                experiment_name=self.experiment_name,
                started_by=self.created_by,
                notes=self.notes
            )
            
            print(f"🗄️ Database integration initialized for run: {self.run_name}")
            
        except Exception as e:
            print(f"⚠️ Database integration failed: {e}")
            print("Training will continue without database logging")
            self.db_logger = None
    
    def train(self, train_dataloader, eval_dataloader=None, **kwargs):
        """Enhanced training with database integration"""
        
        # Start database logging
        if self.db_logger:
            self.run_id = self.db_logger.start_training()
        
        # Call parent training method with database hooks
        try:
            result = super().train(train_dataloader, eval_dataloader, **kwargs)
            
            # Complete database logging
            if self.db_logger and hasattr(self, 'final_model_path'):
                final_metrics = {
                    'final_loss': getattr(self, 'best_loss', None),
                    'final_eval_loss': getattr(self, 'best_eval_loss', None)
                }
                
                self.db_logger.complete_training(
                    final_model_path=self.final_model_path,
                    final_metrics=final_metrics
                )
            
            return result
            
        except Exception as e:
            # Log training failure
            if self.db_logger:
                try:
                    self.db_logger.database.update_training_run(
                        run_id=self.run_id,
                        status='failed',
                        notes=f"Training failed: {str(e)}"
                    )
                except:
                    pass
            raise
    
    def training_step(self, batch, step: int) -> Dict[str, float]:
        """Enhanced training step with database logging"""
        
        # Perform training step
        metrics = super().training_step(batch, step)
        
        # Log to database
        if self.db_logger and step % 10 == 0:  # Log every 10 steps
            try:
                # Calculate tokens processed (approximate)
                batch_size = batch['input_ids'].size(0)
                seq_len = batch['input_ids'].size(1)
                tokens_processed = batch_size * seq_len
                
                self.db_logger.log_step(step, metrics, tokens_processed)
                
            except Exception as e:
                print(f"⚠️ Failed to log metrics to database: {e}")
        
        return metrics
    
    def save_checkpoint(
        self,
        step: int,
        loss: float,
        model_path: str = None,
        is_best: bool = False,
        is_final: bool = False,
        additional_metrics: Dict[str, float] = None
    ) -> str:
        """Enhanced checkpoint saving with database registration"""
        
        # Save checkpoint using parent method
        if model_path is None:
            model_path = os.path.join(
                self.config.output_dir,
                f"checkpoint-{step}.pt"
            )
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save model state
        checkpoint = {
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'config': self.config.__dict__,
            'model_config': self.model.config.__dict__ if hasattr(self.model, 'config') else {},
        }
        
        if hasattr(self, 'scheduler') and self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, model_path)
        
        # Register in database
        if self.db_logger:
            try:
                metrics = {'loss': loss}
                if additional_metrics:
                    metrics.update(additional_metrics)
                
                model_id = self.db_logger.save_checkpoint(
                    checkpoint_path=model_path,
                    step=step,
                    metrics=metrics,
                    is_best=is_best,
                    is_final=is_final
                )
                
                self.model_checkpoints.append({
                    'step': step,
                    'path': model_path,
                    'model_id': model_id,
                    'loss': loss,
                    'is_best': is_best,
                    'is_final': is_final
                })
                
                if is_final:
                    self.final_model_path = model_path
                
            except Exception as e:
                print(f"⚠️ Failed to register checkpoint in database: {e}")
        
        print(f"💾 Checkpoint saved: {model_path}")
        return model_path
    
    def evaluate(self, eval_dataloader, step: int = None) -> Dict[str, float]:
        """Enhanced evaluation with database logging"""
        
        # Perform evaluation
        eval_metrics = super().evaluate(eval_dataloader)
        
        # Log evaluation metrics to database
        if self.db_logger and step is not None:
            try:
                # Prefix eval metrics
                eval_metrics_prefixed = {
                    f"eval_{k}": v for k, v in eval_metrics.items()
                }
                
                self.db_logger.log_step(step, eval_metrics_prefixed)
                
            except Exception as e:
                print(f"⚠️ Failed to log eval metrics to database: {e}")
        
        return eval_metrics
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary"""
        
        summary = {
            'run_name': self.run_name,
            'run_id': self.run_id,
            'experiment_name': self.experiment_name,
            'dataset_name': self.dataset_name,
            'created_by': self.created_by,
            'checkpoints_saved': len(self.model_checkpoints),
            'database_logging': self.db_logger is not None
        }
        
        # Add database summary if available
        if self.db_logger:
            db_summary = self.db_logger.get_training_summary()
            summary.update(db_summary)
        
        return summary
    
    def search_related_runs(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Search for related training runs"""
        
        if not self.db_logger:
            return []
        
        try:
            # Search for runs with similar configuration
            database = self.db_logger.database
            
            # Find runs with similar model configuration
            similar_runs = database.execute_query("""
                SELECT tr.id, tr.run_name, tr.status, tr.best_loss, tr.started_at,
                       tr.model_config, tr.training_config
                FROM training_runs tr
                WHERE tr.id != ? AND tr.status = 'completed'
                ORDER BY tr.started_at DESC
                LIMIT ?
            """, (self.run_id or 0, limit))
            
            return similar_runs
            
        except Exception as e:
            print(f"⚠️ Failed to search related runs: {e}")
            return []
    
    def get_best_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Get the best checkpoint from this training run"""
        
        if not self.model_checkpoints:
            return None
        
        # Find checkpoint with lowest loss
        best_checkpoint = min(
            self.model_checkpoints,
            key=lambda x: x['loss']
        )
        
        return best_checkpoint
    
    def load_from_database(self, run_id: int, checkpoint_step: int = None):
        """Load model state from database-tracked checkpoint"""
        
        if not self.db_logger:
            raise ValueError("Database integration not available")
        
        try:
            database = self.db_logger.database
            
            # Get run information
            run_info = database.execute_query(
                "SELECT * FROM training_runs WHERE id = ?",
                (run_id,)
            )
            
            if not run_info:
                raise ValueError(f"Training run {run_id} not found")
            
            run = run_info[0]
            
            # Get checkpoints for this run
            checkpoints = database.execute_query("""
                SELECT mr.checkpoint_path, mr.id as model_id, mb.metric_value as loss
                FROM model_registry mr
                LEFT JOIN model_benchmarks mb ON mr.id = mb.model_id AND mb.benchmark_type = 'loss'
                WHERE mr.tags LIKE ?
                ORDER BY mr.created_at DESC
            """, (f"%{run['run_name']}%",))
            
            if not checkpoints:
                raise ValueError(f"No checkpoints found for run {run_id}")
            
            # Select checkpoint
            if checkpoint_step:
                # Find checkpoint closest to requested step
                checkpoint = min(
                    checkpoints,
                    key=lambda x: abs(int(x['checkpoint_path'].split('-')[-1].split('.')[0]) - checkpoint_step)
                )
            else:
                # Use latest checkpoint
                checkpoint = checkpoints[0]
            
            # Load checkpoint
            checkpoint_data = torch.load(checkpoint['checkpoint_path'])
            self.model.load_state_dict(checkpoint_data['model_state_dict'])
            
            if 'optimizer_state_dict' in checkpoint_data and hasattr(self, 'optimizer'):
                self.optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
            
            print(f"✅ Loaded checkpoint from run {run_id}: {checkpoint['checkpoint_path']}")
            
            return checkpoint
            
        except Exception as e:
            print(f"❌ Failed to load from database: {e}")
            raise
