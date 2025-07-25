"""
Training Integration with A.D.A.M. SLM Database
Provides database-aware training utilities
"""

import os
import time
import json
from typing import Dict, Any, Optional, List
from datetime import datetime

from . import (
    get_default_database, get_database_manager,
    start_training_session, log_training_metrics, 
    complete_training_session, register_model_checkpoint
)


class DatabaseTrainingLogger:
    """
    Training logger that integrates with A.D.A.M. SLM database
    Automatically tracks training runs, metrics, and model checkpoints
    """
    
    def __init__(
        self,
        run_name: str,
        model_config: dict,
        training_config: dict,
        dataset_name: str = None,
        experiment_name: str = None,
        started_by: str = "system",
        notes: str = None
    ):
        self.run_name = run_name
        self.model_config = model_config
        self.training_config = training_config
        self.dataset_name = dataset_name
        self.experiment_name = experiment_name
        self.started_by = started_by
        self.notes = notes
        
        # Database components
        self.database = get_default_database()
        self.manager = get_database_manager()
        
        # Training state
        self.run_id = None
        self.start_time = None
        self.step_count = 0
        self.best_loss = float('inf')
        self.total_tokens = 0
        
        # Metrics tracking
        self.metrics_history = []
        self.checkpoint_paths = []
        
    def start_training(self) -> int:
        """Start training session in database"""
        self.start_time = time.time()
        
        # Create training run
        self.run_id = start_training_session(
            run_name=self.run_name,
            model_config=self.model_config,
            training_config=self.training_config,
            dataset_name=self.dataset_name,
            started_by=self.started_by,
            notes=self.notes
        )
        
        print(f"🏃 Started training run: {self.run_name} (ID: {self.run_id})")
        
        # Link to experiment if specified
        if self.experiment_name:
            try:
                # Create or get experiment
                experiment_id = self._get_or_create_experiment()
                
                # Link run to experiment
                self.database.execute_insert("""
                    INSERT INTO experiment_runs (experiment_id, training_run_id)
                    VALUES (?, ?)
                """, (experiment_id, self.run_id))
                
                print(f"🧪 Linked to experiment: {self.experiment_name}")
                
            except Exception as e:
                print(f"⚠️ Failed to link experiment: {e}")
        
        return self.run_id
    
    def log_step(
        self,
        step: int,
        metrics: Dict[str, float],
        tokens_processed: int = None
    ):
        """Log training step metrics"""
        self.step_count = step
        
        # Track tokens
        if tokens_processed:
            self.total_tokens += tokens_processed
        
        # Track best loss
        if 'loss' in metrics:
            self.best_loss = min(self.best_loss, metrics['loss'])
        elif 'train_loss' in metrics:
            self.best_loss = min(self.best_loss, metrics['train_loss'])
        
        # Log to database
        log_training_metrics(self.run_id, step, metrics)
        
        # Store in history
        self.metrics_history.append({
            'step': step,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics.copy(),
            'tokens_processed': tokens_processed
        })
        
        # Update training run progress
        self._update_training_progress()
    
    def save_checkpoint(
        self,
        checkpoint_path: str,
        step: int,
        metrics: Dict[str, float] = None,
        is_best: bool = False,
        is_final: bool = False
    ) -> int:
        """Save checkpoint and register in database"""
        
        # Determine version
        version = f"step-{step}"
        if is_best:
            version += "-best"
        if is_final:
            version += "-final"
        
        # Register checkpoint as model
        try:
            model_id = register_model_checkpoint(
                model_name=f"{self.run_name}-checkpoint",
                version=version,
                checkpoint_path=checkpoint_path,
                config_dict=self.model_config,
                created_by=self.started_by,
                description=f"Checkpoint from training run {self.run_name} at step {step}",
                tags=['checkpoint', 'training', self.run_name, f'step-{step}']
            )
            
            self.checkpoint_paths.append({
                'step': step,
                'path': checkpoint_path,
                'model_id': model_id,
                'is_best': is_best,
                'is_final': is_final,
                'metrics': metrics or {}
            })
            
            print(f"💾 Checkpoint saved: {checkpoint_path} (Model ID: {model_id})")
            return model_id
            
        except Exception as e:
            print(f"⚠️ Failed to register checkpoint: {e}")
            return None
    
    def complete_training(
        self,
        final_model_path: str = None,
        final_metrics: Dict[str, float] = None
    ) -> int:
        """Complete training session"""
        
        if not self.run_id:
            raise ValueError("Training not started")
        
        # Calculate training time
        training_time = time.time() - self.start_time if self.start_time else None
        
        # Prepare final metrics
        final_metrics = final_metrics or {}
        if self.best_loss != float('inf'):
            final_metrics['best_loss'] = self.best_loss
        
        # Complete training run
        final_model_id = None
        if final_model_path:
            final_model_id = complete_training_session(
                run_id=self.run_id,
                final_model_path=final_model_path,
                final_metrics=final_metrics,
                training_time_seconds=training_time,
                total_tokens_processed=self.total_tokens
            )
        else:
            # Update training run without final model
            self.database.update_training_run(
                run_id=self.run_id,
                status='completed',
                current_step=self.step_count,
                best_loss=self.best_loss,
                training_time_seconds=training_time,
                total_tokens_processed=self.total_tokens
            )
        
        print(f"✅ Training completed: {self.run_name}")
        if final_model_id:
            print(f"🎯 Final model registered (ID: {final_model_id})")
        
        return final_model_id
    
    def _get_or_create_experiment(self) -> int:
        """Get or create experiment"""
        # Check if experiment exists
        existing = self.database.execute_query(
            "SELECT id FROM experiments WHERE name = ?",
            (self.experiment_name,)
        )
        
        if existing:
            return existing[0]['id']
        
        # Create new experiment
        experiment_id = self.database.execute_insert("""
            INSERT INTO experiments (
                name, description, objective, created_by, status
            ) VALUES (?, ?, ?, ?, ?)
        """, (
            self.experiment_name,
            f"Experiment containing run: {self.run_name}",
            "Training experiment",
            1,  # Admin user
            'running'
        ))
        
        return experiment_id
    
    def _update_training_progress(self):
        """Update training run progress in database"""
        if not self.run_id:
            return
        
        try:
            self.database.update_training_run(
                run_id=self.run_id,
                current_step=self.step_count,
                best_loss=self.best_loss if self.best_loss != float('inf') else None
            )
        except Exception as e:
            print(f"⚠️ Failed to update progress: {e}")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get training summary"""
        return {
            'run_id': self.run_id,
            'run_name': self.run_name,
            'steps_completed': self.step_count,
            'best_loss': self.best_loss if self.best_loss != float('inf') else None,
            'total_tokens': self.total_tokens,
            'checkpoints_saved': len(self.checkpoint_paths),
            'training_time': time.time() - self.start_time if self.start_time else None,
            'metrics_logged': len(self.metrics_history)
        }


class DatabaseModelRegistry:
    """
    Model registry for managing A.D.A.M. SLM models in database
    """
    
    def __init__(self):
        self.database = get_default_database()
        self.manager = get_database_manager()
    
    def register_model(
        self,
        model_name: str,
        version: str,
        model_path: str,
        config: dict,
        description: str = None,
        tags: List[str] = None,
        benchmarks: Dict[str, float] = None
    ) -> int:
        """Register a model in the database"""
        
        # Register model
        model_id = register_model_checkpoint(
            model_name=model_name,
            version=version,
            checkpoint_path=model_path,
            config_dict=config,
            description=description,
            tags=tags or []
        )
        
        # Add benchmarks if provided
        if benchmarks:
            for benchmark_type, value in benchmarks.items():
                self.database.add_model_benchmark(
                    model_id=model_id,
                    benchmark_type=benchmark_type,
                    metric_value=value,
                    dataset_name="evaluation_set"
                )
        
        return model_id
    
    def get_model_info(self, model_id: int) -> Dict[str, Any]:
        """Get detailed model information"""
        model_info = self.database.execute_query(
            "SELECT * FROM model_registry WHERE id = ?",
            (model_id,)
        )
        
        if not model_info:
            return None
        
        model = model_info[0]
        
        # Get benchmarks
        benchmarks = self.database.execute_query(
            "SELECT * FROM model_benchmarks WHERE model_id = ?",
            (model_id,)
        )
        
        # Parse JSON fields
        if model['architecture_config']:
            model['architecture_config'] = json.loads(model['architecture_config'])
        if model['tags']:
            model['tags'] = json.loads(model['tags'])
        
        model['benchmarks'] = benchmarks
        return model
    
    def list_models(
        self,
        model_type: str = None,
        tags: List[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """List models with filtering"""
        
        conditions = []
        params = []
        
        if model_type:
            conditions.append("model_type = ?")
            params.append(model_type)
        
        where_clause = ""
        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)
        
        query = f"""
            SELECT id, model_name, version, model_type, parameter_count,
                   created_at, description, tags
            FROM model_registry
            {where_clause}
            ORDER BY created_at DESC
            LIMIT ?
        """
        params.append(limit)
        
        models = self.database.execute_query(query, tuple(params))
        
        # Filter by tags if specified
        if tags:
            filtered_models = []
            for model in models:
                try:
                    model_tags = json.loads(model.get('tags', '[]'))
                    if any(tag in model_tags for tag in tags):
                        filtered_models.append(model)
                except:
                    continue
            models = filtered_models
        
        return models
    
    def get_best_model(
        self,
        benchmark_type: str = 'loss',
        model_type: str = None
    ) -> Optional[Dict[str, Any]]:
        """Get best performing model by benchmark"""
        
        conditions = ["mb.benchmark_type = ?"]
        params = [benchmark_type]
        
        if model_type:
            conditions.append("mr.model_type = ?")
            params.append(model_type)
        
        where_clause = "WHERE " + " AND ".join(conditions)
        
        query = f"""
            SELECT mr.*, mb.metric_value
            FROM model_registry mr
            JOIN model_benchmarks mb ON mr.id = mb.model_id
            {where_clause}
            ORDER BY mb.metric_value ASC
            LIMIT 1
        """
        
        result = self.database.execute_query(query, tuple(params))
        return result[0] if result else None
