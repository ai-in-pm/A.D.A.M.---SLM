"""
Database manager for ADAM SLM with high-level operations
"""

import os
import json
import hashlib
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import logging

from database import AdamSLMDatabase


class DatabaseManager:
    """
    High-level database manager for ADAM SLM
    
    Provides convenient methods for common operations and integrates
    with the ADAM SLM training and inference systems.
    """
    
    def __init__(self, db_path: str = "adam_slm.db"):
        self.db = AdamSLMDatabase(db_path)
        self.logger = logging.getLogger(__name__)
        
        # Create default admin user if database is empty
        self._ensure_default_user()
        
    def _ensure_default_user(self):
        """Create default admin user if no users exist"""
        users = self.db.execute_query("SELECT COUNT(*) as count FROM users")
        if users[0]['count'] == 0:
            self.db.create_user(
                username="admin",
                email="admin@adamslm.local",
                full_name="ADAM SLM Administrator",
                role="admin",
                preferences={"theme": "dark", "notifications": True}
            )
            self.logger.info("Created default admin user")
            
    # ========================================================================
    # MODEL LIFECYCLE MANAGEMENT
    # ========================================================================
    
    def register_model_from_config(
        self,
        model_name: str,
        version: str,
        config_dict: Dict,
        checkpoint_path: str,
        tokenizer_path: str = None,
        created_by_username: str = "admin",
        description: str = None,
        tags: List[str] = None
    ) -> int:
        """Register a model from ADAM SLM config"""
        
        # Get user ID
        user = self.db.get_user(username=created_by_username)
        user_id = user['id'] if user else None
        
        # Calculate model size
        model_size_mb = 0
        if os.path.exists(checkpoint_path):
            model_size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
            
        # Extract model type and parameters from config
        model_type = f"adam-slm-{config_dict.get('model_size', 'custom')}"
        parameter_count = self._estimate_parameters(config_dict)
        
        return self.db.register_model(
            model_name=model_name,
            version=version,
            model_type=model_type,
            architecture_config=config_dict,
            parameter_count=parameter_count,
            model_size_mb=model_size_mb,
            checkpoint_path=checkpoint_path,
            tokenizer_path=tokenizer_path,
            created_by=user_id,
            description=description,
            tags=tags
        )
        
    def _estimate_parameters(self, config: Dict) -> int:
        """Estimate parameter count from model config"""
        try:
            d_model = config.get('d_model', 768)
            n_layers = config.get('n_layers', 12)
            vocab_size = config.get('vocab_size', 50257)
            d_ff = config.get('d_ff', d_model * 4)
            
            # Rough estimation
            embedding_params = vocab_size * d_model
            attention_params = n_layers * (4 * d_model * d_model)  # Q, K, V, O projections
            ffn_params = n_layers * (2 * d_model * d_ff)  # Up and down projections
            norm_params = n_layers * 2 * d_model  # Layer norms
            
            total = embedding_params + attention_params + ffn_params + norm_params
            return int(total)
            
        except Exception:
            return 0
            
    def start_training_run(
        self,
        run_name: str,
        model_config: Dict,
        training_config: Dict,
        dataset_name: str = None,
        base_model_name: str = None,
        started_by_username: str = "admin",
        notes: str = None
    ) -> int:
        """Start a new training run"""
        
        # Get user ID
        user = self.db.get_user(username=started_by_username)
        user_id = user['id'] if user else None
        
        # Get dataset ID if provided
        dataset_id = None
        if dataset_name:
            datasets = self.db.execute_query(
                "SELECT id FROM datasets WHERE name = ?", (dataset_name,)
            )
            if datasets:
                dataset_id = datasets[0]['id']
                
        # Get base model ID if provided
        base_model_id = None
        if base_model_name:
            models = self.db.execute_query(
                "SELECT id FROM model_registry WHERE model_name = ? ORDER BY created_at DESC LIMIT 1",
                (base_model_name,)
            )
            if models:
                base_model_id = models[0]['id']
                
        # Create training run
        return self.db.create_training_run(
            run_name=run_name,
            training_config=training_config,
            base_model_id=base_model_id,
            dataset_id=dataset_id,
            started_by=user_id,
            total_steps=training_config.get('max_steps'),
            notes=notes
        )
        
    def complete_training_run(
        self,
        run_id: int,
        final_model_path: str,
        final_loss: float,
        training_time_seconds: int,
        total_tokens_processed: int = None,
        gpu_hours: float = None
    ) -> Optional[int]:
        """Complete a training run and register the final model"""
        
        # Update training run
        self.db.update_training_run(
            run_id=run_id,
            status="completed",
            final_loss=final_loss,
            training_time_seconds=training_time_seconds,
            total_tokens_processed=total_tokens_processed,
            gpu_hours=gpu_hours
        )
        
        # Get training run details
        run = self.db.get_training_run(run_id)
        if not run:
            return None
            
        # Register the trained model
        model_name = f"{run['run_name']}_model"
        version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Get training config from the actual training_runs table
        training_run_details = self.db.execute_query(
            "SELECT training_config FROM training_runs WHERE id = ?", (run_id,)
        )
        training_config = {}
        if training_run_details and training_run_details[0]['training_config']:
            training_config = json.loads(training_run_details[0]['training_config'])
        
        # Create model config (you might want to load this from the actual model)
        model_config = {
            "trained_from_run": run_id,
            "final_loss": final_loss,
            "training_time": training_time_seconds,
            "total_tokens": total_tokens_processed
        }
        
        return self.register_model_from_config(
            model_name=model_name,
            version=version,
            config_dict=model_config,
            checkpoint_path=final_model_path,
            created_by_username=run.get('started_by_username', 'admin'),
            description=f"Model trained from run: {run['run_name']}",
            tags=["trained", "auto-generated"]
        )
        
    # ========================================================================
    # DATASET MANAGEMENT
    # ========================================================================
    
    def register_dataset_from_path(
        self,
        name: str,
        source_path: str,
        dataset_type: str = "text",
        description: str = None,
        created_by_username: str = "admin",
        analyze_content: bool = True
    ) -> int:
        """Register a dataset from file path with automatic analysis"""
        
        # Get user ID
        user = self.db.get_user(username=created_by_username)
        user_id = user['id'] if user else None
        
        # Analyze dataset if requested
        statistics = {}
        total_samples = None
        total_tokens = None
        avg_sequence_length = None
        
        if analyze_content and os.path.exists(source_path):
            try:
                statistics = self._analyze_dataset(source_path, dataset_type)
                total_samples = statistics.get('total_samples')
                total_tokens = statistics.get('total_tokens')
                avg_sequence_length = statistics.get('avg_sequence_length')
            except Exception as e:
                self.logger.warning(f"Could not analyze dataset: {e}")
                
        return self.db.register_dataset(
            name=name,
            description=description,
            dataset_type=dataset_type,
            source_path=source_path,
            total_samples=total_samples,
            total_tokens=total_tokens,
            avg_sequence_length=avg_sequence_length,
            created_by=user_id,
            statistics=statistics
        )
        
    def _analyze_dataset(self, file_path: str, dataset_type: str) -> Dict:
        """Analyze dataset content and return statistics"""
        statistics = {
            'file_size_mb': os.path.getsize(file_path) / (1024 * 1024),
            'analysis_date': datetime.now().isoformat()
        }
        
        if dataset_type == "text":
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Basic text statistics
            lines = content.split('\n')
            words = content.split()
            
            statistics.update({
                'total_lines': len(lines),
                'total_words': len(words),
                'total_characters': len(content),
                'avg_line_length': sum(len(line) for line in lines) / len(lines) if lines else 0,
                'avg_word_length': sum(len(word) for word in words) / len(words) if words else 0
            })
            
            # Estimate tokens (rough approximation)
            estimated_tokens = len(words) * 1.3  # Rough BPE estimation
            statistics.update({
                'estimated_tokens': int(estimated_tokens),
                'total_samples': len(lines),
                'total_tokens': int(estimated_tokens),
                'avg_sequence_length': estimated_tokens / len(lines) if lines else 0
            })
            
        return statistics
        
    # ========================================================================
    # EXPERIMENT MANAGEMENT
    # ========================================================================
    
    def create_experiment_with_runs(
        self,
        name: str,
        description: str,
        objective: str,
        hypothesis: str,
        run_configs: List[Dict],
        created_by_username: str = "admin"
    ) -> Tuple[int, List[int]]:
        """Create an experiment and associated training runs"""
        
        # Get user ID
        user = self.db.get_user(username=created_by_username)
        user_id = user['id'] if user else None
        
        # Create experiment
        experiment_id = self.db.create_experiment(
            name=name,
            description=description,
            objective=objective,
            hypothesis=hypothesis,
            created_by=user_id
        )
        
        # Create training runs
        run_ids = []
        for i, run_config in enumerate(run_configs):
            run_name = f"{name}_run_{i+1}"
            
            run_id = self.start_training_run(
                run_name=run_name,
                model_config=run_config.get('model_config', {}),
                training_config=run_config.get('training_config', {}),
                dataset_name=run_config.get('dataset_name'),
                base_model_name=run_config.get('base_model_name'),
                started_by_username=created_by_username,
                notes=f"Part of experiment: {name}"
            )
            
            # Link run to experiment
            self.db.add_run_to_experiment(
                experiment_id=experiment_id,
                training_run_id=run_id,
                run_purpose=run_config.get('purpose', f"Run {i+1}")
            )
            
            run_ids.append(run_id)
            
        return experiment_id, run_ids
        
    # ========================================================================
    # ANALYTICS AND REPORTING
    # ========================================================================
    
    def get_dashboard_stats(self) -> Dict:
        """Get dashboard statistics"""
        
        stats = {}
        
        # Model statistics
        model_stats = self.db.execute_query("""
            SELECT 
                COUNT(*) as total_models,
                COUNT(DISTINCT model_type) as unique_types,
                SUM(parameter_count) as total_parameters,
                AVG(parameter_count) as avg_parameters
            FROM model_registry 
            WHERE is_active = 1
        """)[0]
        
        # Training statistics
        training_stats = self.db.execute_query("""
            SELECT 
                COUNT(*) as total_runs,
                COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed_runs,
                COUNT(CASE WHEN status = 'running' THEN 1 END) as active_runs,
                AVG(CASE WHEN status = 'completed' THEN training_time_seconds END) as avg_training_time
            FROM training_runs
        """)[0]
        
        # Dataset statistics
        dataset_stats = self.db.execute_query("""
            SELECT 
                COUNT(*) as total_datasets,
                SUM(total_samples) as total_samples,
                SUM(total_tokens) as total_tokens,
                AVG(avg_sequence_length) as avg_sequence_length
            FROM datasets
        """)[0]
        
        # Recent activity
        recent_models = self.db.execute_query("""
            SELECT model_name, version, created_at 
            FROM model_registry 
            WHERE is_active = 1 
            ORDER BY created_at DESC 
            LIMIT 5
        """)
        
        recent_runs = self.db.execute_query("""
            SELECT run_name, status, started_at 
            FROM training_runs 
            ORDER BY started_at DESC 
            LIMIT 5
        """)
        
        return {
            'models': model_stats,
            'training': training_stats,
            'datasets': dataset_stats,
            'recent_models': recent_models,
            'recent_runs': recent_runs,
            'generated_at': datetime.now().isoformat()
        }
