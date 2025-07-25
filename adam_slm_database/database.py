"""
Main database class for ADAM SLM
"""

import sqlite3
import json
import os
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from contextlib import contextmanager
import logging


class AdamSLMDatabase:
    """
    Sophisticated SQLite database for ADAM SLM
    
    Features:
    - Model versioning and metadata
    - Training run tracking
    - Dataset management
    - Experiment logging
    - Performance monitoring
    - User management
    - Session handling
    """
    
    def __init__(self, db_path: str = "adam_slm.db", auto_create: bool = True):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        
        if auto_create:
            self.initialize_database()
            
    def initialize_database(self):
        """Initialize database with schema"""
        schema_path = os.path.join(os.path.dirname(__file__), "schema.sql")
        
        with open(schema_path, 'r') as f:
            schema_sql = f.read()
            
        with self.get_connection() as conn:
            conn.executescript(schema_sql)
            conn.commit()
            
        self.logger.info(f"Database initialized: {self.db_path}")
        
    @contextmanager
    def get_connection(self):
        """Get database connection with proper error handling"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row  # Enable dict-like access
            conn.execute("PRAGMA foreign_keys = ON")  # Enable foreign keys
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            self.logger.error(f"Database error: {e}")
            raise
        finally:
            if conn:
                conn.close()
                
    def execute_query(self, query: str, params: tuple = ()) -> List[Dict]:
        """Execute a SELECT query and return results"""
        with self.get_connection() as conn:
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
            
    def execute_update(self, query: str, params: tuple = ()) -> int:
        """Execute an INSERT/UPDATE/DELETE query and return affected rows"""
        with self.get_connection() as conn:
            cursor = conn.execute(query, params)
            conn.commit()
            return cursor.rowcount
            
    def execute_insert(self, query: str, params: tuple = ()) -> int:
        """Execute an INSERT query and return the new row ID"""
        with self.get_connection() as conn:
            cursor = conn.execute(query, params)
            conn.commit()
            return cursor.lastrowid
            
    # ========================================================================
    # USER MANAGEMENT
    # ========================================================================
    
    def create_user(
        self,
        username: str,
        email: Optional[str] = None,
        full_name: Optional[str] = None,
        role: str = "user",
        preferences: Optional[Dict] = None
    ) -> int:
        """Create a new user"""
        api_key = secrets.token_urlsafe(32)
        preferences_json = json.dumps(preferences) if preferences else None
        
        query = """
        INSERT INTO users (username, email, full_name, role, preferences, api_key)
        VALUES (?, ?, ?, ?, ?, ?)
        """
        
        return self.execute_insert(
            query, (username, email, full_name, role, preferences_json, api_key)
        )
        
    def get_user(self, user_id: Optional[int] = None, username: Optional[str] = None) -> Optional[Dict]:
        """Get user by ID or username"""
        if user_id:
            query = "SELECT * FROM users WHERE id = ? AND is_active = 1"
            params = (user_id,)
        elif username:
            query = "SELECT * FROM users WHERE username = ? AND is_active = 1"
            params = (username,)
        else:
            raise ValueError("Must provide either user_id or username")
            
        results = self.execute_query(query, params)
        return results[0] if results else None
        
    def create_session(self, user_id: int, ip_address: str = None, user_agent: str = None) -> str:
        """Create a new user session"""
        session_token = secrets.token_urlsafe(64)
        expires_at = datetime.now() + timedelta(days=30)  # 30-day expiry
        
        query = """
        INSERT INTO sessions (user_id, session_token, expires_at, ip_address, user_agent)
        VALUES (?, ?, ?, ?, ?)
        """
        
        self.execute_insert(
            query, (user_id, session_token, expires_at, ip_address, user_agent)
        )
        
        return session_token
        
    # ========================================================================
    # MODEL MANAGEMENT
    # ========================================================================
    
    def register_model(
        self,
        model_name: str,
        version: str,
        model_type: str,
        architecture_config: Dict,
        parameter_count: int,
        model_size_mb: float,
        checkpoint_path: str,
        tokenizer_path: str = None,
        created_by: int = None,
        description: str = None,
        tags: List[str] = None,
        parent_model_id: int = None
    ) -> int:
        """Register a new model in the database"""
        
        config_json = json.dumps(architecture_config)
        tags_json = json.dumps(tags) if tags else None
        
        query = """
        INSERT INTO model_registry (
            model_name, version, model_type, architecture_config,
            parameter_count, model_size_mb, checkpoint_path, tokenizer_path,
            created_by, description, tags, parent_model_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        return self.execute_insert(query, (
            model_name, version, model_type, config_json,
            parameter_count, model_size_mb, checkpoint_path, tokenizer_path,
            created_by, description, tags_json, parent_model_id
        ))
        
    def get_model(self, model_id: int) -> Optional[Dict]:
        """Get model by ID"""
        query = "SELECT * FROM model_registry WHERE id = ? AND is_active = 1"
        results = self.execute_query(query, (model_id,))
        
        if results:
            model = results[0]
            # Parse JSON fields
            model['architecture_config'] = json.loads(model['architecture_config'])
            if model['tags']:
                model['tags'] = json.loads(model['tags'])
            return model
        return None
        
    def list_models(self, model_type: str = None, limit: int = 100) -> List[Dict]:
        """List models with optional filtering"""
        if model_type:
            query = """
            SELECT * FROM model_summary 
            WHERE model_type = ? 
            ORDER BY created_at DESC 
            LIMIT ?
            """
            params = (model_type, limit)
        else:
            query = "SELECT * FROM model_summary ORDER BY created_at DESC LIMIT ?"
            params = (limit,)
            
        return self.execute_query(query, params)
        
    def add_model_benchmark(
        self,
        model_id: int,
        benchmark_type: str,
        metric_value: float,
        dataset_name: str = None,
        metric_details: Dict = None,
        hardware_info: Dict = None,
        notes: str = None
    ) -> int:
        """Add benchmark results for a model"""
        
        metric_details_json = json.dumps(metric_details) if metric_details else None
        hardware_info_json = json.dumps(hardware_info) if hardware_info else None
        
        query = """
        INSERT INTO model_benchmarks (
            model_id, benchmark_type, dataset_name, metric_value,
            metric_details, hardware_info, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        
        return self.execute_insert(query, (
            model_id, benchmark_type, dataset_name, metric_value,
            metric_details_json, hardware_info_json, notes
        ))
        
    # ========================================================================
    # TRAINING MANAGEMENT
    # ========================================================================
    
    def create_training_run(
        self,
        run_name: str,
        training_config: Dict,
        model_id: int = None,
        base_model_id: int = None,
        dataset_id: int = None,
        started_by: int = None,
        total_steps: int = None,
        checkpoint_dir: str = None,
        wandb_run_id: str = None,
        notes: str = None
    ) -> int:
        """Create a new training run"""
        
        config_json = json.dumps(training_config)
        
        query = """
        INSERT INTO training_runs (
            run_name, model_id, base_model_id, training_config,
            dataset_id, started_by, total_steps, checkpoint_dir,
            wandb_run_id, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        return self.execute_insert(query, (
            run_name, model_id, base_model_id, config_json,
            dataset_id, started_by, total_steps, checkpoint_dir,
            wandb_run_id, notes
        ))
        
    def update_training_run(
        self,
        run_id: int,
        status: str = None,
        current_epoch: int = None,
        current_step: int = None,
        best_loss: float = None,
        best_metric: float = None,
        final_loss: float = None,
        total_tokens_processed: int = None,
        training_time_seconds: int = None,
        gpu_hours: float = None,
        error_message: str = None
    ) -> int:
        """Update training run progress"""
        
        # Build dynamic update query
        updates = []
        params = []
        
        if status is not None:
            updates.append("status = ?")
            params.append(status)
            if status in ['completed', 'failed', 'stopped']:
                updates.append("completed_at = CURRENT_TIMESTAMP")
                
        if current_epoch is not None:
            updates.append("current_epoch = ?")
            params.append(current_epoch)
            
        if current_step is not None:
            updates.append("current_step = ?")
            params.append(current_step)
            
        if best_loss is not None:
            updates.append("best_loss = ?")
            params.append(best_loss)
            
        if best_metric is not None:
            updates.append("best_metric = ?")
            params.append(best_metric)
            
        if final_loss is not None:
            updates.append("final_loss = ?")
            params.append(final_loss)
            
        if total_tokens_processed is not None:
            updates.append("total_tokens_processed = ?")
            params.append(total_tokens_processed)
            
        if training_time_seconds is not None:
            updates.append("training_time_seconds = ?")
            params.append(training_time_seconds)
            
        if gpu_hours is not None:
            updates.append("gpu_hours = ?")
            params.append(gpu_hours)
            
        if error_message is not None:
            updates.append("error_message = ?")
            params.append(error_message)
            
        if not updates:
            return 0

        params.append(run_id)
        
        query = f"UPDATE training_runs SET {', '.join(updates)} WHERE id = ?"
        
        return self.execute_update(query, tuple(params))
        
    def log_training_metric(
        self,
        training_run_id: int,
        step: int,
        metric_name: str,
        metric_value: float,
        epoch: int = None
    ) -> int:
        """Log a training metric"""
        
        query = """
        INSERT INTO training_metrics (training_run_id, step, epoch, metric_name, metric_value)
        VALUES (?, ?, ?, ?, ?)
        """
        
        return self.execute_insert(query, (
            training_run_id, step, epoch, metric_name, metric_value
        ))
        
    def get_training_run(self, run_id: int) -> Optional[Dict]:
        """Get training run details"""
        query = "SELECT * FROM training_summary WHERE id = ?"
        results = self.execute_query(query, (run_id,))
        return results[0] if results else None
        
    def list_training_runs(
        self,
        status: str = None,
        model_id: int = None,
        limit: int = 100
    ) -> List[Dict]:
        """List training runs with optional filtering"""
        
        conditions = []
        params = []
        
        if status:
            conditions.append("status = ?")
            params.append(status)
            
        if model_id:
            conditions.append("model_id = ?")
            params.append(model_id)
            
        where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""
        params.append(limit)
        
        query = f"""
        SELECT * FROM training_summary 
        {where_clause}
        ORDER BY started_at DESC 
        LIMIT ?
        """
        
        return self.execute_query(query, tuple(params))

    # ========================================================================
    # DATASET MANAGEMENT
    # ========================================================================

    def register_dataset(
        self,
        name: str,
        description: str = None,
        dataset_type: str = "text",
        source_path: str = None,
        processed_path: str = None,
        total_samples: int = None,
        total_tokens: int = None,
        avg_sequence_length: float = None,
        vocabulary_size: int = None,
        language: str = "en",
        license: str = None,
        created_by: int = None,
        preprocessing_config: Dict = None,
        statistics: Dict = None,
        is_public: bool = False,
        tags: List[str] = None
    ) -> int:
        """Register a new dataset"""

        preprocessing_json = json.dumps(preprocessing_config) if preprocessing_config else None
        statistics_json = json.dumps(statistics) if statistics else None
        tags_json = json.dumps(tags) if tags else None

        query = """
        INSERT INTO datasets (
            name, description, dataset_type, source_path, processed_path,
            total_samples, total_tokens, avg_sequence_length, vocabulary_size,
            language, license, created_by, preprocessing_config, statistics,
            is_public, tags
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        return self.execute_insert(query, (
            name, description, dataset_type, source_path, processed_path,
            total_samples, total_tokens, avg_sequence_length, vocabulary_size,
            language, license, created_by, preprocessing_json, statistics_json,
            is_public, tags_json
        ))

    def add_dataset_split(
        self,
        dataset_id: int,
        split_name: str,
        file_path: str,
        sample_count: int,
        token_count: int,
        split_ratio: float
    ) -> int:
        """Add a dataset split (train/validation/test)"""

        query = """
        INSERT INTO dataset_splits (
            dataset_id, split_name, file_path, sample_count, token_count, split_ratio
        ) VALUES (?, ?, ?, ?, ?, ?)
        """

        return self.execute_insert(query, (
            dataset_id, split_name, file_path, sample_count, token_count, split_ratio
        ))

    def get_dataset(self, dataset_id: int) -> Optional[Dict]:
        """Get dataset details"""
        query = "SELECT * FROM dataset_stats WHERE id = ?"
        results = self.execute_query(query, (dataset_id,))

        if results:
            dataset = results[0]
            # Get splits
            splits_query = "SELECT * FROM dataset_splits WHERE dataset_id = ?"
            splits = self.execute_query(splits_query, (dataset_id,))
            dataset['splits'] = splits
            return dataset
        return None

    def list_datasets(self, dataset_type: str = None, is_public: bool = None, limit: int = 100) -> List[Dict]:
        """List datasets with optional filtering"""

        conditions = []
        params = []

        if dataset_type:
            conditions.append("dataset_type = ?")
            params.append(dataset_type)

        if is_public is not None:
            conditions.append("is_public = ?")
            params.append(is_public)

        where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""
        params.append(limit)

        query = f"""
        SELECT * FROM dataset_stats
        {where_clause}
        ORDER BY created_at DESC
        LIMIT ?
        """

        return self.execute_query(query, tuple(params))

    # ========================================================================
    # EXPERIMENT MANAGEMENT
    # ========================================================================

    def create_experiment(
        self,
        name: str,
        description: str = None,
        objective: str = None,
        hypothesis: str = None,
        methodology: str = None,
        created_by: int = None,
        tags: List[str] = None
    ) -> int:
        """Create a new experiment"""

        tags_json = json.dumps(tags) if tags else None

        query = """
        INSERT INTO experiments (
            name, description, objective, hypothesis, methodology, created_by, tags
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """

        return self.execute_insert(query, (
            name, description, objective, hypothesis, methodology, created_by, tags_json
        ))

    def add_run_to_experiment(
        self,
        experiment_id: int,
        training_run_id: int,
        run_purpose: str = None
    ) -> int:
        """Add a training run to an experiment"""

        query = """
        INSERT INTO experiment_runs (experiment_id, training_run_id, run_purpose)
        VALUES (?, ?, ?)
        """

        return self.execute_insert(query, (experiment_id, training_run_id, run_purpose))

    def update_experiment(
        self,
        experiment_id: int,
        status: str = None,
        results_summary: str = None,
        conclusions: str = None
    ) -> int:
        """Update experiment results"""

        updates = []
        params = []

        if status is not None:
            updates.append("status = ?")
            params.append(status)

        if results_summary is not None:
            updates.append("results_summary = ?")
            params.append(results_summary)

        if conclusions is not None:
            updates.append("conclusions = ?")
            params.append(conclusions)

        if not updates:
            return 0

        updates.append("updated_at = CURRENT_TIMESTAMP")
        params.append(experiment_id)

        query = f"UPDATE experiments SET {', '.join(updates)} WHERE id = ?"

        return self.execute_update(query, tuple(params))

    def get_experiment(self, experiment_id: int) -> Optional[Dict]:
        """Get experiment details with associated runs"""
        query = "SELECT * FROM experiments WHERE id = ?"
        results = self.execute_query(query, (experiment_id,))

        if results:
            experiment = results[0]

            # Get associated training runs
            runs_query = """
            SELECT tr.*, er.run_purpose, er.added_at
            FROM experiment_runs er
            JOIN training_summary tr ON er.training_run_id = tr.id
            WHERE er.experiment_id = ?
            ORDER BY er.added_at
            """
            runs = self.execute_query(runs_query, (experiment_id,))
            experiment['training_runs'] = runs

            # Parse JSON fields
            if experiment['tags']:
                experiment['tags'] = json.loads(experiment['tags'])

            return experiment
        return None
