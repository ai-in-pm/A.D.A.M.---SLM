"""
Data models for ADAM SLM database
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional
import json


@dataclass
class User:
    """User data model"""
    id: Optional[int] = None
    username: str = ""
    email: Optional[str] = None
    full_name: Optional[str] = None
    role: str = "user"
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    is_active: bool = True
    preferences: Optional[Dict] = None
    api_key: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'full_name': self.full_name,
            'role': self.role,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'is_active': self.is_active,
            'preferences': json.dumps(self.preferences) if self.preferences else None,
            'api_key': self.api_key
        }


@dataclass
class Session:
    """Session data model"""
    id: Optional[int] = None
    user_id: int = 0
    session_token: str = ""
    created_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    is_active: bool = True


@dataclass
class ModelRegistry:
    """Model registry data model"""
    id: Optional[int] = None
    model_name: str = ""
    version: str = ""
    model_type: str = ""
    architecture_config: Dict = field(default_factory=dict)
    parameter_count: Optional[int] = None
    model_size_mb: Optional[float] = None
    checkpoint_path: str = ""
    tokenizer_path: Optional[str] = None
    created_by: Optional[int] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    is_active: bool = True
    parent_model_id: Optional[int] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'model_name': self.model_name,
            'version': self.version,
            'model_type': self.model_type,
            'architecture_config': json.dumps(self.architecture_config),
            'parameter_count': self.parameter_count,
            'model_size_mb': self.model_size_mb,
            'checkpoint_path': self.checkpoint_path,
            'tokenizer_path': self.tokenizer_path,
            'created_by': self.created_by,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'description': self.description,
            'tags': json.dumps(self.tags),
            'is_active': self.is_active,
            'parent_model_id': self.parent_model_id
        }


@dataclass
class TrainingRun:
    """Training run data model"""
    id: Optional[int] = None
    run_name: str = ""
    model_id: Optional[int] = None
    base_model_id: Optional[int] = None
    training_config: Dict = field(default_factory=dict)
    dataset_id: Optional[int] = None
    started_by: Optional[int] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: str = "running"
    current_epoch: int = 0
    current_step: int = 0
    total_steps: Optional[int] = None
    best_loss: Optional[float] = None
    best_metric: Optional[float] = None
    final_loss: Optional[float] = None
    total_tokens_processed: Optional[int] = None
    training_time_seconds: Optional[int] = None
    gpu_hours: Optional[float] = None
    error_message: Optional[str] = None
    logs_path: Optional[str] = None
    checkpoint_dir: Optional[str] = None
    wandb_run_id: Optional[str] = None
    notes: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'run_name': self.run_name,
            'model_id': self.model_id,
            'base_model_id': self.base_model_id,
            'training_config': json.dumps(self.training_config),
            'dataset_id': self.dataset_id,
            'started_by': self.started_by,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'status': self.status,
            'current_epoch': self.current_epoch,
            'current_step': self.current_step,
            'total_steps': self.total_steps,
            'best_loss': self.best_loss,
            'best_metric': self.best_metric,
            'final_loss': self.final_loss,
            'total_tokens_processed': self.total_tokens_processed,
            'training_time_seconds': self.training_time_seconds,
            'gpu_hours': self.gpu_hours,
            'error_message': self.error_message,
            'logs_path': self.logs_path,
            'checkpoint_dir': self.checkpoint_dir,
            'wandb_run_id': self.wandb_run_id,
            'notes': self.notes
        }


@dataclass
class Dataset:
    """Dataset data model"""
    id: Optional[int] = None
    name: str = ""
    description: Optional[str] = None
    dataset_type: str = "text"
    source_path: Optional[str] = None
    processed_path: Optional[str] = None
    total_samples: Optional[int] = None
    total_tokens: Optional[int] = None
    avg_sequence_length: Optional[float] = None
    vocabulary_size: Optional[int] = None
    language: str = "en"
    license: Optional[str] = None
    created_by: Optional[int] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    preprocessing_config: Optional[Dict] = None
    statistics: Optional[Dict] = None
    is_public: bool = False
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'dataset_type': self.dataset_type,
            'source_path': self.source_path,
            'processed_path': self.processed_path,
            'total_samples': self.total_samples,
            'total_tokens': self.total_tokens,
            'avg_sequence_length': self.avg_sequence_length,
            'vocabulary_size': self.vocabulary_size,
            'language': self.language,
            'license': self.license,
            'created_by': self.created_by,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'preprocessing_config': json.dumps(self.preprocessing_config) if self.preprocessing_config else None,
            'statistics': json.dumps(self.statistics) if self.statistics else None,
            'is_public': self.is_public,
            'tags': json.dumps(self.tags)
        }


@dataclass
class Experiment:
    """Experiment data model"""
    id: Optional[int] = None
    name: str = ""
    description: Optional[str] = None
    objective: Optional[str] = None
    hypothesis: Optional[str] = None
    methodology: Optional[str] = None
    created_by: Optional[int] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    status: str = "active"
    results_summary: Optional[str] = None
    conclusions: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'objective': self.objective,
            'hypothesis': self.hypothesis,
            'methodology': self.methodology,
            'created_by': self.created_by,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'status': self.status,
            'results_summary': self.results_summary,
            'conclusions': self.conclusions,
            'tags': json.dumps(self.tags)
        }


@dataclass
class PerformanceMetric:
    """Performance metric data model"""
    id: Optional[int] = None
    model_id: int = 0
    benchmark_type: str = ""
    dataset_name: Optional[str] = None
    metric_value: float = 0.0
    metric_details: Optional[Dict] = None
    benchmark_date: Optional[datetime] = None
    hardware_info: Optional[Dict] = None
    notes: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'model_id': self.model_id,
            'benchmark_type': self.benchmark_type,
            'dataset_name': self.dataset_name,
            'metric_value': self.metric_value,
            'metric_details': json.dumps(self.metric_details) if self.metric_details else None,
            'benchmark_date': self.benchmark_date.isoformat() if self.benchmark_date else None,
            'hardware_info': json.dumps(self.hardware_info) if self.hardware_info else None,
            'notes': self.notes
        }


@dataclass
class InferenceSession:
    """Inference session data model"""
    id: Optional[int] = None
    model_id: int = 0
    user_id: Optional[int] = None
    session_name: Optional[str] = None
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    total_requests: int = 0
    total_tokens_generated: int = 0
    avg_response_time_ms: Optional[float] = None
    hardware_info: Optional[Dict] = None
    configuration: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'model_id': self.model_id,
            'user_id': self.user_id,
            'session_name': self.session_name,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'ended_at': self.ended_at.isoformat() if self.ended_at else None,
            'total_requests': self.total_requests,
            'total_tokens_generated': self.total_tokens_generated,
            'avg_response_time_ms': self.avg_response_time_ms,
            'hardware_info': json.dumps(self.hardware_info) if self.hardware_info else None,
            'configuration': json.dumps(self.configuration) if self.configuration else None
        }
