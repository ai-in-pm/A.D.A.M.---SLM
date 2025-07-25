"""
Training utilities for A.D.A.M. SLM
"""

from .trainer import AdamTrainer
from .config import TrainingConfig, get_training_config, TRAINING_CONFIGS
from .data import AdamDataLoader, create_dataloader
from .utils import get_lr_scheduler, compute_loss, evaluate_model

# Database-aware training components
try:
    from .database_trainer import DatabaseAwareTrainer
    DATABASE_TRAINING_AVAILABLE = True
except ImportError:
    DATABASE_TRAINING_AVAILABLE = False

__all__ = [
    "AdamTrainer",
    "TrainingConfig",
    "get_training_config",
    "TRAINING_CONFIGS",
    "AdamDataLoader",
    "create_dataloader",
    "get_lr_scheduler",
    "compute_loss",
    "evaluate_model",
]

if DATABASE_TRAINING_AVAILABLE:
    __all__.append("DatabaseAwareTrainer")
