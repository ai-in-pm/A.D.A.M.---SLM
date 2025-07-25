"""
A.D.A.M. - Applied Decision Architecture Matrix - Small Language Model

A sophisticated small language model implementation with state-of-the-art features:
- Rotary Position Embeddings (RoPE)
- SwiGLU activation functions
- RMSNorm for improved stability
- Grouped Query Attention (GQA)
- KV-Cache optimization
- Mixed precision training
- Advanced training utilities

Author: AI in PM
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "AI in PM"
__email__ = "ai.in.pm@example.com"

from .models import AdamSLM, AdamSLMConfig
from .training import AdamTrainer, TrainingConfig
from .inference import AdamInference
from .tokenization import AdamTokenizer

# Database integration
try:
    from .database import (
        get_default_database, get_database_manager, get_analytics,
        get_file_manager, DatabaseConfig, initialize_database,
        register_model_checkpoint, start_training_session,
        log_training_metrics, complete_training_session,
        import_file, search_knowledge_base, get_dashboard_stats
    )

    from .database.training_integration import DatabaseTrainingLogger, DatabaseModelRegistry
    from .database.knowledge_base import KnowledgeBase, search_papers, get_knowledge_stats

    DATABASE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Database integration not available: {e}")
    DATABASE_AVAILABLE = False

__all__ = [
    # Core model components
    "AdamSLM",
    "AdamSLMConfig",
    "AdamTrainer",
    "TrainingConfig",
    "AdamInference",
    "AdamTokenizer",
]

# Add database components if available
if DATABASE_AVAILABLE:
    __all__.extend([
        # Database integration
        "get_default_database",
        "get_database_manager",
        "get_analytics",
        "get_file_manager",
        "DatabaseConfig",
        "initialize_database",

        # Training integration
        "DatabaseTrainingLogger",
        "DatabaseModelRegistry",
        "register_model_checkpoint",
        "start_training_session",
        "log_training_metrics",
        "complete_training_session",

        # Knowledge base
        "KnowledgeBase",
        "search_papers",
        "search_knowledge_base",
        "get_knowledge_stats",

        # File management
        "import_file",

        # Analytics
        "get_dashboard_stats",
    ])
