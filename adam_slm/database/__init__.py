"""
A.D.A.M. SLM Database Integration
Sophisticated database system for AI model lifecycle management
"""

import os
import sys

# Add the database system to the path
_database_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "adam_slm_database")
if _database_path not in sys.path:
    sys.path.insert(0, _database_path)

# Import core database components
from database import AdamSLMDatabase
from manager import DatabaseManager
from analytics import DatabaseAnalytics
from file_manager import FileManager
from file_converter import FileConverter
from migrations import DatabaseMigrations

# Import models for type hints
from models import *

__all__ = [
    "AdamSLMDatabase",
    "DatabaseManager", 
    "DatabaseAnalytics",
    "FileManager",
    "FileConverter",
    "DatabaseMigrations",
    "get_default_database",
    "initialize_database",
    "DatabaseConfig",
]

# Default database configuration
class DatabaseConfig:
    """Configuration for ADAM SLM database"""
    
    def __init__(
        self,
        database_path: str = None,
        auto_initialize: bool = True,
        enable_file_management: bool = True,
        enable_analytics: bool = True,
        storage_root: str = None
    ):
        # Default to sophisticated database in the database system directory
        if database_path is None:
            database_path = os.path.join(_database_path, "databases", "adamslm_sophisticated.sqlite")
            
        self.database_path = database_path
        self.auto_initialize = auto_initialize
        self.enable_file_management = enable_file_management
        self.enable_analytics = enable_analytics
        
        # Default storage root
        if storage_root is None:
            storage_root = os.path.join(_database_path, "file_storage")
        self.storage_root = storage_root

# Global database instance
_default_database = None
_default_config = None

def get_default_database() -> AdamSLMDatabase:
    """Get the default ADAM SLM database instance"""
    global _default_database, _default_config
    
    if _default_database is None:
        if _default_config is None:
            _default_config = DatabaseConfig()
        _default_database = initialize_database(_default_config)
    
    return _default_database

def initialize_database(config: DatabaseConfig = None) -> AdamSLMDatabase:
    """Initialize ADAM SLM database with configuration"""
    global _default_database, _default_config
    
    if config is None:
        config = DatabaseConfig()
    
    _default_config = config
    
    # Create database instance
    database = AdamSLMDatabase(config.database_path)
    
    # Initialize schema if needed
    if config.auto_initialize:
        try:
            # Check if database is properly initialized
            tables = database.execute_query("SELECT name FROM sqlite_master WHERE type='table'")
            if len(tables) < 10:  # If less than 10 tables, probably needs initialization
                print("🔄 Initializing database schema...")
                _initialize_schema(database)
                print("✅ Database schema initialized")
        except Exception as e:
            print(f"⚠️ Database initialization warning: {e}")
    
    _default_database = database
    return database

def _initialize_schema(database: AdamSLMDatabase):
    """Initialize database schema from SQL file"""
    schema_path = os.path.join(_database_path, "schema.sql")
    
    if os.path.exists(schema_path):
        with open(schema_path, 'r') as f:
            schema_sql = f.read()
        
        # Execute schema
        with database.get_connection() as conn:
            conn.executescript(schema_sql)
            conn.commit()
    else:
        raise FileNotFoundError(f"Schema file not found: {schema_path}")

# Convenience functions for common operations
def get_database_manager() -> DatabaseManager:
    """Get database manager instance"""
    config = _default_config or DatabaseConfig()
    return DatabaseManager(config.database_path)

def get_analytics() -> DatabaseAnalytics:
    """Get database analytics instance"""
    return DatabaseAnalytics(get_default_database())

def get_file_manager() -> FileManager:
    """Get file manager instance"""
    config = _default_config or DatabaseConfig()
    return FileManager(get_default_database(), config.storage_root)

def get_file_converter() -> FileConverter:
    """Get file converter instance"""
    return FileConverter()

# Database integration utilities
def register_model_checkpoint(
    model_name: str,
    version: str,
    checkpoint_path: str,
    config_dict: dict,
    created_by: str = "system",
    description: str = None,
    tags: list = None
) -> int:
    """Register a model checkpoint in the database"""
    manager = get_database_manager()
    
    return manager.register_model_from_config(
        model_name=model_name,
        version=version,
        config_dict=config_dict,
        checkpoint_path=checkpoint_path,
        created_by_username=created_by,
        description=description,
        tags=tags or []
    )

def start_training_session(
    run_name: str,
    model_config: dict,
    training_config: dict,
    dataset_name: str = None,
    started_by: str = "system",
    notes: str = None
) -> int:
    """Start a training session and return run ID"""
    manager = get_database_manager()
    
    return manager.start_training_run(
        run_name=run_name,
        model_config=model_config,
        training_config=training_config,
        dataset_name=dataset_name,
        started_by_username=started_by,
        notes=notes
    )

def log_training_metrics(
    run_id: int,
    step: int,
    metrics: dict
):
    """Log training metrics for a run"""
    database = get_default_database()
    
    for metric_name, metric_value in metrics.items():
        database.log_training_metric(run_id, step, metric_name, metric_value)

def complete_training_session(
    run_id: int,
    final_model_path: str,
    final_metrics: dict,
    training_time_seconds: float = None,
    total_tokens_processed: int = None
) -> int:
    """Complete a training session and register final model"""
    manager = get_database_manager()
    
    return manager.complete_training_run(
        run_id=run_id,
        final_model_path=final_model_path,
        final_loss=final_metrics.get('final_loss'),
        training_time_seconds=training_time_seconds,
        total_tokens_processed=total_tokens_processed
    )

def import_file(
    file_path: str,
    file_type: str = None,
    description: str = None,
    tags: list = None,
    created_by: str = "system"
) -> int:
    """Import a file into the database"""
    file_manager = get_file_manager()
    
    return file_manager.register_file(
        file_path=file_path,
        file_type=file_type,
        description=description,
        tags=tags or [],
        created_by=1,  # Default to admin user
        copy_to_storage=True,
        process_immediately=True
    )

def search_knowledge_base(query: str, limit: int = 10) -> list:
    """Search the knowledge base for relevant content"""
    database = get_default_database()
    
    search_query = """
        SELECT fc.file_id, fr.filename, fr.description,
               SUBSTR(fc.extracted_text, 1, 500) as preview
        FROM file_content fc
        JOIN file_registry fr ON fc.file_id = fr.id
        WHERE fc.extracted_text LIKE ? 
        ORDER BY fc.word_count DESC
        LIMIT ?
    """
    
    return database.execute_query(search_query, (f"%{query}%", limit))

def get_dashboard_stats() -> dict:
    """Get dashboard statistics"""
    manager = get_database_manager()
    return manager.get_dashboard_stats()

# Version info
__version__ = "1.0.0"
