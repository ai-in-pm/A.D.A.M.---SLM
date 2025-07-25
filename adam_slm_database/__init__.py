"""
ADAM SLM Database System

A sophisticated SQLite database system for managing ADAM SLM models, training runs,
datasets, experiments, and performance metrics.

Features:
- Model versioning and metadata storage
- Training run tracking and metrics
- Dataset management and statistics
- Experiment logging and comparison
- Performance benchmarking
- User and session management
- Advanced querying and analytics
"""

__version__ = "1.0.0"
__author__ = "AI in PM"

from .database import AdamSLMDatabase
from .models import (
    ModelRegistry, TrainingRun, Dataset, Experiment,
    PerformanceMetric, User, Session
)
from .manager import DatabaseManager
from .analytics import DatabaseAnalytics
from .migrations import DatabaseMigrations

__all__ = [
    "AdamSLMDatabase",
    "ModelRegistry",
    "TrainingRun", 
    "Dataset",
    "Experiment",
    "PerformanceMetric",
    "User",
    "Session",
    "DatabaseManager",
    "DatabaseAnalytics",
    "DatabaseMigrations",
]
