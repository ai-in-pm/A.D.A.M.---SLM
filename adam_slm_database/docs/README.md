# ADAM SLM Database System

🗄️ A sophisticated SQLite database system for managing ADAM SLM models, training runs, datasets, experiments, and performance metrics.

## ✨ Features

### 🏗️ Comprehensive Schema
- **15+ interconnected tables** with proper relationships
- **User management** with roles and sessions
- **Model versioning** and lineage tracking
- **Training run** monitoring and metrics
- **Dataset management** with splits and statistics
- **Experiment tracking** and comparison
- **Performance benchmarking** and analytics
- **System monitoring** and health checks

### 🚀 Advanced Capabilities
- **Migration system** for schema updates
- **Analytics engine** for insights and trends
- **High-level manager** for common operations
- **Performance optimization** with indexes and views
- **Data integrity** with foreign key constraints
- **JSON support** for flexible metadata storage

## 📊 Database Schema

### Core Tables

#### Users & Sessions
- `users` - User accounts with roles and preferences
- `sessions` - Active user sessions with tokens

#### Model Management
- `model_registry` - Model versions and metadata
- `model_benchmarks` - Performance benchmarks

#### Training Management
- `training_runs` - Training job tracking
- `training_metrics` - Detailed training metrics

#### Dataset Management
- `datasets` - Dataset catalog and statistics
- `dataset_splits` - Train/validation/test splits

#### Experiment Tracking
- `experiments` - Research experiments
- `experiment_runs` - Linking runs to experiments

#### Inference & Deployment
- `inference_sessions` - Model usage tracking
- `inference_requests` - Individual inference calls

#### System Monitoring
- `system_metrics` - Performance monitoring
- `operation_logs` - Database operation audit

### Views & Analytics
- `model_summary` - Model overview with statistics
- `training_summary` - Training run summaries
- `dataset_stats` - Dataset usage analytics

## 🚀 Quick Start

### Basic Usage

```python
from adam_slm_database import AdamSLMDatabase, DatabaseManager

# Create database
db = AdamSLMDatabase("my_adam_slm.db")

# High-level manager
manager = DatabaseManager("my_adam_slm.db")

# Register a model
model_id = manager.register_model_from_config(
    model_name="my-adam-slm",
    version="1.0.0",
    config_dict={"d_model": 768, "n_layers": 12},
    checkpoint_path="/path/to/model.pt"
)

# Start training run
run_id = manager.start_training_run(
    run_name="experiment_1",
    model_config={"d_model": 768},
    training_config={"learning_rate": 5e-4}
)

# Get dashboard stats
stats = manager.get_dashboard_stats()
print(f"Total models: {stats['models']['total_models']}")
```

### Analytics

```python
from adam_slm_database import DatabaseAnalytics

analytics = DatabaseAnalytics(db)

# Compare model performance
comparison = analytics.get_model_performance_comparison([1, 2, 3])

# Training trends
trends = analytics.get_training_trends(days=30)

# Dataset usage analysis
usage = analytics.get_dataset_usage_analysis()
```

### Migrations

```python
from adam_slm_database import DatabaseMigrations

migrations = DatabaseMigrations(db)

# Check current version
version = migrations.get_current_version()

# Apply all pending migrations
migrations.migrate_to_latest()

# Backup before migration
migrations.backup_database("backup.db")
```

## 📋 API Reference

### AdamSLMDatabase

Core database operations:

```python
# User management
user_id = db.create_user(username="alice", role="researcher")
user = db.get_user(user_id=user_id)
session_token = db.create_session(user_id)

# Model management
model_id = db.register_model(
    model_name="my-model",
    version="1.0.0",
    model_type="adam-slm-base",
    architecture_config={...},
    parameter_count=145_000_000,
    checkpoint_path="/path/to/model.pt"
)

# Training management
run_id = db.create_training_run(
    run_name="training_run_1",
    training_config={...}
)

db.log_training_metric(run_id, step=100, metric_name="loss", metric_value=1.5)
db.update_training_run(run_id, status="completed", final_loss=1.2)

# Dataset management
dataset_id = db.register_dataset(
    name="my_dataset",
    dataset_type="text",
    total_samples=100000
)
```

### DatabaseManager

High-level operations:

```python
manager = DatabaseManager("database.db")

# Model lifecycle
model_id = manager.register_model_from_config(...)
run_id = manager.start_training_run(...)
final_model_id = manager.complete_training_run(...)

# Dataset management
dataset_id = manager.register_dataset_from_path(
    name="my_dataset",
    source_path="/data/dataset.txt",
    analyze_content=True
)

# Experiment management
experiment_id, run_ids = manager.create_experiment_with_runs(
    name="hyperparameter_search",
    description="Testing different learning rates",
    run_configs=[...]
)

# Analytics
stats = manager.get_dashboard_stats()
```

### DatabaseAnalytics

Advanced analytics:

```python
analytics = DatabaseAnalytics(db)

# Model analysis
comparison = analytics.get_model_performance_comparison([1, 2, 3])
lineage = analytics.get_model_lineage(model_id)

# Training analysis
trends = analytics.get_training_trends(days=30)
hyperparams = analytics.get_hyperparameter_analysis()

# Dataset analysis
usage = analytics.get_dataset_usage_analysis()

# System analysis
performance = analytics.get_system_performance_report(hours=24)
```

## 🎯 Use Cases

### Research & Development
- Track model experiments and iterations
- Compare performance across different architectures
- Analyze training trends and optimization
- Manage dataset versions and preprocessing

### Production Deployment
- Monitor model performance in production
- Track inference usage and latency
- Manage model versions and rollbacks
- System health monitoring

### Team Collaboration
- Multi-user access with role-based permissions
- Shared experiment tracking
- Centralized model and dataset registry
- Audit trail for all operations

## 🔧 Advanced Features

### Migration System
- Automatic schema versioning
- Safe database upgrades
- Rollback capabilities
- Backup and restore

### Performance Optimization
- Strategic indexing for fast queries
- Materialized views for analytics
- Connection pooling support
- Query optimization

### Data Integrity
- Foreign key constraints
- Transaction support
- Input validation
- Error handling

## 📈 Monitoring & Analytics

### Dashboard Metrics
- Model count and distribution
- Training success rates
- Resource utilization
- Recent activity

### Performance Tracking
- Training convergence analysis
- Hyperparameter impact studies
- Dataset effectiveness comparison
- System resource monitoring

### Reporting
- Automated report generation
- Custom analytics queries
- Export capabilities
- Visualization support

## 🛠️ Development

### Running the Demo

```bash
cd adam_slm_database
python demo.py
```

This will create a demo database with sample data and showcase all features.

### Testing

```python
# Run comprehensive tests
python -m pytest tests/

# Test specific components
python -m pytest tests/test_database.py
python -m pytest tests/test_analytics.py
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built for the ADAM SLM project
- Inspired by MLOps best practices
- Designed for research and production use

---

**ADAM SLM Database** - Sophisticated data management for AI model development! 🗄️
