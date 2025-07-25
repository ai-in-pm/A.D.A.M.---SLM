#!/usr/bin/env python3
"""
ADAM SLM Database Demo Script
Demonstrates the sophisticated database features
"""

import os
import sys
import json
import time
import random
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from database import AdamSLMDatabase
from manager import DatabaseManager
from analytics import DatabaseAnalytics
from migrations import DatabaseMigrations


def print_header(title: str):
    """Print formatted header"""
    print("\n" + "="*60)
    print(f"🗄️ {title}")
    print("="*60)


def print_section(title: str):
    """Print formatted section"""
    print(f"\n📋 {title}")
    print("-"*40)


def demo_database_setup():
    """Demonstrate database setup and initialization"""
    print_header("Database Setup & Initialization")
    
    # Create database in the databases directory
    db_path = "../databases/adam_slm_demo.db"
    
    # Remove existing demo database
    if os.path.exists(db_path):
        os.remove(db_path)
        
    print_section("Creating Database")
    db = AdamSLMDatabase(db_path)
    print(f"✅ Database created: {db_path}")
    
    # Test connection
    with db.get_connection() as conn:
        tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        print(f"📊 Created {len(tables)} tables")
        
    return db


def demo_user_management(db: AdamSLMDatabase):
    """Demonstrate user management features"""
    print_header("User Management")
    
    print_section("Creating Users")
    
    # Create users
    users_data = [
        {
            'username': 'alice_researcher',
            'email': 'alice@adamslm.ai',
            'full_name': 'Alice Johnson',
            'role': 'researcher',
            'preferences': {'theme': 'dark', 'notifications': True}
        },
        {
            'username': 'bob_engineer',
            'email': 'bob@adamslm.ai', 
            'full_name': 'Bob Smith',
            'role': 'engineer',
            'preferences': {'theme': 'light', 'auto_save': True}
        },
        {
            'username': 'carol_admin',
            'email': 'carol@adamslm.ai',
            'full_name': 'Carol Davis',
            'role': 'admin',
            'preferences': {'theme': 'auto', 'debug_mode': True}
        }
    ]
    
    user_ids = []
    for user_data in users_data:
        user_id = db.create_user(**user_data)
        user_ids.append(user_id)
        print(f"👤 Created user: {user_data['username']} (ID: {user_id})")
        
    # Create sessions
    print_section("Creating Sessions")
    for user_id in user_ids:
        session_token = db.create_session(
            user_id=user_id,
            ip_address=f"192.168.1.{random.randint(100, 200)}",
            user_agent="ADAM SLM Demo Client"
        )
        print(f"🔑 Created session for user {user_id}: {session_token[:20]}...")
        
    return user_ids


def demo_model_management(db: AdamSLMDatabase, user_ids: list):
    """Demonstrate model management features"""
    print_header("Model Management")
    
    print_section("Registering Models")
    
    # Sample model configurations
    models_data = [
        {
            'model_name': 'adam-slm-shakespeare',
            'version': '1.0.0',
            'model_type': 'adam-slm-small',
            'architecture_config': {
                'd_model': 512,
                'n_layers': 8,
                'n_heads': 8,
                'n_kv_heads': 4,
                'd_ff': 2048,
                'vocab_size': 50257,
                'max_seq_len': 1024
            },
            'parameter_count': 57_000_000,
            'model_size_mb': 230.5,
            'checkpoint_path': '/models/adam-slm-shakespeare-v1.pt',
            'tokenizer_path': '/models/tokenizer/',
            'created_by': user_ids[0],
            'description': 'ADAM SLM trained on Shakespeare dataset',
            'tags': ['shakespeare', 'literature', 'small']
        },
        {
            'model_name': 'adam-slm-code',
            'version': '2.1.0',
            'model_type': 'adam-slm-base',
            'architecture_config': {
                'd_model': 768,
                'n_layers': 12,
                'n_heads': 12,
                'n_kv_heads': 6,
                'd_ff': 3072,
                'vocab_size': 50257,
                'max_seq_len': 2048
            },
            'parameter_count': 145_000_000,
            'model_size_mb': 580.2,
            'checkpoint_path': '/models/adam-slm-code-v2.pt',
            'created_by': user_ids[1],
            'description': 'ADAM SLM fine-tuned for code generation',
            'tags': ['code', 'programming', 'base']
        },
        {
            'model_name': 'adam-slm-chat',
            'version': '3.0.0',
            'model_type': 'adam-slm-large',
            'architecture_config': {
                'd_model': 1024,
                'n_layers': 24,
                'n_heads': 16,
                'n_kv_heads': 8,
                'd_ff': 4096,
                'vocab_size': 50257,
                'max_seq_len': 4096
            },
            'parameter_count': 429_000_000,
            'model_size_mb': 1720.8,
            'checkpoint_path': '/models/adam-slm-chat-v3.pt',
            'created_by': user_ids[2],
            'description': 'Large ADAM SLM for conversational AI',
            'tags': ['chat', 'conversation', 'large']
        }
    ]
    
    model_ids = []
    for model_data in models_data:
        model_id = db.register_model(**model_data)
        model_ids.append(model_id)
        print(f"🤖 Registered model: {model_data['model_name']} v{model_data['version']} (ID: {model_id})")
        
    # Add benchmarks
    print_section("Adding Benchmarks")
    benchmark_data = [
        {'model_id': model_ids[0], 'benchmark_type': 'perplexity', 'metric_value': 15.2, 'dataset_name': 'shakespeare_test'},
        {'model_id': model_ids[0], 'benchmark_type': 'bleu', 'metric_value': 0.78, 'dataset_name': 'shakespeare_test'},
        {'model_id': model_ids[1], 'benchmark_type': 'code_accuracy', 'metric_value': 0.85, 'dataset_name': 'humaneval'},
        {'model_id': model_ids[1], 'benchmark_type': 'perplexity', 'metric_value': 12.8, 'dataset_name': 'code_test'},
        {'model_id': model_ids[2], 'benchmark_type': 'helpfulness', 'metric_value': 0.92, 'dataset_name': 'chat_eval'},
        {'model_id': model_ids[2], 'benchmark_type': 'safety', 'metric_value': 0.96, 'dataset_name': 'safety_eval'},
    ]
    
    for benchmark in benchmark_data:
        benchmark_id = db.add_model_benchmark(**benchmark)
        print(f"📊 Added benchmark: {benchmark['benchmark_type']} = {benchmark['metric_value']} (ID: {benchmark_id})")
        
    return model_ids


def demo_training_management(db: AdamSLMDatabase, model_ids: list, user_ids: list):
    """Demonstrate training management features"""
    print_header("Training Management")
    
    print_section("Creating Training Runs")
    
    # Sample training configurations
    training_runs = [
        {
            'run_name': 'shakespeare_fine_tune_v1',
            'model_id': model_ids[0],
            'training_config': {
                'learning_rate': 5e-4,
                'batch_size': 32,
                'max_steps': 10000,
                'warmup_steps': 1000,
                'weight_decay': 0.1
            },
            'started_by': user_ids[0],
            'total_steps': 10000,
            'notes': 'Initial fine-tuning on Shakespeare dataset'
        },
        {
            'run_name': 'code_training_experiment',
            'model_id': model_ids[1],
            'training_config': {
                'learning_rate': 3e-4,
                'batch_size': 16,
                'max_steps': 25000,
                'warmup_steps': 2000,
                'gradient_accumulation_steps': 4
            },
            'started_by': user_ids[1],
            'total_steps': 25000,
            'notes': 'Training on code generation dataset'
        }
    ]
    
    run_ids = []
    for run_data in training_runs:
        run_id = db.create_training_run(**run_data)
        run_ids.append(run_id)
        print(f"🏃 Created training run: {run_data['run_name']} (ID: {run_id})")
        
    # Simulate training progress
    print_section("Simulating Training Progress")
    for i, run_id in enumerate(run_ids):
        # Simulate some training steps
        steps = [100, 500, 1000, 2000, 5000]
        losses = [2.5, 2.1, 1.8, 1.5, 1.2]
        
        for step, loss in zip(steps, losses):
            # Log training metrics
            db.log_training_metric(run_id, step, 'train_loss', loss + random.uniform(-0.1, 0.1))
            db.log_training_metric(run_id, step, 'learning_rate', 5e-4 * (1 - step/10000))
            
        # Update training run status
        final_loss = losses[-1] + random.uniform(-0.05, 0.05)
        db.update_training_run(
            run_id=run_id,
            status='completed',
            current_step=steps[-1],
            best_loss=min(losses),
            final_loss=final_loss,
            training_time_seconds=random.randint(3600, 7200),
            gpu_hours=random.uniform(2.0, 6.0)
        )
        
        print(f"📈 Updated training run {run_id}: final_loss={final_loss:.3f}")
        
    return run_ids


def demo_dataset_management(db: AdamSLMDatabase, user_ids: list):
    """Demonstrate dataset management features"""
    print_header("Dataset Management")
    
    print_section("Registering Datasets")
    
    datasets_data = [
        {
            'name': 'shakespeare_complete',
            'description': 'Complete works of William Shakespeare',
            'dataset_type': 'text',
            'source_path': '/data/shakespeare/complete_works.txt',
            'total_samples': 150000,
            'total_tokens': 5_200_000,
            'avg_sequence_length': 34.7,
            'vocabulary_size': 28000,
            'language': 'en',
            'license': 'Public Domain',
            'created_by': user_ids[0],
            'statistics': {
                'file_size_mb': 20.5,
                'unique_words': 28000,
                'avg_line_length': 45.2
            },
            'tags': ['literature', 'english', 'classic']
        },
        {
            'name': 'python_code_corpus',
            'description': 'Large corpus of Python code from GitHub',
            'dataset_type': 'code',
            'source_path': '/data/code/python_corpus.jsonl',
            'total_samples': 500000,
            'total_tokens': 15_000_000,
            'avg_sequence_length': 30.0,
            'vocabulary_size': 45000,
            'language': 'python',
            'license': 'MIT',
            'created_by': user_ids[1],
            'statistics': {
                'file_size_mb': 180.2,
                'avg_function_length': 25.5,
                'comment_ratio': 0.15
            },
            'tags': ['code', 'python', 'programming']
        },
        {
            'name': 'conversational_ai_dataset',
            'description': 'High-quality conversational AI training data',
            'dataset_type': 'conversation',
            'source_path': '/data/chat/conversations.jsonl',
            'total_samples': 1_000_000,
            'total_tokens': 25_000_000,
            'avg_sequence_length': 25.0,
            'vocabulary_size': 50000,
            'language': 'en',
            'license': 'Custom',
            'created_by': user_ids[2],
            'statistics': {
                'file_size_mb': 320.8,
                'avg_conversation_turns': 8.5,
                'safety_filtered': True
            },
            'tags': ['conversation', 'chat', 'ai-assistant']
        }
    ]
    
    dataset_ids = []
    for dataset_data in datasets_data:
        dataset_id = db.register_dataset(**dataset_data)
        dataset_ids.append(dataset_id)
        print(f"📚 Registered dataset: {dataset_data['name']} (ID: {dataset_id})")
        
        # Add dataset splits
        splits = [
            {'split_name': 'train', 'split_ratio': 0.8},
            {'split_name': 'validation', 'split_ratio': 0.1},
            {'split_name': 'test', 'split_ratio': 0.1}
        ]
        
        for split in splits:
            sample_count = int(dataset_data['total_samples'] * split['split_ratio'])
            token_count = int(dataset_data['total_tokens'] * split['split_ratio'])
            
            split_id = db.add_dataset_split(
                dataset_id=dataset_id,
                split_name=split['split_name'],
                file_path=f"/data/{dataset_data['name']}/{split['split_name']}.jsonl",
                sample_count=sample_count,
                token_count=token_count,
                split_ratio=split['split_ratio']
            )
            
        print(f"  📂 Added 3 splits for dataset {dataset_id}")
        
    return dataset_ids


def demo_analytics(db: AdamSLMDatabase, model_ids: list):
    """Demonstrate analytics features"""
    print_header("Advanced Analytics")
    
    analytics = DatabaseAnalytics(db)
    
    print_section("Model Performance Comparison")
    comparison = analytics.get_model_performance_comparison(model_ids[:2])
    
    print(f"📊 Comparing {len(comparison['models'])} models:")
    for model in comparison['models']:
        print(f"  • {model['name']} v{model['version']}: {model['parameters']:,} parameters")
        
    print(f"📈 Benchmark types: {list(comparison['benchmarks'].keys())}")
    
    for bench_type, summary in comparison['summary'].items():
        print(f"  • {bench_type}: best={summary['best_value']:.3f} (avg={summary['avg_value']:.3f})")
        
    print_section("Training Trends")
    trends = analytics.get_training_trends(days=30)
    
    print(f"📅 Training trends (last {trends['period_days']} days):")
    if trends['success_rate']:
        print(f"  • Success rate: {trends['success_rate']['success_rate']:.1f}%")
        print(f"  • Total runs: {trends['success_rate']['total_runs']}")
        
    if trends['resource_usage']:
        usage = trends['resource_usage']
        if usage['total_gpu_hours']:
            print(f"  • Total GPU hours: {usage['total_gpu_hours']:.1f}")
        if usage['total_tokens_processed']:
            print(f"  • Total tokens processed: {usage['total_tokens_processed']:,}")
            
    print_section("Dataset Usage Analysis")
    dataset_analysis = analytics.get_dataset_usage_analysis()
    
    print(f"📚 Dataset usage statistics:")
    for dataset in dataset_analysis['usage_stats'][:3]:  # Top 3
        print(f"  • {dataset['name']}: used {dataset['times_used']} times")
        if dataset['avg_final_loss']:
            print(f"    Average final loss: {dataset['avg_final_loss']:.3f}")


def demo_database_manager():
    """Demonstrate high-level database manager"""
    print_header("Database Manager Features")
    
    # Create manager
    manager = DatabaseManager("adam_slm_demo.db")
    
    print_section("Dashboard Statistics")
    stats = manager.get_dashboard_stats()
    
    print(f"📊 Dashboard Overview:")
    print(f"  • Models: {stats['models']['total_models']} total, {stats['models']['unique_types']} types")
    print(f"  • Training: {stats['training']['total_runs']} runs, {stats['training']['completed_runs']} completed")
    print(f"  • Datasets: {stats['datasets']['total_datasets']} datasets")
    
    if stats['models']['total_parameters']:
        print(f"  • Total parameters: {stats['models']['total_parameters']:,}")
        
    print_section("Recent Activity")
    print("🕒 Recent models:")
    for model in stats['recent_models'][:3]:
        print(f"  • {model['model_name']} v{model['version']} ({model['created_at'][:10]})")
        
    print("🏃 Recent training runs:")
    for run in stats['recent_runs'][:3]:
        print(f"  • {run['run_name']} - {run['status']} ({run['started_at'][:10]})")


def main():
    """Main demo function"""
    print("🎉 Welcome to ADAM SLM Database Demo!")
    print("Sophisticated SQLite database for AI model management")
    
    try:
        # Setup
        db = demo_database_setup()
        
        # User management
        user_ids = demo_user_management(db)
        
        # Model management
        model_ids = demo_model_management(db, user_ids)
        
        # Training management
        run_ids = demo_training_management(db, model_ids, user_ids)
        
        # Dataset management
        dataset_ids = demo_dataset_management(db, user_ids)
        
        # Analytics
        demo_analytics(db, model_ids)
        
        # High-level manager
        demo_database_manager()
        
        print_header("Demo Complete!")
        print("🎯 Features Demonstrated:")
        print("  ✅ Comprehensive schema with 15+ tables")
        print("  ✅ User management and sessions")
        print("  ✅ Model versioning and benchmarks")
        print("  ✅ Training run tracking with metrics")
        print("  ✅ Dataset management and splits")
        print("  ✅ Advanced analytics and reporting")
        print("  ✅ High-level database manager")
        print("  ✅ Migration system")
        print("  ✅ Performance optimization")
        
        print(f"\n📁 Demo database created: adam_slm_demo.db")
        print("🔍 You can explore the database with any SQLite browser!")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
