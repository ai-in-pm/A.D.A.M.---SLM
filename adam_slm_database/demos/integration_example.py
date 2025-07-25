#!/usr/bin/env python3
"""
ADAM SLM Database Integration Example
Shows how to integrate the sophisticated database with ADAM SLM training and inference
"""

import sys
import os
import json
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from database import AdamSLMDatabase
from manager import DatabaseManager
from analytics import DatabaseAnalytics

# Import ADAM SLM components (would work if running from main directory)
try:
    from adam_slm.models import AdamSLM, get_config
    from adam_slm.tokenization import AdamTokenizer
    from adam_slm.training import get_training_config
    from adam_slm.inference import AdamInference, GenerationConfig
    ADAM_SLM_AVAILABLE = True
except ImportError:
    print("⚠️ ADAM SLM not available - showing database integration concepts")
    ADAM_SLM_AVAILABLE = False


def print_header(title: str):
    """Print formatted header"""
    print("\n" + "="*60)
    print(f"🔗 {title}")
    print("="*60)


def print_section(title: str):
    """Print formatted section"""
    print(f"\n📋 {title}")
    print("-"*40)


def demo_training_integration():
    """Demonstrate training integration with database"""
    print_header("Training Integration")
    
    # Initialize database manager
    manager = DatabaseManager("../databases/adamslm_sophisticated.sqlite")
    
    print_section("Starting Training Run")
    
    # Model configuration
    model_config = {
        "d_model": 512,
        "n_layers": 8,
        "n_heads": 8,
        "n_kv_heads": 4,
        "d_ff": 2048,
        "vocab_size": 50257,
        "max_seq_len": 1024,
        "use_rope": True,
        "use_swiglu": True,
        "use_rms_norm": True,
        "use_gqa": True
    }
    
    # Training configuration
    training_config = {
        "learning_rate": 5e-4,
        "batch_size": 16,
        "max_steps": 5000,
        "warmup_steps": 500,
        "weight_decay": 0.1,
        "gradient_accumulation_steps": 2,
        "eval_steps": 500,
        "save_steps": 1000
    }
    
    # Start training run
    run_id = manager.start_training_run(
        run_name="adam_slm_shakespeare_experiment",
        model_config=model_config,
        training_config=training_config,
        dataset_name="training_corpus",
        started_by_username="admin",
        notes="Training ADAM SLM on Shakespeare dataset with sophisticated features"
    )
    
    print(f"🏃 Started training run: {run_id}")
    
    # Simulate training progress
    print_section("Simulating Training Progress")
    
    db = manager.db
    
    # Log training metrics at different steps
    training_steps = [
        (100, {"train_loss": 3.2, "learning_rate": 4.5e-4, "grad_norm": 1.2}),
        (500, {"train_loss": 2.8, "learning_rate": 4.0e-4, "grad_norm": 0.9, "eval_loss": 2.9}),
        (1000, {"train_loss": 2.4, "learning_rate": 3.5e-4, "grad_norm": 0.8, "eval_loss": 2.5}),
        (2000, {"train_loss": 2.0, "learning_rate": 2.5e-4, "grad_norm": 0.7, "eval_loss": 2.1}),
        (5000, {"train_loss": 1.6, "learning_rate": 1.0e-4, "grad_norm": 0.6, "eval_loss": 1.7}),
    ]
    
    for step, metrics in training_steps:
        for metric_name, metric_value in metrics.items():
            db.log_training_metric(run_id, step, metric_name, metric_value)
        
        # Update training run progress
        db.update_training_run(
            run_id=run_id,
            current_step=step,
            best_loss=min([m.get("eval_loss", m.get("train_loss", 999)) for s, m in training_steps[:training_steps.index((step, metrics))+1]])
        )
        
        print(f"📊 Step {step}: loss={metrics['train_loss']:.3f}, lr={metrics['learning_rate']:.2e}")
    
    # Complete training run
    final_model_path = f"/models/adam_slm_shakespeare_{run_id}.pt"
    
    final_model_id = manager.complete_training_run(
        run_id=run_id,
        final_model_path=final_model_path,
        final_loss=1.6,
        training_time_seconds=3600,
        total_tokens_processed=10_000_000,
        gpu_hours=2.5
    )
    
    print(f"✅ Training completed! Final model ID: {final_model_id}")
    
    return run_id, final_model_id


def demo_inference_integration():
    """Demonstrate inference integration with database"""
    print_header("Inference Integration")
    
    db = AdamSLMDatabase("adamslm_sophisticated.sqlite")
    
    print_section("Creating Inference Session")
    
    # Get a model from database
    models = db.list_models(limit=1)
    if not models:
        print("❌ No models found in database")
        return
        
    model = models[0]
    print(f"🤖 Using model: {model['model_name']} v{model['version']}")
    
    # Create inference session
    session_id = db.execute_insert("""
        INSERT INTO inference_sessions (
            model_id, user_id, session_name, configuration
        ) VALUES (?, ?, ?, ?)
    """, (
        model['id'],
        1,  # admin user
        "shakespeare_generation_session",
        json.dumps({
            "max_new_tokens": 100,
            "temperature": 0.8,
            "top_k": 50,
            "top_p": 0.9
        })
    ))
    
    print(f"🎯 Created inference session: {session_id}")
    
    # Simulate inference requests
    print_section("Simulating Inference Requests")
    
    test_prompts = [
        "To be or not to be,",
        "Romeo, Romeo, wherefore art thou",
        "All the world's a stage,",
        "Now is the winter of our discontent",
        "Friends, Romans, countrymen,"
    ]
    
    total_tokens_generated = 0
    total_requests = 0
    
    for prompt in test_prompts:
        # Simulate generation (would use actual ADAM SLM model in real integration)
        generated_text = prompt + " [generated text would appear here...]"
        prompt_tokens = len(prompt.split())
        generated_tokens = 25  # Simulated
        response_time_ms = 150  # Simulated
        
        # Log inference request
        request_id = db.execute_insert("""
            INSERT INTO inference_requests (
                session_id, prompt_text, generated_text, prompt_tokens,
                generated_tokens, response_time_ms, temperature, top_k, top_p
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            session_id, prompt, generated_text, prompt_tokens,
            generated_tokens, response_time_ms, 0.8, 50, 0.9
        ))
        
        total_tokens_generated += generated_tokens
        total_requests += 1
        
        print(f"💭 Request {request_id}: '{prompt[:30]}...' -> {generated_tokens} tokens")
    
    # Update session statistics
    avg_response_time = 150.0  # Simulated average
    
    db.execute_update("""
        UPDATE inference_sessions 
        SET total_requests = ?, total_tokens_generated = ?, avg_response_time_ms = ?, ended_at = CURRENT_TIMESTAMP
        WHERE id = ?
    """, (total_requests, total_tokens_generated, avg_response_time, session_id))
    
    print(f"📊 Session completed: {total_requests} requests, {total_tokens_generated} tokens")
    
    return session_id


def demo_analytics_integration():
    """Demonstrate analytics integration"""
    print_header("Analytics Integration")
    
    analytics = DatabaseAnalytics(AdamSLMDatabase("adamslm_sophisticated.sqlite"))
    
    print_section("Training Performance Analysis")
    
    # Get training trends
    trends = analytics.get_training_trends(days=1)  # Last day
    
    print("📈 Training Trends:")
    if trends['success_rate']:
        print(f"  • Success rate: {trends['success_rate']['success_rate']:.1f}%")
        print(f"  • Total runs: {trends['success_rate']['total_runs']}")
    
    if trends['resource_usage']:
        usage = trends['resource_usage']
        if usage.get('total_gpu_hours'):
            print(f"  • GPU hours used: {usage['total_gpu_hours']:.1f}")
        if usage.get('total_tokens_processed'):
            print(f"  • Tokens processed: {usage['total_tokens_processed']:,}")
    
    print_section("Model Performance Comparison")
    
    # Get all models for comparison
    db = AdamSLMDatabase("adamslm_sophisticated.sqlite")
    models = db.list_models()
    model_ids = [m['id'] for m in models]
    
    if len(model_ids) > 1:
        comparison = analytics.get_model_performance_comparison(model_ids)
        
        print(f"🔍 Comparing {len(comparison['models'])} models:")
        for model in comparison['models']:
            print(f"  • {model['name']}: {model['parameters']:,} parameters")
            
        for bench_type, summary in comparison['summary'].items():
            print(f"  • {bench_type}: best={summary['best_value']:.3f}")
    else:
        print("ℹ️ Need multiple models for comparison")
    
    print_section("System Performance Report")
    
    # Get system performance (would have real metrics in production)
    performance = analytics.get_system_performance_report(hours=1)
    
    print("🖥️ System Performance:")
    print(f"  • Report period: {performance['period_hours']} hours")
    print(f"  • Active sessions: {performance['active_sessions']}")
    print(f"  • Database tables: {len(performance['table_stats'])}")
    
    for table_stat in performance['table_stats'][:5]:  # Top 5 tables
        print(f"    - {table_stat['table_name']}: {table_stat['row_count']} rows")


def demo_experiment_tracking():
    """Demonstrate experiment tracking"""
    print_header("Experiment Tracking")
    
    manager = DatabaseManager("adamslm_sophisticated.sqlite")
    
    print_section("Creating Hyperparameter Experiment")
    
    # Define experiment with multiple runs
    run_configs = [
        {
            'model_config': {'d_model': 512, 'n_layers': 8},
            'training_config': {'learning_rate': 1e-3, 'batch_size': 16},
            'purpose': 'High learning rate baseline'
        },
        {
            'model_config': {'d_model': 512, 'n_layers': 8},
            'training_config': {'learning_rate': 5e-4, 'batch_size': 16},
            'purpose': 'Medium learning rate'
        },
        {
            'model_config': {'d_model': 512, 'n_layers': 8},
            'training_config': {'learning_rate': 1e-4, 'batch_size': 16},
            'purpose': 'Low learning rate'
        }
    ]
    
    experiment_id, run_ids = manager.create_experiment_with_runs(
        name="learning_rate_ablation",
        description="Systematic study of learning rate impact on ADAM SLM training",
        objective="Find optimal learning rate for Shakespeare dataset",
        hypothesis="Medium learning rate (5e-4) will provide best convergence",
        run_configs=run_configs,
        created_by_username="admin"
    )
    
    print(f"🧪 Created experiment: {experiment_id}")
    print(f"🏃 Created {len(run_ids)} training runs: {run_ids}")
    
    # Simulate experiment completion
    db = manager.db
    
    # Update experiment with results
    db.update_experiment(
        experiment_id=experiment_id,
        status="completed",
        results_summary="Medium learning rate (5e-4) achieved best final loss of 1.6",
        conclusions="Hypothesis confirmed. 5e-4 learning rate optimal for this dataset size and model architecture."
    )
    
    print("✅ Experiment completed with results logged")
    
    return experiment_id


def main():
    """Main integration demo"""
    print("🔗 ADAM SLM Database Integration Demo")
    print("Sophisticated database integration with AI model lifecycle")
    
    if not ADAM_SLM_AVAILABLE:
        print("\n⚠️ Note: ADAM SLM modules not available")
        print("This demo shows database integration concepts")
    
    try:
        # Training integration
        run_id, model_id = demo_training_integration()
        
        # Inference integration
        session_id = demo_inference_integration()
        
        # Analytics integration
        demo_analytics_integration()
        
        # Experiment tracking
        experiment_id = demo_experiment_tracking()
        
        print_header("Integration Demo Complete!")
        print("🎯 Integration Features Demonstrated:")
        print("  ✅ Training run lifecycle management")
        print("  ✅ Real-time metrics logging")
        print("  ✅ Model versioning and lineage")
        print("  ✅ Inference session tracking")
        print("  ✅ Performance analytics")
        print("  ✅ Experiment organization")
        print("  ✅ System monitoring")
        
        print("\n📊 Database Contents:")
        manager = DatabaseManager("adamslm_sophisticated.sqlite")
        stats = manager.get_dashboard_stats()
        
        print(f"  • Models: {stats['models']['total_models']}")
        print(f"  • Training runs: {stats['training']['total_runs']}")
        print(f"  • Datasets: {stats['datasets']['total_datasets']}")
        print(f"  • Parameters: {stats['models']['total_parameters']:,}")
        
        print("\n🚀 Ready for Production Integration!")
        print("The sophisticated database is now ready to support:")
        print("  • Full ADAM SLM training pipelines")
        print("  • Production inference deployments")
        print("  • Research experiment tracking")
        print("  • Performance monitoring and optimization")
        
    except Exception as e:
        print(f"\n❌ Integration demo failed: {e}")
        raise


if __name__ == "__main__":
    main()
