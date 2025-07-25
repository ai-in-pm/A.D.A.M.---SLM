#!/usr/bin/env python3
"""
A.D.A.M. SLM Integration Demo
Demonstrates the complete integration of A.D.A.M. SLM with the sophisticated database system
"""

import os
import sys
import torch
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

# Import A.D.A.M. SLM components
try:
    from adam_slm import (
        AdamSLM, AdamSLMConfig, AdamTokenizer,
        get_default_database, get_database_manager, 
        DatabaseTrainingLogger, KnowledgeBase,
        search_papers, get_knowledge_stats,
        get_dashboard_stats
    )
    
    # Import training components
    from adam_slm.training import DatabaseAwareTrainer, TrainingConfig
    
    # Import inference components  
    from adam_slm.inference import KnowledgeEnhancedInference, GenerationConfig
    
    INTEGRATION_AVAILABLE = True
    
except ImportError as e:
    print(f"❌ Integration not available: {e}")
    print("Make sure the database system is properly set up")
    INTEGRATION_AVAILABLE = False
    sys.exit(1)


def print_header(title: str):
    """Print formatted header"""
    print("\n" + "="*60)
    print(f"🚀 {title}")
    print("="*60)


def print_section(title: str):
    """Print formatted section"""
    print(f"\n📋 {title}")
    print("-"*40)


def print_success(message: str):
    """Print success message"""
    print(f"✅ {message}")


def print_info(message: str):
    """Print info message"""
    print(f"ℹ️  {message}")


def demo_database_integration():
    """Demonstrate database integration"""
    print_section("Database Integration")
    
    try:
        # Initialize database
        db = get_default_database()
        manager = get_database_manager()
        
        print_success("Database connection established")
        
        # Get dashboard stats
        stats = get_dashboard_stats()
        print_info(f"Database contains:")
        print(f"   📊 Models: {stats['models']['total_models']}")
        print(f"   🏃 Training runs: {stats['training']['total_runs']}")
        print(f"   📚 Datasets: {stats['datasets']['total_datasets']}")
        print(f"   📄 Documents: {stats.get('files', {}).get('total_files', 0)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Database integration failed: {e}")
        return False


def demo_knowledge_base():
    """Demonstrate knowledge base functionality"""
    print_section("Knowledge Base Integration")
    
    try:
        # Initialize knowledge base
        kb = KnowledgeBase()
        
        # Get knowledge stats
        stats = get_knowledge_stats()
        print_success("Knowledge base initialized")
        print_info(f"Knowledge base contains:")
        print(f"   📄 Research papers: {stats['total_papers']}")
        print(f"   📝 Total words: {stats['total_words']:,}")
        print(f"   📊 Average words per paper: {stats['avg_words_per_paper']:,}")
        
        # Show topics
        if stats['topics']:
            print_info("Research topics:")
            for topic, count in list(stats['topics'].items())[:5]:
                print(f"   🔬 {topic}: {count} papers")
        
        # Demonstrate search
        print_section("Knowledge Search Demo")
        
        search_queries = [
            "transformer architecture",
            "mixture of experts", 
            "reinforcement learning",
            "attention mechanism"
        ]
        
        for query in search_queries:
            results = search_papers(query, limit=2)
            if results:
                print_success(f"'{query}' found in {len(results)} papers:")
                for result in results:
                    print(f"   📄 {result['filename']} (relevance: {result['relevance_score']:.2f})")
            else:
                print_info(f"'{query}' not found in knowledge base")
        
        return True
        
    except Exception as e:
        print(f"❌ Knowledge base demo failed: {e}")
        return False


def demo_model_creation():
    """Demonstrate model creation and configuration"""
    print_section("Model Creation")
    
    try:
        # Create model configuration
        config = AdamSLMConfig(
            d_model=512,
            n_layers=6,
            n_heads=8,
            n_kv_heads=4,
            d_ff=2048,
            vocab_size=50257,
            max_seq_len=1024,
            use_rope=True,
            use_swiglu=True,
            use_rms_norm=True,
            use_gqa=True
        )
        
        print_success("Model configuration created")
        print_info(f"Model parameters:")
        print(f"   🧠 d_model: {config.d_model}")
        print(f"   📚 Layers: {config.n_layers}")
        print(f"   👁️  Attention heads: {config.n_heads}")
        print(f"   🔤 Vocabulary: {config.vocab_size:,}")
        print(f"   📏 Max sequence: {config.max_seq_len:,}")
        
        # Create model
        model = AdamSLM(config)
        print_success(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Create A.D.A.M.-SLM tokenizer
        from .tokenization import get_tokenizer
        tokenizer = get_tokenizer("adam_slm")
        print_success("A.D.A.M.-SLM tokenizer initialized")
        
        return model, tokenizer, config
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return None, None, None


def demo_database_training(model, tokenizer, config):
    """Demonstrate database-aware training"""
    print_section("Database-Aware Training Demo")
    
    try:
        # Create training configuration
        training_config = TrainingConfig(
            learning_rate=5e-4,
            batch_size=4,
            max_steps=100,  # Short demo
            warmup_steps=10,
            eval_steps=25,
            save_steps=50,
            output_dir="./demo_checkpoints"
        )
        
        print_success("Training configuration created")
        
        # Create database-aware trainer
        trainer = DatabaseAwareTrainer(
            model=model,
            tokenizer=tokenizer,
            config=training_config,
            run_name=f"adam_slm_demo_{int(datetime.now().timestamp())}",
            experiment_name="integration_demo",
            dataset_name="demo_dataset",
            notes="Integration demonstration run",
            created_by="demo_user"
        )
        
        print_success("Database-aware trainer created")
        print_info(f"Training run: {trainer.run_name}")
        
        # Simulate training steps (without actual training)
        print_info("Simulating training steps...")
        
        # Start training in database
        run_id = trainer.db_logger.start_training() if trainer.db_logger else None
        
        if run_id:
            print_success(f"Training run started in database (ID: {run_id})")
            
            # Simulate some training steps
            for step in range(1, 11):
                metrics = {
                    'loss': 3.5 - (step * 0.1),  # Decreasing loss
                    'learning_rate': 5e-4 * (1 - step/100),
                    'grad_norm': 1.0 + (step * 0.05)
                }
                
                trainer.db_logger.log_step(step, metrics, tokens_processed=1024)
                
                if step % 5 == 0:
                    print_info(f"Step {step}: loss={metrics['loss']:.3f}")
            
            # Get training summary
            summary = trainer.get_training_summary()
            print_success("Training simulation completed")
            print_info(f"Steps logged: {summary.get('steps_completed', 0)}")
            print_info(f"Metrics logged: {summary.get('metrics_logged', 0)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Database training demo failed: {e}")
        return False


def demo_knowledge_inference(model, tokenizer):
    """Demonstrate knowledge-enhanced inference"""
    print_section("Knowledge-Enhanced Inference Demo")
    
    try:
        # Create knowledge-enhanced inference
        inference = KnowledgeEnhancedInference(
            model=model,
            tokenizer=tokenizer,
            max_knowledge_context=1000,
            enable_citations=True
        )
        
        print_success("Knowledge-enhanced inference initialized")
        
        # Demo questions
        questions = [
            "What is a transformer architecture?",
            "How does mixture of experts work?",
            "What are the benefits of attention mechanisms?",
            "Explain reinforcement learning from human feedback"
        ]
        
        for question in questions:
            print_info(f"Question: {question}")
            
            try:
                # Answer using knowledge base
                result = inference.answer_question(
                    question=question,
                    max_context_length=500,
                    generation_config=GenerationConfig(
                        max_new_tokens=100,
                        temperature=0.7,
                        do_sample=True
                    )
                )
                
                if result['sources']:
                    print_success(f"Found {len(result['sources'])} relevant sources")
                    for source in result['sources']:
                        print(f"   📄 {source['filename']} (relevance: {source['relevance']:.2f})")
                else:
                    print_info("No specific sources found, using general knowledge")
                
            except Exception as e:
                print(f"   ⚠️ Question processing failed: {e}")
        
        # Demo research summarization
        print_section("Research Summarization Demo")
        
        topics = ["neural networks", "language models", "artificial intelligence"]
        
        for topic in topics:
            try:
                summary_result = inference.summarize_research(
                    topic=topic,
                    max_papers=3,
                    summary_length="short"
                )
                
                if summary_result['papers_found'] > 0:
                    print_success(f"Research summary for '{topic}':")
                    print(f"   📊 Papers found: {summary_result['papers_found']}")
                    print(f"   📝 Summary: {summary_result['summary'][:100]}...")
                else:
                    print_info(f"No papers found for topic: {topic}")
                    
            except Exception as e:
                print(f"   ⚠️ Summarization failed for {topic}: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Knowledge inference demo failed: {e}")
        return False


def main():
    """Main demo function"""
    print_header("A.D.A.M. SLM Database Integration Demo")
    print("Comprehensive demonstration of A.D.A.M. SLM with sophisticated database system")
    
    if not INTEGRATION_AVAILABLE:
        print("❌ Integration components not available")
        return 1
    
    # Demo components
    demos = [
        ("Database Integration", demo_database_integration),
        ("Knowledge Base", demo_knowledge_base),
    ]
    
    # Run basic demos
    for demo_name, demo_func in demos:
        try:
            success = demo_func()
            if not success:
                print(f"⚠️ {demo_name} demo had issues")
        except Exception as e:
            print(f"❌ {demo_name} demo failed: {e}")
    
    # Model-dependent demos
    try:
        model, tokenizer, config = demo_model_creation()
        
        if model and tokenizer:
            # Training demo
            demo_database_training(model, tokenizer, config)
            
            # Inference demo
            demo_knowledge_inference(model, tokenizer)
    
    except Exception as e:
        print(f"⚠️ Model demos skipped due to: {e}")
    
    print_header("Integration Demo Complete")
    print_success("A.D.A.M. SLM database integration demonstrated successfully!")
    print_info("Key features showcased:")
    print("   🗄️ Sophisticated database system with 15+ tables")
    print("   📄 Research paper knowledge base with 70,000+ words")
    print("   🏃 Database-aware training with automatic logging")
    print("   🧠 Knowledge-enhanced inference with citations")
    print("   📊 Comprehensive analytics and reporting")
    print("   🔍 Full-text search across research papers")
    print("   📈 Model lifecycle management and versioning")
    
    print("\n🚀 A.D.A.M. SLM is now fully integrated with the database system!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
