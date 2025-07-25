#!/usr/bin/env python3
"""
Example script for training ADAM SLM
"""

import torch
import os
import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from adam_slm.models import AdamSLM, get_config
from adam_slm.training import AdamTrainer, get_training_config, create_dataloader
from adam_slm.tokenization import AdamTokenizer
from adam_slm.inference import AdamInference, GenerationConfig


def download_shakespeare_data():
    """Download Shakespeare dataset for training"""
    import urllib.request
    
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    filename = "shakespeare.txt"
    
    if not os.path.exists(filename):
        print("Downloading Shakespeare dataset...")
        urllib.request.urlretrieve(url, filename)
        print(f"Downloaded {filename}")
    else:
        print(f"Using existing {filename}")
        
    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read()
        
    return text


def main():
    parser = argparse.ArgumentParser(description="Train ADAM SLM")
    parser.add_argument("--model_size", default="adam-slm-small", choices=["adam-slm-small", "adam-slm-base", "adam-slm-large"])
    parser.add_argument("--training_config", default="small", choices=["debug", "small", "base", "large"])
    parser.add_argument("--output_dir", default="./checkpoints", help="Output directory for checkpoints")
    parser.add_argument("--max_steps", type=int, default=None, help="Maximum training steps")
    parser.add_argument("--eval_steps", type=int, default=None, help="Evaluation frequency")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=None, help="Learning rate")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases logging")
    parser.add_argument("--resume_from", type=str, default=None, help="Resume training from checkpoint")
    
    args = parser.parse_args()
    
    print("🚀 ADAM SLM Training Script")
    print("=" * 50)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model configuration
    model_config = get_config(args.model_size)
    print(f"Model: {args.model_size}")
    print(f"Parameters: ~{model_config.d_model * model_config.n_layers * 12 // 1000}K")
    
    # Load training configuration
    training_config = get_training_config(args.training_config)
    
    # Override with command line arguments
    if args.output_dir:
        training_config.output_dir = args.output_dir
    if args.max_steps:
        training_config.max_steps = args.max_steps
    if args.eval_steps:
        training_config.eval_steps = args.eval_steps
    if args.batch_size:
        training_config.batch_size = args.batch_size
    if args.learning_rate:
        training_config.learning_rate = args.learning_rate
    if args.use_wandb:
        training_config.report_to = ["wandb"]
        training_config.run_name = f"adam-slm-{args.model_size}-{args.training_config}"
        
    print(f"Training config: {args.training_config}")
    print(f"Batch size: {training_config.batch_size}")
    print(f"Learning rate: {training_config.learning_rate}")
    print(f"Max steps: {training_config.max_steps}")
    
    # Create model
    print("\n📦 Creating model...")
    model = AdamSLM(model_config)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Setup tokenizer
    print("\n🔤 Setting up tokenizer...")
    tokenizer = AdamTokenizer("gpt2")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    # Load and prepare data
    print("\n📚 Loading data...")
    text_data = download_shakespeare_data()
    print(f"Text length: {len(text_data):,} characters")
    
    # Create data loaders
    train_loader, eval_loader = create_dataloader(
        texts=[text_data],
        tokenizer=tokenizer,
        max_length=model_config.max_seq_len,
        batch_size=training_config.batch_size,
        train_test_split=0.1,  # 10% for validation
        num_workers=training_config.dataloader_num_workers,
    )
    
    print(f"Training batches: {len(train_loader)}")
    print(f"Evaluation batches: {len(eval_loader)}")
    
    # Create trainer
    print("\n🏋️ Setting up trainer...")
    trainer = AdamTrainer(
        model=model,
        config=training_config,
        train_dataloader=train_loader,
        eval_dataloader=eval_loader,
        tokenizer=tokenizer,
        device=device,
    )
    
    # Resume from checkpoint if specified
    if args.resume_from:
        print(f"📂 Resuming from checkpoint: {args.resume_from}")
        # Implementation would go here
        
    # Start training
    print("\n🚀 Starting training...")
    print("=" * 50)
    
    try:
        training_stats = trainer.train()
        print("\n✅ Training completed!")
        print(f"Final loss: {training_stats['final_loss']:.4f}")
        print(f"Total steps: {training_stats['total_steps']}")
        print(f"Epochs: {training_stats['epochs']}")
        
    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        raise
        
    # Test inference
    print("\n🧪 Testing inference...")
    
    # Load best model for inference
    best_model_path = os.path.join(training_config.output_dir, "best_model.pt")
    if os.path.exists(best_model_path):
        inference = AdamInference.from_pretrained(
            model_path=best_model_path,
            tokenizer=tokenizer,
            device=device,
            generation_config=GenerationConfig(
                max_new_tokens=100,
                temperature=0.8,
                top_k=50,
                do_sample=True,
            )
        )
        
        # Generate some sample text
        test_prompts = [
            "To be or not to be,",
            "Romeo, Romeo, wherefore art thou",
            "All the world's a stage,",
        ]
        
        print("\n📝 Sample generations:")
        print("-" * 30)
        
        for prompt in test_prompts:
            generated = inference.generate(prompt)
            print(f"Prompt: {prompt}")
            print(f"Generated: {generated}")
            print("-" * 30)
            
        # Show inference stats
        stats = inference.get_stats()
        print(f"\n📊 Inference stats:")
        print(f"Tokens per second: {stats['tokens_per_second']:.1f}")
        
    print("\n🎉 All done!")


if __name__ == "__main__":
    main()
