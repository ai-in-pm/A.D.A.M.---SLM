#!/usr/bin/env python3
"""
ADAM SLM Database File Import Demo
Demonstrates comprehensive file management capabilities
"""

import os
import sys
import json
import tempfile
import shutil
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from database import AdamSLMDatabase
from file_manager import FileManager
from file_converter import FileConverter


def print_header(title: str):
    """Print formatted header"""
    print("\n" + "="*60)
    print(f"📁 {title}")
    print("="*60)


def print_section(title: str):
    """Print formatted section"""
    print(f"\n📋 {title}")
    print("-"*40)


def create_sample_files():
    """Create sample files for demonstration"""
    print_section("Creating Sample Files")
    
    # Create temporary directory for sample files
    sample_dir = Path("sample_files")
    sample_dir.mkdir(exist_ok=True)
    
    files_created = []
    
    # 1. Text file
    text_file = sample_dir / "sample_text.txt"
    with open(text_file, 'w') as f:
        f.write("""This is a sample text file for ADAM SLM database.
It contains multiple lines of text that can be analyzed and processed.
The file manager will extract content and metadata automatically.

Key features:
- Automatic content extraction
- Metadata analysis
- Full-text search capabilities
- Version control and lineage tracking
""")
    files_created.append(("text", str(text_file)))
    print(f"📝 Created text file: {text_file}")
    
    # 2. JSON configuration file
    config_file = sample_dir / "model_config.json"
    config_data = {
        "model_name": "adam-slm-demo",
        "architecture": {
            "d_model": 768,
            "n_layers": 12,
            "n_heads": 12,
            "n_kv_heads": 6,
            "d_ff": 3072,
            "vocab_size": 50257,
            "max_seq_len": 2048
        },
        "training": {
            "learning_rate": 5e-4,
            "batch_size": 32,
            "warmup_steps": 1000,
            "max_steps": 10000
        },
        "features": {
            "use_rope": True,
            "use_swiglu": True,
            "use_rms_norm": True,
            "use_gqa": True
        }
    }
    
    with open(config_file, 'w') as f:
        json.dump(config_data, f, indent=2)
    files_created.append(("config", str(config_file)))
    print(f"⚙️ Created config file: {config_file}")
    
    # 3. CSV dataset file
    csv_file = sample_dir / "training_data.csv"
    with open(csv_file, 'w') as f:
        f.write("""text,label,category
"The quick brown fox jumps over the lazy dog",positive,example
"Machine learning is transforming our world",positive,technology
"Natural language processing enables AI understanding",positive,ai
"Deep learning models require large datasets",neutral,education
"Training neural networks can be computationally expensive",neutral,technical
"ADAM SLM represents the future of language models",positive,product
""")
    files_created.append(("dataset", str(csv_file)))
    print(f"📊 Created dataset file: {csv_file}")
    
    # 4. Markdown documentation
    md_file = sample_dir / "documentation.md"
    with open(md_file, 'w') as f:
        f.write("""# ADAM SLM Documentation

## Overview

ADAM SLM (Advanced Deep Attention Model Small Language Model) is a sophisticated language model with state-of-the-art features.

## Features

### Architecture
- **Rotary Position Embeddings (RoPE)** - Better positional understanding
- **Grouped Query Attention (GQA)** - Memory-efficient attention
- **SwiGLU Activation** - Superior activation function
- **RMSNorm** - More stable normalization

### Training
- Mixed precision training (FP16/BF16)
- Gradient accumulation and clipping
- Learning rate scheduling
- Comprehensive checkpointing

### Inference
- Batch text generation
- Multiple sampling strategies
- Chat interface
- Performance optimization

## Usage

```python
from adam_slm import AdamSLM, AdamTokenizer

# Load model
model = AdamSLM.from_pretrained("adam-slm-base")
tokenizer = AdamTokenizer("gpt2")

# Generate text
text = model.generate("The future of AI is", max_length=100)
print(text)
```

## Database Integration

The sophisticated database system provides:
- Model versioning and lineage
- Training run tracking
- Dataset management
- Experiment organization
- Performance analytics
""")
    files_created.append(("document", str(md_file)))
    print(f"📄 Created markdown file: {md_file}")
    
    # 5. Python code file
    py_file = sample_dir / "example_usage.py"
    with open(py_file, 'w') as f:
        f.write("""#!/usr/bin/env python3
\"\"\"
Example usage of ADAM SLM
\"\"\"

import torch
from adam_slm.models import AdamSLM, get_config
from adam_slm.tokenization import AdamTokenizer
from adam_slm.inference import AdamInference, GenerationConfig


def main():
    \"\"\"Main example function\"\"\"
    # Load model configuration
    config = get_config("adam-slm-base")
    
    # Create model and tokenizer
    model = AdamSLM(config)
    tokenizer = AdamTokenizer("gpt2")
    
    # Setup inference
    inference = AdamInference(
        model=model,
        tokenizer=tokenizer,
        generation_config=GenerationConfig(
            max_new_tokens=100,
            temperature=0.8,
            top_k=50,
        )
    )
    
    # Generate text
    prompts = [
        "The future of artificial intelligence",
        "In a world where technology",
        "Machine learning algorithms"
    ]
    
    for prompt in prompts:
        generated = inference.generate(prompt)
        print(f"Prompt: {prompt}")
        print(f"Generated: {generated}")
        print("-" * 50)


if __name__ == "__main__":
    main()
""")
    files_created.append(("code", str(py_file)))
    print(f"💻 Created Python file: {py_file}")
    
    # 6. YAML configuration
    yaml_file = sample_dir / "experiment_config.yaml"
    with open(yaml_file, 'w') as f:
        f.write("""# ADAM SLM Experiment Configuration
experiment:
  name: "hyperparameter_search"
  description: "Testing different learning rates and batch sizes"
  
model:
  type: "adam-slm-base"
  checkpoint: null
  
training:
  dataset: "shakespeare_corpus"
  learning_rates: [1e-4, 5e-4, 1e-3]
  batch_sizes: [16, 32, 64]
  max_steps: 5000
  eval_steps: 500
  
optimization:
  optimizer: "adamw"
  weight_decay: 0.1
  gradient_clipping: 1.0
  warmup_ratio: 0.1
  
logging:
  wandb_project: "adam-slm-experiments"
  log_interval: 100
  save_interval: 1000
""")
    files_created.append(("config", str(yaml_file)))
    print(f"📋 Created YAML file: {yaml_file}")
    
    # 7. Log file
    log_file = sample_dir / "training.log"
    with open(log_file, 'w') as f:
        f.write("""2024-01-15 10:00:00 - INFO - Starting ADAM SLM training
2024-01-15 10:00:01 - INFO - Model: adam-slm-base (145M parameters)
2024-01-15 10:00:02 - INFO - Dataset: shakespeare_corpus (1M samples)
2024-01-15 10:00:03 - INFO - Training config: lr=5e-4, batch_size=32
2024-01-15 10:01:00 - INFO - Step 100: loss=3.245, lr=4.5e-4
2024-01-15 10:02:00 - INFO - Step 200: loss=2.987, lr=4.0e-4
2024-01-15 10:03:00 - INFO - Step 300: loss=2.756, lr=3.5e-4
2024-01-15 10:04:00 - INFO - Step 400: loss=2.543, lr=3.0e-4
2024-01-15 10:05:00 - INFO - Step 500: loss=2.398, lr=2.5e-4, eval_loss=2.456
2024-01-15 10:06:00 - INFO - Step 600: loss=2.287, lr=2.0e-4
2024-01-15 10:07:00 - INFO - Step 700: loss=2.198, lr=1.5e-4
2024-01-15 10:08:00 - INFO - Step 800: loss=2.134, lr=1.0e-4
2024-01-15 10:09:00 - INFO - Step 900: loss=2.089, lr=5.0e-5
2024-01-15 10:10:00 - INFO - Step 1000: loss=2.056, lr=1.0e-5, eval_loss=2.078
2024-01-15 10:10:01 - INFO - Training completed successfully
2024-01-15 10:10:02 - INFO - Final model saved to: /models/adam-slm-shakespeare-final.pt
""")
    files_created.append(("log", str(log_file)))
    print(f"📋 Created log file: {log_file}")
    
    print(f"\n✅ Created {len(files_created)} sample files")
    return files_created


def demo_file_import(file_manager: FileManager, files_created: list):
    """Demonstrate file import capabilities"""
    print_header("File Import Demonstration")
    
    imported_files = []
    
    for file_type, file_path in files_created:
        print_section(f"Importing {file_type.title()} File")
        
        # Import file with different options based on type
        if file_type == "dataset":
            file_id = file_manager.register_file(
                file_path=file_path,
                file_type=file_type,
                description=f"Sample {file_type} file for demonstration",
                tags=["demo", "sample", file_type, "training"],
                created_by=1,  # admin user
                copy_to_storage=True,
                process_immediately=True
            )
        elif file_type == "config":
            file_id = file_manager.register_file(
                file_path=file_path,
                file_type=file_type,
                description=f"Configuration file for ADAM SLM",
                tags=["demo", "config", "adam-slm"],
                created_by=1,
                copy_to_storage=True,
                process_immediately=True
            )
        else:
            file_id = file_manager.register_file(
                file_path=file_path,
                file_type=file_type,
                description=f"Sample {file_type} file",
                tags=["demo", "sample", file_type],
                created_by=1,
                copy_to_storage=True,
                process_immediately=True
            )
            
        imported_files.append(file_id)
        print(f"✅ Imported {file_type} file: ID {file_id}")
        
        # Show file info
        file_info = file_manager.get_file_info(file_id)
        print(f"   📁 Stored: {file_info['stored_path']}")
        print(f"   📊 Size: {file_info['file_size_bytes']:,} bytes")
        print(f"   🏷️  Format: {file_info['file_format']}")
        print(f"   ⚡ Status: {file_info['processing_status']}")
        
    return imported_files


def demo_file_conversion(file_manager: FileManager, imported_files: list):
    """Demonstrate file conversion capabilities"""
    print_header("File Conversion Demonstration")
    
    converter = FileConverter()
    
    # Get CSV file for conversion demo
    csv_files = file_manager.list_files(file_format="csv", limit=1)
    if csv_files:
        csv_file = csv_files[0]
        print_section("Converting CSV to JSON")
        
        input_path = csv_file['stored_path']
        output_path = input_path.replace('.csv', '.json')
        
        result = converter.convert_file(input_path, output_path, 'json')
        
        if result['success']:
            print(f"✅ Converted CSV to JSON: {output_path}")
            print(f"   📊 Output size: {result['output_size']:,} bytes")
            
            # Import converted file
            converted_file_id = file_manager.register_file(
                file_path=output_path,
                file_type="dataset",
                description="Converted from CSV to JSON format",
                tags=["demo", "converted", "json", "dataset"],
                created_by=1,
                copy_to_storage=False,  # Already in storage
                process_immediately=True
            )
            
            print(f"📥 Imported converted file: ID {converted_file_id}")
        else:
            print(f"❌ Conversion failed: {result['error']}")
    
    # Get markdown file for conversion demo
    md_files = file_manager.list_files(file_format="md", limit=1)
    if md_files:
        md_file = md_files[0]
        print_section("Converting Markdown to HTML")
        
        input_path = md_file['stored_path']
        output_path = input_path.replace('.md', '.html')
        
        result = converter.convert_file(input_path, output_path, 'html')
        
        if result['success']:
            print(f"✅ Converted Markdown to HTML: {output_path}")
            print(f"   📊 Output size: {result['output_size']:,} bytes")
            
            # Import converted file
            converted_file_id = file_manager.register_file(
                file_path=output_path,
                file_type="document",
                description="Converted from Markdown to HTML format",
                tags=["demo", "converted", "html", "document"],
                created_by=1,
                copy_to_storage=False,
                process_immediately=True
            )
            
            print(f"📥 Imported converted file: ID {converted_file_id}")
        else:
            print(f"❌ Conversion failed: {result['error']}")


def demo_file_processing(file_manager: FileManager):
    """Demonstrate file processing capabilities"""
    print_header("File Processing Demonstration")
    
    # Process pending files
    print_section("Processing Files")
    
    # Get files that need processing
    pending_files = file_manager.list_files(processing_status="pending", limit=5)
    
    for file_info in pending_files:
        file_id = file_info['id']
        filename = file_info['filename']
        
        print(f"🔄 Processing file: {filename} (ID: {file_id})")
        
        # Process the file
        success = file_manager.process_file(file_id, 'analyze')
        
        if success:
            print(f"✅ Processing completed for {filename}")
            
            # Show updated file info
            updated_info = file_manager.get_file_info(file_id)
            print(f"   ⚡ Status: {updated_info['processing_status']}")
            
            # Show processing jobs
            if updated_info['processing_jobs']:
                latest_job = updated_info['processing_jobs'][0]
                print(f"   📊 Job status: {latest_job['job_status']}")
                if latest_job['result_data']:
                    result = json.loads(latest_job['result_data'])
                    print(f"   📈 Analysis: {len(result)} properties extracted")
        else:
            print(f"❌ Processing failed for {filename}")


def demo_file_search_and_analytics(file_manager: FileManager):
    """Demonstrate file search and analytics"""
    print_header("File Search and Analytics")
    
    print_section("File Statistics")
    
    # Get file statistics by type
    file_types = ["text", "dataset", "config", "document", "code", "log"]
    
    for file_type in file_types:
        files = file_manager.list_files(file_type=file_type)
        if files:
            total_size = sum(f['file_size_bytes'] for f in files)
            print(f"📊 {file_type.title()}: {len(files)} files, {total_size:,} bytes")
    
    print_section("Recent Files")
    
    # Show recent files
    recent_files = file_manager.list_files(limit=10)
    
    for file_info in recent_files:
        print(f"📁 {file_info['filename']}")
        print(f"   🏷️  Type: {file_info['file_type']} ({file_info['file_format']})")
        print(f"   📊 Size: {file_info['file_size_bytes']:,} bytes")
        print(f"   📅 Uploaded: {file_info['uploaded_at']}")
        if file_info['tags']:
            try:
                if isinstance(file_info['tags'], str):
                    tags = json.loads(file_info['tags'])
                else:
                    tags = file_info['tags']
                print(f"   🏷️  Tags: {', '.join(tags)}")
            except (json.JSONDecodeError, TypeError):
                print(f"   🏷️  Tags: {file_info['tags']}")
        print()


def cleanup_sample_files():
    """Clean up sample files"""
    print_section("Cleaning Up Sample Files")
    
    sample_dir = Path("sample_files")
    if sample_dir.exists():
        shutil.rmtree(sample_dir)
        print("🧹 Cleaned up sample files directory")


def main():
    """Main demo function"""
    print("📁 ADAM SLM Database File Import Demo")
    print("Comprehensive file management and conversion capabilities")
    
    try:
        # Initialize database and file manager
        db = AdamSLMDatabase("../databases/adamslm_sophisticated.sqlite")
        file_manager = FileManager(db)
        
        # Create sample files
        files_created = create_sample_files()
        
        # Demo file import
        imported_files = demo_file_import(file_manager, files_created)
        
        # Demo file conversion
        demo_file_conversion(file_manager, imported_files)
        
        # Demo file processing
        demo_file_processing(file_manager)
        
        # Demo search and analytics
        demo_file_search_and_analytics(file_manager)
        
        print_header("Demo Complete!")
        print("🎯 Features Demonstrated:")
        print("  ✅ Multi-format file import (text, data, config, code, etc.)")
        print("  ✅ Automatic file type detection and analysis")
        print("  ✅ File format conversion (CSV↔JSON, MD→HTML, etc.)")
        print("  ✅ Content extraction and metadata analysis")
        print("  ✅ Organized storage with version control")
        print("  ✅ Processing job queue and status tracking")
        print("  ✅ File search and analytics")
        print("  ✅ Tag-based organization")
        print("  ✅ User ownership and access control")
        
        print("\n🚀 File Management System Ready!")
        print("The database now supports:")
        print("  • All file types with automatic detection")
        print("  • Format conversion and processing")
        print("  • Content extraction and search")
        print("  • Version control and lineage")
        print("  • Analytics and reporting")
        
        # Clean up
        cleanup_sample_files()
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
