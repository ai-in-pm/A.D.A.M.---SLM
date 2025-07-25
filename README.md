# A.D.A.M. - Applied Decision Architecture Matrix - Small Language Model

🚀 A sophisticated small language model implementation with state-of-the-art features, built from the ground up with modern transformer architectures and training techniques.

## 🔓 **No API Keys Required!**

**A.D.A.M. SLM is a completely self-contained language model that runs locally on your machine.** You don't need:
- ❌ OpenAI API keys
- ❌ Anthropic API keys
- ❌ Google API keys
- ❌ Any external LLM service subscriptions
- ❌ Internet connection for inference

**✅ Everything runs locally with your own trained models!**

Disclaimer: The development of this GitHub Repository was inspired by "LLMs from Scratch" by Sebastian Raschka. The codebase was forked from the original repository and has been significantly modified and expanded upon. The original repository can be found at https://github.com/rasbt/LLMs-from-scratch. The author of this repository is not affiliated with Sebastian Raschka or the original repository. The author of this repository is Darrell Mesa, and can be contacted at darrell.mesa@pm-ss.org.
## ✨ Features

### 🏗️ Advanced Architecture
- **Rotary Position Embeddings (RoPE)** - Better positional understanding than learned embeddings
- **Grouped Query Attention (GQA)** - Efficient attention mechanism reducing memory usage
- **SwiGLU Activation** - Superior activation function used in modern LLMs
- **RMSNorm** - More stable normalization than LayerNorm
- **KV-Cache Optimization** - Fast inference with key-value caching

### 🎯 Training Features
- **Mixed Precision Training** - FP16/BF16 support for faster training
- **Gradient Accumulation** - Train with larger effective batch sizes
- **Learning Rate Scheduling** - Cosine annealing with warmup
- **Gradient Clipping** - Stable training with gradient norm clipping
- **Advanced Optimizers** - AdamW with proper weight decay handling
- **Checkpointing** - Automatic model saving and resuming

### 🔧 Modern Engineering
- **Modular Design** - Clean, extensible codebase
- **Type Hints** - Full type annotation for better development
- **Configuration Management** - Flexible config system
- **Comprehensive Logging** - Weights & Biases integration
- **Batch Inference** - Efficient batch text generation
- **Model Compilation** - PyTorch 2.0 compile support

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/ai-in-pm/adam-slm.git
cd adam-slm

# Install in development mode
pip install -e .

# Or install from PyPI (when available)
pip install adam-slm
```

### 💬 **Chat with A.D.A.M. (Easiest Way - No API Keys!)**

```bash
# Navigate to project directory
cd D:/science_projects/adam_slm

# Start chatting immediately - completely offline!
python main.py
```

**🔒 Privacy First**: All conversations happen locally on your machine. No data is sent to external services.

### 🎮 **Main.py Usage Options (All Offline)**

```bash
python main.py              # Start interactive chat (default)
python main.py --info       # Show system information
python main.py --demo       # Run demonstration
python main.py --test       # Run integration test
python main.py --check      # Check system status
python main.py --tokenizer  # Test tokenizer system
python main.py --help       # Show all options
```

**🌐 Offline Operation**: All commands work without internet connection or API keys.

### 🧠 **Programmatic Usage**

```python
import torch
from adam_slm.models import AdamSLM, get_config
from adam_slm.tokenization import AdamTokenizer
from adam_slm.inference import AdamInference, GenerationConfig

# Load model configuration
config = get_config("adam-slm-small")

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
text = inference.generate("The future of AI is")
print(text)
```

### Training

```python
from adam_slm.training import AdamTrainer, get_training_config, create_dataloader

# Load training configuration
training_config = get_training_config("small")

# Prepare data
train_loader, eval_loader = create_dataloader(
    texts=["Your training text here..."],
    tokenizer=tokenizer,
    max_length=1024,
    batch_size=32,
    train_test_split=0.1,
)

# Create trainer
trainer = AdamTrainer(
    model=model,
    config=training_config,
    train_dataloader=train_loader,
    eval_dataloader=eval_loader,
    tokenizer=tokenizer,
)

# Start training
trainer.train()
```

## 📊 Model Configurations

| Model | Parameters | Layers | Heads | Embedding Dim | Context Length |
|-------|------------|--------|-------|---------------|----------------|
| adam-slm-small | ~50M | 8 | 8 | 512 | 1024 |
| adam-slm-base | ~124M | 12 | 12 | 768 | 2048 |
| adam-slm-large | ~350M | 24 | 16 | 1024 | 4096 |

## 🎯 Training Example

Train on Shakespeare dataset:

```bash
python examples/train_adam_slm.py \
    --model_size adam-slm-small \
    --training_config small \
    --batch_size 16 \
    --max_steps 10000 \
    --use_wandb
```

## 🏗️ Architecture Details

### Attention Mechanism
- **Grouped Query Attention**: Reduces memory usage by sharing key-value heads
- **Rotary Position Embeddings**: Better handling of positional information
- **Causal Masking**: Proper autoregressive generation

### Feed-Forward Networks
- **SwiGLU Activation**: `swish(W1(x)) * W3(x) -> W2`
- **Gated Linear Units**: Improved information flow
- **Proper Initialization**: Scaled initialization for stable training

### Normalization
- **RMSNorm**: `x / sqrt(mean(x²) + ε) * scale`
- **Pre-normalization**: Applied before attention and FFN layers
- **Stable Training**: Better gradient flow than LayerNorm

## 📈 Performance

### Training Speed
- **Mixed Precision**: Up to 2x faster training
- **Gradient Checkpointing**: Reduced memory usage
- **Efficient Data Loading**: Optimized data pipeline

### Inference Speed
- **KV-Cache**: Faster autoregressive generation
- **Model Compilation**: PyTorch 2.0 compile support
- **Batch Processing**: Efficient batch inference

## 🔒 **Self-Contained & Private**

### 🏠 **Complete Local Operation**
- **No External Dependencies**: Train and run models entirely on your hardware
- **Privacy Guaranteed**: All data processing happens locally
- **Offline Capable**: Works without internet connection
- **No Subscription Fees**: No ongoing costs for API usage
- **Full Control**: You own your models and data

### 🔐 **Data Security**
- **Local Storage**: All models, training data, and conversations stored locally
- **No Telemetry**: No data sent to external servers
- **Air-Gap Compatible**: Can run in completely isolated environments
- **GDPR Compliant**: No external data processing

## 🔧 Advanced Features

### Custom Tokenization (Local Training)
```python
from adam_slm.tokenization import BPETokenizer

# Train custom BPE tokenizer locally
tokenizer = BPETokenizer()
tokenizer.train(texts, vocab_size=32000)
```

### Local Model Optimization
```python
# Optimize for local inference (no API calls)
inference.optimize_for_inference()

# Get local performance stats
stats = inference.get_stats()
print(f"Local tokens/sec: {stats['tokens_per_second']}")
```

### Configuration Management
```python
# Save configuration
config.to_json("model_config.json")

# Load configuration
config = AdamSLMConfig.from_json("model_config.json")
```

## ❓ **Frequently Asked Questions**

### **Q: Do I need OpenAI/Anthropic/Google API keys?**
**A: No!** A.D.A.M. SLM is a completely self-contained system. You train and run your own models locally.

### **Q: Does this require internet connection?**
**A: No!** Once installed, A.D.A.M. SLM works completely offline. All inference and training happens on your local machine.

### **Q: Are there any subscription fees?**
**A: No!** This is open-source software under AGPL-3.0. No ongoing costs, no API usage fees, no subscriptions.

### **Q: What does the AGPL-3.0 license mean for commercial use?**
**A: Commercial use is allowed**, but if you modify the code or use it in a web service, you must share your source code under the same AGPL-3.0 license. This ensures improvements benefit the entire community.

### **Q: Where is my data stored?**
**A: Locally!** All models, training data, conversations, and configurations are stored on your machine.

### **Q: Can I use this in air-gapped environments?**
**A: Yes!** A.D.A.M. SLM is designed to work in completely isolated environments without external dependencies.

## 📚 Documentation

- [Model Architecture](docs/architecture.md)
- [Training Guide](docs/training.md)
- [Inference Guide](docs/inference.md)
- [API Reference](docs/api.md)

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0) - see the [LICENSE](LICENSE) file for details.

**AGPL-3.0 License** - This is a strong copyleft license that requires:
- 🔒 **Source code disclosure**: Any modifications must be shared under the same license
- 🌐 **Network copyleft**: Even web services using this code must provide source code to users
- ⚖️ **Commercial use allowed**: But derivative works must also be AGPL-3.0 licensed
- 📤 **Share improvements**: All enhancements must be contributed back to the community

## 🙏 Acknowledgments

- Built upon concepts from "LLMs from Scratch" by Sebastian Raschka
- Inspired by modern LLM architectures (LLaMA, GPT, etc.)
- Uses tiktoken for tokenization
- Powered by PyTorch

## 📞 Support

- 🐛 [Report Issues](https://github.com/ai-in-pm/adam-slm/issues)
- 💬 [Discussions](https://github.com/ai-in-pm/adam-slm/discussions)
- 📧 Email: darrell.mesa@pm-ss.org

---

**A.D.A.M.** - Where sophistication meets simplicity in language modeling! 🎯

**🔒 Completely Self-Contained • 🌐 No API Keys Required • 🏠 Runs Locally • 🔐 Privacy First**
