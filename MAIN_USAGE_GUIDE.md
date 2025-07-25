# A.D.A.M. SLM Main.py Usage Guide

## 🎯 **Your Gateway to A.D.A.M. SLM!**

The `main.py` file is your **primary entry point** to interact with A.D.A.M. SLM (Applied Decision Architecture Matrix - Small Language Model). It provides multiple ways to access the system's capabilities.

## 🚀 **Quick Start - Chat with A.D.A.M.**

### **Simplest Way to Chat:**
```bash
python main.py
```
This automatically starts the interactive chat interface where you can have natural conversations with A.D.A.M. SLM!

## 📋 **All Available Options**

### 💬 **Chat Interface** (Default)
```bash
python main.py                # Start interactive chat (default)
python main.py --chat         # Explicitly start chat interface
python main.py -c             # Short form
```

**What you get:**
- 🤖 **Natural conversation** with A.D.A.M. SLM
- 📚 **Knowledge-enhanced responses** using 70,000+ words of research
- 🔍 **Research paper search** with `/search` command
- 📊 **System statistics** with `/stats` command
- 🎮 **Special commands** like `/help`, `/history`, `/settings`

### 📊 **System Information**
```bash
python main.py --info         # Show detailed system information
python main.py -i             # Short form
```

**What you get:**
- 🗄️ **Database statistics** (models, training runs, datasets)
- 📚 **Knowledge base stats** (papers, words, topics)
- 🔬 **Research topics** available
- 🚀 **System status** overview

### 🧪 **System Testing**
```bash
python main.py --test         # Run integration test
python main.py -t             # Short form
```

**What you get:**
- ✅ **Component verification** (imports, database, knowledge base)
- 📊 **System health check**
- 🔍 **Integration testing** results
- 🎯 **Pass/fail status** for all components

### 🎬 **Demonstration**
```bash
python main.py --demo         # Run A.D.A.M. SLM demonstration
python main.py -d             # Short form
```

**What you get:**
- 🎯 **Feature showcase** without interaction
- 📚 **Knowledge base overview**
- 🗄️ **Database capabilities** demonstration
- 💡 **Usage examples** and tips

### 🔍 **System Check**
```bash
python main.py --check        # Check if system is working
```

**What you get:**
- ✅ **Component availability** check
- 🗄️ **Database connection** verification
- 📚 **Knowledge base** status
- ⚠️ **Issue identification** if any

### 📋 **Version & Help**
```bash
python main.py --version      # Show version information
python main.py -v             # Short form
python main.py --help         # Show all options
python main.py -h             # Short form
```

## 🎮 **Interactive Chat Commands**

Once you start the chat interface, you can use these special commands:

| Command | Description | Example |
|---------|-------------|---------|
| `/help` | Show all available commands | `/help` |
| `/stats` | Display system statistics | `/stats` |
| `/search <query>` | Search research papers | `/search transformer` |
| `/settings` | Show current chat settings | `/settings` |
| `/history` | View conversation history | `/history` |
| `/clear` | Clear the screen | `/clear` |
| `/quit` | Exit the chat | `/quit` |

## 💬 **Example Chat Session**

```bash
$ python main.py

╔══════════════════════════════════════════════════════════════╗
║     🤖 A.D.A.M. SLM - Applied Decision Architecture Matrix   ║
║                    Small Language Model                      ║
╚══════════════════════════════════════════════════════════════╝

🚀 Starting A.D.A.M. SLM Chat Interface...
✅ Database connected (4 papers, 70,068 words)
✅ Knowledge-enhanced inference ready

💬 You: What is a transformer architecture?
🤖 A.D.A.M. SLM: Based on the research papers, transformers are...
📚 Sources:
   • DeepSeek-v2_Paper.pdf (relevance: 0.85)

💬 You: /search mixture of experts
🤖 A.D.A.M. SLM: Found 2 relevant papers for 'mixture of experts':
1. **DeepSeek-v2_Paper.pdf**
   Relevance: 0.92
   Preview: DeepSeek-V2 employs a mixture-of-experts...

💬 You: /stats
🤖 A.D.A.M. SLM: 📊 **A.D.A.M. SLM System Statistics**
**Database:**
• Models: 2
• Training runs: 5
• Research papers: 4

💬 You: /quit
👋 Thanks for using A.D.A.M. SLM! Goodbye!
```

## 🎯 **Use Cases**

### 🧠 **For AI Research & Learning**
```bash
python main.py
# Ask about: transformers, attention, neural networks, AI history
```

### 📚 **For Knowledge Discovery**
```bash
python main.py
# Use: /search <topic>, explore research papers, get citations
```

### 🔧 **For System Administration**
```bash
python main.py --info    # Check system status
python main.py --test    # Verify everything works
python main.py --check   # Quick health check
```

### 🎬 **For Demonstrations**
```bash
python main.py --demo    # Show capabilities to others
```

## 🛠️ **Troubleshooting**

### **If Chat Doesn't Start:**
```bash
python main.py --check   # Check what's wrong
python main.py --test    # Run full diagnostics
```

### **If Database Issues:**
- Make sure you're in the project root directory: `D:/science_projects/adam_slm`
- Check that `adam_slm_database` directory exists
- Verify database files are present

### **If Import Errors:**
- Ensure you're running from the correct directory
- Check Python path and dependencies
- Try: `python main.py --version` to test basic functionality

## 🚀 **Advanced Usage**

### **Custom Workflows**
```bash
# Quick system check and chat
python main.py --check && python main.py

# Run tests then start demo
python main.py --test && python main.py --demo

# Get info then start chat
python main.py --info && python main.py --chat
```

### **Scripting Integration**
```python
# You can also import and use programmatically
from main import check_system, show_system_info

if check_system():
    show_system_info()
```

## 🎉 **Ready to Chat!**

**Start your conversation with A.D.A.M. SLM:**

```bash
cd D:/science_projects/adam_slm
python main.py
```

### ✅ **What You'll Have Access To:**
- 🤖 **Intelligent AI assistant** with sophisticated reasoning
- 📚 **70,000+ words** of AI research knowledge
- 🗄️ **Enterprise database** with comprehensive analytics
- 🔍 **Full-text search** across research papers
- 💬 **Natural conversation** interface
- 📊 **System insights** and statistics

**A.D.A.M. SLM is ready to assist you with AI research, technical questions, and knowledge discovery!** 🚀✨

---

## 🎯 **Quick Reference**

| What you want to do | Command |
|---------------------|---------|
| **Chat with A.D.A.M.** | `python main.py` |
| **Check if working** | `python main.py --check` |
| **See system info** | `python main.py --info` |
| **Run demonstration** | `python main.py --demo` |
| **Test everything** | `python main.py --test` |
| **Get help** | `python main.py --help` |

**Happy chatting with A.D.A.M. SLM!** 🤖💬
