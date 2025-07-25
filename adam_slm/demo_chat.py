#!/usr/bin/env python3
"""
A.D.A.M. SLM Chat Interface Demo
Demonstrates the chat interface capabilities without interactive input
"""

import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

def demo_chat_interface():
    """Demonstrate the chat interface capabilities"""
    
    print("🎬 A.D.A.M. SLM Chat Interface Demo")
    print("="*50)
    
    try:
        from adam_slm.chat_interface import AdamSLMChatInterface
        
        # Initialize chat interface
        print("🚀 Initializing chat interface...")
        chat = AdamSLMChatInterface()
        
        print("\n✅ Chat interface initialized successfully!")
        
        # Demo queries
        demo_queries = [
            "What is a transformer architecture?",
            "/stats",
            "/search mixture of experts",
            "How does attention work in neural networks?",
            "Tell me about ELIZA"
        ]
        
        print("\n🎯 Demo Queries:")
        for i, query in enumerate(demo_queries, 1):
            print(f"{i}. {query}")
        
        print("\n💡 To start interactive chat:")
        print("   python adam_slm/chat_interface.py")
        
        print("\n📚 Available Knowledge:")
        if chat.knowledge_base:
            try:
                from adam_slm import get_knowledge_stats
                stats = get_knowledge_stats()
                print(f"   • {stats['total_papers']} research papers")
                print(f"   • {stats['total_words']:,} words of content")
                print(f"   • Topics: {', '.join(list(stats['topics'].keys())[:5])}")
            except:
                print("   • Knowledge base available")
        
        print("\n🗄️ Database Status:")
        if chat.database:
            try:
                from adam_slm import get_dashboard_stats
                stats = get_dashboard_stats()
                print(f"   • {stats['models']['total_models']} models")
                print(f"   • {stats['training']['total_runs']} training runs")
                print(f"   • {stats['datasets']['total_datasets']} datasets")
            except:
                print("   • Database connected")
        
        print("\n🎮 Special Commands:")
        commands = [
            "/help - Show all commands",
            "/stats - System statistics", 
            "/search <query> - Search papers",
            "/settings - Chat settings",
            "/history - Conversation history",
            "/quit - Exit chat"
        ]
        
        for cmd in commands:
            print(f"   • {cmd}")
        
        print("\n🚀 Ready to chat! Start with:")
        print("   python adam_slm/chat_interface.py")
        
        return True
        
    except ImportError as e:
        print(f"❌ Chat interface not available: {e}")
        return False
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return False


def main():
    """Main demo function"""
    success = demo_chat_interface()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
