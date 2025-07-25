#!/usr/bin/env python3
"""
A.D.A.M. SLM Chat Launcher
Simple launcher for the A.D.A.M. SLM chat interface
"""

import os
import sys

def main():
    """Launch the ADAM SLM chat interface"""
    try:
        # Import and run the chat interface
        from chat_interface import AdamSLMChatInterface
        
        print("🚀 Starting ADAM SLM Chat Interface...")
        chat = AdamSLMChatInterface()
        chat.start_chat()
        
    except ImportError as e:
        print(f"❌ Failed to import chat interface: {e}")
        print("Make sure you're running from the adam_slm directory.")
        return 1
    except KeyboardInterrupt:
        print("\n\n👋 Chat session ended. Goodbye!")
        return 0
    except Exception as e:
        print(f"❌ Error starting chat: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
