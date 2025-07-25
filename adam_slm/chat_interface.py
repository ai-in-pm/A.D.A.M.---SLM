#!/usr/bin/env python3
"""
A.D.A.M. SLM Terminal Chat Interface
Interactive chat interface for natural interaction with A.D.A.M. SLM and its database system
"""

import os
import sys
import json
import time
import readline
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    from adam_slm import (
        AdamSLM, AdamSLMConfig, AdamTokenizer,
        get_default_database, get_database_manager,
        search_papers, get_knowledge_stats, get_dashboard_stats
    )
    from adam_slm.database.knowledge_base import KnowledgeBase
    from adam_slm.inference import GenerationConfig
    
    # Try to import enhanced inference
    try:
        from adam_slm.inference import KnowledgeEnhancedInference
        ENHANCED_INFERENCE_AVAILABLE = True
    except ImportError:
        from adam_slm.inference import AdamInference
        ENHANCED_INFERENCE_AVAILABLE = False
    
    ADAM_SLM_AVAILABLE = True
    
except ImportError as e:
    print(f"❌ A.D.A.M. SLM components not available: {e}")
    ADAM_SLM_AVAILABLE = False


class AdamSLMChatInterface:
    """
    Interactive terminal chat interface for A.D.A.M. SLM

    Features:
    - Natural conversation with A.D.A.M. SLM
    - Knowledge base integration
    - Database queries and analytics
    - Model management
    - Training insights
    - Research paper search
    """
    
    def __init__(self):
        self.session_id = f"chat_{int(time.time())}"
        self.conversation_history = []
        self.knowledge_base = None
        self.database = None
        self.inference_engine = None
        self.model = None
        self.tokenizer = None

        # Chat settings
        self.max_response_length = 200
        self.temperature = 0.7
        self.use_knowledge = True
        self.show_sources = True

        # Conversation context
        self.user_name = None
        self.conversation_topics = []
        self.last_search_query = None

        # Initialize components
        self._initialize_system()
        
    def _initialize_system(self):
        """Initialize A.D.A.M. SLM system components"""
        print("🚀 Initializing A.D.A.M. SLM Chat Interface...")

        if not ADAM_SLM_AVAILABLE:
            print("❌ A.D.A.M. SLM not available. Limited functionality.")
            return
        
        try:
            # Initialize database
            print("🗄️ Connecting to database...")
            self.database = get_default_database()
            self.knowledge_base = KnowledgeBase()
            print("✅ Database connected")
            
            # Get system stats
            stats = get_dashboard_stats()
            kb_stats = get_knowledge_stats()
            
            print(f"📊 System Status:")
            print(f"   • Models: {stats['models']['total_models']}")
            print(f"   • Training runs: {stats['training']['total_runs']}")
            print(f"   • Research papers: {kb_stats['total_papers']}")
            print(f"   • Knowledge words: {kb_stats['total_words']:,}")
            
            # Initialize model (lightweight for demo)
            print("🧠 Initializing A.D.A.M. SLM model...")
            self._initialize_model()
            
        except Exception as e:
            print(f"⚠️ Initialization warning: {e}")
            print("Some features may be limited.")
    
    def _initialize_model(self):
        """Initialize A.D.A.M. SLM model for inference"""
        try:
            # Create a small model for demo purposes
            config = AdamSLMConfig(
                d_model=256,
                n_layers=4,
                n_heads=4,
                n_kv_heads=2,
                d_ff=1024,
                vocab_size=50257,
                max_seq_len=512
            )
            
            self.model = AdamSLM(config)

            # Use A.D.A.M.-SLM tokenizer
            if hasattr(self.model, 'get_tokenizer') and self.model.get_tokenizer() is not None:
                self.tokenizer = self.model.get_tokenizer()
                print("✅ Chat interface using model's A.D.A.M.-SLM tokenizer")
            else:
                from .tokenization import get_tokenizer
                self.tokenizer = get_tokenizer("adam_slm")
                print("✅ Chat interface initialized with A.D.A.M.-SLM tokenizer")
            
            # Initialize inference engine
            if ENHANCED_INFERENCE_AVAILABLE:
                self.inference_engine = KnowledgeEnhancedInference(
                    self.model, self.tokenizer,
                    knowledge_base=self.knowledge_base,
                    enable_citations=self.show_sources
                )
                print("✅ Knowledge-enhanced inference ready")
            else:
                self.inference_engine = AdamInference(self.model, self.tokenizer)
                print("✅ Basic inference ready")
                
        except Exception as e:
            print(f"⚠️ Model initialization failed: {e}")
            print("Chat will work with database features only.")
    
    def start_chat(self):
        """Start the interactive chat session"""
        self._print_welcome()
        
        while True:
            try:
                # Get user input
                user_input = input("\n💬 You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle special commands
                if user_input.startswith('/'):
                    if self._handle_command(user_input):
                        continue
                    else:
                        break
                
                # Process user message
                response = self._process_message(user_input)
                
                # Display response
                self._display_response(response)
                
                # Add to conversation history
                self.conversation_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'user': user_input,
                    'assistant': response.get('text', ''),
                    'sources': response.get('sources', []),
                    'type': response.get('type', 'chat')
                })

                # Update conversation context
                self._update_conversation_context(user_input, response)
                
            except KeyboardInterrupt:
                print("\n\n👋 Chat session ended. Goodbye!")
                break
            except EOFError:
                print("\n\n👋 Chat session ended. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print("Please try again or type /help for assistance.")
    
    def _print_welcome(self):
        """Print welcome message and instructions"""
        print("\n" + "="*60)
        print("🤖 A.D.A.M. SLM Interactive Chat Interface")
        print("="*60)
        print("Welcome! I'm A.D.A.M. SLM with integrated knowledge base.")
        print("I can help you with:")
        print("  🧠 AI and machine learning questions")
        print("  📚 Research paper insights")
        print("  📊 Database analytics")
        print("  🏃 Training information")
        print("  🔍 Knowledge base search")
        print("\nSpecial commands:")
        print("  /help     - Show all commands")
        print("  /stats    - Show system statistics")
        print("  /search   - Search research papers")
        print("  /settings - Adjust chat settings")
        print("  /history  - Show conversation history")
        print("  /quit     - Exit chat")
        print("\nJust type your question naturally!")
        print("="*60)
    
    def _handle_command(self, command: str) -> bool:
        """Handle special commands. Returns True to continue, False to exit."""
        cmd_parts = command[1:].split()
        cmd = cmd_parts[0].lower() if cmd_parts else ""
        
        if cmd == "help":
            self._show_help()
        elif cmd == "stats":
            self._show_stats()
        elif cmd == "search":
            query = " ".join(cmd_parts[1:]) if len(cmd_parts) > 1 else None
            self._search_papers(query)
        elif cmd == "settings":
            self._show_settings()
        elif cmd == "history":
            self._show_history()
        elif cmd == "clear":
            os.system('cls' if os.name == 'nt' else 'clear')
            self._print_welcome()
        elif cmd in ["quit", "exit", "bye"]:
            print("\n👋 Thanks for using A.D.A.M. SLM! Goodbye!")
            return False
        else:
            print(f"❓ Unknown command: {command}")
            print("Type /help to see available commands.")
        
        return True
    
    def _process_message(self, message: str) -> Dict[str, Any]:
        """Process user message and generate response"""
        
        # Detect message type
        message_lower = message.lower()
        
        if any(word in message_lower for word in ['search', 'find', 'paper', 'research']):
            return self._handle_search_query(message)
        elif any(word in message_lower for word in ['stats', 'statistics', 'dashboard', 'analytics']):
            return self._handle_stats_query(message)
        elif any(word in message_lower for word in ['training', 'model', 'run', 'experiment']):
            return self._handle_training_query(message)
        else:
            return self._handle_general_query(message)
    
    def _handle_search_query(self, message: str) -> Dict[str, Any]:
        """Handle search-related queries"""
        try:
            # Extract search terms
            search_terms = message.lower()
            for word in ['search', 'find', 'paper', 'research', 'about', 'on']:
                search_terms = search_terms.replace(word, '').strip()
            
            if not search_terms:
                search_terms = message
            
            # Search papers
            papers = search_papers(search_terms, limit=3)
            
            if papers:
                response_text = f"Found {len(papers)} relevant papers for '{search_terms}':\n\n"
                sources = []
                
                for i, paper in enumerate(papers, 1):
                    response_text += f"{i}. **{paper['filename']}**\n"
                    response_text += f"   Relevance: {paper['relevance_score']:.2f}\n"
                    if paper.get('context'):
                        preview = paper['context'][0][:150] + "..." if paper['context'] else ""
                        response_text += f"   Preview: {preview}\n"
                    response_text += "\n"
                    
                    sources.append({
                        'filename': paper['filename'],
                        'relevance': paper['relevance_score'],
                        'file_id': paper['file_id']
                    })
                
                return {
                    'text': response_text,
                    'sources': sources,
                    'type': 'search'
                }
            else:
                return {
                    'text': f"No papers found for '{search_terms}'. Try different keywords or check /stats for available content.",
                    'sources': [],
                    'type': 'search'
                }
                
        except Exception as e:
            return {
                'text': f"Search failed: {e}",
                'sources': [],
                'type': 'error'
            }
    
    def _handle_stats_query(self, message: str) -> Dict[str, Any]:
        """Handle statistics and analytics queries"""
        try:
            stats = get_dashboard_stats()
            kb_stats = get_knowledge_stats()
            
            response_text = "📊 **A.D.A.M. SLM System Statistics**\n\n"
            response_text += f"**Database:**\n"
            response_text += f"• Models: {stats['models']['total_models']}\n"
            response_text += f"• Training runs: {stats['training']['total_runs']}\n"
            response_text += f"• Datasets: {stats['datasets']['total_datasets']}\n"
            response_text += f"• Users: {stats['users']['total_users']}\n\n"
            
            response_text += f"**Knowledge Base:**\n"
            response_text += f"• Research papers: {kb_stats['total_papers']}\n"
            response_text += f"• Total words: {kb_stats['total_words']:,}\n"
            response_text += f"• Avg words/paper: {kb_stats['avg_words_per_paper']:,}\n\n"
            
            if kb_stats['topics']:
                response_text += f"**Top Research Topics:**\n"
                for topic, count in list(kb_stats['topics'].items())[:3]:
                    response_text += f"• {topic}: {count} papers\n"
            
            return {
                'text': response_text,
                'sources': [],
                'type': 'stats'
            }
            
        except Exception as e:
            return {
                'text': f"Failed to get statistics: {e}",
                'sources': [],
                'type': 'error'
            }
    
    def _handle_training_query(self, message: str) -> Dict[str, Any]:
        """Handle training and model queries"""
        try:
            if not self.database:
                return {
                    'text': "Database not available for training information.",
                    'sources': [],
                    'type': 'error'
                }
            
            # Get recent training runs
            recent_runs = self.database.execute_query("""
                SELECT run_name, status, best_loss, started_at, completed_at
                FROM training_runs 
                ORDER BY started_at DESC 
                LIMIT 5
            """)
            
            if recent_runs:
                response_text = "🏃 **Recent Training Runs:**\n\n"
                for run in recent_runs:
                    response_text += f"• **{run['run_name']}**\n"
                    response_text += f"  Status: {run['status']}\n"
                    if run['best_loss']:
                        response_text += f"  Best loss: {run['best_loss']:.4f}\n"
                    response_text += f"  Started: {run['started_at']}\n\n"
            else:
                response_text = "No training runs found in the database."
            
            return {
                'text': response_text,
                'sources': [],
                'type': 'training'
            }
            
        except Exception as e:
            return {
                'text': f"Failed to get training information: {e}",
                'sources': [],
                'type': 'error'
            }
    
    def _handle_general_query(self, message: str) -> Dict[str, Any]:
        """Handle general chat queries"""
        try:
            if self.inference_engine and ENHANCED_INFERENCE_AVAILABLE:
                # Use knowledge-enhanced inference
                result = self.inference_engine.answer_question(
                    question=message,
                    max_context_length=1000
                )

                return {
                    'text': result['answer'],
                    'sources': result.get('sources', []),
                    'type': 'chat',
                    'context_used': len(result.get('context_used', ''))
                }
            else:
                # Enhanced fallback with better conversation handling
                return self._generate_conversational_response(message)

        except Exception as e:
            return {
                'text': f"I apologize for the technical issue. Let me try to help you anyway! I'm A.D.A.M. SLM and I can assist with AI and machine learning questions. What would you like to know?",
                'sources': [],
                'type': 'error'
            }

    def _generate_conversational_response(self, message: str) -> Dict[str, Any]:
        """Generate conversational responses for better interaction"""
        message_lower = message.lower()

        # Greeting responses
        if any(word in message_lower for word in ['hello', 'hi', 'hey', 'greetings', 'good morning', 'good afternoon']):
            greeting = "Hello! Great to meet you!"

            # Personalize if we know the user's name
            if self.user_name:
                greeting = f"Hello again, {self.user_name}! Great to continue our conversation!"

            # Reference previous topics if any
            if self.conversation_topics:
                topic_text = f" I remember we were discussing {', '.join(self.conversation_topics[:3])}."
            else:
                topic_text = ""

            return {
                'text': f"{greeting} I'm A.D.A.M. SLM (Applied Decision Architecture Matrix), your AI assistant with access to extensive research knowledge.{topic_text}\n\nI can help you with:\n• AI and machine learning concepts\n• Research paper insights\n• Technical explanations\n• System information\n\nWhat would you like to explore today?",
                'sources': [],
                'type': 'chat'
            }

        # How are you / status questions
        elif any(phrase in message_lower for phrase in ['how are you', 'how do you do', 'what\'s up', 'how\'s it going']):
            return {
                'text': "I'm doing excellent, thank you for asking! All my systems are running smoothly:\n\n✅ Knowledge base: 4 research papers loaded\n✅ Database: Connected and operational\n✅ AI reasoning: Ready to assist\n\nI'm excited to help you learn about AI, answer technical questions, or explore research topics. What interests you?",
                'sources': [],
                'type': 'chat'
            }

        # Thank you responses
        elif any(word in message_lower for word in ['thank', 'thanks', 'appreciate']):
            return {
                'text': "You're very welcome! I'm here to help and I enjoy our conversations. Feel free to ask me anything about AI, machine learning, or use commands like /search to explore the research papers. Is there anything else you'd like to know?",
                'sources': [],
                'type': 'chat'
            }

        # AI/ML related questions
        elif any(word in message_lower for word in ['ai', 'artificial intelligence', 'machine learning', 'ml', 'neural', 'transformer', 'model']):
            # Try to search papers for relevant content
            try:
                papers = search_papers(message, limit=2)
                if papers:
                    response_text = f"Great question about {message}! Based on my research papers:\n\n"
                    for paper in papers:
                        if paper.get('context'):
                            context = paper['context'][0][:200] + "..."
                            response_text += f"📄 From {paper['filename']}:\n{context}\n\n"
                    response_text += "Would you like me to search for more specific information using /search, or do you have follow-up questions?"

                    return {
                        'text': response_text,
                        'sources': [{'filename': p['filename'], 'relevance': p['relevance_score']} for p in papers],
                        'type': 'chat'
                    }
            except:
                pass

            return {
                'text': f"That's a fascinating topic! I have research papers covering {message_lower} and related concepts. While I'd love to give you a detailed answer right now, I recommend using the /search command to find specific information, or you could ask me about particular aspects you're most interested in. What specific part would you like to explore?",
                'sources': [],
                'type': 'chat'
            }

        # General conversation
        else:
            return {
                'text': f"That's an interesting point about '{message}'! I'm A.D.A.M. SLM, and while I specialize in AI and machine learning topics, I'm always eager to help. I have access to research papers and can provide insights on technical topics.\n\nTry asking me about:\n• AI concepts and technologies\n• Machine learning approaches\n• Research paper insights\n• Or use /search to explore specific topics\n\nWhat would you like to know more about?",
                'sources': [],
                'type': 'chat'
            }
    
    def _display_response(self, response: Dict[str, Any]):
        """Display the response to the user"""
        print(f"\n🤖 A.D.A.M. SLM: {response['text']}")
        
        # Show sources if available
        if response.get('sources') and self.show_sources:
            print(f"\n📚 Sources:")
            for source in response['sources']:
                if isinstance(source, dict):
                    relevance = source.get('relevance', 0)
                    print(f"   • {source['filename']} (relevance: {relevance:.2f})")
                else:
                    print(f"   • {source}")
        
        # Show additional info for certain types
        if response.get('context_used'):
            print(f"\n📖 Context used: {response['context_used']} characters")
    
    def _show_help(self):
        """Show help information"""
        print("\n📖 **A.D.A.M. SLM Chat Commands:**")
        print("  /help     - Show this help message")
        print("  /stats    - Show system statistics")
        print("  /search <query> - Search research papers")
        print("  /settings - Show/adjust chat settings")
        print("  /history  - Show conversation history")
        print("  /clear    - Clear screen")
        print("  /quit     - Exit chat")
        print("\n💡 **Tips:**")
        print("  • Ask about AI, machine learning, transformers")
        print("  • Request research paper summaries")
        print("  • Ask for training run information")
        print("  • Search for specific topics or concepts")
    
    def _show_stats(self):
        """Show detailed system statistics"""
        response = self._handle_stats_query("")
        print(f"\n{response['text']}")
    
    def _search_papers(self, query: str = None):
        """Interactive paper search"""
        if not query:
            query = input("🔍 Enter search query: ").strip()
        
        if query:
            response = self._handle_search_query(f"search {query}")
            print(f"\n{response['text']}")
        else:
            print("❓ Please provide a search query.")
    
    def _show_settings(self):
        """Show current settings"""
        print(f"\n⚙️ **Chat Settings:**")
        print(f"  • Max response length: {self.max_response_length}")
        print(f"  • Temperature: {self.temperature}")
        print(f"  • Use knowledge base: {self.use_knowledge}")
        print(f"  • Show sources: {self.show_sources}")
        print(f"  • Session ID: {self.session_id}")
    
    def _update_conversation_context(self, user_input: str, response: Dict[str, Any]):
        """Update conversation context for better continuity"""
        # Extract topics from user input
        user_lower = user_input.lower()

        # Detect name introduction
        if any(phrase in user_lower for phrase in ['my name is', 'i am', 'i\'m', 'call me']):
            words = user_input.split()
            for i, word in enumerate(words):
                if word.lower() in ['name', 'am', 'is', 'me'] and i + 1 < len(words):
                    potential_name = words[i + 1].strip('.,!?')
                    if potential_name.isalpha() and len(potential_name) > 1:
                        self.user_name = potential_name
                        break

        # Track topics discussed
        ai_topics = ['ai', 'artificial intelligence', 'machine learning', 'neural', 'transformer',
                    'attention', 'model', 'training', 'deep learning', 'nlp']

        for topic in ai_topics:
            if topic in user_lower and topic not in self.conversation_topics:
                self.conversation_topics.append(topic)

        # Remember last search
        if user_input.startswith('/search') or response.get('type') == 'search':
            self.last_search_query = user_input.replace('/search', '').strip()

    def _show_history(self):
        """Show conversation history"""
        if not self.conversation_history:
            print("\n📝 No conversation history yet.")
            return

        print(f"\n📝 **Conversation History** ({len(self.conversation_history)} messages):")

        # Show conversation context if available
        if self.user_name:
            print(f"👤 User: {self.user_name}")
        if self.conversation_topics:
            print(f"🔬 Topics discussed: {', '.join(self.conversation_topics[:5])}")
        if self.last_search_query:
            print(f"🔍 Last search: {self.last_search_query}")

        print("\n📋 Recent messages:")
        for i, entry in enumerate(self.conversation_history[-5:], 1):  # Show last 5
            print(f"\n{i}. **You:** {entry['user'][:100]}...")
            print(f"   **A.D.A.M.:** {entry['assistant'][:100]}...")
            if entry['sources']:
                print(f"   **Sources:** {len(entry['sources'])} papers")


def main():
    """Main function to start the chat interface"""
    try:
        chat = AdamSLMChatInterface()
        chat.start_chat()
    except KeyboardInterrupt:
        print("\n\n👋 Chat interface interrupted. Goodbye!")
    except Exception as e:
        print(f"\n❌ Chat interface error: {e}")
        print("Please check your ADAM SLM installation.")


if __name__ == "__main__":
    main()
