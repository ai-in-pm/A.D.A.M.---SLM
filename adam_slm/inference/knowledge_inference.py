"""
Knowledge-Enhanced Inference for A.D.A.M. SLM
Integrates research paper knowledge base for enhanced generation
"""

import re
import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from .inference import AdamInference
from .generation import GenerationConfig
from ..database.knowledge_base import KnowledgeBase, search_papers


class KnowledgeEnhancedInference(AdamInference):
    """
    Enhanced A.D.A.M. SLM inference with knowledge base integration

    Features:
    - Automatic knowledge retrieval for prompts
    - Context-aware generation with research paper content
    - Citation and reference integration
    - Knowledge-grounded responses
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        knowledge_base: KnowledgeBase = None,
        max_knowledge_context: int = 2000,
        knowledge_relevance_threshold: float = 0.3,
        enable_citations: bool = True,
        **kwargs
    ):
        super().__init__(model, tokenizer, **kwargs)
        
        # Knowledge base integration
        self.knowledge_base = knowledge_base or KnowledgeBase()
        self.max_knowledge_context = max_knowledge_context
        self.knowledge_relevance_threshold = knowledge_relevance_threshold
        self.enable_citations = enable_citations
        
        # Knowledge retrieval cache
        self.knowledge_cache = {}
        
    def generate_with_knowledge(
        self,
        prompt: str,
        generation_config: GenerationConfig = None,
        retrieve_knowledge: bool = True,
        max_knowledge_results: int = 3,
        knowledge_weight: float = 0.7,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate text with knowledge base enhancement
        
        Args:
            prompt: Input prompt
            generation_config: Generation configuration
            retrieve_knowledge: Whether to retrieve relevant knowledge
            max_knowledge_results: Maximum knowledge snippets to include
            knowledge_weight: Weight for knowledge vs. prompt (0.0-1.0)
            
        Returns:
            Dictionary with generated text, knowledge used, and metadata
        """
        
        # Extract key concepts from prompt for knowledge retrieval
        knowledge_context = ""
        knowledge_sources = []
        
        if retrieve_knowledge:
            knowledge_results = self._retrieve_relevant_knowledge(
                prompt, max_knowledge_results
            )
            
            if knowledge_results:
                knowledge_context, knowledge_sources = self._format_knowledge_context(
                    knowledge_results, knowledge_weight
                )
        
        # Construct enhanced prompt
        enhanced_prompt = self._construct_enhanced_prompt(
            prompt, knowledge_context, knowledge_weight
        )
        
        # Generate with enhanced prompt
        generation_result = self.generate(
            enhanced_prompt,
            generation_config=generation_config,
            **kwargs
        )
        
        # Post-process to add citations if enabled
        if self.enable_citations and knowledge_sources:
            generated_text = self._add_citations(
                generation_result, knowledge_sources
            )
        else:
            generated_text = generation_result
        
        return {
            'generated_text': generated_text,
            'original_prompt': prompt,
            'enhanced_prompt': enhanced_prompt,
            'knowledge_used': knowledge_sources,
            'knowledge_context_length': len(knowledge_context),
            'generation_metadata': {
                'model_name': getattr(self.model, 'name', 'adam-slm'),
                'generation_time': datetime.now().isoformat(),
                'knowledge_enhanced': retrieve_knowledge,
                'num_knowledge_sources': len(knowledge_sources)
            }
        }
    
    def answer_question(
        self,
        question: str,
        context_type: str = "research",
        max_context_length: int = 1500,
        generation_config: GenerationConfig = None
    ) -> Dict[str, Any]:
        """
        Answer a question using knowledge base context

        Args:
            question: Question to answer
            context_type: Type of context to retrieve ('research', 'technical', etc.)
            max_context_length: Maximum context length
            generation_config: Generation configuration

        Returns:
            Answer with supporting knowledge and sources
        """

        try:
            # Search for relevant knowledge
            knowledge_results = self.knowledge_base.search(
                question,
                limit=5,
                content_type='document',
                min_relevance=self.knowledge_relevance_threshold
            )

            if not knowledge_results:
                # Provide a helpful response without knowledge base
                return {
                    'question': question,
                    'answer': self._generate_fallback_response(question),
                    'context_used': "",
                    'sources': [],
                    'context_length': 0,
                    'num_sources': 0,
                    'generation_metadata': {
                        'model_name': getattr(self.model, 'name', 'a.d.a.m.-slm'),
                        'generation_time': datetime.now().isoformat(),
                        'qa_mode': True,
                        'fallback': True
                    }
                }

            # Build context from most relevant sources
            context_parts = []
            sources = []
            total_length = 0

            for result in knowledge_results:
                if total_length >= max_context_length:
                    break

                # Add context from this source
                for context_snippet in result.get('context', []):
                    if total_length + len(context_snippet) <= max_context_length:
                        context_parts.append(context_snippet)
                        total_length += len(context_snippet)

                        if result not in sources:
                            sources.append({
                                'filename': result['filename'],
                                'relevance': result['relevance_score'],
                                'file_id': result['file_id']
                            })

            # Generate response based on available context
            if context_parts:
                context_text = "\n\n".join(context_parts)
                answer = f"Based on the research papers, here's what I found about '{question}':\n\n"
                answer += f"From the available research: {context_parts[0][:300]}..."
            else:
                context_text = ""
                answer = self._generate_fallback_response(question)

            return {
                'question': question,
                'answer': answer,
                'context_used': context_text,
                'sources': sources,
                'context_length': len(context_text),
                'num_sources': len(sources),
                'generation_metadata': {
                    'model_name': getattr(self.model, 'name', 'a.d.a.m.-slm'),
                    'generation_time': datetime.now().isoformat(),
                    'qa_mode': True
                }
            }

        except Exception as e:
            # Fallback error handling
            return {
                'question': question,
                'answer': f"I apologize, but I encountered an issue processing your question. However, I'm here to help! Could you try rephrasing your question or ask about AI, machine learning, transformers, or other topics I have knowledge about?",
                'context_used': "",
                'sources': [],
                'context_length': 0,
                'num_sources': 0,
                'generation_metadata': {
                    'model_name': getattr(self.model, 'name', 'a.d.a.m.-slm'),
                    'generation_time': datetime.now().isoformat(),
                    'qa_mode': True,
                    'error': str(e)
                }
            }

    def _generate_fallback_response(self, question: str) -> str:
        """Generate a helpful fallback response"""
        question_lower = question.lower()

        # Greeting responses
        if any(word in question_lower for word in ['hello', 'hi', 'hey', 'greetings']):
            return "Hello! I'm A.D.A.M. SLM (Applied Decision Architecture Matrix). I'm here to help you with questions about AI, machine learning, transformers, and other technical topics. I have access to research papers and can provide detailed explanations. What would you like to know?"

        # General AI/ML questions
        elif any(word in question_lower for word in ['ai', 'artificial intelligence', 'machine learning', 'ml']):
            return "I'd be happy to help with AI and machine learning questions! I have access to research papers covering topics like transformers, neural networks, and various AI approaches. Could you be more specific about what aspect you're interested in?"

        # Technical questions
        elif any(word in question_lower for word in ['transformer', 'attention', 'neural', 'model', 'training']):
            return "That's a great technical question! I have research papers that cover these topics in detail. While I don't have specific information readily available right now, I can help you search through the research papers using the /search command, or you could ask about specific aspects of these technologies."

        # Default response
        else:
            return "I'm A.D.A.M. SLM, and I'm here to help! I specialize in AI and machine learning topics and have access to research papers covering transformers, neural networks, and various AI approaches. Feel free to ask me about these topics, or use /search to find specific information in the research papers. What would you like to know?"
    
    def summarize_research(
        self,
        topic: str,
        max_papers: int = 5,
        summary_length: str = "medium",
        generation_config: GenerationConfig = None
    ) -> Dict[str, Any]:
        """
        Generate a research summary on a given topic
        
        Args:
            topic: Research topic to summarize
            max_papers: Maximum number of papers to include
            summary_length: "short", "medium", or "long"
            generation_config: Generation configuration
            
        Returns:
            Research summary with paper sources
        """
        
        # Search for papers on the topic
        papers = search_papers(topic, limit=max_papers)
        
        if not papers:
            return {
                'topic': topic,
                'summary': f"No research papers found on the topic: {topic}",
                'papers_found': 0,
                'sources': []
            }
        
        # Extract key information from papers
        paper_summaries = []
        sources = []
        
        for paper in papers:
            # Get paper content preview
            contexts = paper.get('context', [])
            if contexts:
                paper_summary = contexts[0][:300] + "..." if len(contexts[0]) > 300 else contexts[0]
                paper_summaries.append(f"From {paper['filename']}: {paper_summary}")
                
                sources.append({
                    'filename': paper['filename'],
                    'relevance': paper['relevance_score'],
                    'file_id': paper['file_id']
                })
        
        # Determine summary length
        length_instructions = {
            "short": "Provide a concise 2-3 sentence summary.",
            "medium": "Provide a comprehensive paragraph summary.",
            "long": "Provide a detailed multi-paragraph analysis."
        }
        
        length_instruction = length_instructions.get(summary_length, length_instructions["medium"])
        
        # Construct summary prompt
        papers_text = "\n\n".join(paper_summaries)
        summary_prompt = f"""Based on the following research papers about {topic}, {length_instruction}

Research Papers:
{papers_text}

Summary of research on {topic}:"""
        
        # Generate summary
        summary = self.generate(summary_prompt, generation_config=generation_config)
        
        return {
            'topic': topic,
            'summary': summary,
            'papers_found': len(papers),
            'sources': sources,
            'summary_length': summary_length,
            'generation_metadata': {
                'model_name': getattr(self.model, 'name', 'adam-slm'),
                'generation_time': datetime.now().isoformat(),
                'research_summary': True
            }
        }
    
    def _retrieve_relevant_knowledge(
        self,
        prompt: str,
        max_results: int = 3
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant knowledge for a prompt"""
        
        # Extract key terms from prompt
        key_terms = self._extract_key_terms(prompt)
        
        # Search knowledge base
        all_results = []
        for term in key_terms:
            if term in self.knowledge_cache:
                results = self.knowledge_cache[term]
            else:
                results = self.knowledge_base.search(
                    term,
                    limit=max_results,
                    content_type='document',
                    min_relevance=self.knowledge_relevance_threshold
                )
                self.knowledge_cache[term] = results
            
            all_results.extend(results)
        
        # Remove duplicates and sort by relevance
        seen_ids = set()
        unique_results = []
        for result in all_results:
            if result['file_id'] not in seen_ids:
                seen_ids.add(result['file_id'])
                unique_results.append(result)
        
        # Sort by relevance and limit
        unique_results.sort(key=lambda x: x['relevance_score'], reverse=True)
        return unique_results[:max_results]
    
    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms from text for knowledge retrieval"""
        
        # Simple key term extraction
        # In production, this could use more sophisticated NLP
        
        # Remove common words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'can', 'may', 'might', 'must', 'this', 'that', 'these', 'those'
        }
        
        # Extract words
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Filter and return key terms
        key_terms = []
        for word in words:
            if len(word) > 3 and word not in stop_words:
                key_terms.append(word)
        
        # Also look for multi-word technical terms
        technical_patterns = [
            r'\b(?:machine|deep|neural|artificial)\s+(?:learning|network|intelligence)\b',
            r'\b(?:natural|language)\s+(?:processing|model)\b',
            r'\b(?:transformer|attention|embedding)\b',
            r'\b(?:training|inference|generation)\b'
        ]
        
        for pattern in technical_patterns:
            matches = re.findall(pattern, text.lower())
            key_terms.extend(matches)
        
        return list(set(key_terms))[:5]  # Return top 5 unique terms
    
    def _format_knowledge_context(
        self,
        knowledge_results: List[Dict[str, Any]],
        knowledge_weight: float
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Format knowledge results into context string"""
        
        context_parts = []
        sources = []
        total_length = 0
        
        for result in knowledge_results:
            if total_length >= self.max_knowledge_context:
                break
            
            # Add context snippets
            for context in result.get('context', []):
                if total_length + len(context) <= self.max_knowledge_context:
                    context_parts.append(context)
                    total_length += len(context)
            
            # Track source
            sources.append({
                'filename': result['filename'],
                'relevance': result['relevance_score'],
                'file_id': result['file_id']
            })
        
        context_text = "\n\n".join(context_parts)
        return context_text, sources
    
    def _construct_enhanced_prompt(
        self,
        original_prompt: str,
        knowledge_context: str,
        knowledge_weight: float
    ) -> str:
        """Construct enhanced prompt with knowledge context"""
        
        if not knowledge_context:
            return original_prompt
        
        if knowledge_weight > 0.5:
            # Knowledge-heavy prompt
            enhanced_prompt = f"""Based on the following research context:

{knowledge_context}

{original_prompt}"""
        else:
            # Prompt-heavy with knowledge support
            enhanced_prompt = f"""{original_prompt}

Additional context from research:
{knowledge_context}"""
        
        return enhanced_prompt
    
    def _add_citations(
        self,
        generated_text: str,
        knowledge_sources: List[Dict[str, Any]]
    ) -> str:
        """Add citations to generated text"""
        
        if not knowledge_sources:
            return generated_text
        
        # Simple citation addition
        citations = []
        for i, source in enumerate(knowledge_sources, 1):
            citations.append(f"[{i}] {source['filename']}")
        
        citation_text = "\n\nSources:\n" + "\n".join(citations)
        
        return generated_text + citation_text
