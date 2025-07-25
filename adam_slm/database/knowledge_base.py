"""
Knowledge Base Integration for A.D.A.M. SLM
Provides access to research papers and knowledge retrieval
"""

import re
import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from . import get_default_database, get_file_manager, search_knowledge_base


class KnowledgeBase:
    """
    A.D.A.M. SLM Knowledge Base for research paper retrieval and analysis
    """
    
    def __init__(self):
        self.database = get_default_database()
        self.file_manager = get_file_manager()
    
    def search(
        self,
        query: str,
        limit: int = 10,
        content_type: str = None,
        min_relevance: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Search knowledge base for relevant content
        
        Args:
            query: Search query
            limit: Maximum number of results
            content_type: Filter by content type ('research', 'document', etc.)
            min_relevance: Minimum relevance score (0.0-1.0)
            
        Returns:
            List of search results with relevance scores
        """
        
        # Build search query
        search_conditions = ["fc.extracted_text LIKE ?"]
        search_params = [f"%{query}%"]
        
        if content_type:
            search_conditions.append("fr.file_type = ?")
            search_params.append(content_type)
        
        where_clause = " AND ".join(search_conditions)
        
        search_query = f"""
            SELECT fc.file_id, fr.filename, fr.description, fr.tags,
                   fc.word_count, fc.character_count, fc.extraction_method,
                   fc.extracted_text
            FROM file_content fc
            JOIN file_registry fr ON fc.file_id = fr.id
            WHERE {where_clause}
            ORDER BY fc.word_count DESC
            LIMIT ?
        """
        search_params.append(limit)
        
        results = self.database.execute_query(search_query, tuple(search_params))
        
        # Calculate relevance scores and extract context
        processed_results = []
        for result in results:
            relevance_score = self._calculate_relevance(query, result['extracted_text'])
            
            if relevance_score >= min_relevance:
                # Extract relevant context
                context = self._extract_context(query, result['extracted_text'])
                
                # Parse tags
                tags = []
                if result['tags']:
                    try:
                        tags = json.loads(result['tags'])
                    except:
                        tags = []
                
                processed_results.append({
                    'file_id': result['file_id'],
                    'filename': result['filename'],
                    'description': result['description'],
                    'tags': tags,
                    'word_count': result['word_count'],
                    'relevance_score': relevance_score,
                    'context': context,
                    'extraction_method': result['extraction_method']
                })
        
        # Sort by relevance score
        processed_results.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        return processed_results
    
    def get_research_papers(self) -> List[Dict[str, Any]]:
        """Get all research papers in the knowledge base"""
        
        papers_query = """
            SELECT fr.*, fc.word_count, fc.character_count
            FROM file_registry fr
            LEFT JOIN file_content fc ON fr.id = fc.file_id
            WHERE fr.file_type = 'document' 
            AND (fr.tags LIKE '%research%' OR fr.tags LIKE '%ai-paper%')
            ORDER BY fr.uploaded_at DESC
        """
        
        papers = self.database.execute_query(papers_query)
        
        # Process papers
        processed_papers = []
        for paper in papers:
            # Parse metadata
            metadata = {}
            if paper['metadata']:
                try:
                    metadata = json.loads(paper['metadata'])
                except:
                    metadata = {}
            
            # Parse tags
            tags = []
            if paper['tags']:
                try:
                    tags = json.loads(paper['tags'])
                except:
                    tags = []
            
            processed_papers.append({
                'file_id': paper['id'],
                'filename': paper['filename'],
                'title': metadata.get('title', paper['filename']),
                'authors': metadata.get('authors', 'Unknown'),
                'year': metadata.get('year', 'Unknown'),
                'topic': metadata.get('topic', 'General'),
                'description': paper['description'],
                'tags': tags,
                'word_count': paper['word_count'],
                'character_count': paper['character_count'],
                'uploaded_at': paper['uploaded_at'],
                'metadata': metadata
            })
        
        return processed_papers
    
    def get_paper_content(self, file_id: int) -> Optional[str]:
        """Get full content of a research paper"""
        
        content_query = """
            SELECT extracted_text FROM file_content 
            WHERE file_id = ? 
            ORDER BY extracted_at DESC 
            LIMIT 1
        """
        
        result = self.database.execute_query(content_query, (file_id,))
        return result[0]['extracted_text'] if result else None
    
    def find_related_papers(
        self,
        file_id: int,
        similarity_threshold: float = 0.3,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Find papers related to a given paper"""
        
        # Get the paper's content and metadata
        paper_info = self.database.execute_query(
            "SELECT filename, tags, description FROM file_registry WHERE id = ?",
            (file_id,)
        )
        
        if not paper_info:
            return []
        
        paper = paper_info[0]
        
        # Extract keywords from tags and description
        keywords = []
        if paper['tags']:
            try:
                tags = json.loads(paper['tags'])
                keywords.extend(tags)
            except:
                pass
        
        if paper['description']:
            # Extract key terms from description
            description_words = re.findall(r'\b\w+\b', paper['description'].lower())
            keywords.extend([w for w in description_words if len(w) > 4])
        
        # Search for papers with similar keywords
        related_papers = []
        for keyword in set(keywords):
            if keyword in ['research', 'ai-paper', 'paper', 'the', 'and', 'for', 'with']:
                continue  # Skip common words
            
            search_results = self.search(keyword, limit=10)
            for result in search_results:
                if result['file_id'] != file_id:  # Exclude the original paper
                    related_papers.append(result)
        
        # Remove duplicates and sort by relevance
        seen_ids = set()
        unique_papers = []
        for paper in related_papers:
            if paper['file_id'] not in seen_ids:
                seen_ids.add(paper['file_id'])
                unique_papers.append(paper)
        
        # Sort by relevance and limit results
        unique_papers.sort(key=lambda x: x['relevance_score'], reverse=True)
        return unique_papers[:limit]
    
    def get_knowledge_summary(self) -> Dict[str, Any]:
        """Get summary of knowledge base contents"""
        
        # Count papers by topic
        papers = self.get_research_papers()
        
        topics = {}
        years = {}
        total_words = 0
        total_papers = len(papers)
        
        for paper in papers:
            # Count by topic
            topic = paper['topic']
            topics[topic] = topics.get(topic, 0) + 1
            
            # Count by year
            year = str(paper['year'])
            years[year] = years.get(year, 0) + 1
            
            # Sum word counts
            if paper['word_count']:
                total_words += paper['word_count']
        
        # Get most common search terms (from recent searches)
        # This would require tracking search queries, simplified for now
        
        return {
            'total_papers': total_papers,
            'total_words': total_words,
            'avg_words_per_paper': total_words // total_papers if total_papers > 0 else 0,
            'topics': dict(sorted(topics.items(), key=lambda x: x[1], reverse=True)),
            'years': dict(sorted(years.items(), key=lambda x: x[1], reverse=True)),
            'last_updated': datetime.now().isoformat()
        }
    
    def extract_concepts(self, text: str, limit: int = 10) -> List[str]:
        """Extract key concepts from text"""
        
        # Simple concept extraction using patterns
        # In a real implementation, this could use NLP libraries
        
        # Look for technical terms (capitalized words, acronyms)
        concepts = []
        
        # Find acronyms (2+ uppercase letters)
        acronyms = re.findall(r'\b[A-Z]{2,}\b', text)
        concepts.extend(acronyms)
        
        # Find capitalized terms (potential proper nouns/concepts)
        capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        concepts.extend(capitalized)
        
        # Find technical terms with common AI/ML patterns
        ai_patterns = [
            r'\b\w*neural\w*\b', r'\b\w*network\w*\b', r'\b\w*learning\w*\b',
            r'\b\w*model\w*\b', r'\b\w*algorithm\w*\b', r'\b\w*attention\w*\b',
            r'\b\w*transformer\w*\b', r'\b\w*embedding\w*\b'
        ]
        
        for pattern in ai_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            concepts.extend(matches)
        
        # Clean and deduplicate
        cleaned_concepts = []
        for concept in concepts:
            concept = concept.strip()
            if len(concept) > 2 and concept.lower() not in ['the', 'and', 'for', 'with']:
                cleaned_concepts.append(concept)
        
        # Return most frequent concepts
        from collections import Counter
        concept_counts = Counter(cleaned_concepts)
        return [concept for concept, count in concept_counts.most_common(limit)]
    
    def _calculate_relevance(self, query: str, text: str) -> float:
        """Calculate relevance score between query and text"""
        
        query_terms = query.lower().split()
        text_lower = text.lower()
        
        # Count term matches
        matches = 0
        total_terms = len(query_terms)
        
        for term in query_terms:
            if term in text_lower:
                matches += 1
        
        # Basic relevance score
        base_score = matches / total_terms if total_terms > 0 else 0
        
        # Boost score for exact phrase matches
        if query.lower() in text_lower:
            base_score += 0.3
        
        # Boost score for multiple occurrences
        for term in query_terms:
            occurrences = text_lower.count(term)
            if occurrences > 1:
                base_score += min(0.1 * (occurrences - 1), 0.3)
        
        return min(base_score, 1.0)
    
    def _extract_context(self, query: str, text: str, context_size: int = 200) -> List[str]:
        """Extract relevant context snippets from text"""
        
        query_terms = query.lower().split()
        text_lower = text.lower()
        contexts = []
        
        # Find positions of query terms
        for term in query_terms:
            start = 0
            while True:
                pos = text_lower.find(term, start)
                if pos == -1:
                    break
                
                # Extract context around the term
                context_start = max(0, pos - context_size // 2)
                context_end = min(len(text), pos + len(term) + context_size // 2)
                
                context = text[context_start:context_end].strip()
                
                # Clean up context (remove partial words at edges)
                words = context.split()
                if len(words) > 1:
                    if context_start > 0:
                        words = words[1:]  # Remove partial first word
                    if context_end < len(text):
                        words = words[:-1]  # Remove partial last word
                    
                    context = ' '.join(words)
                    if context and context not in contexts:
                        contexts.append(context)
                
                start = pos + 1
        
        return contexts[:3]  # Return top 3 contexts


# Convenience functions
def search_papers(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Quick search for research papers"""
    kb = KnowledgeBase()
    return kb.search(query, limit=limit, content_type='document')

def get_paper_by_title(title_part: str) -> Optional[Dict[str, Any]]:
    """Find paper by partial title match"""
    kb = KnowledgeBase()
    papers = kb.get_research_papers()
    
    for paper in papers:
        if title_part.lower() in paper['title'].lower():
            return paper
    
    return None

def get_knowledge_stats() -> Dict[str, Any]:
    """Get knowledge base statistics"""
    kb = KnowledgeBase()
    return kb.get_knowledge_summary()
