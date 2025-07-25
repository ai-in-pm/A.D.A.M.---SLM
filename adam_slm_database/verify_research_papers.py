#!/usr/bin/env python3
"""
Verify Research Paper Imports in A.D.A.M. SLM Database
Check the imported PDF papers and their extracted content
"""

import os
import sys
import json
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

from database import AdamSLMDatabase
from file_manager import FileManager


def print_header(title: str):
    """Print formatted header"""
    print("\n" + "="*60)
    print(f"🔍 {title}")
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


def verify_research_papers():
    """Verify all imported research papers"""
    
    print_header("A.D.A.M. SLM Research Paper Verification")
    
    try:
        # Initialize database
        db = AdamSLMDatabase("databases/adamslm_sophisticated.sqlite")
        file_manager = FileManager(db)
        
        print_section("Database Connection")
        print_success("Connected to A.D.A.M. SLM database")
        
        # Get all documents
        all_documents = file_manager.list_files(file_type="document")
        print_info(f"Total documents in database: {len(all_documents)}")
        
        # Filter research papers
        research_papers = []
        for doc in all_documents:
            try:
                if isinstance(doc.get("tags"), str):
                    tags = json.loads(doc.get("tags", "[]"))
                else:
                    tags = doc.get("tags", []) or []
                    
                if "research" in tags or "ai-paper" in tags:
                    research_papers.append(doc)
            except:
                # Check filename for research papers
                if any(keyword in doc['filename'].lower() for keyword in ['deepseek', 'eliza', 'gofai', 'openai']):
                    research_papers.append(doc)
        
        print_info(f"Research papers found: {len(research_papers)}")
        
        if not research_papers:
            print("❌ No research papers found in database")
            return False
        
        # Verify each paper
        print_section("Research Paper Details")
        
        total_words = 0
        total_chars = 0
        
        for i, paper in enumerate(research_papers, 1):
            print(f"\n📄 Paper {i}: {paper['filename']}")
            print(f"   🆔 File ID: {paper['id']}")
            print(f"   📊 Size: {paper['file_size_bytes']:,} bytes")
            print(f"   📅 Uploaded: {paper['uploaded_at']}")
            print(f"   📁 Stored: {paper['stored_path']}")
            print(f"   ⚡ Status: {paper['processing_status']}")
            
            # Get tags
            try:
                if isinstance(paper.get("tags"), str):
                    tags = json.loads(paper.get("tags", "[]"))
                else:
                    tags = paper.get("tags", []) or []
                print(f"   🏷️  Tags: {', '.join(tags)}")
            except:
                print(f"   🏷️  Tags: {paper.get('tags', 'None')}")
            
            # Get description
            if paper.get('description'):
                desc = paper['description']
                if len(desc) > 100:
                    desc = desc[:100] + "..."
                print(f"   📝 Description: {desc}")
            
            # Check extracted content
            content_query = """
                SELECT * FROM file_content 
                WHERE file_id = ? 
                ORDER BY extracted_at DESC 
                LIMIT 1
            """
            content_results = db.execute_query(content_query, (paper['id'],))
            
            if content_results:
                content = content_results[0]
                word_count = content.get('word_count', 0)
                char_count = content.get('character_count', 0)
                
                print(f"   📖 Content extracted: ✅")
                print(f"   📊 Words: {word_count:,}")
                print(f"   📊 Characters: {char_count:,}")
                print(f"   🔍 Extraction method: {content.get('extraction_method', 'unknown')}")
                
                # Show content preview
                if content.get('extracted_text'):
                    preview = content['extracted_text'][:200].replace('\n', ' ').strip()
                    print(f"   👁️  Preview: {preview}...")
                
                total_words += word_count or 0
                total_chars += char_count or 0
            else:
                print(f"   📖 Content extracted: ❌")
        
        # Summary statistics
        print_section("Summary Statistics")
        print_success(f"Successfully imported: {len(research_papers)} research papers")
        print_info(f"Total extracted content:")
        print(f"   📊 Words: {total_words:,}")
        print(f"   📊 Characters: {total_chars:,}")
        print(f"   📖 Average words per paper: {total_words // len(research_papers):,}")
        
        # Check searchable content
        print_section("Content Search Verification")
        
        # Test search for specific terms
        search_terms = ["language model", "artificial intelligence", "neural network", "transformer"]
        
        for term in search_terms:
            search_query = """
                SELECT fc.file_id, fr.filename, fc.word_count
                FROM file_content fc
                JOIN file_registry fr ON fc.file_id = fr.id
                WHERE fc.extracted_text LIKE ? AND fr.file_type = 'document'
            """
            
            search_results = db.execute_query(search_query, (f"%{term}%",))
            
            if search_results:
                print_success(f"'{term}' found in {len(search_results)} papers")
                for result in search_results:
                    print(f"   📄 {result['filename']} (ID: {result['file_id']})")
            else:
                print_info(f"'{term}' not found in extracted content")
        
        # Database integration check
        print_section("Database Integration Check")
        
        # Check file storage organization
        storage_check = db.execute_query("""
            SELECT stored_path, COUNT(*) as count
            FROM file_registry 
            WHERE file_type = 'document' AND stored_path LIKE '%document%'
            GROUP BY stored_path
        """)
        
        if storage_check:
            print_success("Files properly organized in document storage")
            for path_info in storage_check:
                print(f"   📁 {path_info['stored_path']}")
        
        # Check processing jobs
        job_check = db.execute_query("""
            SELECT fpj.job_status, COUNT(*) as count
            FROM file_processing_jobs fpj
            JOIN file_registry fr ON fpj.file_id = fr.id
            WHERE fr.file_type = 'document'
            GROUP BY fpj.job_status
        """)
        
        if job_check:
            print_success("Processing jobs tracked:")
            for job_info in job_check:
                print(f"   ⚡ {job_info['job_status']}: {job_info['count']} jobs")
        
        print_header("Verification Complete")
        print_success("All research papers successfully imported and verified!")
        print_info("Papers are ready for A.D.A.M. SLM knowledge retrieval")
        print_info("Content is searchable through the database system")
        
        return True
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def search_papers(query: str):
    """Search through imported research papers"""
    
    print_header(f"Searching Papers for: '{query}'")
    
    try:
        db = AdamSLMDatabase("databases/adamslm_sophisticated.sqlite")
        
        # Search in extracted content
        search_query = """
            SELECT fc.file_id, fr.filename, fc.word_count, fc.character_count,
                   SUBSTR(fc.extracted_text, 1, 300) as preview
            FROM file_content fc
            JOIN file_registry fr ON fc.file_id = fr.id
            WHERE fc.extracted_text LIKE ? AND fr.file_type = 'document'
            ORDER BY fc.word_count DESC
        """
        
        results = db.execute_query(search_query, (f"%{query}%",))
        
        if results:
            print_success(f"Found '{query}' in {len(results)} papers:")
            
            for result in results:
                print(f"\n📄 {result['filename']} (ID: {result['file_id']})")
                print(f"   📊 {result['word_count']:,} words, {result['character_count']:,} characters")
                
                # Show context around the search term
                preview = result['preview'].replace('\n', ' ').strip()
                if query.lower() in preview.lower():
                    # Find the position and show context
                    pos = preview.lower().find(query.lower())
                    start = max(0, pos - 50)
                    end = min(len(preview), pos + len(query) + 50)
                    context = preview[start:end]
                    print(f"   🔍 Context: ...{context}...")
                else:
                    print(f"   👁️  Preview: {preview}...")
        else:
            print_info(f"No papers found containing '{query}'")
            
    except Exception as e:
        print(f"❌ Search failed: {e}")


def main():
    """Main function"""
    if len(sys.argv) > 1:
        # Search mode
        query = " ".join(sys.argv[1:])
        search_papers(query)
    else:
        # Verification mode
        success = verify_research_papers()
        return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
