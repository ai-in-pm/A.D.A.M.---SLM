#!/usr/bin/env python3
"""
Research Paper PDF Processing for A.D.A.M. SLM Database
Extracts and imports PDF research papers into the knowledge base
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

from database import AdamSLMDatabase
from file_manager import FileManager
from file_converter import FileConverter


def print_header(title: str):
    """Print formatted header"""
    print("\n" + "="*60)
    print(f"📄 {title}")
    print("="*60)


def print_section(title: str):
    """Print formatted section"""
    print(f"\n📋 {title}")
    print("-"*40)


def print_success(message: str):
    """Print success message"""
    print(f"✅ {message}")


def print_error(message: str):
    """Print error message"""
    print(f"❌ {message}")


def print_info(message: str):
    """Print info message"""
    print(f"ℹ️  {message}")


def extract_paper_metadata(filename: str) -> dict:
    """Extract metadata from paper filename"""
    metadata = {
        "source": "research_paper",
        "knowledge_base": True,
        "extraction_date": datetime.now().isoformat()
    }
    
    # Extract paper-specific metadata based on filename
    if "DeepSeek-v2" in filename:
        metadata.update({
            "title": "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model",
            "authors": "DeepSeek AI",
            "topic": "mixture-of-experts",
            "model_type": "language_model",
            "architecture": "moe",
            "year": "2024"
        })
    elif "Eliza" in filename:
        metadata.update({
            "title": "ELIZA - A Computer Program for the Study of Natural Language Communication",
            "authors": "Joseph Weizenbaum",
            "topic": "natural_language_processing",
            "historical": True,
            "year": "1966"
        })
    elif "GOFAI" in filename:
        metadata.update({
            "title": "Good Old-Fashioned Artificial Intelligence",
            "topic": "symbolic_ai",
            "approach": "symbolic",
            "historical": True
        })
    elif "OpenAI_o1" in filename:
        metadata.update({
            "title": "OpenAI o1 System Card",
            "authors": "OpenAI",
            "topic": "reasoning_model",
            "model_type": "reasoning",
            "year": "2024"
        })
    
    return metadata


def process_pdf_paper(
    file_manager: FileManager,
    pdf_path: str,
    user_id: int = 1
) -> tuple[bool, int, dict]:
    """
    Process a single PDF research paper
    
    Returns:
        (success, file_id, extraction_info)
    """
    filename = os.path.basename(pdf_path)
    print_section(f"Processing: {filename}")
    
    # Extract metadata
    metadata = extract_paper_metadata(filename)
    print_info(f"Paper: {metadata.get('title', filename)}")
    
    # Prepare tags
    tags = [
        "research", "ai-paper", "knowledge-base", "pdf-extracted"
    ]
    
    # Add topic-specific tags
    if metadata.get("topic"):
        tags.append(metadata["topic"])
    if metadata.get("model_type"):
        tags.append(metadata["model_type"])
    if metadata.get("architecture"):
        tags.append(metadata["architecture"])
    if metadata.get("historical"):
        tags.append("historical")
    if metadata.get("year"):
        tags.append(f"year-{metadata['year']}")
    
    # Prepare description
    description = f"Research paper: {metadata.get('title', filename)} - Extracted for A.D.A.M. SLM knowledge base"
    if metadata.get("authors"):
        description += f" by {metadata['authors']}"
    
    try:
        # Import the PDF file
        print_info("Importing PDF file...")
        file_id = file_manager.register_file(
            file_path=pdf_path,
            file_type="document",
            description=description,
            tags=tags,
            created_by=user_id,
            metadata=metadata,
            copy_to_storage=True,
            process_immediately=True
        )
        
        print_success(f"PDF imported with ID: {file_id}")
        
        # Wait a moment for processing to complete
        import time
        time.sleep(2)
        
        # Check if processing completed and extract text
        file_info = file_manager.get_file_info(file_id)
        
        extraction_info = {
            "file_id": file_id,
            "filename": filename,
            "stored_path": file_info["stored_path"],
            "processing_status": file_info["processing_status"],
            "file_size": file_info["file_size_bytes"]
        }
        
        # Try to extract text content manually if needed
        if file_info["processing_status"] != "completed":
            print_info("Manual text extraction...")
            converter = FileConverter()
            
            # Convert PDF to text
            text_output_path = pdf_path.replace('.pdf', '_extracted.txt')
            result = converter.convert_file(pdf_path, text_output_path, 'txt')
            
            if result['success']:
                print_success(f"Text extracted to: {text_output_path}")
                
                # Read extracted text
                with open(text_output_path, 'r', encoding='utf-8') as f:
                    extracted_text = f.read()
                
                # Store in file_content table
                db = file_manager.db
                content_id = db.execute_insert("""
                    INSERT INTO file_content (
                        file_id, content_type, extracted_text, 
                        extraction_method, word_count, character_count
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    file_id, 'text', extracted_text, 'pdf_converter',
                    len(extracted_text.split()), len(extracted_text)
                ))
                
                extraction_info.update({
                    "text_extracted": True,
                    "content_id": content_id,
                    "word_count": len(extracted_text.split()),
                    "character_count": len(extracted_text),
                    "text_preview": extracted_text[:500] + "..." if len(extracted_text) > 500 else extracted_text
                })
                
                print_success(f"Content stored in database (ID: {content_id})")
                print_info(f"Extracted {len(extracted_text.split()):,} words, {len(extracted_text):,} characters")
                
                # Clean up temporary text file
                try:
                    os.remove(text_output_path)
                except:
                    pass
                    
            else:
                print_error(f"Text extraction failed: {result.get('error', 'Unknown error')}")
                extraction_info["text_extracted"] = False
        
        return True, file_id, extraction_info
        
    except Exception as e:
        print_error(f"Failed to process {filename}: {str(e)}")
        return False, 0, {"error": str(e)}


def process_all_research_papers():
    """Process all PDF research papers in the NoLLM_NoAPIKey directory"""
    
    print_header("A.D.A.M. SLM Research Paper Processing")
    print("Extracting and importing PDF research papers into knowledge base")
    
    # Define source directory and papers
    source_dir = Path("NoLLM_NoAPIKey")
    papers = [
        "DeepSeek-v2_Paper.pdf",
        "Eliza_Paper.pdf", 
        "GOFAI_Paper.pdf",
        "OpenAI_o1_System_Card_Paper.pdf"
    ]
    
    # Check if source directory exists
    if not source_dir.exists():
        print_error(f"Source directory not found: {source_dir}")
        return False
    
    # Initialize database and file manager
    try:
        print_section("Initializing Database System")
        db = AdamSLMDatabase("databases/adamslm_sophisticated.sqlite")
        file_manager = FileManager(db)
        print_success("Database system initialized")
        
        # Get admin user
        admin_user = db.get_user(username="admin")
        if not admin_user:
            print_error("Admin user not found in database")
            return False
        
        user_id = admin_user['id']
        print_info(f"Using user: admin (ID: {user_id})")
        
    except Exception as e:
        print_error(f"Failed to initialize database: {e}")
        return False
    
    # Process each paper
    results = []
    successful_imports = 0
    
    for paper_filename in papers:
        paper_path = source_dir / paper_filename
        
        if not paper_path.exists():
            print_error(f"Paper not found: {paper_path}")
            results.append({
                "filename": paper_filename,
                "success": False,
                "error": "File not found"
            })
            continue
        
        # Process the paper
        success, file_id, extraction_info = process_pdf_paper(
            file_manager, str(paper_path), user_id
        )
        
        if success:
            successful_imports += 1
            
        results.append({
            "filename": paper_filename,
            "success": success,
            "file_id": file_id,
            **extraction_info
        })
    
    # Print summary
    print_header("Processing Complete")
    print_info(f"Successfully imported: {successful_imports}/{len(papers)} papers")
    
    for result in results:
        if result["success"]:
            print_success(f"{result['filename']} → ID: {result['file_id']}")
            if result.get("word_count"):
                print(f"   📊 {result['word_count']:,} words, {result['character_count']:,} characters")
        else:
            print_error(f"{result['filename']} → {result.get('error', 'Failed')}")
    
    # Show database stats
    if successful_imports > 0:
        print_section("Database Statistics")
        
        # Count documents
        documents = file_manager.list_files(file_type="document")
        research_papers = [d for d in documents if "research" in (d.get("tags") or "")]
        
        print_info(f"Total documents in database: {len(documents)}")
        print_info(f"Research papers: {len(research_papers)}")
        
        # Show recent imports
        recent_papers = file_manager.list_files(file_type="document", limit=5)
        print_info("Recent document imports:")
        for doc in recent_papers:
            try:
                if isinstance(doc.get("tags"), str):
                    tags = json.loads(doc.get("tags", "[]"))
                else:
                    tags = doc.get("tags", [])

                if "research" in tags:
                    print(f"   📄 {doc['filename']} (ID: {doc['id']})")
            except (json.JSONDecodeError, TypeError):
                # Skip if tags can't be parsed
                print(f"   📄 {doc['filename']} (ID: {doc['id']})")
    
    print_header("Knowledge Base Ready")
    print("🧠 Research papers are now available for A.D.A.M. SLM knowledge retrieval")
    print("🔍 Content is searchable through the file_content table")
    print("📊 Analytics available through the database system")
    
    return successful_imports == len(papers)


def main():
    """Main function"""
    try:
        success = process_all_research_papers()
        return 0 if success else 1
    except Exception as e:
        print_error(f"Processing failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
