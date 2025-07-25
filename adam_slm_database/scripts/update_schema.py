#!/usr/bin/env python3
"""
Update database schema to include file management tables
"""

import sqlite3
import os
from database import AdamSLMDatabase


def update_database_schema():
    """Update the database schema with file management tables"""
    
    print("🔄 Updating database schema with file management tables...")
    
    # File management schema
    file_schema = """
    -- File registry for managing all types of files
    CREATE TABLE IF NOT EXISTS file_registry (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename VARCHAR(255) NOT NULL,
        original_path VARCHAR(500),
        stored_path VARCHAR(500),
        file_type VARCHAR(50) NOT NULL, -- 'dataset', 'model', 'config', 'log', 'image', 'document', etc.
        file_format VARCHAR(20), -- 'txt', 'json', 'csv', 'pt', 'png', 'pdf', etc.
        file_size_bytes INTEGER,
        mime_type VARCHAR(100),
        checksum_md5 VARCHAR(32),
        checksum_sha256 VARCHAR(64),
        created_by INTEGER,
        uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        description TEXT,
        metadata TEXT, -- JSON with file-specific metadata
        tags TEXT, -- JSON array of tags
        is_processed BOOLEAN DEFAULT 0,
        processing_status VARCHAR(20) DEFAULT 'pending', -- 'pending', 'processing', 'completed', 'failed'
        processing_log TEXT,
        parent_file_id INTEGER, -- For derived files
        version INTEGER DEFAULT 1,
        is_active BOOLEAN DEFAULT 1,
        access_level VARCHAR(20) DEFAULT 'private', -- 'public', 'private', 'restricted'
        FOREIGN KEY (created_by) REFERENCES users(id),
        FOREIGN KEY (parent_file_id) REFERENCES file_registry(id)
    );

    -- File relationships for linking files together
    CREATE TABLE IF NOT EXISTS file_relationships (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        source_file_id INTEGER NOT NULL,
        target_file_id INTEGER NOT NULL,
        relationship_type VARCHAR(50) NOT NULL, -- 'derived_from', 'part_of', 'related_to', 'backup_of'
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        metadata TEXT, -- JSON with relationship-specific data
        FOREIGN KEY (source_file_id) REFERENCES file_registry(id) ON DELETE CASCADE,
        FOREIGN KEY (target_file_id) REFERENCES file_registry(id) ON DELETE CASCADE,
        UNIQUE(source_file_id, target_file_id, relationship_type)
    );

    -- File processing jobs for async operations
    CREATE TABLE IF NOT EXISTS file_processing_jobs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_id INTEGER NOT NULL,
        job_type VARCHAR(50) NOT NULL, -- 'import', 'convert', 'analyze', 'extract', 'compress'
        job_status VARCHAR(20) DEFAULT 'queued', -- 'queued', 'running', 'completed', 'failed', 'cancelled'
        priority INTEGER DEFAULT 5, -- 1-10, higher is more priority
        started_at TIMESTAMP,
        completed_at TIMESTAMP,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        created_by INTEGER,
        job_config TEXT, -- JSON with job-specific configuration
        progress_percent INTEGER DEFAULT 0,
        error_message TEXT,
        result_data TEXT, -- JSON with job results
        FOREIGN KEY (file_id) REFERENCES file_registry(id) ON DELETE CASCADE,
        FOREIGN KEY (created_by) REFERENCES users(id)
    );

    -- File content extraction for searchable text
    CREATE TABLE IF NOT EXISTS file_content (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_id INTEGER NOT NULL,
        content_type VARCHAR(50), -- 'text', 'metadata', 'summary', 'keywords'
        extracted_text TEXT,
        language VARCHAR(10) DEFAULT 'en',
        confidence_score REAL, -- 0.0-1.0 for extraction confidence
        extraction_method VARCHAR(50), -- 'direct', 'ocr', 'parser', 'ai'
        extracted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        word_count INTEGER,
        character_count INTEGER,
        FOREIGN KEY (file_id) REFERENCES file_registry(id) ON DELETE CASCADE
    );

    -- File access logs for audit trail
    CREATE TABLE IF NOT EXISTS file_access_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_id INTEGER NOT NULL,
        user_id INTEGER,
        access_type VARCHAR(20) NOT NULL, -- 'view', 'download', 'edit', 'delete'
        access_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        ip_address VARCHAR(45),
        user_agent TEXT,
        success BOOLEAN DEFAULT 1,
        error_message TEXT,
        FOREIGN KEY (file_id) REFERENCES file_registry(id) ON DELETE CASCADE,
        FOREIGN KEY (user_id) REFERENCES users(id)
    );

    -- File management indexes
    CREATE INDEX IF NOT EXISTS idx_file_registry_type ON file_registry(file_type);
    CREATE INDEX IF NOT EXISTS idx_file_registry_format ON file_registry(file_format);
    CREATE INDEX IF NOT EXISTS idx_file_registry_uploaded ON file_registry(uploaded_at);
    CREATE INDEX IF NOT EXISTS idx_file_registry_checksum ON file_registry(checksum_sha256);
    CREATE INDEX IF NOT EXISTS idx_file_registry_status ON file_registry(processing_status);
    CREATE INDEX IF NOT EXISTS idx_file_relationships_source ON file_relationships(source_file_id);
    CREATE INDEX IF NOT EXISTS idx_file_relationships_target ON file_relationships(target_file_id);
    CREATE INDEX IF NOT EXISTS idx_file_processing_jobs_status ON file_processing_jobs(job_status);
    CREATE INDEX IF NOT EXISTS idx_file_processing_jobs_priority ON file_processing_jobs(priority DESC);
    CREATE INDEX IF NOT EXISTS idx_file_content_file ON file_content(file_id);
    CREATE INDEX IF NOT EXISTS idx_file_access_logs_file ON file_access_logs(file_id);
    CREATE INDEX IF NOT EXISTS idx_file_access_logs_timestamp ON file_access_logs(access_timestamp);
    """
    
    # Connect to database and execute schema
    db_path = "../databases/adamslm_sophisticated.sqlite"
    
    try:
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        
        # Execute the schema
        conn.executescript(file_schema)
        conn.commit()
        
        # Verify tables were created
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'file_%'")
        tables = cursor.fetchall()
        
        print(f"✅ Created {len(tables)} file management tables:")
        for table in tables:
            print(f"   • {table[0]}")
            
        conn.close()
        
        print("✅ Database schema updated successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Failed to update schema: {e}")
        return False


def main():
    """Main function"""
    print("🗄️ A.D.A.M. SLM Database Schema Update")
    print("Adding file management capabilities")
    
    if update_database_schema():
        print("\n🚀 Database is now ready for file management!")
        print("You can now:")
        print("  • Import files of all types")
        print("  • Convert between formats")
        print("  • Extract content and metadata")
        print("  • Track file relationships")
        print("  • Monitor processing jobs")
    else:
        print("\n❌ Schema update failed!")
        return 1
        
    return 0


if __name__ == "__main__":
    exit(main())
