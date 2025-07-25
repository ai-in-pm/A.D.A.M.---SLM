-- ADAM SLM Database Schema
-- Sophisticated SQLite database for managing ADAM SLM models and experiments

-- Enable foreign key constraints
PRAGMA foreign_keys = ON;

-- ============================================================================
-- CORE TABLES
-- ============================================================================

-- Users table for multi-user support
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE,
    full_name VARCHAR(100),
    role VARCHAR(20) DEFAULT 'user', -- 'admin', 'researcher', 'user'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT 1,
    preferences TEXT, -- JSON string for user preferences
    api_key VARCHAR(64) UNIQUE -- For API access
);

-- Sessions table for tracking user sessions
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    session_token VARCHAR(128) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP NOT NULL,
    ip_address VARCHAR(45),
    user_agent TEXT,
    is_active BOOLEAN DEFAULT 1,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- ============================================================================
-- MODEL MANAGEMENT
-- ============================================================================

-- Model registry for tracking different model versions
CREATE TABLE IF NOT EXISTS model_registry (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name VARCHAR(100) NOT NULL,
    version VARCHAR(20) NOT NULL,
    model_type VARCHAR(50) NOT NULL, -- 'adam-slm-small', 'adam-slm-base', etc.
    architecture_config TEXT NOT NULL, -- JSON string of model config
    parameter_count INTEGER,
    model_size_mb REAL,
    checkpoint_path VARCHAR(500),
    tokenizer_path VARCHAR(500),
    created_by INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    description TEXT,
    tags TEXT, -- JSON array of tags
    is_active BOOLEAN DEFAULT 1,
    parent_model_id INTEGER, -- For model lineage
    FOREIGN KEY (created_by) REFERENCES users(id),
    FOREIGN KEY (parent_model_id) REFERENCES model_registry(id),
    UNIQUE(model_name, version)
);

-- Model performance benchmarks
CREATE TABLE IF NOT EXISTS model_benchmarks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id INTEGER NOT NULL,
    benchmark_type VARCHAR(50) NOT NULL, -- 'perplexity', 'bleu', 'rouge', etc.
    dataset_name VARCHAR(100),
    metric_value REAL NOT NULL,
    metric_details TEXT, -- JSON with additional metrics
    benchmark_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    hardware_info TEXT, -- JSON with hardware details
    notes TEXT,
    FOREIGN KEY (model_id) REFERENCES model_registry(id) ON DELETE CASCADE
);

-- ============================================================================
-- TRAINING MANAGEMENT
-- ============================================================================

-- Training runs for tracking model training
CREATE TABLE IF NOT EXISTS training_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_name VARCHAR(100) NOT NULL,
    model_id INTEGER,
    base_model_id INTEGER, -- Model used as starting point
    training_config TEXT NOT NULL, -- JSON string of training config
    dataset_id INTEGER,
    started_by INTEGER,
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    status VARCHAR(20) DEFAULT 'running', -- 'running', 'completed', 'failed', 'stopped'
    current_epoch INTEGER DEFAULT 0,
    current_step INTEGER DEFAULT 0,
    total_steps INTEGER,
    best_loss REAL,
    best_metric REAL,
    final_loss REAL,
    total_tokens_processed INTEGER,
    training_time_seconds INTEGER,
    gpu_hours REAL,
    error_message TEXT,
    logs_path VARCHAR(500),
    checkpoint_dir VARCHAR(500),
    wandb_run_id VARCHAR(100),
    notes TEXT,
    FOREIGN KEY (model_id) REFERENCES model_registry(id),
    FOREIGN KEY (base_model_id) REFERENCES model_registry(id),
    FOREIGN KEY (dataset_id) REFERENCES datasets(id),
    FOREIGN KEY (started_by) REFERENCES users(id)
);

-- Training metrics for detailed tracking
CREATE TABLE IF NOT EXISTS training_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    training_run_id INTEGER NOT NULL,
    step INTEGER NOT NULL,
    epoch INTEGER,
    metric_name VARCHAR(50) NOT NULL, -- 'train_loss', 'eval_loss', 'learning_rate', etc.
    metric_value REAL NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (training_run_id) REFERENCES training_runs(id) ON DELETE CASCADE
);

-- ============================================================================
-- DATASET MANAGEMENT
-- ============================================================================

-- Datasets table for managing training data
CREATE TABLE IF NOT EXISTS datasets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name VARCHAR(100) UNIQUE NOT NULL,
    description TEXT,
    dataset_type VARCHAR(50), -- 'text', 'conversation', 'instruction', etc.
    source_path VARCHAR(500),
    processed_path VARCHAR(500),
    total_samples INTEGER,
    total_tokens INTEGER,
    avg_sequence_length REAL,
    vocabulary_size INTEGER,
    language VARCHAR(10) DEFAULT 'en',
    license VARCHAR(100),
    created_by INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    preprocessing_config TEXT, -- JSON string
    statistics TEXT, -- JSON with detailed stats
    is_public BOOLEAN DEFAULT 0,
    tags TEXT, -- JSON array
    FOREIGN KEY (created_by) REFERENCES users(id)
);

-- Dataset splits for train/validation/test
CREATE TABLE IF NOT EXISTS dataset_splits (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id INTEGER NOT NULL,
    split_name VARCHAR(20) NOT NULL, -- 'train', 'validation', 'test'
    file_path VARCHAR(500),
    sample_count INTEGER,
    token_count INTEGER,
    split_ratio REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (dataset_id) REFERENCES datasets(id) ON DELETE CASCADE,
    UNIQUE(dataset_id, split_name)
);

-- ============================================================================
-- EXPERIMENT TRACKING
-- ============================================================================

-- Experiments for organizing related training runs
CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name VARCHAR(100) UNIQUE NOT NULL,
    description TEXT,
    objective TEXT, -- What we're trying to achieve
    hypothesis TEXT, -- What we expect to happen
    methodology TEXT, -- How we're testing
    created_by INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(20) DEFAULT 'active', -- 'active', 'completed', 'archived'
    results_summary TEXT,
    conclusions TEXT,
    tags TEXT, -- JSON array
    FOREIGN KEY (created_by) REFERENCES users(id)
);

-- Link training runs to experiments
CREATE TABLE IF NOT EXISTS experiment_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL,
    training_run_id INTEGER NOT NULL,
    run_purpose TEXT, -- Why this run is part of the experiment
    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (experiment_id) REFERENCES experiments(id) ON DELETE CASCADE,
    FOREIGN KEY (training_run_id) REFERENCES training_runs(id) ON DELETE CASCADE,
    UNIQUE(experiment_id, training_run_id)
);

-- ============================================================================
-- INFERENCE AND DEPLOYMENT
-- ============================================================================

-- Inference sessions for tracking model usage
CREATE TABLE IF NOT EXISTS inference_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id INTEGER NOT NULL,
    user_id INTEGER,
    session_name VARCHAR(100),
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ended_at TIMESTAMP,
    total_requests INTEGER DEFAULT 0,
    total_tokens_generated INTEGER DEFAULT 0,
    avg_response_time_ms REAL,
    hardware_info TEXT, -- JSON
    configuration TEXT, -- JSON with generation config
    FOREIGN KEY (model_id) REFERENCES model_registry(id),
    FOREIGN KEY (user_id) REFERENCES users(id)
);

-- Individual inference requests
CREATE TABLE IF NOT EXISTS inference_requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    request_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    prompt_text TEXT,
    generated_text TEXT,
    prompt_tokens INTEGER,
    generated_tokens INTEGER,
    response_time_ms INTEGER,
    temperature REAL,
    top_k INTEGER,
    top_p REAL,
    max_tokens INTEGER,
    success BOOLEAN DEFAULT 1,
    error_message TEXT,
    FOREIGN KEY (session_id) REFERENCES inference_sessions(id) ON DELETE CASCADE
);

-- ============================================================================
-- FILE MANAGEMENT
-- ============================================================================

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

-- ============================================================================
-- SYSTEM MONITORING
-- ============================================================================

-- System performance metrics
CREATE TABLE IF NOT EXISTS system_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metric_type VARCHAR(50), -- 'cpu', 'memory', 'gpu', 'disk', 'network'
    metric_name VARCHAR(50),
    metric_value REAL,
    unit VARCHAR(20),
    hostname VARCHAR(100),
    process_id INTEGER,
    additional_info TEXT -- JSON
);

-- Database operations log
CREATE TABLE IF NOT EXISTS operation_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    user_id INTEGER,
    operation_type VARCHAR(50), -- 'CREATE', 'UPDATE', 'DELETE', 'QUERY'
    table_name VARCHAR(50),
    record_id INTEGER,
    operation_details TEXT, -- JSON
    ip_address VARCHAR(45),
    user_agent TEXT,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

-- ============================================================================
-- INDEXES FOR PERFORMANCE
-- ============================================================================

-- Model registry indexes
CREATE INDEX IF NOT EXISTS idx_model_registry_name ON model_registry(model_name);
CREATE INDEX IF NOT EXISTS idx_model_registry_type ON model_registry(model_type);
CREATE INDEX IF NOT EXISTS idx_model_registry_created ON model_registry(created_at);

-- Training runs indexes
CREATE INDEX IF NOT EXISTS idx_training_runs_status ON training_runs(status);
CREATE INDEX IF NOT EXISTS idx_training_runs_started ON training_runs(started_at);
CREATE INDEX IF NOT EXISTS idx_training_runs_model ON training_runs(model_id);

-- Training metrics indexes
CREATE INDEX IF NOT EXISTS idx_training_metrics_run_step ON training_metrics(training_run_id, step);
CREATE INDEX IF NOT EXISTS idx_training_metrics_name ON training_metrics(metric_name);

-- Dataset indexes
CREATE INDEX IF NOT EXISTS idx_datasets_name ON datasets(name);
CREATE INDEX IF NOT EXISTS idx_datasets_type ON datasets(dataset_type);
CREATE INDEX IF NOT EXISTS idx_datasets_created ON datasets(created_at);

-- Experiment indexes
CREATE INDEX IF NOT EXISTS idx_experiments_status ON experiments(status);
CREATE INDEX IF NOT EXISTS idx_experiments_created ON experiments(created_at);

-- Inference indexes
CREATE INDEX IF NOT EXISTS idx_inference_sessions_model ON inference_sessions(model_id);
CREATE INDEX IF NOT EXISTS idx_inference_sessions_started ON inference_sessions(started_at);
CREATE INDEX IF NOT EXISTS idx_inference_requests_timestamp ON inference_requests(request_timestamp);

-- System metrics indexes
CREATE INDEX IF NOT EXISTS idx_system_metrics_timestamp ON system_metrics(timestamp);
CREATE INDEX IF NOT EXISTS idx_system_metrics_type ON system_metrics(metric_type);

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

-- ============================================================================
-- VIEWS FOR COMMON QUERIES
-- ============================================================================

-- Model summary view
CREATE VIEW IF NOT EXISTS model_summary AS
SELECT 
    mr.id,
    mr.model_name,
    mr.version,
    mr.model_type,
    mr.parameter_count,
    mr.model_size_mb,
    mr.created_at,
    u.username as created_by_username,
    COUNT(tr.id) as training_runs_count,
    MAX(tr.completed_at) as last_training_date,
    AVG(mb.metric_value) as avg_benchmark_score
FROM model_registry mr
LEFT JOIN users u ON mr.created_by = u.id
LEFT JOIN training_runs tr ON mr.id = tr.model_id
LEFT JOIN model_benchmarks mb ON mr.id = mb.model_id
WHERE mr.is_active = 1
GROUP BY mr.id;

-- Training run summary view
CREATE VIEW IF NOT EXISTS training_summary AS
SELECT 
    tr.id,
    tr.run_name,
    tr.status,
    tr.started_at,
    tr.completed_at,
    tr.current_step,
    tr.total_steps,
    tr.best_loss,
    tr.final_loss,
    mr.model_name,
    mr.model_type,
    d.name as dataset_name,
    u.username as started_by_username,
    CASE 
        WHEN tr.completed_at IS NOT NULL 
        THEN (julianday(tr.completed_at) - julianday(tr.started_at)) * 24 * 3600
        ELSE NULL 
    END as duration_seconds
FROM training_runs tr
LEFT JOIN model_registry mr ON tr.model_id = mr.id
LEFT JOIN datasets d ON tr.dataset_id = d.id
LEFT JOIN users u ON tr.started_by = u.id;

-- Dataset statistics view
CREATE VIEW IF NOT EXISTS dataset_stats AS
SELECT 
    d.id,
    d.name,
    d.dataset_type,
    d.total_samples,
    d.total_tokens,
    d.avg_sequence_length,
    d.created_at,
    u.username as created_by_username,
    COUNT(tr.id) as used_in_training_runs
FROM datasets d
LEFT JOIN users u ON d.created_by = u.id
LEFT JOIN training_runs tr ON d.id = tr.dataset_id
GROUP BY d.id;
