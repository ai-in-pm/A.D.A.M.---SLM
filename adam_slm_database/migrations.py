"""
Database migrations for ADAM SLM
"""

import os
import sqlite3
from datetime import datetime
from typing import List, Dict, Any
import logging

from database import AdamSLMDatabase


class DatabaseMigrations:
    """
    Database migration system for ADAM SLM
    
    Handles schema updates, data migrations, and version management.
    """
    
    def __init__(self, db: AdamSLMDatabase):
        self.db = db
        self.logger = logging.getLogger(__name__)
        
        # Ensure migration tracking table exists
        self._create_migration_table()
        
    def _create_migration_table(self):
        """Create migration tracking table"""
        query = """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            version VARCHAR(20) NOT NULL UNIQUE,
            description TEXT,
            applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            rollback_sql TEXT
        )
        """
        
        with self.db.get_connection() as conn:
            conn.execute(query)
            conn.commit()
            
    def get_current_version(self) -> str:
        """Get current database schema version"""
        try:
            result = self.db.execute_query(
                "SELECT version FROM schema_migrations ORDER BY applied_at DESC LIMIT 1"
            )
            return result[0]['version'] if result else "0.0.0"
        except Exception:
            return "0.0.0"
            
    def get_applied_migrations(self) -> List[Dict]:
        """Get list of applied migrations"""
        return self.db.execute_query(
            "SELECT * FROM schema_migrations ORDER BY applied_at"
        )
        
    def apply_migration(
        self,
        version: str,
        description: str,
        migration_sql: str,
        rollback_sql: str = None
    ) -> bool:
        """Apply a database migration"""
        
        # Check if migration already applied
        existing = self.db.execute_query(
            "SELECT id FROM schema_migrations WHERE version = ?", (version,)
        )
        
        if existing:
            self.logger.info(f"Migration {version} already applied")
            return True
            
        try:
            with self.db.get_connection() as conn:
                # Apply migration
                conn.executescript(migration_sql)
                
                # Record migration
                conn.execute("""
                    INSERT INTO schema_migrations (version, description, rollback_sql)
                    VALUES (?, ?, ?)
                """, (version, description, rollback_sql))
                
                conn.commit()
                
            self.logger.info(f"Applied migration {version}: {description}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to apply migration {version}: {e}")
            return False
            
    def rollback_migration(self, version: str) -> bool:
        """Rollback a specific migration"""
        
        # Get migration details
        migration = self.db.execute_query(
            "SELECT * FROM schema_migrations WHERE version = ?", (version,)
        )
        
        if not migration:
            self.logger.error(f"Migration {version} not found")
            return False
            
        migration = migration[0]
        rollback_sql = migration['rollback_sql']
        
        if not rollback_sql:
            self.logger.error(f"No rollback SQL for migration {version}")
            return False
            
        try:
            with self.db.get_connection() as conn:
                # Apply rollback
                conn.executescript(rollback_sql)
                
                # Remove migration record
                conn.execute(
                    "DELETE FROM schema_migrations WHERE version = ?", (version,)
                )
                
                conn.commit()
                
            self.logger.info(f"Rolled back migration {version}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to rollback migration {version}: {e}")
            return False
            
    def migrate_to_latest(self) -> bool:
        """Apply all pending migrations"""
        
        current_version = self.get_current_version()
        migrations = self._get_pending_migrations(current_version)
        
        if not migrations:
            self.logger.info("No pending migrations")
            return True
            
        success = True
        for migration in migrations:
            if not self.apply_migration(**migration):
                success = False
                break
                
        return success
        
    def _get_pending_migrations(self, current_version: str) -> List[Dict]:
        """Get list of pending migrations"""
        
        # Define migrations in order
        migrations = [
            {
                'version': '1.0.0',
                'description': 'Initial schema',
                'migration_sql': self._get_initial_schema(),
                'rollback_sql': self._get_initial_rollback()
            },
            {
                'version': '1.1.0',
                'description': 'Add inference tracking',
                'migration_sql': self._get_inference_schema(),
                'rollback_sql': self._get_inference_rollback()
            },
            {
                'version': '1.2.0',
                'description': 'Add system monitoring',
                'migration_sql': self._get_monitoring_schema(),
                'rollback_sql': self._get_monitoring_rollback()
            },
            {
                'version': '1.3.0',
                'description': 'Add advanced indexes',
                'migration_sql': self._get_indexes_schema(),
                'rollback_sql': self._get_indexes_rollback()
            }
        ]
        
        # Filter out already applied migrations
        applied_versions = {m['version'] for m in self.get_applied_migrations()}
        pending = [m for m in migrations if m['version'] not in applied_versions]
        
        return pending
        
    def _get_initial_schema(self) -> str:
        """Get initial schema SQL"""
        schema_path = os.path.join(os.path.dirname(__file__), "schema.sql")
        with open(schema_path, 'r') as f:
            return f.read()
            
    def _get_initial_rollback(self) -> str:
        """Get initial schema rollback SQL"""
        return """
        DROP VIEW IF EXISTS dataset_stats;
        DROP VIEW IF EXISTS training_summary;
        DROP VIEW IF EXISTS model_summary;
        
        DROP TABLE IF EXISTS operation_logs;
        DROP TABLE IF EXISTS system_metrics;
        DROP TABLE IF EXISTS inference_requests;
        DROP TABLE IF EXISTS inference_sessions;
        DROP TABLE IF EXISTS experiment_runs;
        DROP TABLE IF EXISTS experiments;
        DROP TABLE IF EXISTS dataset_splits;
        DROP TABLE IF EXISTS datasets;
        DROP TABLE IF EXISTS training_metrics;
        DROP TABLE IF EXISTS training_runs;
        DROP TABLE IF EXISTS model_benchmarks;
        DROP TABLE IF EXISTS model_registry;
        DROP TABLE IF EXISTS sessions;
        DROP TABLE IF EXISTS users;
        """
        
    def _get_inference_schema(self) -> str:
        """Get inference tracking schema additions"""
        return """
        -- Add inference performance tracking
        ALTER TABLE inference_sessions ADD COLUMN performance_score REAL;
        ALTER TABLE inference_sessions ADD COLUMN error_rate REAL;
        
        -- Add request categorization
        ALTER TABLE inference_requests ADD COLUMN request_category VARCHAR(50);
        ALTER TABLE inference_requests ADD COLUMN complexity_score REAL;
        
        -- Create inference performance view
        CREATE VIEW IF NOT EXISTS inference_performance AS
        SELECT 
            is.id,
            is.model_id,
            is.session_name,
            is.total_requests,
            is.total_tokens_generated,
            is.avg_response_time_ms,
            AVG(ir.response_time_ms) as actual_avg_response_time,
            COUNT(CASE WHEN ir.success = 0 THEN 1 END) * 100.0 / COUNT(*) as error_rate,
            SUM(ir.generated_tokens) as total_tokens_in_requests
        FROM inference_sessions is
        LEFT JOIN inference_requests ir ON is.id = ir.session_id
        GROUP BY is.id;
        """
        
    def _get_inference_rollback(self) -> str:
        """Get inference tracking rollback SQL"""
        return """
        DROP VIEW IF EXISTS inference_performance;
        
        -- Note: SQLite doesn't support DROP COLUMN, so we'd need to recreate tables
        -- For now, just document that these columns were added
        """
        
    def _get_monitoring_schema(self) -> str:
        """Get system monitoring schema additions"""
        return """
        -- Add monitoring alerts table
        CREATE TABLE IF NOT EXISTS monitoring_alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            alert_type VARCHAR(50) NOT NULL,
            severity VARCHAR(20) NOT NULL, -- 'low', 'medium', 'high', 'critical'
            message TEXT NOT NULL,
            details TEXT, -- JSON
            triggered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            resolved_at TIMESTAMP,
            resolved_by INTEGER,
            is_active BOOLEAN DEFAULT 1,
            FOREIGN KEY (resolved_by) REFERENCES users(id)
        );
        
        -- Add system health checks
        CREATE TABLE IF NOT EXISTS health_checks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            check_name VARCHAR(100) NOT NULL,
            check_type VARCHAR(50) NOT NULL, -- 'database', 'disk', 'memory', 'model'
            status VARCHAR(20) NOT NULL, -- 'healthy', 'warning', 'critical'
            last_check TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            check_details TEXT, -- JSON
            next_check TIMESTAMP
        );
        
        -- Create monitoring dashboard view
        CREATE VIEW IF NOT EXISTS monitoring_dashboard AS
        SELECT 
            'alerts' as metric_type,
            COUNT(CASE WHEN is_active = 1 THEN 1 END) as active_count,
            COUNT(CASE WHEN severity = 'critical' AND is_active = 1 THEN 1 END) as critical_count,
            MAX(triggered_at) as last_alert
        FROM monitoring_alerts
        UNION ALL
        SELECT 
            'health_checks' as metric_type,
            COUNT(*) as active_count,
            COUNT(CASE WHEN status = 'critical' THEN 1 END) as critical_count,
            MAX(last_check) as last_alert
        FROM health_checks;
        """
        
    def _get_monitoring_rollback(self) -> str:
        """Get monitoring rollback SQL"""
        return """
        DROP VIEW IF EXISTS monitoring_dashboard;
        DROP TABLE IF EXISTS health_checks;
        DROP TABLE IF EXISTS monitoring_alerts;
        """
        
    def _get_indexes_schema(self) -> str:
        """Get advanced indexes schema"""
        return """
        -- Additional performance indexes
        CREATE INDEX IF NOT EXISTS idx_training_metrics_composite ON training_metrics(training_run_id, metric_name, step);
        CREATE INDEX IF NOT EXISTS idx_model_benchmarks_composite ON model_benchmarks(model_id, benchmark_type, benchmark_date);
        CREATE INDEX IF NOT EXISTS idx_inference_requests_session_time ON inference_requests(session_id, request_timestamp);
        CREATE INDEX IF NOT EXISTS idx_operation_logs_user_time ON operation_logs(user_id, timestamp);
        CREATE INDEX IF NOT EXISTS idx_system_metrics_composite ON system_metrics(metric_type, metric_name, timestamp);
        
        -- Full-text search indexes (if supported)
        CREATE INDEX IF NOT EXISTS idx_models_description_fts ON model_registry(description);
        CREATE INDEX IF NOT EXISTS idx_experiments_description_fts ON experiments(description);
        CREATE INDEX IF NOT EXISTS idx_datasets_description_fts ON datasets(description);
        """
        
    def _get_indexes_rollback(self) -> str:
        """Get indexes rollback SQL"""
        return """
        DROP INDEX IF EXISTS idx_datasets_description_fts;
        DROP INDEX IF EXISTS idx_experiments_description_fts;
        DROP INDEX IF EXISTS idx_models_description_fts;
        DROP INDEX IF EXISTS idx_system_metrics_composite;
        DROP INDEX IF EXISTS idx_operation_logs_user_time;
        DROP INDEX IF EXISTS idx_inference_requests_session_time;
        DROP INDEX IF EXISTS idx_model_benchmarks_composite;
        DROP INDEX IF EXISTS idx_training_metrics_composite;
        """
        
    def backup_database(self, backup_path: str) -> bool:
        """Create database backup before migrations"""
        try:
            with self.db.get_connection() as conn:
                backup = sqlite3.connect(backup_path)
                conn.backup(backup)
                backup.close()
                
            self.logger.info(f"Database backed up to {backup_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to backup database: {e}")
            return False
            
    def restore_database(self, backup_path: str) -> bool:
        """Restore database from backup"""
        try:
            if not os.path.exists(backup_path):
                self.logger.error(f"Backup file not found: {backup_path}")
                return False
                
            # Close current connection and replace database file
            backup = sqlite3.connect(backup_path)
            
            with self.db.get_connection() as conn:
                backup.backup(conn)
                
            backup.close()
            
            self.logger.info(f"Database restored from {backup_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to restore database: {e}")
            return False
