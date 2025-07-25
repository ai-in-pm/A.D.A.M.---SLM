"""
Analytics and reporting for ADAM SLM database
"""

import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import statistics

from database import AdamSLMDatabase


class DatabaseAnalytics:
    """
    Advanced analytics for ADAM SLM database
    
    Provides insights into model performance, training trends,
    dataset usage, and system metrics.
    """
    
    def __init__(self, db: AdamSLMDatabase):
        self.db = db
        
    # ========================================================================
    # MODEL ANALYTICS
    # ========================================================================
    
    def get_model_performance_comparison(self, model_ids: List[int]) -> Dict:
        """Compare performance across multiple models"""
        
        comparison = {
            'models': [],
            'benchmarks': {},
            'summary': {}
        }
        
        for model_id in model_ids:
            # Get model info
            model = self.db.get_model(model_id)
            if not model:
                continue
                
            # Get benchmarks
            benchmarks = self.db.execute_query("""
                SELECT benchmark_type, metric_value, dataset_name, benchmark_date
                FROM model_benchmarks 
                WHERE model_id = ?
                ORDER BY benchmark_date DESC
            """, (model_id,))
            
            model_data = {
                'id': model_id,
                'name': model['model_name'],
                'version': model['version'],
                'type': model['model_type'],
                'parameters': model['parameter_count'],
                'benchmarks': benchmarks
            }
            
            comparison['models'].append(model_data)
            
            # Aggregate benchmarks by type
            for benchmark in benchmarks:
                bench_type = benchmark['benchmark_type']
                if bench_type not in comparison['benchmarks']:
                    comparison['benchmarks'][bench_type] = []
                    
                comparison['benchmarks'][bench_type].append({
                    'model_id': model_id,
                    'model_name': model['model_name'],
                    'value': benchmark['metric_value'],
                    'dataset': benchmark['dataset_name']
                })
                
        # Calculate summary statistics
        for bench_type, values in comparison['benchmarks'].items():
            metric_values = [v['value'] for v in values]
            if metric_values:
                comparison['summary'][bench_type] = {
                    'best_value': max(metric_values),
                    'worst_value': min(metric_values),
                    'avg_value': statistics.mean(metric_values),
                    'std_dev': statistics.stdev(metric_values) if len(metric_values) > 1 else 0,
                    'best_model': max(values, key=lambda x: x['value'])['model_name']
                }
                
        return comparison
        
    def get_model_lineage(self, model_id: int) -> Dict:
        """Get model family tree and evolution"""
        
        lineage = {
            'root_model': None,
            'ancestors': [],
            'descendants': [],
            'siblings': []
        }
        
        # Get current model
        current_model = self.db.get_model(model_id)
        if not current_model:
            return lineage
            
        # Find root model (traverse up the parent chain)
        root_id = model_id
        ancestors = []
        
        while True:
            model = self.db.get_model(root_id)
            if not model or not model['parent_model_id']:
                lineage['root_model'] = model
                break
            ancestors.append(model)
            root_id = model['parent_model_id']
            
        lineage['ancestors'] = list(reversed(ancestors))
        
        # Find descendants (models that have this as parent)
        descendants = self.db.execute_query("""
            SELECT * FROM model_registry 
            WHERE parent_model_id = ? AND is_active = 1
            ORDER BY created_at
        """, (model_id,))
        
        lineage['descendants'] = descendants
        
        # Find siblings (models with same parent)
        if current_model['parent_model_id']:
            siblings = self.db.execute_query("""
                SELECT * FROM model_registry 
                WHERE parent_model_id = ? AND id != ? AND is_active = 1
                ORDER BY created_at
            """, (current_model['parent_model_id'], model_id))
            
            lineage['siblings'] = siblings
            
        return lineage
        
    # ========================================================================
    # TRAINING ANALYTICS
    # ========================================================================
    
    def get_training_trends(self, days: int = 30) -> Dict:
        """Analyze training trends over time"""
        
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # Training runs over time
        runs_by_day = self.db.execute_query("""
            SELECT 
                DATE(started_at) as date,
                COUNT(*) as runs_started,
                COUNT(CASE WHEN status = 'completed' THEN 1 END) as runs_completed,
                AVG(CASE WHEN status = 'completed' THEN training_time_seconds END) as avg_training_time
            FROM training_runs 
            WHERE started_at >= ?
            GROUP BY DATE(started_at)
            ORDER BY date
        """, (cutoff_date,))
        
        # Success rate trends
        success_rate = self.db.execute_query("""
            SELECT 
                COUNT(CASE WHEN status = 'completed' THEN 1 END) * 100.0 / COUNT(*) as success_rate,
                COUNT(*) as total_runs
            FROM training_runs 
            WHERE started_at >= ?
        """, (cutoff_date,))
        
        # Loss improvement trends
        loss_trends = self.db.execute_query("""
            SELECT 
                tr.id,
                tr.run_name,
                tr.started_at,
                tr.best_loss,
                tr.final_loss,
                mr.model_type
            FROM training_runs tr
            LEFT JOIN model_registry mr ON tr.model_id = mr.id
            WHERE tr.started_at >= ? AND tr.status = 'completed'
            ORDER BY tr.started_at
        """, (cutoff_date,))
        
        # Resource utilization
        resource_usage = self.db.execute_query("""
            SELECT 
                SUM(training_time_seconds) as total_training_seconds,
                SUM(gpu_hours) as total_gpu_hours,
                AVG(gpu_hours) as avg_gpu_hours_per_run,
                SUM(total_tokens_processed) as total_tokens_processed
            FROM training_runs 
            WHERE started_at >= ? AND status = 'completed'
        """, (cutoff_date,))
        
        return {
            'period_days': days,
            'runs_by_day': runs_by_day,
            'success_rate': success_rate[0] if success_rate else {},
            'loss_trends': loss_trends,
            'resource_usage': resource_usage[0] if resource_usage else {},
            'generated_at': datetime.now().isoformat()
        }
        
    def get_hyperparameter_analysis(self, experiment_id: Optional[int] = None) -> Dict:
        """Analyze hyperparameter impact on training outcomes"""
        
        # Get training runs with their configs
        if experiment_id:
            query = """
                SELECT tr.*, er.run_purpose
                FROM training_runs tr
                JOIN experiment_runs er ON tr.id = er.training_run_id
                WHERE er.experiment_id = ? AND tr.status = 'completed'
            """
            params = (experiment_id,)
        else:
            query = """
                SELECT * FROM training_runs 
                WHERE status = 'completed' AND training_config IS NOT NULL
                LIMIT 100
            """
            params = ()
            
        runs = self.db.execute_query(query, params)
        
        analysis = {
            'hyperparameters': {},
            'correlations': {},
            'best_configs': []
        }
        
        # Extract hyperparameters and outcomes
        configs_and_outcomes = []
        
        for run in runs:
            try:
                config = json.loads(run['training_config'])
                outcome = {
                    'final_loss': run['final_loss'],
                    'best_loss': run['best_loss'],
                    'training_time': run['training_time_seconds'],
                    'run_id': run['id']
                }
                
                configs_and_outcomes.append({
                    'config': config,
                    'outcome': outcome,
                    'run': run
                })
                
            except (json.JSONDecodeError, TypeError):
                continue
                
        # Analyze each hyperparameter
        if configs_and_outcomes:
            # Group by hyperparameter values
            hyperparam_groups = {}
            
            for item in configs_and_outcomes:
                config = item['config']
                outcome = item['outcome']
                
                for param, value in config.items():
                    if isinstance(value, (int, float, str)):
                        if param not in hyperparam_groups:
                            hyperparam_groups[param] = {}
                            
                        value_str = str(value)
                        if value_str not in hyperparam_groups[param]:
                            hyperparam_groups[param][value_str] = []
                            
                        hyperparam_groups[param][value_str].append(outcome)
                        
            # Calculate statistics for each hyperparameter
            for param, value_groups in hyperparam_groups.items():
                param_analysis = {}
                
                for value, outcomes in value_groups.items():
                    if len(outcomes) > 0:
                        final_losses = [o['final_loss'] for o in outcomes if o['final_loss'] is not None]
                        
                        if final_losses:
                            param_analysis[value] = {
                                'count': len(outcomes),
                                'avg_final_loss': statistics.mean(final_losses),
                                'min_final_loss': min(final_losses),
                                'max_final_loss': max(final_losses),
                                'std_final_loss': statistics.stdev(final_losses) if len(final_losses) > 1 else 0
                            }
                            
                analysis['hyperparameters'][param] = param_analysis
                
            # Find best configurations
            best_runs = sorted(
                configs_and_outcomes,
                key=lambda x: x['outcome']['final_loss'] or float('inf')
            )[:5]
            
            analysis['best_configs'] = [
                {
                    'run_id': item['run']['id'],
                    'run_name': item['run']['run_name'],
                    'final_loss': item['outcome']['final_loss'],
                    'config': item['config']
                }
                for item in best_runs
            ]
            
        return analysis
        
    # ========================================================================
    # DATASET ANALYTICS
    # ========================================================================
    
    def get_dataset_usage_analysis(self) -> Dict:
        """Analyze dataset usage patterns"""
        
        # Dataset usage frequency
        usage_stats = self.db.execute_query("""
            SELECT 
                d.id,
                d.name,
                d.dataset_type,
                d.total_samples,
                d.total_tokens,
                COUNT(tr.id) as times_used,
                AVG(tr.final_loss) as avg_final_loss,
                MIN(tr.final_loss) as best_final_loss
            FROM datasets d
            LEFT JOIN training_runs tr ON d.id = tr.dataset_id
            GROUP BY d.id
            ORDER BY times_used DESC
        """)
        
        # Dataset performance correlation
        performance_by_size = self.db.execute_query("""
            SELECT 
                d.total_tokens,
                d.total_samples,
                AVG(tr.final_loss) as avg_loss,
                COUNT(tr.id) as run_count
            FROM datasets d
            JOIN training_runs tr ON d.id = tr.dataset_id
            WHERE tr.status = 'completed' AND tr.final_loss IS NOT NULL
            GROUP BY d.id
            HAVING run_count > 0
            ORDER BY d.total_tokens
        """)
        
        # Dataset type effectiveness
        type_effectiveness = self.db.execute_query("""
            SELECT 
                d.dataset_type,
                COUNT(tr.id) as total_runs,
                AVG(tr.final_loss) as avg_final_loss,
                MIN(tr.final_loss) as best_final_loss,
                AVG(tr.training_time_seconds) as avg_training_time
            FROM datasets d
            JOIN training_runs tr ON d.id = tr.dataset_id
            WHERE tr.status = 'completed'
            GROUP BY d.dataset_type
            ORDER BY avg_final_loss
        """)
        
        return {
            'usage_stats': usage_stats,
            'performance_by_size': performance_by_size,
            'type_effectiveness': type_effectiveness,
            'generated_at': datetime.now().isoformat()
        }
        
    # ========================================================================
    # SYSTEM ANALYTICS
    # ========================================================================
    
    def get_system_performance_report(self, hours: int = 24) -> Dict:
        """Generate system performance report"""
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        # Resource utilization over time
        resource_metrics = self.db.execute_query("""
            SELECT 
                metric_type,
                metric_name,
                AVG(metric_value) as avg_value,
                MAX(metric_value) as max_value,
                MIN(metric_value) as min_value,
                COUNT(*) as sample_count
            FROM system_metrics 
            WHERE timestamp >= ?
            GROUP BY metric_type, metric_name
            ORDER BY metric_type, metric_name
        """, (cutoff_time,))
        
        # Database operation statistics
        operation_stats = self.db.execute_query("""
            SELECT 
                operation_type,
                table_name,
                COUNT(*) as operation_count,
                COUNT(DISTINCT user_id) as unique_users
            FROM operation_logs 
            WHERE timestamp >= ?
            GROUP BY operation_type, table_name
            ORDER BY operation_count DESC
        """, (cutoff_time,))
        
        # Active sessions
        active_sessions = self.db.execute_query("""
            SELECT COUNT(*) as active_sessions
            FROM sessions 
            WHERE is_active = 1 AND expires_at > CURRENT_TIMESTAMP
        """)
        
        # Database size and growth
        db_stats = self.db.execute_query("""
            SELECT 
                name as table_name,
                COUNT(*) as row_count
            FROM sqlite_master 
            WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
        """)
        
        # Add row counts for each table
        table_stats = []
        for table in db_stats:
            try:
                count_result = self.db.execute_query(f"SELECT COUNT(*) as count FROM {table['table_name']}")
                table_stats.append({
                    'table_name': table['table_name'],
                    'row_count': count_result[0]['count']
                })
            except Exception:
                table_stats.append({
                    'table_name': table['table_name'],
                    'row_count': 0
                })
                
        return {
            'period_hours': hours,
            'resource_metrics': resource_metrics,
            'operation_stats': operation_stats,
            'active_sessions': active_sessions[0]['active_sessions'] if active_sessions else 0,
            'table_stats': table_stats,
            'generated_at': datetime.now().isoformat()
        }
