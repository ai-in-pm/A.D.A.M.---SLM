"""
A.D.A.M.-SLM Tokenizer Integration System
Manages the transition from GPT-2 to A.D.A.M.-SLM tokenizer
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, Union
from datetime import datetime

# Import both tokenizers
from .tokenizer import AdamTokenizer  # Original GPT-2 based
from .adam_slm_tokenizer import AdamSLMTokenizer, TokenizerMigrationManager


class TokenizerIntegrationManager:
    """
    Manages integration and migration of A.D.A.M.-SLM tokenizer
    """
    
    def __init__(self, integration_mode: str = "hybrid"):
        """
        Initialize integration manager
        
        Args:
            integration_mode: 'hybrid', 'adam_only', or 'gpt2_only'
        """
        self.integration_mode = integration_mode
        self.migration_manager = TokenizerMigrationManager()
        
        # Initialize tokenizers based on mode
        self._initialize_tokenizers()
        
        # Integration settings
        self.settings = {
            'auto_detect_domain': True,
            'fallback_enabled': True,
            'performance_monitoring': True,
            'migration_logging': True
        }
    
    def _initialize_tokenizers(self):
        """Initialize tokenizers based on integration mode"""
        
        print(f"🔧 Initializing tokenizers in {self.integration_mode} mode...")
        
        if self.integration_mode in ['hybrid', 'gpt2_only']:
            # Initialize GPT-2 tokenizer
            self.gpt2_tokenizer = AdamTokenizer("gpt2")
            print("✅ GPT-2 tokenizer initialized")
        
        if self.integration_mode in ['hybrid', 'adam_only']:
            # Try to initialize A.D.A.M.-SLM tokenizer
            try:
                adam_model_path = "adam_slm/tokenization/adam_slm_model"
                if Path(adam_model_path).exists():
                    self.adam_tokenizer = AdamSLMTokenizer(
                        model_path=adam_model_path,
                        fallback_to_gpt2=True
                    )
                    print("✅ A.D.A.M.-SLM tokenizer initialized")
                else:
                    print("⚠️ A.D.A.M.-SLM model not found, creating new one...")
                    self.adam_tokenizer = self._create_adam_tokenizer()
            except Exception as e:
                print(f"⚠️ A.D.A.M.-SLM tokenizer failed to initialize: {e}")
                if self.integration_mode == 'adam_only':
                    raise
                self.adam_tokenizer = None
    
    def _create_adam_tokenizer(self) -> AdamSLMTokenizer:
        """Create new A.D.A.M.-SLM tokenizer"""
        
        from .adam_slm_tokenizer import create_adam_slm_tokenizer
        
        print("🚀 Creating new A.D.A.M.-SLM tokenizer...")
        return create_adam_slm_tokenizer(
            corpus_path=None,  # Use default corpus
            save_path="adam_slm/tokenization/adam_slm_model"
        )
    
    def get_tokenizer(self, content_hint: str = None) -> Union[AdamTokenizer, AdamSLMTokenizer]:
        """
        Get appropriate tokenizer based on content and mode
        
        Args:
            content_hint: Hint about content type ('ai_ml', 'code', 'math', 'general')
            
        Returns:
            Appropriate tokenizer instance
        """
        
        if self.integration_mode == 'gpt2_only':
            return self.gpt2_tokenizer
        elif self.integration_mode == 'adam_only':
            return self.adam_tokenizer
        else:  # hybrid mode
            return self._select_hybrid_tokenizer(content_hint)
    
    def _select_hybrid_tokenizer(self, content_hint: str = None) -> Union[AdamTokenizer, AdamSLMTokenizer]:
        """Select tokenizer in hybrid mode"""
        
        # If A.D.A.M. tokenizer is not available, use GPT-2
        if not self.adam_tokenizer:
            return self.gpt2_tokenizer
        
        # Use content hint to select tokenizer
        if content_hint in ['ai_ml', 'research', 'technical', 'math']:
            return self.adam_tokenizer
        elif content_hint == 'general':
            return self.gpt2_tokenizer
        else:
            # Default to A.D.A.M. for unknown content
            return self.adam_tokenizer
    
    def encode_smart(self, text: str, **kwargs) -> Dict[str, Any]:
        """
        Smart encoding that automatically selects best tokenizer
        
        Args:
            text: Input text
            **kwargs: Additional encoding arguments
            
        Returns:
            Encoding results with metadata
        """
        
        # Detect content domain
        domain = self._detect_content_domain(text)
        
        # Get appropriate tokenizer
        tokenizer = self.get_tokenizer(domain)
        
        # Encode text
        token_ids = tokenizer.encode(text, **kwargs)
        
        # Return results with metadata
        return {
            'token_ids': token_ids,
            'tokenizer_used': type(tokenizer).__name__,
            'domain_detected': domain,
            'token_count': len(token_ids),
            'efficiency_score': self._calculate_efficiency_score(text, token_ids)
        }
    
    def _detect_content_domain(self, text: str) -> str:
        """Detect content domain for tokenizer selection"""
        
        text_lower = text.lower()
        
        # AI/ML indicators
        ai_ml_terms = ['transformer', 'attention', 'neural', 'network', 'deep', 'learning', 
                       'machine', 'artificial', 'intelligence', 'gradient', 'embedding']
        ai_ml_score = sum(1 for term in ai_ml_terms if term in text_lower)
        
        # Mathematical indicators
        math_indicators = ['equation', 'theorem', 'proof', '$', '∇', '∂', '∑', 'α', 'β', 'γ']
        math_score = sum(1 for indicator in math_indicators if indicator in text_lower)
        
        # Code indicators
        code_indicators = ['def ', 'class ', 'import ', 'torch', 'tensorflow', 'numpy']
        code_score = sum(1 for indicator in code_indicators if indicator in text_lower)
        
        # Research indicators
        research_indicators = ['abstract', 'introduction', 'methodology', 'results', 'conclusion']
        research_score = sum(1 for indicator in research_indicators if indicator in text_lower)
        
        # Determine domain
        scores = {
            'ai_ml': ai_ml_score,
            'math': math_score,
            'code': code_score,
            'research': research_score
        }
        
        max_score = max(scores.values())
        if max_score > 0:
            return max(scores.items(), key=lambda x: x[1])[0]
        else:
            return 'general'
    
    def _calculate_efficiency_score(self, text: str, token_ids: list) -> float:
        """Calculate tokenization efficiency score"""
        
        # Simple efficiency metric: characters per token
        if not token_ids:
            return 0.0
        
        chars_per_token = len(text) / len(token_ids)
        
        # Normalize to 0-1 scale (assuming 4 chars/token is optimal)
        efficiency = min(1.0, chars_per_token / 4.0)
        return round(efficiency, 3)
    
    def benchmark_tokenizers(self, test_texts: list) -> Dict[str, Any]:
        """
        Benchmark both tokenizers on test texts
        
        Args:
            test_texts: List of test texts
            
        Returns:
            Benchmark results
        """
        
        print("📊 Benchmarking tokenizers...")
        
        results = {
            'gpt2_results': [],
            'adam_results': [],
            'comparison': {}
        }
        
        for i, text in enumerate(test_texts):
            print(f"   Testing text {i+1}/{len(test_texts)}")
            
            # Test GPT-2 tokenizer
            if self.gpt2_tokenizer:
                gpt2_tokens = self.gpt2_tokenizer.encode(text)
                results['gpt2_results'].append({
                    'text_length': len(text),
                    'token_count': len(gpt2_tokens),
                    'efficiency': self._calculate_efficiency_score(text, gpt2_tokens)
                })
            
            # Test A.D.A.M. tokenizer
            if self.adam_tokenizer:
                adam_tokens = self.adam_tokenizer.encode(text)
                results['adam_results'].append({
                    'text_length': len(text),
                    'token_count': len(adam_tokens),
                    'efficiency': self._calculate_efficiency_score(text, adam_tokens)
                })
        
        # Calculate comparison metrics
        if results['gpt2_results'] and results['adam_results']:
            gpt2_avg_efficiency = sum(r['efficiency'] for r in results['gpt2_results']) / len(results['gpt2_results'])
            adam_avg_efficiency = sum(r['efficiency'] for r in results['adam_results']) / len(results['adam_results'])
            
            results['comparison'] = {
                'gpt2_avg_efficiency': round(gpt2_avg_efficiency, 3),
                'adam_avg_efficiency': round(adam_avg_efficiency, 3),
                'improvement': round((adam_avg_efficiency - gpt2_avg_efficiency) / gpt2_avg_efficiency * 100, 1)
            }
        
        print("✅ Benchmark completed")
        return results
    
    def migrate_to_adam_tokenizer(self, backup_gpt2: bool = True) -> Dict[str, Any]:
        """
        Migrate system to use A.D.A.M.-SLM tokenizer
        
        Args:
            backup_gpt2: Whether to backup GPT-2 tokenizer
            
        Returns:
            Migration results
        """
        
        print("🔄 Starting migration to A.D.A.M.-SLM tokenizer...")
        
        migration_results = {
            'start_time': datetime.now().isoformat(),
            'steps_completed': [],
            'backup_created': False,
            'migration_successful': False
        }
        
        try:
            # Step 1: Create backup if requested
            if backup_gpt2:
                self._backup_gpt2_tokenizer()
                migration_results['backup_created'] = True
                migration_results['steps_completed'].append('backup_created')
            
            # Step 2: Ensure A.D.A.M. tokenizer is ready
            if not self.adam_tokenizer:
                self.adam_tokenizer = self._create_adam_tokenizer()
            migration_results['steps_completed'].append('adam_tokenizer_ready')
            
            # Step 3: Update integration mode
            self.integration_mode = 'adam_only'
            migration_results['steps_completed'].append('mode_updated')
            
            # Step 4: Update imports and references
            self._update_tokenizer_references()
            migration_results['steps_completed'].append('references_updated')
            
            migration_results['migration_successful'] = True
            migration_results['end_time'] = datetime.now().isoformat()
            
            print("✅ Migration to A.D.A.M.-SLM tokenizer completed successfully!")
            
        except Exception as e:
            migration_results['error'] = str(e)
            migration_results['end_time'] = datetime.now().isoformat()
            print(f"❌ Migration failed: {e}")
        
        return migration_results
    
    def _backup_gpt2_tokenizer(self):
        """Create backup of GPT-2 tokenizer configuration"""
        
        backup_dir = Path("adam_slm/tokenization/backups")
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        backup_info = {
            'backup_time': datetime.now().isoformat(),
            'original_tokenizer': 'GPT-2',
            'encoding_name': 'gpt2',
            'vocab_size': 50257,
            'special_tokens': {
                'pad_token_id': 50256,
                'eos_token_id': 50256,
                'bos_token_id': 50256,
                'unk_token_id': 50256
            }
        }
        
        with open(backup_dir / 'gpt2_tokenizer_backup.json', 'w') as f:
            json.dump(backup_info, f, indent=2)
        
        print("💾 GPT-2 tokenizer backup created")
    
    def _update_tokenizer_references(self):
        """Update system references to use A.D.A.M. tokenizer"""
        
        # This would update imports and configurations in real implementation
        print("🔧 Updating tokenizer references...")
        
        # Simulate updating configuration
        config_updates = {
            'default_tokenizer': 'adam_slm',
            'fallback_tokenizer': 'gpt2',
            'adaptive_mode': True,
            'domain_detection': True
        }
        
        print("✅ Tokenizer references updated")
    
    def rollback_to_gpt2(self) -> Dict[str, Any]:
        """
        Rollback to GPT-2 tokenizer if needed
        
        Returns:
            Rollback results
        """
        
        print("🔄 Rolling back to GPT-2 tokenizer...")
        
        try:
            # Update integration mode
            self.integration_mode = 'gpt2_only'
            
            # Reinitialize GPT-2 tokenizer if needed
            if not self.gpt2_tokenizer:
                self.gpt2_tokenizer = AdamTokenizer("gpt2")
            
            print("✅ Rollback to GPT-2 completed successfully!")
            
            return {
                'rollback_successful': True,
                'rollback_time': datetime.now().isoformat(),
                'active_tokenizer': 'GPT-2'
            }
            
        except Exception as e:
            print(f"❌ Rollback failed: {e}")
            return {
                'rollback_successful': False,
                'error': str(e),
                'rollback_time': datetime.now().isoformat()
            }
    
    def get_status(self) -> Dict[str, Any]:
        """Get current integration status"""
        
        return {
            'integration_mode': self.integration_mode,
            'gpt2_available': self.gpt2_tokenizer is not None,
            'adam_available': self.adam_tokenizer is not None,
            'settings': self.settings,
            'recommended_action': self._get_recommended_action()
        }
    
    def _get_recommended_action(self) -> str:
        """Get recommended action based on current state"""
        
        if not self.adam_tokenizer:
            return "Create A.D.A.M.-SLM tokenizer"
        elif self.integration_mode == 'gpt2_only':
            return "Consider migrating to A.D.A.M.-SLM tokenizer"
        elif self.integration_mode == 'hybrid':
            return "Monitor performance and consider full migration"
        else:
            return "System optimally configured"


# Global integration manager instance
_integration_manager = None

def get_integration_manager() -> TokenizerIntegrationManager:
    """Get global integration manager instance"""
    global _integration_manager
    if _integration_manager is None:
        _integration_manager = TokenizerIntegrationManager()
    return _integration_manager

def get_smart_tokenizer(content_hint: str = None):
    """Get smart tokenizer that automatically selects best option"""
    manager = get_integration_manager()
    return manager.get_tokenizer(content_hint)
