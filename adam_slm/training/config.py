"""
Training configuration for ADAM SLM
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import json


@dataclass
class TrainingConfig:
    """Configuration for training ADAM SLM"""
    
    # Training hyperparameters
    learning_rate: float = 5e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    
    # Learning rate scheduling
    lr_scheduler_type: str = "cosine"  # "cosine", "linear", "constant"
    warmup_steps: int = 1000
    max_steps: Optional[int] = None
    min_lr_ratio: float = 0.1
    
    # Training dynamics
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    # Mixed precision
    use_amp: bool = True
    amp_dtype: str = "float16"  # "float16" or "bfloat16"
    
    # Evaluation
    eval_steps: int = 500
    eval_batch_size: Optional[int] = None
    save_steps: int = 1000
    logging_steps: int = 100
    
    # Checkpointing
    output_dir: str = "./checkpoints"
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    
    # Data
    max_seq_length: int = 1024
    dataloader_num_workers: int = 0
    dataloader_pin_memory: bool = True
    
    # Regularization
    dropout: float = 0.1
    attention_dropout: float = 0.0
    
    # Advanced training features
    use_gradient_checkpointing: bool = False
    use_deepspeed: bool = False
    deepspeed_config: Optional[Dict[str, Any]] = None
    
    # Logging and monitoring
    report_to: List[str] = None
    run_name: Optional[str] = None
    logging_dir: Optional[str] = None
    
    # Seed for reproducibility
    seed: int = 42
    
    def __post_init__(self):
        """Post-initialization setup"""
        if self.eval_batch_size is None:
            self.eval_batch_size = self.batch_size
            
        if self.report_to is None:
            self.report_to = []
            
        if self.logging_dir is None:
            self.logging_dir = f"{self.output_dir}/logs"
            
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "TrainingConfig":
        """Create config from dictionary"""
        return cls(**config_dict)
        
    @classmethod
    def from_json(cls, json_path: str) -> "TrainingConfig":
        """Load config from JSON file"""
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        
    def to_json(self, json_path: str):
        """Save config to JSON file"""
        with open(json_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


# Predefined training configurations
TRAINING_CONFIGS = {
    "debug": TrainingConfig(
        learning_rate=1e-3,
        batch_size=4,
        gradient_accumulation_steps=1,
        max_steps=100,
        eval_steps=50,
        save_steps=50,
        logging_steps=10,
        warmup_steps=10,
    ),
    
    "small": TrainingConfig(
        learning_rate=5e-4,
        batch_size=16,
        gradient_accumulation_steps=2,
        max_steps=10000,
        eval_steps=500,
        save_steps=1000,
        logging_steps=100,
        warmup_steps=500,
    ),
    
    "base": TrainingConfig(
        learning_rate=3e-4,
        batch_size=32,
        gradient_accumulation_steps=4,
        max_steps=50000,
        eval_steps=1000,
        save_steps=2000,
        logging_steps=100,
        warmup_steps=2000,
    ),
    
    "large": TrainingConfig(
        learning_rate=1e-4,
        batch_size=64,
        gradient_accumulation_steps=8,
        max_steps=100000,
        eval_steps=2000,
        save_steps=5000,
        logging_steps=100,
        warmup_steps=5000,
        use_gradient_checkpointing=True,
    ),
}


def get_training_config(config_name: str) -> TrainingConfig:
    """Get predefined training configuration by name"""
    if config_name not in TRAINING_CONFIGS:
        raise ValueError(f"Unknown training config: {config_name}. Available: {list(TRAINING_CONFIGS.keys())}")
    return TRAINING_CONFIGS[config_name]
