"""
Inference utilities for A.D.A.M. SLM
"""

from .inference import AdamInference
from .generation import GenerationConfig, generate_text, batch_generate
from .utils import load_model_for_inference, optimize_for_inference

# Knowledge-enhanced inference
try:
    from .knowledge_inference import KnowledgeEnhancedInference
    KNOWLEDGE_INFERENCE_AVAILABLE = True
except ImportError:
    KNOWLEDGE_INFERENCE_AVAILABLE = False

__all__ = [
    "AdamInference",
    "GenerationConfig",
    "generate_text",
    "batch_generate",
    "load_model_for_inference",
    "optimize_for_inference",
]

if KNOWLEDGE_INFERENCE_AVAILABLE:
    __all__.append("KnowledgeEnhancedInference")
