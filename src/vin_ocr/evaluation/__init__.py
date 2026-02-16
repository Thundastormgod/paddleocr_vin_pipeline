"""
VIN OCR Evaluation Module
=========================

Comprehensive evaluation tools for VIN OCR models.

Usage:
    # Single model evaluation
    python -m src.vin_ocr.evaluation.evaluate --data_dir ./data/test
    
    # Multi-model comparison
    from src.vin_ocr.evaluation.multi_model_evaluation import MultiModelEvaluator
    evaluator = MultiModelEvaluator()
    results = evaluator.evaluate_all()
"""

# Import from actual implementation files
# evaluate.py - Main evaluation script with CLI
# multi_model_evaluation.py - Multi-model comparison

__all__ = [
    "evaluate",
    "multi_model_evaluation",
    "metrics",  # Add metrics module
]
