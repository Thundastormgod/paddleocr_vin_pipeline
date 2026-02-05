"""
Hyperparameter Tuning Module
============================

Provides Optuna-based hyperparameter optimization for VIN OCR models.

Classes:
    OptunaHyperparameterTuner: Main tuning class for PaddleOCR and DeepSeek models
    TuningConfig: Configuration for hyperparameter search

Usage:
    from src.vin_ocr.training.hyperparameter_tuning import OptunaHyperparameterTuner
    
    tuner = OptunaHyperparameterTuner(model_type='paddleocr')
    best_params = tuner.optimize(n_trials=50)
"""

from .optuna_tuning import (
    OptunaHyperparameterTuner,
    TuningConfig,
    PaddleOCRSearchSpace,
    DeepSeekSearchSpace,
)

__all__ = [
    'OptunaHyperparameterTuner',
    'TuningConfig',
    'PaddleOCRSearchSpace',
    'DeepSeekSearchSpace',
]
