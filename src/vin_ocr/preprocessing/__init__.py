"""
VIN Image Preprocessing Module
==============================

Provides image preprocessing specifically optimized for VIN plate recognition.

Classes:
    VINPreprocessor: Main preprocessing class with multiple strategies
    PreprocessConfig: Configuration for preprocessing parameters

Usage:
    from src.vin_ocr.preprocessing import VINPreprocessor
    
    preprocessor = VINPreprocessor()
    processed = preprocessor.process(image)
"""

from .vin_preprocessor import (
    VINPreprocessor,
    PreprocessConfig,
    PreprocessStrategy,
)

__all__ = [
    'VINPreprocessor',
    'PreprocessConfig', 
    'PreprocessStrategy',
]
