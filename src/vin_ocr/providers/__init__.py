"""
VIN OCR Providers Module
========================

OCR provider abstraction layer for multiple backends.

Supported providers:
- PaddleOCR (PP-OCRv3, v4, v5 and custom fine-tuned models)
- DeepSeek (vision-language model via API)

All providers include built-in VIN-optimized preprocessing with
multiple strategies for different image conditions:
- ENGRAVED: Optimized for engraved/stamped metal plates (default)
- STANDARD: Basic preprocessing for clean images  
- LOW_CONTRAST: For faded or low-contrast images
- ADAPTIVE: Automatically selects best strategy
- AGGRESSIVE: Heavy processing for difficult images

Usage:
    from vin_ocr.providers import OCRProviderFactory, PreprocessStrategy
    
    # Create provider
    provider = OCRProviderFactory.create("paddleocr", device="gpu")
    
    # Recognize VIN (uses ENGRAVED preprocessing by default)
    result = provider.recognize(image_path)
    print(result.text, result.confidence)
    
    # Use specific preprocessing strategy
    result = provider.recognize(
        image_path, 
        preprocess_strategy=PreprocessStrategy.LOW_CONTRAST
    )
"""

from .ocr_providers import (
    OCRProviderType,
    OCRResult,
    OCRProvider,
    PaddleOCRProvider,
    DeepSeekOCRProvider,
    OCRProviderFactory,
    PaddleOCRConfig,
    DeepSeekOCRConfig,
    ProviderConfig,
)

# Re-export preprocessing components for convenience
from ..preprocessing import (
    PreprocessStrategy,
    PreprocessConfig,
    VINPreprocessor,
)

__all__ = [
    # Providers
    "OCRProviderType",
    "OCRResult",
    "OCRProvider",
    "PaddleOCRProvider",
    "DeepSeekOCRProvider",
    "OCRProviderFactory",
    # Configs
    "PaddleOCRConfig",
    "DeepSeekOCRConfig",
    "ProviderConfig",
    # Preprocessing
    "PreprocessStrategy",
    "PreprocessConfig",
    "VINPreprocessor",
]
