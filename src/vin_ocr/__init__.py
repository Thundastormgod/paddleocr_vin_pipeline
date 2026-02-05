"""
VIN OCR Pipeline
================

Enterprise-grade VIN (Vehicle Identification Number) recognition system.

Package Structure:
    vin_ocr/
    ├── core/           # VIN utilities, constants, validation
    ├── pipeline/       # Main VIN recognition pipeline
    ├── inference/      # Inference backends (ONNX, Paddle)
    ├── training/       # Training scripts
    ├── evaluation/     # Evaluation tools
    ├── providers/      # OCR backends
    ├── utils/          # Data preparation
    └── web/            # Streamlit web UI

Quick Start:
    # Inference
    from vin_ocr.inference import VINInference, ONNXVINRecognizer
    
    inference = VINInference("output/model/inference")
    result = inference.recognize("image.jpg")
    print(result['vin'])
    
    # Validation
    from vin_ocr.core import validate_vin
    result = validate_vin("1HGBH41JXMN109186")
    print(result.is_fully_valid)

Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "VIN OCR Team"

# Core exports (lightweight, always available)
from .core import (
    VINConstants,
    VIN_LENGTH,
    VIN_VALID_CHARS,
    VINValidationResult,
    validate_vin,
    validate_vin_format,
    extract_vin_from_filename,
    extract_vin_from_text,
    calculate_check_digit,
    validate_checksum,
    levenshtein_distance,
    correct_vin,
)

__all__ = [
    "__version__",
    "__author__",
    # Core
    "VINConstants",
    "VIN_LENGTH",
    "VIN_VALID_CHARS",
    "VINValidationResult",
    "validate_vin",
    "validate_vin_format",
    "extract_vin_from_filename",
    "extract_vin_from_text",
    "calculate_check_digit",
    "validate_checksum",
    "levenshtein_distance",
    "correct_vin",
]

# Lazy imports for inference (heavier dependencies)
def __getattr__(name: str):
    """Lazy import for inference modules."""
    if name == "VINInference":
        from .inference.paddle_inference import VINInference
        return VINInference
    elif name == "ONNXVINRecognizer":
        from .inference.onnx_inference import ONNXVINRecognizer
        return ONNXVINRecognizer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
