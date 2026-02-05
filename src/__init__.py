"""
VIN OCR Pipeline - Source Package
=================================

Re-exports from vin_ocr subpackage.

Usage:
    from src.vin_ocr import validate_vin, extract_vin_from_filename
    from src.vin_ocr.pipeline.vin_pipeline import VINOCRPipeline
"""

from .vin_ocr import (
    __version__,
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
