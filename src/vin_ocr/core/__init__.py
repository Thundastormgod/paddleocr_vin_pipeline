"""
VIN OCR Core Module
===================

Core VIN utilities, constants, and validation logic.
Single Source of Truth for all VIN-related functionality.
"""

from .charset import (
    BLANK_INDEX,
    BLANK_TOKEN,
    VIN_CHARSET,
    load_char_dict,
    num_classes,
)
from .vin_utils import (
    # Constants
    VINConstants,
    VIN_LENGTH,
    VIN_VALID_CHARS,
    VIN_INVALID_CHARS,
    # Validation
    VINValidationResult,
    validate_vin,
    validate_vin_format,
    validate_vin_checksum,
    # Extraction
    extract_vin_from_filename,
    extract_vin_from_text,
    # Checksum
    calculate_check_digit,
    validate_checksum,
    # Similarity
    levenshtein_distance,
    # Correction
    RuleBasedCorrector,
    correct_vin,
    get_corrector,
)

__all__ = [
    # Charset (single source of truth for char<->index mapping)
    "BLANK_INDEX",
    "BLANK_TOKEN",
    "VIN_CHARSET",
    "load_char_dict",
    "num_classes",
    # Constants
    "VINConstants",
    "VIN_LENGTH",
    "VIN_VALID_CHARS", 
    "VIN_INVALID_CHARS",
    # Validation
    "VINValidationResult",
    "validate_vin",
    "validate_vin_format",
    "validate_vin_checksum",
    # Extraction
    "extract_vin_from_filename",
    "extract_vin_from_text",
    # Checksum
    "calculate_check_digit",
    "validate_checksum",
    # Similarity
    "levenshtein_distance",
    # Correction
    "RuleBasedCorrector",
    "correct_vin",
    "get_corrector",
]
