"""
VIN OCR Pipeline - Single File Version
======================================

A complete PaddleOCR-based pipeline for Vehicle Identification Number (VIN) recognition
from engraved metal plates.

Usage:
    from vin_pipeline import VINOCRPipeline
    
    pipeline = VINOCRPipeline()
    result = pipeline.recognize('path/to/image.jpg')
    print(result['vin'])

Author: JRL-VIN Project
License: MIT
Version: 1.0.0
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import re
import logging
import time
from contextlib import contextmanager

# Configure module logger
logger = logging.getLogger(__name__)

# PaddleOCR import with graceful handling
try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False
    PaddleOCR = None  # type: ignore


class PipelineError(Exception):
    """Base exception for pipeline errors."""
    pass


class ImageLoadError(PipelineError):
    """Raised when image cannot be loaded."""
    pass


class OCREngineError(PipelineError):
    """Raised when OCR engine fails."""
    pass


class ConfigurationError(PipelineError):
    """Raised when pipeline is misconfigured."""
    pass


# =============================================================================
# VIN CONSTANTS & CONFIGURATION
# =============================================================================

class PreprocessMode(str, Enum):
    """Preprocessing mode enumeration for type safety."""
    NONE = 'none'
    FAST = 'fast'
    BALANCED = 'balanced'
    ENGRAVED = 'engraved'


@dataclass(frozen=True)
class VINConfig:
    """Immutable VIN format configuration."""
    LENGTH: int = 17
    VALID_CHARS: frozenset = field(default_factory=lambda: frozenset('0123456789ABCDEFGHJKLMNPRSTUVWXYZ'))
    INVALID_CHARS: frozenset = field(default_factory=lambda: frozenset('IOQ'))
    CHECK_DIGIT_POSITION: int = 9  # 1-indexed
    SEQUENTIAL_START: int = 12    # 1-indexed, positions 12-17
    SEQUENTIAL_END: int = 17


@dataclass
class CLAHEConfig:
    """CLAHE preprocessing configuration."""
    clip_limit: float = 2.0
    tile_size: Tuple[int, int] = (8, 8)


@dataclass
class OCRConfig:
    """PaddleOCR configuration."""
    lang: str = 'en'
    text_det_box_thresh: float = 0.3
    use_doc_orientation_classify: bool = False
    use_doc_unwarping: bool = False
    use_textline_orientation: bool = False
    

# Global VIN configuration instance
VIN_CONFIG = VINConfig()

# Valid VIN characters (I, O, Q are not allowed in VINs) - for backward compatibility
VIN_VALID_CHARS = set('0123456789ABCDEFGHJKLMNPRSTUVWXYZ')
VIN_INVALID_CHARS = {'I', 'O', 'Q'}
VIN_LENGTH = 17

# Character confusion mappings (common OCR errors on engraved plates)
INVALID_CHAR_FIXES: Dict[str, str] = {
    'I': '1',
    'O': '0', 
    'Q': '0',
}

# Common artifact characters to remove - compiled patterns for performance
ARTIFACT_PATTERNS: List[re.Pattern] = [
    re.compile(r'^[*#XYT]+'),     # Artifacts at start
    re.compile(r'[*#]+$'),         # Artifacts at end
    re.compile(r'^[IYTFA][*#]*'),  # Common prefix artifacts
]

# Position-based character confusion (for ambiguous cases)
# Positions 12-17 (sequential number) should be digits
DIGIT_POSITIONS: frozenset = frozenset({12, 13, 14, 15, 16, 17})  # 1-indexed

# Letter to digit mappings for sequential number section
LETTER_TO_DIGIT_MAP: Dict[str, str] = {
    'S': '5', 'G': '6', 'B': '8', 'A': '4',
    'L': '1', 'Z': '2', 'E': '3', 'O': '0',
    'I': '1', 'D': '0', 'C': '6', 'T': '7',
}


# =============================================================================
# PREPROCESSOR
# =============================================================================

class VINImagePreprocessor:
    """
    Image preprocessing for engraved VIN plates.
    
    Uses CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance
    contrast for better OCR recognition on metal surfaces.
    
    Thread Safety: This class is thread-safe for concurrent use.
    """
    
    # Maximum image dimension to prevent OOM
    MAX_DIMENSION: int = 4096
    
    def __init__(
        self,
        mode: Union[str, PreprocessMode] = PreprocessMode.ENGRAVED,
        clahe_config: Optional[CLAHEConfig] = None,
    ):
        """
        Initialize preprocessor.
        
        Args:
            mode: Preprocessing mode ('none', 'fast', 'balanced', 'engraved')
            clahe_config: CLAHE configuration (uses defaults if None)
            
        Raises:
            ConfigurationError: If mode is invalid
        """
        # Validate and normalize mode
        if isinstance(mode, str):
            try:
                self.mode = PreprocessMode(mode.lower())
            except ValueError:
                valid_modes = [m.value for m in PreprocessMode]
                raise ConfigurationError(
                    f"Invalid preprocessing mode: '{mode}'. "
                    f"Valid modes: {valid_modes}"
                )
        else:
            self.mode = mode
            
        self.clahe_config = clahe_config or CLAHEConfig()
        
        # Create CLAHE object
        self.clahe = cv2.createCLAHE(
            clipLimit=self.clahe_config.clip_limit,
            tileGridSize=self.clahe_config.tile_size
        )
        
        logger.debug(f"Preprocessor initialized with mode={self.mode.value}")
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for OCR.
        
        Args:
            image: Input image (BGR or grayscale)
            
        Returns:
            Preprocessed image
            
        Raises:
            ValueError: If image is invalid
        """
        if image is None or image.size == 0:
            raise ValueError("Input image is empty or None")
            
        # Check image dimensions to prevent OOM
        h, w = image.shape[:2]
        if max(h, w) > self.MAX_DIMENSION:
            scale = self.MAX_DIMENSION / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            logger.warning(f"Image resized from {w}x{h} to {new_w}x{new_h} to prevent OOM")
        
        if self.mode == PreprocessMode.NONE:
            return image
            
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        if self.mode == PreprocessMode.FAST:
            # Just normalize
            normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
            # Convert back to 3-channel for PaddleOCR compatibility
            return cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)
            
        # Apply CLAHE for better contrast on engraved surfaces
        enhanced = self.clahe.apply(gray)
        
        if self.mode == PreprocessMode.BALANCED:
            # Convert back to 3-channel for PaddleOCR compatibility
            return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
            
        if self.mode == PreprocessMode.ENGRAVED:
            # Additional processing for engraved plates
            # Denoise while preserving edges
            denoised = cv2.bilateralFilter(enhanced, 5, 50, 50)
            # Convert back to 3-channel for PaddleOCR compatibility
            return cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
        
        # Convert back to 3-channel for PaddleOCR compatibility    
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


# =============================================================================
# POSTPROCESSOR
# =============================================================================

@dataclass
class VINResult:
    """Structured VIN recognition result."""
    vin: str
    raw_ocr: str
    confidence: float
    is_valid_length: bool
    checksum_valid: bool
    corrections: List[str] = field(default_factory=list)
    processing_time_ms: float = 0.0
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'vin': self.vin,
            'raw_ocr': self.raw_ocr,
            'confidence': self.confidence,
            'is_valid_length': self.is_valid_length,
            'checksum_valid': self.checksum_valid,
            'corrections': self.corrections,
            'processing_time_ms': self.processing_time_ms,
            'error': self.error,
        }


class VINPostProcessor:
    """
    VIN validation and correction after OCR.
    
    Handles:
    - Artifact removal (*, #, X prefixes/suffixes)
    - Invalid character replacement (I→1, O→0, Q→0)
    - Position-based confusion correction
    - VIN checksum validation
    
    Thread Safety: This class is thread-safe for concurrent use.
    """
    
    # VIN checksum weights by position (NHTSA standard)
    CHECKSUM_WEIGHTS: Tuple[int, ...] = (8, 7, 6, 5, 4, 3, 2, 10, 0, 9, 8, 7, 6, 5, 4, 3, 2)
    
    # Character to value mapping for checksum (ISO 3779)
    CHAR_VALUES: Dict[str, int] = {
        'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8,
        'J': 1, 'K': 2, 'L': 3, 'M': 4, 'N': 5, 'P': 7, 'R': 9,
        'S': 2, 'T': 3, 'U': 4, 'V': 5, 'W': 6, 'X': 7, 'Y': 8, 'Z': 9,
        '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9
    }
    
    def __init__(self, verbose: bool = False):
        """
        Initialize postprocessor.
        
        Args:
            verbose: Print correction steps
        """
        self.verbose = verbose
    
    def process(self, raw_text: str, confidence: float = 0.0) -> Dict[str, Any]:
        """
        Process raw OCR output to extract and correct VIN.
        
        Args:
            raw_text: Raw OCR text output
            confidence: OCR confidence score
            
        Returns:
            Dict with vin, confidence, raw_ocr, corrections
        """
        corrections = []
        
        # Step 1: Clean and uppercase, remove whitespace and newlines
        text = raw_text.upper().strip()
        # Remove newlines, tabs, and other whitespace characters
        text = ''.join(text.split())
        original = text
        
        # Step 2: Remove artifacts
        text = self._remove_artifacts(text)
        if text != original:
            corrections.append(f"Removed artifacts: '{original}' → '{text}'")
        
        # Step 3: Fix invalid characters
        text_before = text
        text = self._fix_invalid_chars(text)
        if text != text_before:
            corrections.append(f"Fixed invalid chars: '{text_before}' → '{text}'")
        
        # Step 4: Apply position-based corrections
        text_before = text
        text = self._apply_position_corrections(text)
        if text != text_before:
            corrections.append(f"Position corrections: '{text_before}' → '{text}'")
        
        # Step 5: Validate length
        is_valid_length = len(text) == VIN_LENGTH
        
        # Step 6: Validate checksum (if correct length)
        checksum_valid = False
        if is_valid_length:
            checksum_valid = self._validate_checksum(text)
        
        if self.verbose:
            for c in corrections:
                print(f"  [Postprocess] {c}")
        
        return {
            'vin': text,
            'raw_ocr': raw_text,
            'confidence': confidence,
            'is_valid_length': is_valid_length,
            'checksum_valid': checksum_valid,
            'corrections': corrections
        }
    
    def _remove_artifacts(self, text: str) -> str:
        """Remove common artifact characters."""
        for pattern in ARTIFACT_PATTERNS:
            text = pattern.sub('', text)
        
        # Remove any remaining artifact chars throughout
        text = text.replace('*', '').replace('#', '')
        
        return text
    
    def _fix_invalid_chars(self, text: str) -> str:
        """Replace invalid VIN characters."""
        result = []
        for char in text:
            if char in INVALID_CHAR_FIXES:
                result.append(INVALID_CHAR_FIXES[char])
            else:
                result.append(char)
        return ''.join(result)
    
    def _apply_position_corrections(self, text: str) -> str:
        """Apply position-based character corrections."""
        if len(text) != VIN_LENGTH:
            return text
            
        result = list(text)
        
        # Positions 12-17 should be digits (sequential number)
        for pos in DIGIT_POSITIONS:
            idx = pos - 1  # Convert to 0-indexed
            if idx < len(result):
                char = result[idx]
                if char in LETTER_TO_DIGIT_MAP:
                    result[idx] = LETTER_TO_DIGIT_MAP[char]
        
        return ''.join(result)
    
    def _validate_checksum(self, vin: str) -> bool:
        """
        Validate VIN checksum (position 9).
        
        The checksum is calculated by:
        1. Assigning a value to each character (ISO 3779)
        2. Multiplying each value by its position weight (NHTSA)
        3. Summing all products
        4. Taking mod 11 (result 10 = 'X')
        
        Args:
            vin: VIN string to validate
            
        Returns:
            True if checksum is valid, False otherwise
        """
        if len(vin) != VIN_LENGTH:
            return False
            
        try:
            total = 0
            for i, char in enumerate(vin):
                value = self.CHAR_VALUES.get(char)
                if value is None:
                    logger.debug(f"Unknown character '{char}' at position {i+1}")
                    return False
                weight = self.CHECKSUM_WEIGHTS[i]
                total += value * weight
            
            remainder = total % 11
            expected = 'X' if remainder == 10 else str(remainder)
            
            is_valid = vin[8] == expected  # Position 9 (0-indexed: 8)
            
            if not is_valid:
                logger.debug(
                    f"Checksum mismatch: expected '{expected}', got '{vin[8]}' "
                    f"(sum={total}, mod11={remainder})"
                )
            
            return is_valid
            
        except (IndexError, TypeError) as e:
            logger.warning(f"Checksum validation error: {e}")
            return False


# =============================================================================
# MAIN PIPELINE
# =============================================================================

@contextmanager
def _timer():
    """Context manager for timing operations."""
    start = time.perf_counter()
    elapsed = {'ms': 0.0}
    yield elapsed
    elapsed['ms'] = (time.perf_counter() - start) * 1000


class VINOCRPipeline:
    """
    Complete VIN OCR Pipeline.
    
    Combines:
    - Image preprocessing (CLAHE contrast enhancement)
    - PaddleOCR (PP-OCRv4 model)
    - VIN postprocessing (correction and validation)
    
    Thread Safety: NOT thread-safe due to PaddleOCR internals.
    Create separate instances for parallel processing.
    
    Example:
        pipeline = VINOCRPipeline()
        result = pipeline.recognize('vin_image.jpg')
        print(result['vin'])  # "SAL1P9EU2SA606664"
    """
    
    def __init__(
        self,
        preprocess_mode: Union[str, PreprocessMode] = PreprocessMode.ENGRAVED,
        enable_postprocess: bool = True,
        verbose: bool = False,
        ocr_config: Optional[OCRConfig] = None,
    ):
        """
        Initialize the VIN OCR pipeline.
        
        Args:
            preprocess_mode: 'none', 'fast', 'balanced', 'engraved'
            enable_postprocess: Apply VIN correction
            verbose: Print processing steps
            ocr_config: PaddleOCR configuration (uses defaults if None)
            
        Raises:
            ConfigurationError: If PaddleOCR is not installed
        """
        if not PADDLEOCR_AVAILABLE:
            raise ConfigurationError(
                "PaddleOCR is not installed. "
                "Install with: pip install paddleocr paddlepaddle"
            )
        
        # Normalize mode
        if isinstance(preprocess_mode, str):
            self.preprocess_mode = preprocess_mode.lower()
        else:
            self.preprocess_mode = preprocess_mode.value
            
        self.enable_postprocess = enable_postprocess
        self.verbose = verbose
        self.ocr_config = ocr_config or OCRConfig()
        
        # Initialize components
        self.preprocessor = VINImagePreprocessor(mode=self.preprocess_mode)
        self.postprocessor = VINPostProcessor(verbose=verbose)
        
        # Initialize PaddleOCR with optimized settings
        logger.info("Initializing PaddleOCR...")
        if self.verbose:
            print("Initializing PaddleOCR...")
            
        try:
            self.ocr = PaddleOCR(
                lang=self.ocr_config.lang,
                use_doc_orientation_classify=self.ocr_config.use_doc_orientation_classify,
                use_doc_unwarping=self.ocr_config.use_doc_unwarping,
                use_textline_orientation=self.ocr_config.use_textline_orientation,
                text_det_box_thresh=self.ocr_config.text_det_box_thresh,
            )
        except Exception as e:
            logger.error(f"Failed to initialize PaddleOCR: {e}")
            raise OCREngineError(f"Failed to initialize PaddleOCR: {e}") from e
        
        logger.info("Pipeline initialized successfully")
        if self.verbose:
            print("Pipeline initialized.")
    
    def recognize(self, image_path: Union[str, Path, np.ndarray]) -> Dict[str, Any]:
        """
        Recognize VIN from an image.
        
        Args:
            image_path: Path to image file or numpy array
            
        Returns:
            Dict containing:
                - vin: Corrected VIN string
                - confidence: OCR confidence score
                - raw_ocr: Original OCR output
                - is_valid_length: Whether VIN is 17 characters
                - checksum_valid: Whether VIN checksum is valid
                - corrections: List of corrections applied
                - processing_time_ms: Time taken in milliseconds
                - error: Error message if any
        """
        with _timer() as elapsed:
            try:
                return self._recognize_internal(image_path, elapsed)
            except Exception as e:
                logger.exception(f"Recognition failed: {e}")
                return {
                    'vin': '',
                    'confidence': 0.0,
                    'raw_ocr': '',
                    'is_valid_length': False,
                    'checksum_valid': False,
                    'corrections': [],
                    'processing_time_ms': elapsed['ms'],
                    'error': str(e),
                }
    
    def _recognize_internal(
        self, 
        image_path: Union[str, Path, np.ndarray],
        elapsed: Dict[str, float]
    ) -> Dict[str, Any]:
        """Internal recognition logic with timing."""
        # Load image
        if isinstance(image_path, np.ndarray):
            image = image_path
            source = "numpy_array"
        else:
            path = Path(image_path)
            if not path.exists():
                raise ImageLoadError(f"Image file not found: {path}")
            
            # Handle unicode paths
            try:
                image = cv2.imread(str(path))
            except Exception as e:
                raise ImageLoadError(f"Failed to read image: {e}") from e
                
            if image is None:
                raise ImageLoadError(
                    f"Could not decode image: {path}. "
                    "Verify it's a valid image format (jpg, png, etc.)"
                )
            source = str(path)
        
        logger.debug(f"Loaded image from {source}, shape={image.shape}")
        
        # Preprocess
        if self.verbose:
            print(f"Preprocessing (mode: {self.preprocess_mode})...")
        processed = self.preprocessor.preprocess(image)
        
        # Run OCR
        if self.verbose:
            print("Running OCR...")
            
        try:
            result = self.ocr.predict(processed)
        except Exception as e:
            raise OCREngineError(f"OCR prediction failed: {e}") from e
        
        # Extract text and confidence
        raw_text, confidence = self._extract_ocr_result(result)
        
        logger.debug(f"Raw OCR: '{raw_text}' (confidence: {confidence:.2f})")
        if self.verbose:
            print(f"Raw OCR: '{raw_text}' (conf: {confidence:.2f})")
        
        # Postprocess
        if self.enable_postprocess:
            if self.verbose:
                print("Postprocessing...")
            output = self.postprocessor.process(raw_text, confidence)
        else:
            output = {
                'vin': raw_text,
                'raw_ocr': raw_text,
                'confidence': confidence,
                'is_valid_length': len(raw_text) == VIN_LENGTH,
                'checksum_valid': False,
                'corrections': []
            }
        
        # Add timing
        output['processing_time_ms'] = elapsed['ms']
        output['error'] = None
        
        return output
    
    def recognize_batch(
        self,
        image_paths: List[Union[str, Path]],
        show_progress: bool = True,
        continue_on_error: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Recognize VINs from multiple images.
        
        Args:
            image_paths: List of image paths
            show_progress: Print progress
            continue_on_error: Continue processing if an image fails
            
        Returns:
            List of recognition results
        """
        results = []
        total = len(image_paths)
        successful = 0
        failed = 0
        
        for i, path in enumerate(image_paths):
            if show_progress:
                print(f"Processing {i+1}/{total}: {Path(path).name}")
            
            try:
                result = self.recognize(path)
                result['file'] = str(path)
                results.append(result)
                
                if result.get('error'):
                    failed += 1
                else:
                    successful += 1
                    
            except Exception as e:
                logger.error(f"Failed to process {path}: {e}")
                if continue_on_error:
                    results.append({
                        'file': str(path),
                        'vin': '',
                        'confidence': 0.0,
                        'raw_ocr': '',
                        'error': str(e),
                    })
                    failed += 1
                else:
                    raise
        
        if show_progress:
            print(f"Completed: {successful} successful, {failed} failed")
            
        return results
    
    def _extract_ocr_result(self, result: Any) -> Tuple[str, float]:
        """
        Extract text and confidence from PaddleOCR result.
        
        Handles PaddleOCR v3.x result format.
        """
        if not result:
            logger.debug("Empty OCR result")
            return '', 0.0
        
        # Handle PaddleOCR v3.x format (list of dicts)
        if isinstance(result, list):
            if len(result) == 0:
                return '', 0.0
            result = result[0]
        
        if isinstance(result, dict):
            texts = result.get('rec_texts', [])
            scores = result.get('rec_scores', [])
            
            if texts:
                # Combine all detected text
                full_text = ''.join(texts)
                avg_score = float(np.mean(scores)) if scores else 0.0
                return full_text, avg_score
        
        logger.warning(f"Unexpected OCR result format: {type(result)}")
        return '', 0.0


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def validate_vin(vin: str) -> Dict[str, Any]:
    """
    Validate a VIN string.
    
    Performs comprehensive validation including:
    - Length check (must be 17 characters)
    - Character validity (no I, O, Q)
    - Checksum validation (position 9)
    
    Args:
        vin: VIN string to validate
        
    Returns:
        Dict with validation results:
            - vin: Normalized VIN string
            - is_valid_length: True if 17 characters
            - has_valid_chars: True if no invalid characters
            - invalid_chars: List of invalid characters found
            - checksum_valid: True if checksum passes
            - is_fully_valid: True if all checks pass
    """
    if not isinstance(vin, str):
        return {
            'vin': '',
            'is_valid_length': False,
            'has_valid_chars': False,
            'invalid_chars': [],
            'checksum_valid': False,
            'is_fully_valid': False,
            'error': f'Expected string, got {type(vin).__name__}'
        }
    
    vin = vin.upper().strip()
    
    # Check length
    is_valid_length = len(vin) == VIN_LENGTH
    
    # Check characters
    invalid_chars = [c for c in vin if c not in VIN_VALID_CHARS]
    has_valid_chars = len(invalid_chars) == 0
    
    # Check checksum
    processor = VINPostProcessor()
    checksum_valid = processor._validate_checksum(vin) if is_valid_length else False
    
    return {
        'vin': vin,
        'is_valid_length': is_valid_length,
        'has_valid_chars': has_valid_chars,
        'invalid_chars': invalid_chars,
        'checksum_valid': checksum_valid,
        'is_fully_valid': is_valid_length and has_valid_chars and checksum_valid
    }


def decode_vin(vin: str) -> Dict[str, Any]:
    """
    Decode VIN structure into its component parts.
    
    VIN Structure (ISO 3779):
    - Position 1-3: WMI (World Manufacturer Identifier)
    - Position 4-8: VDS (Vehicle Descriptor Section)
    - Position 9: Check digit
    - Position 10: Model year
    - Position 11: Plant code
    - Position 12-17: VIS Sequential number
    
    Note: Model year codes repeat every 30 years (A=1980/2010, B=1981/2011, etc.)
    This function returns the more recent interpretation (2010+) for letter codes.
    Use the 7th character (vehicle attributes) to disambiguate if needed.
    
    Args:
        vin: VIN string to decode
        
    Returns:
        Dict with decoded VIN components
    """
    if not isinstance(vin, str):
        return {'error': f'Expected string, got {type(vin).__name__}'}
        
    vin = vin.upper().strip()
    
    if len(vin) != VIN_LENGTH:
        return {'error': f'Invalid VIN length: {len(vin)} (expected {VIN_LENGTH})'}
    
    # Model year decode table
    # Letter codes (A-Y excluding I, O, Q, U, Z) cycle every 30 years
    # 1980-2000 and 2010-2030 use the same letters
    # Digit codes 1-9 are used for 2001-2009 (and will repeat 2031-2039)
    # We return the modern interpretation (2010+) for letters
    MODEL_YEAR_CODES_MODERN = {
        # 2001-2009 (digits - unambiguous)
        '1': 2001, '2': 2002, '3': 2003, '4': 2004,
        '5': 2005, '6': 2006, '7': 2007, '8': 2008, '9': 2009,
        # 2010-2030 (letters - also valid for 1980-2000)
        'A': 2010, 'B': 2011, 'C': 2012, 'D': 2013, 'E': 2014,
        'F': 2015, 'G': 2016, 'H': 2017, 'J': 2018, 'K': 2019,
        'L': 2020, 'M': 2021, 'N': 2022, 'P': 2023, 'R': 2024,
        'S': 2025, 'T': 2026, 'V': 2027, 'W': 2028, 'X': 2029,
        'Y': 2030,
    }
    
    # Legacy mapping for reference (subtract 30 from modern for 1980s/1990s)
    MODEL_YEAR_CODES_LEGACY = {
        'A': 1980, 'B': 1981, 'C': 1982, 'D': 1983, 'E': 1984,
        'F': 1985, 'G': 1986, 'H': 1987, 'J': 1988, 'K': 1989,
        'L': 1990, 'M': 1991, 'N': 1992, 'P': 1993, 'R': 1994,
        'S': 1995, 'T': 1996, 'V': 1997, 'W': 1998, 'X': 1999,
        'Y': 2000,
    }
    
    year_code = vin[9]
    model_year_modern = MODEL_YEAR_CODES_MODERN.get(year_code)
    model_year_legacy = MODEL_YEAR_CODES_LEGACY.get(year_code)
    
    # Determine which year to report
    if model_year_modern is not None:
        model_year = model_year_modern
        # If letter code, note ambiguity
        if year_code.isalpha() and model_year_legacy:
            model_year_note = f"{model_year_modern} (or {model_year_legacy})"
        else:
            model_year_note = str(model_year_modern)
    else:
        model_year = f'Unknown ({year_code})'
        model_year_note = model_year
    
    return {
        'vin': vin,
        'wmi': vin[0:3],           # World Manufacturer Identifier
        'vds': vin[3:9],           # Vehicle Descriptor Section
        'check_digit': vin[8],     # Check digit
        'model_year_code': year_code,
        'model_year': model_year,
        'model_year_display': model_year_note,  # Human-readable with ambiguity note
        'plant_code': vin[10],     # Assembly plant
        'sequential': vin[11:17],  # Sequential number
        'vis': vin[9:17],          # Vehicle Identifier Section
    }


def calculate_check_digit(vin_without_check: str) -> str:
    """
    Calculate the correct check digit for a VIN.
    
    Useful for correcting VINs with invalid check digits.
    
    Args:
        vin_without_check: VIN string (position 9 will be ignored)
        
    Returns:
        Correct check digit (0-9 or X)
        
    Raises:
        ValueError: If VIN contains invalid characters
    """
    if len(vin_without_check) < VIN_LENGTH:
        raise ValueError(f"VIN too short: {len(vin_without_check)} characters")
    
    vin = vin_without_check.upper()[:VIN_LENGTH]
    
    # Use CHAR_VALUES from VINPostProcessor
    char_values = VINPostProcessor.CHAR_VALUES
    weights = VINPostProcessor.CHECKSUM_WEIGHTS
    
    total = 0
    for i, char in enumerate(vin):
        if i == 8:  # Skip check digit position
            continue
        if char not in char_values:
            raise ValueError(f"Invalid character '{char}' at position {i+1}")
        total += char_values[char] * weights[i]
    
    remainder = total % 11
    return 'X' if remainder == 10 else str(remainder)


# =============================================================================
# CLI INTERFACE
# =============================================================================

def _setup_logging(verbose: bool) -> None:
    """Configure logging based on verbosity."""
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def main() -> int:
    """Main CLI entry point."""
    import argparse
    import sys
    import json
    
    parser = argparse.ArgumentParser(
        description='VIN OCR Pipeline - Recognize VINs from images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python vin_pipeline.py image.jpg
  python vin_pipeline.py image.jpg --mode engraved -v
  python vin_pipeline.py image.jpg --json
  python vin_pipeline.py --validate "SAL1P9EU2SA606664"
        """
    )
    parser.add_argument('image', nargs='?', help='Path to VIN image')
    parser.add_argument('--mode', default='engraved', 
                       choices=['none', 'fast', 'balanced', 'engraved'],
                       help='Preprocessing mode (default: engraved)')
    parser.add_argument('--no-postprocess', action='store_true',
                       help='Disable postprocessing')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Verbose output')
    parser.add_argument('--json', action='store_true',
                       help='Output as JSON')
    parser.add_argument('--validate', metavar='VIN',
                       help='Validate a VIN string instead of processing an image')
    parser.add_argument('--decode', metavar='VIN',
                       help='Decode a VIN string structure')
    parser.add_argument('--version', action='version', version='VIN OCR Pipeline 1.0.0')
    
    args = parser.parse_args()
    
    # Setup logging
    _setup_logging(args.verbose)
    
    # Handle --validate flag
    if args.validate:
        result = validate_vin(args.validate)
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            status = "VALID" if result['is_fully_valid'] else "INVALID"
            print(f"VIN: {result['vin']} - {status}")
            print(f"  Length: {'OK' if result['is_valid_length'] else 'INVALID'}")
            char_status = 'OK' if result['has_valid_chars'] else f"INVALID ({result['invalid_chars']})"
            print(f"  Characters: {char_status}")
            print(f"  Checksum: {'OK' if result['checksum_valid'] else 'INVALID'}")
        return 0 if result['is_fully_valid'] else 1
    
    # Handle --decode flag
    if args.decode:
        result = decode_vin(args.decode)
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            if 'error' in result:
                print(f"Error: {result['error']}")
                return 1
            print(f"VIN: {result['vin']}")
            print(f"  WMI (Manufacturer): {result['wmi']}")
            print(f"  VDS (Descriptor): {result['vds']}")
            print(f"  Check Digit: {result['check_digit']}")
            print(f"  Model Year: {result['model_year_display']}")
            print(f"  Plant Code: {result['plant_code']}")
            print(f"  Sequential: {result['sequential']}")
        return 0
    
    # Require image for OCR
    if not args.image:
        parser.error("Image path required (or use --validate/--decode)")
        return 1
    
    # Check image exists
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}", file=sys.stderr)
        return 1
    
    # Create pipeline
    try:
        pipeline = VINOCRPipeline(
            preprocess_mode=args.mode,
            enable_postprocess=not args.no_postprocess,
            verbose=args.verbose
        )
    except ConfigurationError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        return 1
    
    # Process image
    result = pipeline.recognize(args.image)
    
    # Output
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("\n" + "="*50)
        if result.get('error'):
            print(f"Error: {result['error']}")
            return 1
        print(f"VIN: {result['vin']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Valid Length: {result.get('is_valid_length', 'N/A')}")
        print(f"Checksum Valid: {result.get('checksum_valid', 'N/A')}")
        print(f"Processing Time: {result.get('processing_time_ms', 0):.0f}ms")
        
        if result.get('corrections'):
            print("\nCorrections applied:")
            for c in result['corrections']:
                print(f"  - {c}")
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
