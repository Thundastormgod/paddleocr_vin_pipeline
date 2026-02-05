"""
VIN OCR Pipeline - Single File Version
======================================

A complete PaddleOCR-based pipeline for Vehicle Identification Number (VIN) recognition
from engraved metal plates.

Usage:
    from src.vin_ocr.pipeline.vin_pipeline import VINOCRPipeline
    
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
from config import get_config

# Import unified VIN preprocessing module
from ..preprocessing import VINPreprocessor, PreprocessConfig, PreprocessStrategy

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
    """
    Base exception for pipeline errors.
    
    Provides structured error information with error codes for programmatic handling.
    """
    
    def __init__(self, message: str, error_code: str = "PIPELINE_ERROR", context: Optional[Dict[str, Any]] = None):
        self.message = message
        self.error_code = error_code
        self.context = context or {}
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for JSON serialization."""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "context": self.context,
        }


class ImageLoadError(PipelineError):
    """Raised when image cannot be loaded."""
    
    def __init__(self, file_path: str, reason: str = "Unknown error"):
        super().__init__(
            message=f"Failed to load image: {file_path}. Reason: {reason}",
            error_code="IMAGE_LOAD_ERROR",
            context={"file_path": file_path, "reason": reason}
        )
        self.file_path = file_path
        self.reason = reason


class OCREngineError(PipelineError):
    """Raised when OCR engine fails."""
    
    def __init__(self, message: str, engine: str = "PaddleOCR", details: Optional[str] = None):
        super().__init__(
            message=f"OCR engine error ({engine}): {message}",
            error_code="OCR_ENGINE_ERROR",
            context={"engine": engine, "details": details}
        )
        self.engine = engine
        self.details = details


class ConfigurationError(PipelineError):
    """Raised when pipeline is misconfigured."""
    
    def __init__(self, message: str, config_key: Optional[str] = None, expected: Optional[str] = None):
        super().__init__(
            message=f"Configuration error: {message}",
            error_code="CONFIG_ERROR",
            context={"config_key": config_key, "expected": expected}
        )
        self.config_key = config_key
        self.expected = expected


# =============================================================================
# VIN CONSTANTS & CONFIGURATION
# =============================================================================

# Backward compatibility alias - PreprocessMode maps to PreprocessStrategy
PreprocessMode = PreprocessStrategy


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
    ocr_version: str = 'PP-OCRv3'  # Use v3 for better VIN recognition
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
# PREPROCESSOR (Unified Module Wrapper)
# =============================================================================

class VINImagePreprocessor:
    """
    Image preprocessing for engraved VIN plates.
    
    This is a backward-compatible wrapper around the unified VINPreprocessor module.
    Uses CLAHE, morphological operations, and adaptive processing for optimal OCR.
    
    Available strategies:
    - NONE: No preprocessing
    - STANDARD: Basic preprocessing for clean images
    - ENGRAVED: Optimized for engraved/stamped metal plates (default)
    - LOW_CONTRAST: For faded or low-contrast images
    - ADAPTIVE: Automatically selects best strategy
    - AGGRESSIVE: Heavy processing with binarization
    
    Thread Safety: This class is thread-safe for concurrent use.
    """
    
    # Maximum image dimension to prevent OOM
    MAX_DIMENSION: int = 4096
    
    # Strategy mapping for backward compatibility (old modes -> new strategies)
    _MODE_TO_STRATEGY = {
        'none': PreprocessStrategy.NONE,
        'fast': PreprocessStrategy.STANDARD,  # fast maps to standard
        'balanced': PreprocessStrategy.STANDARD,  # balanced maps to standard
        'engraved': PreprocessStrategy.ENGRAVED,
        'standard': PreprocessStrategy.STANDARD,
        'low_contrast': PreprocessStrategy.LOW_CONTRAST,
        'adaptive': PreprocessStrategy.ADAPTIVE,
        'aggressive': PreprocessStrategy.AGGRESSIVE,
    }
    
    def __init__(
        self,
        mode: Union[str, PreprocessStrategy] = PreprocessStrategy.ENGRAVED,
        clahe_config: Optional[CLAHEConfig] = None,
        max_dimension: Optional[int] = None,
    ):
        """
        Initialize preprocessor.
        
        Args:
            mode: Preprocessing mode/strategy
            clahe_config: CLAHE configuration (uses defaults if None)
            max_dimension: Maximum image dimension (default 4096)
            
        Raises:
            ConfigurationError: If mode is invalid
        """
        # Validate and normalize mode to strategy
        if isinstance(mode, str):
            mode_lower = mode.lower()
            if mode_lower in self._MODE_TO_STRATEGY:
                self.strategy = self._MODE_TO_STRATEGY[mode_lower]
            else:
                try:
                    self.strategy = PreprocessStrategy(mode_lower)
                except ValueError:
                    valid_modes = list(self._MODE_TO_STRATEGY.keys())
                    raise ConfigurationError(
                        f"Invalid preprocessing mode: '{mode}'. "
                        f"Valid modes: {valid_modes}"
                    )
        elif isinstance(mode, PreprocessStrategy):
            self.strategy = mode
        else:
            self.strategy = PreprocessStrategy.ENGRAVED
        
        # For backward compatibility
        self.mode = self.strategy
            
        self.clahe_config = clahe_config or CLAHEConfig()
        self.max_dimension = max_dimension or self.MAX_DIMENSION
        
        # Create unified preprocessor with config
        preprocess_config = PreprocessConfig(
            strategy=self.strategy,
            target_width=1024,
            clahe_clip_limit=self.clahe_config.clip_limit,
            clahe_tile_size=self.clahe_config.tile_size,
        )
        self._preprocessor = VINPreprocessor(config=preprocess_config)
        
        logger.debug(f"Preprocessor initialized with strategy={self.strategy.value}")
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for OCR.
        
        Args:
            image: Input image (BGR or grayscale)
            
        Returns:
            Preprocessed image (BGR format, or unchanged for NONE mode)
            
        Raises:
            ValueError: If image is invalid
        """
        if image is None or image.size == 0:
            raise ValueError("Input image is empty or None")
        
        # NONE mode: return image unchanged (for backward compatibility)
        if self.strategy == PreprocessStrategy.NONE:
            return image
            
        # Check image dimensions to prevent OOM
        h, w = image.shape[:2]
        if max(h, w) > self.max_dimension:
            scale = self.max_dimension / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            logger.warning(f"Image resized from {w}x{h} to {new_w}x{new_h} to prevent OOM")
        
        # Ensure 3-channel input for the preprocessor
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # Use unified preprocessor
        return self._preprocessor.process(image)
    
    def analyze_image(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Analyze image characteristics for debugging/tuning.
        
        Args:
            image: Input image
            
        Returns:
            Dict with analysis results
        """
        return self._preprocessor.analyze_image(image)


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
        
        # Step 4: Extract VIN substring if text is too long
        text_before = text
        text = self._extract_vin_substring(text)
        if text != text_before:
            corrections.append(f"Extracted VIN: '{text_before}' → '{text}'")
        
        # Step 5: Apply position-based corrections
        text_before = text
        text = self._apply_position_corrections(text)
        if text != text_before:
            corrections.append(f"Position corrections: '{text_before}' → '{text}'")
        
        # Step 6: Validate length
        is_valid_length = len(text) == VIN_LENGTH
        
        # Step 7: Validate checksum (if correct length)
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
    
    def _extract_vin_substring(self, text: str) -> str:
        """
        Extract 17-character VIN from longer text.
        
        Delegates to vin_utils.extract_vin_from_text() as Single Source of Truth.
        
        Args:
            text: Raw OCR text that may contain extra characters
            
        Returns:
            Best 17-character VIN candidate
        """
        from src.vin_ocr.core.vin_utils import extract_vin_from_text
        return extract_vin_from_text(text)
    
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
        
        Delegates to vin_utils.validate_checksum() as Single Source of Truth.
        
        Args:
            vin: VIN string to validate
            
        Returns:
            True if checksum is valid, False otherwise
        """
        # Import here to avoid circular imports at module level
        from src.vin_ocr.core.vin_utils import validate_checksum
        return validate_checksum(vin)


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
        preprocess_mode: Optional[Union[str, PreprocessMode]] = None,
        enable_postprocess: bool = True,
        verbose: bool = False,
        ocr_config: Optional[OCRConfig] = None,
        use_gpu: bool = False,
    ):
        """
        Initialize the VIN OCR pipeline.
        
        Args:
            preprocess_mode: 'none', 'fast', 'balanced', 'engraved' (uses config default if None)
            enable_postprocess: Apply VIN correction
            verbose: Print processing steps
            ocr_config: PaddleOCR configuration (uses defaults if None)
            use_gpu: Whether to use GPU acceleration
            
        Raises:
            ConfigurationError: If PaddleOCR is not installed
        """
        if not PADDLEOCR_AVAILABLE:
            raise ConfigurationError(
                "PaddleOCR is not installed. "
                "Install with: pip install paddleocr paddlepaddle"
            )
        
        config = get_config()
        
        # Store GPU setting
        self.use_gpu = use_gpu

        # Normalize mode
        if preprocess_mode is None:
            preprocess_mode = config.preprocessing.default_mode

        if isinstance(preprocess_mode, str):
            self.preprocess_mode = preprocess_mode.lower()
        else:
            self.preprocess_mode = preprocess_mode.value
            
        self.enable_postprocess = enable_postprocess
        self.verbose = verbose
        self.ocr_config = ocr_config or OCRConfig(
            lang=config.ocr.language,
            text_det_box_thresh=config.ocr.det_db_box_thresh,
            use_doc_orientation_classify=config.ocr.use_doc_orientation_classify,
            use_doc_unwarping=config.ocr.use_doc_unwarping,
            use_textline_orientation=config.ocr.use_textline_orientation,
        )
        
        # Initialize components
        clahe_config = CLAHEConfig(
            clip_limit=config.preprocessing.clahe_clip_limit,
            tile_size=config.preprocessing.clahe_tile_size,
        )
        self.preprocessor = VINImagePreprocessor(
            mode=self.preprocess_mode,
            clahe_config=clahe_config,
            max_dimension=config.preprocessing.max_image_dimension,
        )
        self.postprocessor = VINPostProcessor(verbose=verbose)
        
        # Initialize PaddleOCR with optimized settings
        logger.info("Initializing PaddleOCR...")
        if self.verbose:
            print("Initializing PaddleOCR...")
        
        # Set Paddle device based on use_gpu setting
        try:
            import paddle
            if self.use_gpu:
                if paddle.device.is_compiled_with_cuda():
                    paddle.device.set_device('gpu')
                    logger.info("Using GPU (CUDA) for PaddleOCR")
                else:
                    logger.warning("GPU requested but CUDA not available, using CPU")
                    paddle.device.set_device('cpu')
            else:
                paddle.device.set_device('cpu')
                logger.info("Using CPU for PaddleOCR")
        except Exception as e:
            logger.warning(f"Failed to set Paddle device: {e}")
            
        try:
            # PaddleOCR 3.0 API - use PP-OCRv3 for better VIN recognition
            # use_gpu is handled via paddle.device.set_device()
            self.ocr = PaddleOCR(
                lang=self.ocr_config.lang,
                ocr_version=self.ocr_config.ocr_version,
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
            
            # Combine all detected text (texts could be empty list)
            full_text = ''.join(texts) if texts else ''
            avg_score = float(np.mean(scores)) if scores else 0.0
            return full_text, avg_score

        # -----------------------------------------------------------------
        # Handle PaddleX / Paddle inference OCRResult objects which are
        # neither plain dicts nor lists but expose rec_texts / rec_scores
        # either via mapping access (obj['rec_texts']) or attributes
        # (obj.rec_texts) or via __getitem__ implemented in their BaseCVResult.
        # -----------------------------------------------------------------
        try:
            # Try mapping-like access first (works for BaseCVResult)
            texts = None
            scores = None
            if hasattr(result, '__getitem__'):
                try:
                    texts = result['rec_texts']
                except Exception:
                    texts = None
                try:
                    scores = result['rec_scores']
                except Exception:
                    scores = None

            # If not available via __getitem__, try attribute access
            if texts is None:
                texts = getattr(result, 'rec_texts', None)
            if scores is None:
                scores = getattr(result, 'rec_scores', None)

            # Check if we successfully accessed texts (could be empty list)
            if texts is not None:
                # rec_texts may be numpy arrays or lists; coerce to list of str
                try:
                    text_list = list(texts)
                except Exception:
                    text_list = [str(texts)] if texts else []

                full_text = ''.join([str(t) for t in text_list])

                # rec_scores can be a numpy array, list, or None
                try:
                    avg_score = float(np.mean(scores)) if scores is not None and len(scores) > 0 else 0.0
                except Exception:
                    try:
                        avg_score = float(np.mean(list(scores))) if scores is not None and len(list(scores)) > 0 else 0.0
                    except Exception:
                        avg_score = 0.0

                return full_text, avg_score
        except Exception as e:
            # Log the actual exception for debugging
            logger.debug(f"Exception extracting OCR result: {e}")
            pass

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
    
    from ..core.vin_utils import validate_vin as _validate_vin
    return _validate_vin(vin).to_dict()


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
    
    Delegates to vin_utils.calculate_check_digit (Single Source of Truth).
    
    Args:
        vin_without_check: VIN string (position 9 will be ignored)
        
    Returns:
        Correct check digit (0-9 or X)
        
    Raises:
        ValueError: If VIN contains invalid characters or is too short
    """
    from src.vin_ocr.core.vin_utils import calculate_check_digit as _calculate_check_digit
    
    if len(vin_without_check) < VIN_LENGTH:
        raise ValueError(f"VIN too short: {len(vin_without_check)} characters")
    
    vin = vin_without_check.upper()[:VIN_LENGTH]
    result = _calculate_check_digit(vin)
    
    if result is None:
        # Find the invalid character for a helpful error message
        from src.vin_ocr.core.vin_utils import VINConstants
        for i, char in enumerate(vin):
            if i == 8:  # Skip check digit position
                continue
            if char not in VINConstants.CHAR_VALUES:
                raise ValueError(f"Invalid character '{char}' at position {i+1}")
        raise ValueError("Invalid VIN characters")
    
    return result


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


# =============================================================================
# MULTI-PROVIDER PIPELINE
# =============================================================================

class MultiProviderVINPipeline:
    """
    VIN OCR Pipeline with multi-provider support.
    
    Supports:
    - PaddleOCR (default, local processing)
    - DeepSeek Vision (API-based)
    - Ensemble mode (combine multiple providers)
    - Future: Tesseract, Google Vision, Azure Vision
    
    Example:
        # Use PaddleOCR (default)
        pipeline = MultiProviderVINPipeline()
        
        # Use DeepSeek
        pipeline = MultiProviderVINPipeline(provider="deepseek", api_key="...")
        
        # Use ensemble of multiple providers
        pipeline = MultiProviderVINPipeline(
            provider="ensemble",
            ensemble_providers=["paddleocr", "deepseek"],
            ensemble_strategy="best"
        )
    """
    
    def __init__(
        self,
        provider: str = "paddleocr",
        preprocess_mode: Optional[Union[str, PreprocessMode]] = None,
        enable_postprocess: bool = True,
        verbose: bool = False,
        # Provider-specific options
        api_key: Optional[str] = None,
        use_gpu: Optional[bool] = None,
        # Ensemble options
        ensemble_providers: Optional[List[str]] = None,
        ensemble_strategy: str = "best",
        **provider_kwargs
    ):
        """
        Initialize multi-provider VIN OCR pipeline.
        
        Args:
            provider: OCR provider ('paddleocr', 'deepseek', 'ensemble')
            preprocess_mode: Image preprocessing mode (uses config default if None)
            enable_postprocess: Apply VIN correction rules
            verbose: Print processing steps
            api_key: API key for cloud providers (DeepSeek, etc.)
            use_gpu: Whether to use GPU acceleration (None = use config default)
            ensemble_providers: List of providers for ensemble mode
            ensemble_strategy: Ensemble strategy ('best', 'vote', 'weighted_vote', 'cascade')
            **provider_kwargs: Additional provider-specific options
        """
        self.provider_name = provider.lower()
        self.verbose = verbose
        self.enable_postprocess = enable_postprocess
        
        config = get_config()

        # Normalize mode
        if preprocess_mode is None:
            preprocess_mode = config.preprocessing.default_mode

        if isinstance(preprocess_mode, str):
            self.preprocess_mode = preprocess_mode.lower()
        else:
            self.preprocess_mode = preprocess_mode.value
        
        # Initialize preprocessor and postprocessor
        clahe_config = CLAHEConfig(
            clip_limit=config.preprocessing.clahe_clip_limit,
            tile_size=config.preprocessing.clahe_tile_size,
        )
        self.preprocessor = VINImagePreprocessor(
            mode=self.preprocess_mode,
            clahe_config=clahe_config,
            max_dimension=config.preprocessing.max_image_dimension,
        )
        self.postprocessor = VINPostProcessor(verbose=verbose)
        
        # Determine GPU setting
        if use_gpu is None:
            use_gpu = config.ocr.use_gpu
        
        # Import and create OCR provider
        if provider.lower() in {"paddleocr", "ensemble"}:
            provider_kwargs.setdefault("det_db_box_thresh", config.ocr.det_db_box_thresh)
            provider_kwargs.setdefault("rec_thresh", config.ocr.rec_thresh)
            provider_kwargs.setdefault("use_gpu", use_gpu)

        self._create_provider(
            provider=provider,
            api_key=api_key,
            ensemble_providers=ensemble_providers,
            ensemble_strategy=ensemble_strategy,
            use_gpu=use_gpu,
            **provider_kwargs
        )
        
        logger.info(f"MultiProviderVINPipeline initialized with {self.ocr_provider.name}")
        if self.verbose:
            print(f"Pipeline initialized with {self.ocr_provider.name}")
    
    def _create_provider(
        self,
        provider: str,
        api_key: Optional[str],
        ensemble_providers: Optional[List[str]],
        ensemble_strategy: str,
        use_gpu: bool = False,
        **kwargs
    ) -> None:
        """Create the OCR provider instance."""
        try:
            from src.vin_ocr.providers.ocr_providers import (
                OCRProviderFactory,
                EnsembleOCRProvider,
                OCRProviderType,
            )
        except ImportError:
            raise ConfigurationError(
                "ocr_providers module not found. "
                "Ensure ocr_providers.py is in the same directory."
            )
        
        provider = provider.lower()
        
        # Ensure use_gpu is in kwargs for providers
        kwargs.setdefault("use_gpu", use_gpu)
        
        if provider == "ensemble":
            # Create ensemble provider
            if not ensemble_providers:
                ensemble_providers = ["paddleocr"]  # Default to single provider
            
            providers = []
            for p in ensemble_providers:
                p_kwargs = {"api_key": api_key} if p == "deepseek" else {}
                p_kwargs.update(kwargs)
                providers.append(
                    OCRProviderFactory.create(p, auto_initialize=True, **p_kwargs)
                )
            
            self.ocr_provider = EnsembleOCRProvider(
                providers=providers,
                strategy=ensemble_strategy
            )
        else:
            # Create single provider
            p_kwargs = {"api_key": api_key} if provider == "deepseek" else {}
            p_kwargs.update(kwargs)
            self.ocr_provider = OCRProviderFactory.create(
                provider, auto_initialize=True, **p_kwargs
            )
    
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
                - provider: Name of OCR provider used
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
                    'provider': self.ocr_provider.name,
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
        
        # Run OCR via provider
        if self.verbose:
            print(f"Running OCR ({self.ocr_provider.name})...")
        
        ocr_result = self.ocr_provider.recognize(processed)
        
        raw_text = ocr_result.text
        confidence = ocr_result.confidence
        
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
        
        # Add timing and provider info
        output['processing_time_ms'] = elapsed['ms']
        output['provider'] = self.ocr_provider.name
        output['error'] = None
        
        return output
    
    @staticmethod
    def list_providers() -> List[str]:
        """List available OCR providers."""
        try:
            from src.vin_ocr.providers.ocr_providers import OCRProviderFactory
            return OCRProviderFactory.list_available()
        except ImportError:
            return ["paddleocr"]  # Fallback


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
    python -m src.vin_ocr.pipeline.vin_pipeline image.jpg
    python -m src.vin_ocr.pipeline.vin_pipeline image.jpg --mode engraved -v
    python -m src.vin_ocr.pipeline.vin_pipeline image.jpg --json
    python -m src.vin_ocr.pipeline.vin_pipeline image.jpg --provider deepseek --api-key YOUR_KEY
    python -m src.vin_ocr.pipeline.vin_pipeline --validate "SAL1P9EU2SA606664"
    python -m src.vin_ocr.pipeline.vin_pipeline --list-providers
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
    parser.add_argument('--version', action='version', version='VIN OCR Pipeline 2.0.0')
    
    # Provider options
    parser.add_argument('--provider', default='paddleocr',
                       choices=['paddleocr', 'deepseek', 'ensemble'],
                       help='OCR provider (default: paddleocr)')
    parser.add_argument('--api-key', metavar='KEY',
                       help='API key for cloud providers (DeepSeek, etc.)')
    parser.add_argument('--ensemble-providers', nargs='+',
                       help='Providers for ensemble mode (e.g., --ensemble-providers paddleocr deepseek)')
    parser.add_argument('--ensemble-strategy', default='best',
                       choices=['best', 'vote', 'cascade'],
                       help='Ensemble strategy (default: best)')
    parser.add_argument('--list-providers', action='store_true',
                       help='List available OCR providers')
    
    args = parser.parse_args()
    
    # Setup logging
    _setup_logging(args.verbose)
    
    # Handle --list-providers flag
    if args.list_providers:
        providers = MultiProviderVINPipeline.list_providers()
        print("Available OCR providers:")
        for p in providers:
            print(f"  - {p}")
        return 0
    
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
        parser.error("Image path required (or use --validate/--decode/--list-providers)")
        return 1
    
    # Check image exists
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}", file=sys.stderr)
        return 1
    
    # Create pipeline - use MultiProviderVINPipeline if non-default provider
    try:
        if args.provider != 'paddleocr' or args.api_key:
            # Use multi-provider pipeline
            pipeline = MultiProviderVINPipeline(
                provider=args.provider,
                preprocess_mode=args.mode,
                enable_postprocess=not args.no_postprocess,
                verbose=args.verbose,
                api_key=args.api_key,
                ensemble_providers=args.ensemble_providers,
                ensemble_strategy=args.ensemble_strategy,
            )
        else:
            # Use legacy PaddleOCR-only pipeline (faster initialization)
            pipeline = VINOCRPipeline(
                preprocess_mode=args.mode,
                enable_postprocess=not args.no_postprocess,
                verbose=args.verbose
            )
    except ConfigurationError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error initializing pipeline: {e}", file=sys.stderr)
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
        if result.get('provider'):
            print(f"Provider: {result['provider']}")
        
        if result.get('corrections'):
            print("\nCorrections applied:")
            for c in result['corrections']:
                print(f"  - {c}")
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
