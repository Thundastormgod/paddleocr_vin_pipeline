"""
VIN Image Preprocessor
======================

Specialized image preprocessing for VIN plate recognition on engraved metal surfaces.

This module provides multiple preprocessing strategies optimized for different
types of VIN plates and image conditions:

- STANDARD: Basic preprocessing for clean images
- ENGRAVED: Optimized for engraved/stamped metal plates
- LOW_CONTRAST: For faded or low-contrast images  
- ADAPTIVE: Automatically selects best strategy based on image analysis

Key techniques:
- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Morphological operations for engraved text enhancement
- Adaptive resizing to optimal OCR dimensions
- Noise reduction while preserving text edges

Author: VIN OCR Pipeline
License: MIT
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple, List, Dict, Any, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class PreprocessStrategy(str, Enum):
    """Preprocessing strategy enumeration."""
    NONE = 'none'
    STANDARD = 'standard'
    ENGRAVED = 'engraved'
    LOW_CONTRAST = 'low_contrast'
    ADAPTIVE = 'adaptive'
    AGGRESSIVE = 'aggressive'


@dataclass
class PreprocessConfig:
    """Configuration for VIN image preprocessing."""
    
    # Strategy selection
    strategy: PreprocessStrategy = PreprocessStrategy.ENGRAVED
    
    # Resizing parameters
    target_width: int = 1024  # Optimal width for OCR
    min_height: int = 32      # Minimum height after resize
    max_height: int = 512     # Maximum height after resize
    maintain_aspect: bool = True
    
    # CLAHE parameters
    clahe_clip_limit: float = 2.0
    clahe_tile_size: Tuple[int, int] = (8, 8)
    
    # Morphological parameters
    morph_kernel_size: Tuple[int, int] = (2, 2)
    morph_iterations: int = 1
    
    # Noise reduction
    denoise_strength: int = 5
    bilateral_d: int = 5
    bilateral_sigma_color: float = 50.0
    bilateral_sigma_space: float = 50.0
    
    # Sharpening
    sharpen_amount: float = 1.0
    unsharp_radius: int = 1
    unsharp_amount: float = 1.5
    
    # Binarization (for aggressive mode)
    use_binarization: bool = False
    binary_block_size: int = 11
    binary_c: int = 2
    
    # Auto-detection thresholds
    low_contrast_threshold: float = 50.0  # Std dev below this = low contrast
    
    # Debug
    save_intermediate: bool = False
    debug_output_dir: str = '/tmp/vin_preprocess_debug'


class VINPreprocessor:
    """
    VIN image preprocessor with multiple strategies.
    
    Provides optimized preprocessing for VIN plate recognition,
    particularly for engraved/stamped metal plates.
    
    Example:
        preprocessor = VINPreprocessor()
        processed = preprocessor.process(image)
        
        # With specific strategy
        preprocessor = VINPreprocessor(strategy=PreprocessStrategy.ENGRAVED)
        processed = preprocessor.process(image)
        
        # With custom config
        config = PreprocessConfig(target_width=1280, clahe_clip_limit=3.0)
        preprocessor = VINPreprocessor(config=config)
    """
    
    def __init__(
        self,
        strategy: Optional[PreprocessStrategy] = None,
        config: Optional[PreprocessConfig] = None,
    ):
        """
        Initialize the VIN preprocessor.
        
        Args:
            strategy: Preprocessing strategy (overrides config.strategy if provided)
            config: Full configuration (uses defaults if None)
        """
        self.config = config or PreprocessConfig()
        
        if strategy is not None:
            self.config.strategy = strategy
        
        # Initialize CLAHE
        self.clahe = cv2.createCLAHE(
            clipLimit=self.config.clahe_clip_limit,
            tileGridSize=self.config.clahe_tile_size
        )
        
        # Initialize morphological kernel
        self.morph_kernel = np.ones(
            self.config.morph_kernel_size, 
            np.uint8
        )
        
        logger.debug(f"VINPreprocessor initialized with strategy={self.config.strategy.value}")
    
    def process(
        self, 
        image: np.ndarray,
        strategy: Optional[PreprocessStrategy] = None,
    ) -> np.ndarray:
        """
        Process image for OCR.
        
        Args:
            image: Input image (BGR format)
            strategy: Override strategy for this call
            
        Returns:
            Preprocessed image (BGR format, compatible with PaddleOCR)
            
        Raises:
            ValueError: If image is invalid
        """
        if image is None or image.size == 0:
            raise ValueError("Input image is empty or None")
        
        # Use provided strategy or default
        active_strategy = strategy or self.config.strategy
        
        # Select processing pipeline based on strategy
        if active_strategy == PreprocessStrategy.NONE:
            return image
        elif active_strategy == PreprocessStrategy.STANDARD:
            return self._process_standard(image)
        elif active_strategy == PreprocessStrategy.ENGRAVED:
            return self._process_engraved(image)
        elif active_strategy == PreprocessStrategy.LOW_CONTRAST:
            return self._process_low_contrast(image)
        elif active_strategy == PreprocessStrategy.ADAPTIVE:
            return self._process_adaptive(image)
        elif active_strategy == PreprocessStrategy.AGGRESSIVE:
            return self._process_aggressive(image)
        else:
            logger.warning(f"Unknown strategy {active_strategy}, using ENGRAVED")
            return self._process_engraved(image)
    
    def _process_standard(self, image: np.ndarray) -> np.ndarray:
        """Standard preprocessing: resize + light enhancement."""
        # Resize to target width
        resized = self._resize_to_target(image)
        
        # Convert to grayscale
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE
        enhanced = self.clahe.apply(gray)
        
        # Convert back to BGR
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    
    def _process_engraved(self, image: np.ndarray) -> np.ndarray:
        """
        Optimized preprocessing for engraved/stamped VIN plates.
        
        Pipeline:
        1. Resize to optimal width (maintains aspect ratio)
        2. Convert to grayscale
        3. Apply CLAHE for contrast enhancement
        4. Apply morphological closing to enhance engraved characters
        5. Optional denoising
        """
        # Step 1: Resize
        resized = self._resize_to_target(image)
        
        if self.config.save_intermediate:
            self._save_debug('01_resized', resized)
        
        # Step 2: Convert to grayscale
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        
        if self.config.save_intermediate:
            self._save_debug('02_gray', gray)
        
        # Step 3: CLAHE
        enhanced = self.clahe.apply(gray)
        
        if self.config.save_intermediate:
            self._save_debug('03_clahe', enhanced)
        
        # Step 4: Morphological closing
        # This helps connect broken strokes in engraved characters
        closed = cv2.morphologyEx(
            enhanced, 
            cv2.MORPH_CLOSE, 
            self.morph_kernel,
            iterations=self.config.morph_iterations
        )
        
        if self.config.save_intermediate:
            self._save_debug('04_morph', closed)
        
        # Step 5: Light bilateral filter to reduce noise while preserving edges
        denoised = cv2.bilateralFilter(
            closed,
            d=self.config.bilateral_d,
            sigmaColor=self.config.bilateral_sigma_color,
            sigmaSpace=self.config.bilateral_sigma_space
        )
        
        if self.config.save_intermediate:
            self._save_debug('05_denoised', denoised)
        
        # Convert back to BGR for PaddleOCR
        return cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
    
    def _process_low_contrast(self, image: np.ndarray) -> np.ndarray:
        """Enhanced preprocessing for low contrast images."""
        # Resize
        resized = self._resize_to_target(image)
        
        # Convert to grayscale
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        
        # More aggressive CLAHE
        strong_clahe = cv2.createCLAHE(
            clipLimit=4.0,  # Higher clip limit
            tileGridSize=(4, 4)  # Smaller tiles
        )
        enhanced = strong_clahe.apply(gray)
        
        # Unsharp masking for additional contrast
        blurred = cv2.GaussianBlur(enhanced, (0, 0), self.config.unsharp_radius)
        sharpened = cv2.addWeighted(
            enhanced, 
            1 + self.config.unsharp_amount,
            blurred, 
            -self.config.unsharp_amount, 
            0
        )
        
        # Morphological enhancement
        closed = cv2.morphologyEx(
            sharpened, 
            cv2.MORPH_CLOSE, 
            self.morph_kernel
        )
        
        return cv2.cvtColor(closed, cv2.COLOR_GRAY2BGR)
    
    def _process_aggressive(self, image: np.ndarray) -> np.ndarray:
        """Aggressive preprocessing with binarization for difficult images."""
        # Start with engraved processing
        processed = self._process_engraved(image)
        
        # Convert to grayscale
        gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        
        # Adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            self.config.binary_block_size,
            self.config.binary_c
        )
        
        # Clean up with morphology
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        
        return cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
    
    def _process_adaptive(self, image: np.ndarray) -> np.ndarray:
        """
        Automatically select best strategy based on image analysis.
        
        Analyzes image characteristics to determine optimal preprocessing:
        - Contrast level
        - Brightness distribution
        - Noise level
        """
        # Analyze image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Check contrast (standard deviation)
        contrast = np.std(gray)
        
        # Check brightness
        brightness = np.mean(gray)
        
        logger.debug(f"Image analysis: contrast={contrast:.1f}, brightness={brightness:.1f}")
        
        # Select strategy based on analysis
        if contrast < self.config.low_contrast_threshold:
            logger.debug("Detected low contrast, using LOW_CONTRAST strategy")
            return self._process_low_contrast(image)
        elif contrast > 80:
            # High contrast - standard is usually sufficient
            logger.debug("Detected high contrast, using STANDARD strategy")
            return self._process_standard(image)
        else:
            # Medium contrast - use engraved (default)
            logger.debug("Using ENGRAVED strategy (default)")
            return self._process_engraved(image)
    
    def _resize_to_target(self, image: np.ndarray) -> np.ndarray:
        """
        Resize image to target width while maintaining aspect ratio.
        
        Args:
            image: Input image
            
        Returns:
            Resized image
        """
        h, w = image.shape[:2]
        
        if not self.config.maintain_aspect:
            return cv2.resize(
                image, 
                (self.config.target_width, self.config.min_height)
            )
        
        # Calculate scale to reach target width
        scale = self.config.target_width / w
        new_h = int(h * scale)
        
        # Clamp height
        new_h = max(self.config.min_height, min(new_h, self.config.max_height))
        
        return cv2.resize(
            image, 
            (self.config.target_width, new_h),
            interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
        )
    
    def _save_debug(self, name: str, image: np.ndarray) -> None:
        """Save intermediate image for debugging."""
        import os
        os.makedirs(self.config.debug_output_dir, exist_ok=True)
        path = os.path.join(self.config.debug_output_dir, f'{name}.jpg')
        cv2.imwrite(path, image)
        logger.debug(f"Saved debug image: {path}")
    
    def process_batch(
        self, 
        images: List[np.ndarray],
        strategy: Optional[PreprocessStrategy] = None,
    ) -> List[np.ndarray]:
        """
        Process multiple images.
        
        Args:
            images: List of input images
            strategy: Override strategy for all images
            
        Returns:
            List of preprocessed images
        """
        return [self.process(img, strategy) for img in images]
    
    def analyze_image(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Analyze image characteristics for debugging/tuning.
        
        Args:
            image: Input image
            
        Returns:
            Dict with analysis results
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        return {
            'width': image.shape[1],
            'height': image.shape[0],
            'channels': image.shape[2] if len(image.shape) > 2 else 1,
            'contrast': float(np.std(gray)),
            'brightness': float(np.mean(gray)),
            'min_pixel': int(gray.min()),
            'max_pixel': int(gray.max()),
            'dynamic_range': int(gray.max() - gray.min()),
            'suggested_strategy': self._suggest_strategy(gray),
        }
    
    def _suggest_strategy(self, gray: np.ndarray) -> str:
        """Suggest optimal strategy based on grayscale image analysis."""
        contrast = np.std(gray)
        
        if contrast < self.config.low_contrast_threshold:
            return PreprocessStrategy.LOW_CONTRAST.value
        elif contrast > 80:
            return PreprocessStrategy.STANDARD.value
        else:
            return PreprocessStrategy.ENGRAVED.value


# Convenience function for quick preprocessing
def preprocess_vin_image(
    image: Union[np.ndarray, str, Path],
    strategy: PreprocessStrategy = PreprocessStrategy.ENGRAVED,
    target_width: int = 1024,
) -> np.ndarray:
    """
    Quick preprocessing function for VIN images.
    
    Args:
        image: Input image (numpy array or path)
        strategy: Preprocessing strategy
        target_width: Target width for resizing
        
    Returns:
        Preprocessed image (BGR format)
    """
    # Load image if path
    if isinstance(image, (str, Path)):
        image = cv2.imread(str(image))
        if image is None:
            raise ValueError(f"Failed to load image: {image}")
    
    config = PreprocessConfig(
        strategy=strategy,
        target_width=target_width,
    )
    
    preprocessor = VINPreprocessor(config=config)
    return preprocessor.process(image)
