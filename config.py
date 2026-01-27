"""
Pipeline Configuration - Centralized Settings
==============================================

All configurable parameters in one place.
Supports environment variable overrides.

Usage:
    from config import get_config
    config = get_config()
    print(config.clahe_clip_limit)

Environment Variables:
    VIN_CLAHE_CLIP_LIMIT=3.0
    VIN_AUGMENTATION_THRESHOLD=100
    VIN_LOG_LEVEL=DEBUG

Author: JRL-VIN Project
Date: January 2026
"""

import os
import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

logger = logging.getLogger(__name__)


def _get_env_float(key: str, default: float) -> float:
    """Get float from environment variable."""
    value = os.environ.get(key)
    if value is not None:
        try:
            return float(value)
        except ValueError:
            logger.warning(f"Invalid float for {key}: {value}, using default {default}")
    return default


def _get_env_int(key: str, default: int) -> int:
    """Get int from environment variable."""
    value = os.environ.get(key)
    if value is not None:
        try:
            return int(value)
        except ValueError:
            logger.warning(f"Invalid int for {key}: {value}, using default {default}")
    return default


def _get_env_bool(key: str, default: bool) -> bool:
    """Get bool from environment variable."""
    value = os.environ.get(key)
    if value is not None:
        return value.lower() in ('true', '1', 'yes', 'on')
    return default


def _get_env_str(key: str, default: str) -> str:
    """Get string from environment variable."""
    return os.environ.get(key, default)


@dataclass
class PreprocessingConfig:
    """Image preprocessing configuration."""
    
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe_clip_limit: float = field(
        default_factory=lambda: _get_env_float('VIN_CLAHE_CLIP_LIMIT', 2.0)
    )
    clahe_tile_size: Tuple[int, int] = (8, 8)
    
    # Bilateral filter for denoising
    bilateral_d: int = 5
    bilateral_sigma_color: int = 50
    bilateral_sigma_space: int = 50
    
    # Maximum image dimension (prevents OOM)
    max_image_dimension: int = field(
        default_factory=lambda: _get_env_int('VIN_MAX_IMAGE_DIM', 4096)
    )
    
    # Default preprocessing mode
    default_mode: str = field(
        default_factory=lambda: _get_env_str('VIN_PREPROCESS_MODE', 'engraved')
    )


@dataclass
class OCRConfig:
    """PaddleOCR configuration."""
    
    language: str = 'en'
    
    # Detection thresholds
    det_db_box_thresh: float = field(
        default_factory=lambda: _get_env_float('VIN_DET_BOX_THRESH', 0.3)
    )
    det_db_thresh: float = 0.3
    det_db_unclip_ratio: float = 1.5
    
    # Recognition threshold
    rec_thresh: float = field(
        default_factory=lambda: _get_env_float('VIN_REC_THRESH', 0.3)
    )
    
    # Disable unnecessary features for VIN (performance optimization)
    use_doc_orientation_classify: bool = False
    use_doc_unwarping: bool = False
    use_textline_orientation: bool = False
    
    # GPU settings
    use_gpu: bool = field(
        default_factory=lambda: _get_env_bool('VIN_USE_GPU', True)
    )
    gpu_mem: int = 500  # MB


@dataclass
class TrainingConfig:
    """Training pipeline configuration."""
    
    # Data augmentation
    augmentation_threshold: int = field(
        default_factory=lambda: _get_env_int('VIN_AUGMENTATION_THRESHOLD', 50)
    )
    min_augmentation_factor: int = field(
        default_factory=lambda: _get_env_int('VIN_MIN_AUGMENTATION_FACTOR', 5)
    )
    max_augmentation_factor: int = field(
        default_factory=lambda: _get_env_int('VIN_MAX_AUGMENTATION_FACTOR', 50)
    )
    
    # Augmentation parameters
    brightness_range: Tuple[float, float] = (0.8, 1.2)
    rotation_range: Tuple[float, float] = (-3.0, 3.0)
    blur_kernel_sizes: Tuple[int, ...] = (3, 5)
    noise_std: float = 5.0
    
    # Training hyperparameters
    default_epochs: int = field(
        default_factory=lambda: _get_env_int('VIN_EPOCHS', 50)
    )
    default_batch_size: int = field(
        default_factory=lambda: _get_env_int('VIN_BATCH_SIZE', 8)
    )
    default_learning_rate: float = field(
        default_factory=lambda: _get_env_float('VIN_LEARNING_RATE', 0.0001)
    )
    
    # Finetune-specific hyperparameters
    finetune_epochs: int = field(
        default_factory=lambda: _get_env_int('VIN_FINETUNE_EPOCHS', 100)
    )
    finetune_batch_size: int = field(
        default_factory=lambda: _get_env_int('VIN_FINETUNE_BATCH_SIZE', 64)
    )
    finetune_learning_rate: float = field(
        default_factory=lambda: _get_env_float('VIN_FINETUNE_LR', 0.0005)
    )
    finetune_warmup_epochs: int = 5
    finetune_weight_decay: float = 1e-5
    
    # Dataset split ratios
    default_train_ratio: float = 0.7
    default_val_ratio: float = 0.15
    default_test_ratio: float = 0.15


@dataclass 
class LoggingConfig:
    """Logging configuration."""
    
    level: str = field(
        default_factory=lambda: _get_env_str('VIN_LOG_LEVEL', 'INFO')
    )
    format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_format: str = '%Y-%m-%d %H:%M:%S'
    
    # File logging (optional)
    log_file: Optional[str] = field(
        default_factory=lambda: os.environ.get('VIN_LOG_FILE')
    )


@dataclass
class PipelineConfig:
    """Complete pipeline configuration."""
    
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # VIN format settings
    vin_length: int = 17
    vin_charset: str = "0123456789ABCDEFGHJKLMNPRSTUVWXYZ"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def save(self, path: Path):
        """Save configuration to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> 'PipelineConfig':
        """Load configuration from JSON file."""
        with open(path) as f:
            data = json.load(f)
        
        config = cls()
        
        # Update preprocessing
        if 'preprocessing' in data:
            for key, value in data['preprocessing'].items():
                if hasattr(config.preprocessing, key):
                    setattr(config.preprocessing, key, value)
        
        # Update OCR
        if 'ocr' in data:
            for key, value in data['ocr'].items():
                if hasattr(config.ocr, key):
                    setattr(config.ocr, key, value)
        
        # Update training
        if 'training' in data:
            for key, value in data['training'].items():
                if hasattr(config.training, key):
                    setattr(config.training, key, value)
        
        return config


# Global configuration instance (singleton pattern)
_config: Optional[PipelineConfig] = None


def get_config() -> PipelineConfig:
    """
    Get the global configuration instance.
    
    Creates a new instance on first call, returns cached instance thereafter.
    """
    global _config
    if _config is None:
        _config = PipelineConfig()
        _setup_logging(_config.logging)
    return _config


def reset_config():
    """Reset configuration to defaults (useful for testing)."""
    global _config
    _config = None


def _setup_logging(config: LoggingConfig):
    """Configure logging based on settings."""
    level = getattr(logging, config.level.upper(), logging.INFO)
    
    handlers = [logging.StreamHandler()]
    
    if config.log_file:
        handlers.append(logging.FileHandler(config.log_file))
    
    logging.basicConfig(
        level=level,
        format=config.format,
        datefmt=config.date_format,
        handlers=handlers,
    )
