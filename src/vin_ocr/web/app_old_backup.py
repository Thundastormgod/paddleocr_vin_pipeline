#!/usr/bin/env python3
"""
VIN OCR Web UI - Streamlit Application
======================================

A user-friendly web interface for VIN recognition using multiple OCR models.

Features:
- Model Selection: Choose from PaddleOCR v3/v4, VIN Pipeline, or DeepSeek-OCR
- Single Image Recognition: Upload and process individual images
- Batch Processing: Process entire folders of images
- Training Interface: Train/fine-tune models on custom datasets
- Results Dashboard: View metrics, comparisons, and export results

Usage:
    streamlit run src/vin_ocr/web/app.py
    
    # Or with custom port:
    streamlit run src/vin_ocr/web/app.py --server.port 8501

Requirements:
    pip install streamlit plotly pandas pillow

Author: JRL-VIN Project
Date: January 2026
"""

import os
import sys
import json
import time
import logging
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import streamlit as st
import pandas as pd

# Setup logging
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
_project_root = Path(__file__).parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Optional imports with graceful fallback
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# Project imports - use absolute imports
try:
    from src.vin_ocr.pipeline.vin_pipeline import VINOCRPipeline
    VIN_PIPELINE_AVAILABLE = True
except ImportError:
    VIN_PIPELINE_AVAILABLE = False

try:
    from src.vin_ocr.evaluation.multi_model_evaluation import MultiModelEvaluator, VINCharValidator
    EVALUATOR_AVAILABLE = True
except ImportError:
    EVALUATOR_AVAILABLE = False

# VIN utilities - canonical functions from Single Source of Truth
try:
    from src.vin_ocr.core.vin_utils import extract_vin_from_filename as _extract_vin_from_filename
    VIN_UTILS_AVAILABLE = True
except ImportError:
    VIN_UTILS_AVAILABLE = False
    _extract_vin_from_filename = None

# Training progress components
try:
    from src.vin_ocr.web.training_components import (
        TrainingUI,
        ProgressTracker,
        TrainingRunner,
        get_global_tracker,
        get_global_runner,
    )
    TRAINING_UI_AVAILABLE = True
except ImportError as e:
    TRAINING_UI_AVAILABLE = False
    print(f"[WARNING] Training UI import failed: {e}")

# PaddleOCR training availability - use TRAINING_AVAILABLE (not PPOCR_TRAIN_AVAILABLE)
# Our custom implementation provides full training without requiring ppocr module
try:
    from src.vin_ocr.training.finetune_paddleocr import TRAINING_AVAILABLE as PADDLE_TRAINING_AVAILABLE
    from src.vin_ocr.training.finetune_paddleocr import PADDLE_AVAILABLE
except Exception:
    PADDLE_TRAINING_AVAILABLE = False
    PADDLE_AVAILABLE = False

# Legacy alias for backwards compatibility (DEPRECATED - use PADDLE_TRAINING_AVAILABLE)
PPOCR_TRAIN_AVAILABLE = PADDLE_TRAINING_AVAILABLE

# Dataset preparation utilities (optional)
try:
    from src.vin_ocr.utils.prepare_dataset import prepare_dataset, load_images_with_labels, create_paddleocr_labels
    DATASET_PREP_AVAILABLE = True
except Exception as e:
    DATASET_PREP_AVAILABLE = False
    print(f"[WARNING] Dataset preparation utilities unavailable: {e}")

# GPU management
try:
    from src.vin_ocr.utils.gpu_utils import get_gpu_manager, GPUManager
    GPU_UTILS_AVAILABLE = True
except ImportError:
    GPU_UTILS_AVAILABLE = False
    get_gpu_manager = None


# =============================================================================
# CENTRALIZED PATH CONFIGURATION (Single Source of Truth)
# =============================================================================
# All paths in the Web UI are resolved relative to _project_root to ensure
# consistent behavior regardless of where streamlit is launched from.

class ProjectPaths:
    """
    Centralized path configuration for the VIN OCR Web UI.
    
    All paths are resolved as absolute paths from the project root.
    This ensures consistent path handling across all operations.
    """
    ROOT = _project_root
    
    # Data directories
    FINETUNE_DATA = ROOT / "finetune_data"
    DAGSHUB_DATA = ROOT / "dagshub_data"
    DATASET = ROOT / "dataset"
    
    # Output directories
    OUTPUT = ROOT / "output"
    OUTPUT_FINETUNE = OUTPUT / "vin_rec_finetune"
    OUTPUT_SCRATCH = OUTPUT / "paddleocr_scratch"
    OUTPUT_DEEPSEEK = OUTPUT / "deepseek_vin_finetune"
    OUTPUT_ONNX = OUTPUT / "onnx"
    OUTPUT_TUNING = OUTPUT / "hyperparameter_tuning"
    
    # Model directories
    MODELS = ROOT / "models"
    MODELS_FINETUNED = MODELS / "finetuned"
    MODELS_ONNX = MODELS / "onnx"
    MODELS_DEEPSEEK_ONNX = MODELS / "deepseek_onnx"
    
    # Results directory
    RESULTS = ROOT / "results"
    
    # Config directory
    CONFIGS = ROOT / "configs"
    
    @classmethod
    def resolve(cls, path_str: str) -> Path:
        """
        Resolve a path string to an absolute path.
        
        If path starts with './' or is relative, resolve from project root.
        If path is already absolute, return as-is.
        
        Args:
            path_str: Path string (can be relative or absolute)
            
        Returns:
            Absolute Path object
        """
        path = Path(path_str)
        if path.is_absolute():
            return path
        # Remove leading './' if present
        if path_str.startswith('./'):
            path_str = path_str[2:]
        return cls.ROOT / path_str
    
    @classmethod
    def get_default_paths(cls) -> Dict[str, str]:
        """Get default paths as strings for UI defaults."""
        return {
            'finetune_data': str(cls.FINETUNE_DATA),
            'train_labels': str(cls.FINETUNE_DATA / "train_labels.txt"),
            'val_labels': str(cls.FINETUNE_DATA / "val_labels.txt"),
            'test_labels': str(cls.FINETUNE_DATA / "test_labels.txt"),
            'train_dir': str(cls.FINETUNE_DATA / "train"),
            'val_dir': str(cls.FINETUNE_DATA / "val"),
            'test_dir': str(cls.FINETUNE_DATA / "test"),
            'output_finetune': str(cls.OUTPUT_FINETUNE),
            'output_deepseek': str(cls.OUTPUT_DEEPSEEK),
            'output_scratch': str(cls.OUTPUT_SCRATCH),
            'output_tuning': str(cls.OUTPUT_TUNING),
        }
    
    @classmethod
    def ensure_directories(cls):
        """Create essential directories if they don't exist."""
        dirs_to_create = [
            cls.OUTPUT,
            cls.OUTPUT_FINETUNE,
            cls.OUTPUT_SCRATCH,
            cls.OUTPUT_DEEPSEEK,
            cls.OUTPUT_ONNX,
            cls.OUTPUT_TUNING,
            cls.RESULTS,
        ]
        for dir_path in dirs_to_create:
            dir_path.mkdir(parents=True, exist_ok=True)


# Initialize default paths
_DEFAULT_PATHS = ProjectPaths.get_default_paths()


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="VIN OCR Recognition System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .vin-display {
        font-family: 'Courier New', monospace;
        font-size: 1.5rem;
        font-weight: bold;
        letter-spacing: 0.2rem;
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 5px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

def init_session_state():
    """Initialize session state variables."""
    if 'models_loaded' not in st.session_state:
        st.session_state.models_loaded = {}
    if 'results_history' not in st.session_state:
        st.session_state.results_history = []
    if 'current_model' not in st.session_state:
        st.session_state.current_model = None
    if 'training_status' not in st.session_state:
        st.session_state.training_status = None
    if 'training_runner' not in st.session_state:
        st.session_state.training_runner = None
    if 'training_tracker' not in st.session_state:
        st.session_state.training_tracker = None
    if 'pipeline_load_error' not in st.session_state:
        st.session_state.pipeline_load_error = None
    
    # GPU settings - initialize from GPU manager if available
    if 'use_gpu' not in st.session_state:
        if GPU_UTILS_AVAILABLE:
            gpu = get_gpu_manager()
            st.session_state.use_gpu = gpu.any_gpu_available  # Default to GPU if available
        else:
            st.session_state.use_gpu = False


# =============================================================================
# FILE VALIDATION UTILITIES
# =============================================================================

SUPPORTED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}


def validate_image_file(file_path: str) -> Tuple[bool, str]:
    """
    Validate that a file exists, is readable, and is a supported image format.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    path = Path(file_path)
    
    # Check if file exists
    if not path.exists():
        return False, f"File not found: {file_path}"
    
    # Check if it's a file (not a directory)
    if not path.is_file():
        return False, f"Path is not a file: {file_path}"
    
    # Check if file is readable
    if not os.access(file_path, os.R_OK):
        return False, f"File is not readable (permission denied): {file_path}"
    
    # Check file extension
    ext = path.suffix.lower()
    if ext not in SUPPORTED_IMAGE_EXTENSIONS:
        return False, f"Unsupported image format '{ext}'. Supported: {', '.join(sorted(SUPPORTED_IMAGE_EXTENSIONS))}"
    
    # Check file size (not empty)
    if path.stat().st_size == 0:
        return False, f"File is empty: {file_path}"
    
    # Try to open with PIL if available
    if PIL_AVAILABLE:
        try:
            with Image.open(file_path) as img:
                img.verify()  # Verify it's a valid image
        except Exception as e:
            return False, f"Invalid or corrupted image file: {e}"
    
    return True, ""


def check_model_cache_health() -> Dict[str, Any]:
    """Check the health of the PaddleOCR model cache directory."""
    cache_dir = Path.home() / ".paddleocr"
    paddlex_dir = Path.home() / ".paddlex" / "official_models"
    
    health = {
        'paddleocr_cache_exists': cache_dir.exists(),
        'paddlex_cache_exists': paddlex_dir.exists(),
        'paddleocr_cache_writable': os.access(cache_dir.parent, os.W_OK),
        'paddlex_cache_writable': os.access(paddlex_dir.parent, os.W_OK) if paddlex_dir.parent.exists() else os.access(paddlex_dir.parent.parent, os.W_OK),
        'cache_paths': {
            'paddleocr': str(cache_dir),
            'paddlex': str(paddlex_dir),
        }
    }
    return health


# =============================================================================
# MODEL MANAGEMENT
# =============================================================================

def clear_model_cache():
    """Clear cached model instances to allow reload."""
    load_vin_pipeline.clear()
    load_paddleocr.clear()
    load_deepseek.clear()
    load_finetuned_paddleocr.clear()
    st.session_state.pipeline_load_error = None
    st.session_state.models_loaded = {}


@st.cache_resource
def load_vin_pipeline(_retry_count: int = 0, _use_gpu: bool = False):
    """
    Load VIN Pipeline (cached).
    
    Args:
        _retry_count: Internal parameter to bust cache on retry (underscore prefix 
                      tells Streamlit not to hash this parameter)
        _use_gpu: Whether to use GPU acceleration
    """
    if not VIN_PIPELINE_AVAILABLE:
        st.session_state.pipeline_load_error = "VIN Pipeline module not available. Check imports."
        return None
    
    try:
        # Set GPU preference if available
        if GPU_UTILS_AVAILABLE:
            gpu = get_gpu_manager()
            gpu.use_gpu = _use_gpu
            gpu.set_paddle_device()
        
        pipeline = VINOCRPipeline(use_gpu=_use_gpu)
        st.session_state.pipeline_load_error = None
        return pipeline
    except Exception as e:
        import traceback
        error_msg = f"Failed to load VIN Pipeline: {e}\n\nTraceback:\n{traceback.format_exc()}"
        st.session_state.pipeline_load_error = error_msg
        return None


@st.cache_resource
def load_paddleocr(version: str = "v4", _retry_count: int = 0, _use_gpu: bool = False):
    """
    Load PaddleOCR model (cached).
    
    Args:
        version: PaddleOCR version ("v3" or "v4")
        _retry_count: Internal parameter to bust cache on retry
        _use_gpu: Whether to use GPU acceleration
    """
    try:
        from paddleocr import PaddleOCR
        import paddle
        
        # Set Paddle device based on GPU preference
        # PaddleOCR 3.0+ doesn't accept use_gpu parameter - use paddle.device.set_device() instead
        use_gpu = _use_gpu
        if GPU_UTILS_AVAILABLE:
            gpu = get_gpu_manager()
            gpu.use_gpu = _use_gpu
            use_gpu = gpu.use_gpu and gpu.status.paddle_gpu
        
        try:
            if use_gpu and paddle.device.is_compiled_with_cuda():
                paddle.device.set_device('gpu')
            else:
                paddle.device.set_device('cpu')
        except Exception as device_err:
            logger.warning(f"Failed to set Paddle device: {device_err}")
        
        if version == "v3":
            return PaddleOCR(
                use_textline_orientation=True,
                lang='en',
                ocr_version='PP-OCRv3',
            )
        else:
            return PaddleOCR(
                use_textline_orientation=True,
                lang='en',
                text_det_thresh=0.3,
                text_det_box_thresh=0.5,
            )
    except Exception as e:
        import traceback
        st.error(f"Failed to load PaddleOCR {version}: {e}")
        st.session_state.pipeline_load_error = f"PaddleOCR {version} load failed:\n{traceback.format_exc()}"
        return None


@st.cache_resource
def load_deepseek(_retry_count: int = 0, _use_gpu: bool = False):
    """
    Load DeepSeek-OCR model (cached).
    
    Args:
        _retry_count: Internal parameter to bust cache on retry
        _use_gpu: Whether to use GPU acceleration
    """
    try:
        from src.vin_ocr.providers.ocr_providers import DeepSeekOCRProvider
        
        # Determine device
        device = "cpu"
        if _use_gpu and GPU_UTILS_AVAILABLE:
            gpu = get_gpu_manager()
            device = gpu.get_device_string(for_torch=True)
        
        provider = DeepSeekOCRProvider(device=device)
        if provider.is_available:
            return provider
        else:
            st.warning("DeepSeek-OCR dependencies not installed. Click 'Download DeepSeek Model' to install.")
            return None
    except Exception as e:
        import traceback
        st.error(f"Failed to load DeepSeek-OCR: {e}")
        st.session_state.pipeline_load_error = f"DeepSeek-OCR load failed:\n{traceback.format_exc()}"
        return None


@st.cache_resource
def load_finetuned_paddleocr(model_path: str, _use_gpu: bool = False):
    """
    Load a fine-tuned PaddleOCR model for inference.
    
    Uses direct Paddle inference API instead of PaddleOCR wrapper
    for better compatibility with custom fine-tuned models.
    
    Args:
        model_path: Path to the model directory
        _use_gpu: Whether to use GPU acceleration
        
    Returns:
        VINInference instance, or None on failure
    """
    try:
        from src.vin_ocr.inference.paddle_inference import VINInference
        
        model_path = Path(model_path)
        logger.info(f"Attempting to load fine-tuned model from: {model_path}")
        
        # Check if this is an inference model directory
        inference_dir = None
        
        # Case 1: Direct inference directory with inference.pdiparams
        if model_path.is_dir():
            if (model_path / "inference.pdiparams").exists():
                inference_dir = str(model_path)
                logger.info(f"Found inference model at: {inference_dir}")
            # Case 2: Directory with inference subdirectory
            elif (model_path / "inference" / "inference.pdiparams").exists():
                inference_dir = str(model_path / "inference")
                logger.info(f"Found inference model at: {inference_dir}")
        
        if inference_dir:
            # Load using direct Paddle inference (more reliable than PaddleOCR wrapper)
            engine = VINInference(inference_dir, use_gpu=_use_gpu)
            logger.info(f"Successfully loaded inference model with direct Paddle API")
            return engine
        else:
            logger.warning(f"Model path {model_path} is not an inference model. "
                          f"Fine-tuned checkpoints (.pdparams) must be exported to inference format first.")
            return None
        
    except Exception as e:
        import traceback
        logger.error(f"Failed to load fine-tuned model: {e}")
        logger.error(traceback.format_exc())
        st.session_state.pipeline_load_error = f"Fine-tuned model load failed: {e}"
        return None


def run_onnx_inference(model_path: str, image_path: str) -> Dict[str, Any]:
    """
    Run inference with an ONNX model.
    
    Args:
        model_path: Path to the .onnx model file
        image_path: Path to the input image
        
    Returns:
        Dict with 'vin', 'confidence', 'raw_text', or 'error'
    """
    result = {'vin': '', 'confidence': 0.0, 'raw_text': '', 'error': None}
    
    try:
        import onnxruntime as ort
        import cv2
        import numpy as np
        
        # Load the ONNX model
        session = ort.InferenceSession(model_path)
        
        # Get input details
        input_info = session.get_inputs()[0]
        input_name = input_info.name
        input_shape = input_info.shape
        
        # Load and preprocess image
        image = cv2.imread(image_path)
        if image is None:
            result['error'] = f"Failed to load image: {image_path}"
            return result
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Determine target size from model input
        if len(input_shape) == 4:
            if isinstance(input_shape[2], int) and isinstance(input_shape[3], int):
                target_h, target_w = input_shape[2], input_shape[3]
            else:
                target_h, target_w = 32, 320  # Default OCR input size
        else:
            target_h, target_w = 32, 320
        
        # Resize maintaining aspect ratio
        h, w = gray.shape[:2]
        ratio = target_h / h
        new_w = min(int(w * ratio), target_w)
        resized = cv2.resize(gray, (new_w, target_h))
        
        # Pad to target width
        if new_w < target_w:
            padded = np.zeros((target_h, target_w), dtype=np.uint8)
            padded[:, :new_w] = resized
            resized = padded
        
        # Normalize and prepare input
        input_data = resized.astype(np.float32) / 255.0
        input_data = np.expand_dims(input_data, axis=0)  # Add batch
        input_data = np.expand_dims(input_data, axis=0)  # Add channel
        
        # Run inference
        output_names = [o.name for o in session.get_outputs()]
        outputs = session.run(output_names, {input_name: input_data})
        
        # Decode output (CTC decoding)
        output = outputs[0]
        pred_indices = np.argmax(output, axis=2)[0]
        
        # VIN valid characters
        char_set = "0123456789ABCDEFGHJKLMNPRSTUVWXYZ"
        blank_idx = len(char_set)
        
        # Simple CTC decode
        decoded = []
        prev_idx = blank_idx
        for idx in pred_indices:
            if idx != blank_idx and idx != prev_idx:
                if idx < len(char_set):
                    decoded.append(char_set[idx])
            prev_idx = idx
        
        raw_text = ''.join(decoded)
        result['raw_text'] = raw_text
        result['vin'] = raw_text[:17] if len(raw_text) >= 17 else raw_text
        result['confidence'] = float(np.mean(np.max(output, axis=2))) if output.size > 0 else 0.0
        
    except ImportError:
        result['error'] = "ONNX Runtime not installed. Run: pip install onnxruntime"
    except Exception as e:
        result['error'] = f"ONNX inference failed: {str(e)}"
    
    return result


def get_available_models() -> Dict[str, bool]:
    """Check which models are available, including fine-tuned models."""
    models = {
        "VIN Pipeline (PP-OCRv5)": VIN_PIPELINE_AVAILABLE,
        "PaddleOCR PP-OCRv4": True,  # Will check on load
        "PaddleOCR PP-OCRv3": True,  # Will check on load
        "DeepSeek-OCR": False,  # Will check on load
    }
    
    # Check DeepSeek availability
    try:
        from src.vin_ocr.providers.ocr_providers import DeepSeekOCRProvider
        provider = DeepSeekOCRProvider()
        models["DeepSeek-OCR"] = provider.is_available
    except:
        pass
    
    # Check for fine-tuned PaddleOCR models
    finetuned_models = get_finetuned_models()
    for model_name, model_path in finetuned_models.items():
        models[model_name] = True
    
    return models


def get_finetuned_models() -> Dict[str, str]:
    """
    Discover fine-tuned models in output directories.
    
    Priority:
    1. Inference models (exported, ready for production)
    2. Training checkpoints (.pdparams files)
    
    Returns:
        Dict mapping display name to model path
    """
    finetuned = {}
    
    output_root = _project_root / "output"
    
    if not output_root.exists():
        return finetuned
    
    # Scan all directories under output/
    for output_dir in output_root.iterdir():
        if not output_dir.is_dir():
            continue
        
        dir_name = output_dir.name
        
        # Priority 1: Check for inference directory (exported model - PREFERRED)
        inference_dir = output_dir / "inference"
        if inference_dir.exists() and (inference_dir / "inference.pdiparams").exists():
            display_name = f"🎯 Fine-tuned: {dir_name} (inference)"
            finetuned[display_name] = str(inference_dir)
            logger.debug(f"Found inference model: {inference_dir}")
        
        # Priority 2: Check for best_model or best_accuracy checkpoint
        for best_name in ["best_model", "best_accuracy"]:
            best_path = output_dir / f"{best_name}.pdparams"
            if best_path.exists():
                display_name = f"🎯 Fine-tuned: {dir_name}/{best_name}"
                finetuned[display_name] = str(output_dir)
                break
        
        # Priority 3: Latest checkpoint
        latest_path = output_dir / "latest.pdparams"
        if latest_path.exists():
            display_name = f"🎯 Fine-tuned: {dir_name}/latest"
            finetuned[display_name] = str(output_dir)
    
    # Also check models/finetuned/ directory
    finetuned_dir = _project_root / "models" / "finetuned"
    if finetuned_dir.exists():
        for model_dir in finetuned_dir.iterdir():
            if model_dir.is_dir():
                # Check for inference model
                inference_dir = model_dir / "inference"
                if inference_dir.exists():
                    display_name = f"🎯 Fine-tuned: {model_dir.name}"
                    finetuned[display_name] = str(inference_dir)
    
    # Check for ONNX models
    onnx_dirs = [
        _project_root / "output" / "onnx",
        _project_root / "models" / "onnx",
        _project_root / "models" / "deepseek_onnx",
    ]
    
    for onnx_dir in onnx_dirs:
        if not onnx_dir.exists():
            continue
        for onnx_file in onnx_dir.glob("*.onnx"):
            model_name = onnx_file.stem
            display_name = f"📦 ONNX: {model_name}"
            finetuned[display_name] = str(onnx_file)
    
    return finetuned


# =============================================================================
# RECOGNITION FUNCTIONS
# =============================================================================

def recognize_with_model(image_path: str, model_name: str) -> Dict[str, Any]:
    """
    Run recognition with selected model.
    
    Includes proper file validation and detailed error reporting.
    """
    import traceback
    start_time = time.time()
    result = {
        'vin': '',
        'confidence': 0.0,
        'model': model_name,
        'processing_time': 0.0,
        'error': None,
        'error_type': None,
        'error_traceback': None,
    }
    
    # Step 1: Validate the image file before processing
    is_valid, validation_error = validate_image_file(image_path)
    if not is_valid:
        result['error'] = validation_error
        result['error_type'] = 'file_validation'
        result['processing_time'] = time.time() - start_time
        return result
    
    try:
        if model_name == "VIN Pipeline (PP-OCRv5)":
            use_gpu = st.session_state.get('use_gpu', False)
            pipeline = load_vin_pipeline(_use_gpu=use_gpu)
            if pipeline is None:
                result['error'] = st.session_state.get('pipeline_load_error', 'Pipeline failed to load (check model download)')
                result['error_type'] = 'model_load'
            else:
                res = pipeline.recognize(image_path)
                result['vin'] = res.get('vin', '')
                result['confidence'] = res.get('confidence', 0.0)
                result['raw_ocr'] = res.get('raw_ocr', '')
                result['corrections'] = res.get('corrections', [])
        
        elif model_name.startswith("PaddleOCR"):
            version = "v3" if "v3" in model_name else "v4"
            use_gpu = st.session_state.get('use_gpu', False)
            engine = load_paddleocr(version, _use_gpu=use_gpu)
            if engine is None:
                result['error'] = st.session_state.get('pipeline_load_error', f'PaddleOCR {version} failed to load')
                result['error_type'] = 'model_load'
            else:
                ocr_result = engine.predict(image_path)
                texts, confidences = [], []
                for item in ocr_result:
                    # PaddleOCR v5 returns dict-like OCRResult objects
                    if isinstance(item, dict) or hasattr(item, 'get'):
                        rec_texts = item.get('rec_texts', [])
                        rec_scores = item.get('rec_scores', [])
                        if rec_texts:
                            texts.extend(rec_texts)
                        if rec_scores:
                            confidences.extend(rec_scores)
                    # Legacy API fallback
                    elif hasattr(item, 'rec_texts'):
                        texts.extend(item.rec_texts or [])
                        confidences.extend(item.rec_scores or [])
                
                combined_text = ' '.join(str(t) for t in texts)
                result['raw_ocr'] = combined_text
                result['confidence'] = sum(confidences) / len(confidences) if confidences else 0.0
                
                # Extract VIN
                if EVALUATOR_AVAILABLE:
                    result['vin'] = VINCharValidator.extract_vin_from_text(combined_text)
                else:
                    result['vin'] = ''.join(c for c in combined_text.upper() if c.isalnum())[:17]
        
        elif model_name == "DeepSeek-OCR":
            use_gpu = st.session_state.get('use_gpu', False)
            provider = load_deepseek(_use_gpu=use_gpu)
            if provider is None:
                result['error'] = st.session_state.get('pipeline_load_error', 'DeepSeek-OCR failed to load')
                result['error_type'] = 'model_load'
            else:
                if not provider._initialized:
                    provider.initialize()
                res = provider.recognize(image_path)
                result['vin'] = res.text if res else ''
                result['confidence'] = res.confidence if res else 0.0
        
        elif model_name.startswith("🎯 Fine-tuned:"):
            # Handle fine-tuned PaddleOCR models
            finetuned_models = get_finetuned_models()
            model_path = finetuned_models.get(model_name)
            
            if not model_path:
                result['error'] = f"Fine-tuned model not found: {model_name}"
                result['error_type'] = 'model_not_found'
            else:
                use_gpu = st.session_state.get('use_gpu', False)
                engine = load_finetuned_paddleocr(model_path, _use_gpu=use_gpu)
                if engine is None:
                    result['error'] = f'Failed to load fine-tuned model: {model_path}'
                    result['error_type'] = 'model_load'
                else:
                    # Use direct Paddle inference API
                    rec_result = engine.recognize(image_path)
                    
                    if rec_result.get('error'):
                        result['error'] = rec_result['error']
                        result['error_type'] = 'inference_error'
                    else:
                        result['raw_ocr'] = rec_result.get('raw_text', '')
                        result['confidence'] = rec_result.get('confidence', 0.0)
                        result['vin'] = rec_result.get('vin', '')
        
        elif model_name.startswith("📦 ONNX:"):
            # Handle ONNX models
            finetuned_models = get_finetuned_models()
            model_path = finetuned_models.get(model_name)
            
            if not model_path:
                result['error'] = f"ONNX model not found: {model_name}"
                result['error_type'] = 'model_not_found'
            else:
                onnx_result = run_onnx_inference(model_path, image_path)
                if onnx_result.get('error'):
                    result['error'] = onnx_result['error']
                    result['error_type'] = 'onnx_inference'
                else:
                    result['vin'] = onnx_result.get('vin', '')
                    result['confidence'] = onnx_result.get('confidence', 0.0)
                    result['raw_ocr'] = onnx_result.get('raw_text', '')
    
    except FileNotFoundError as e:
        result['error'] = f"File not found: {e}"
        result['error_type'] = 'file_not_found'
        result['error_traceback'] = traceback.format_exc()
    except PermissionError as e:
        result['error'] = f"Permission denied: {e}"
        result['error_type'] = 'permission_error'
        result['error_traceback'] = traceback.format_exc()
    except IOError as e:
        result['error'] = f"IO error reading file: {e}"
        result['error_type'] = 'io_error'
        result['error_traceback'] = traceback.format_exc()
    except Exception as e:
        result['error'] = str(e)
        result['error_type'] = 'recognition_error'
        result['error_traceback'] = traceback.format_exc()
    
    result['processing_time'] = time.time() - start_time
    return result


# =============================================================================
# UI COMPONENTS
# =============================================================================

def render_gpu_toggle():
    """Render GPU toggle with status information."""
    if not GPU_UTILS_AVAILABLE:
        st.caption("⚠️ GPU utils not available")
        return
    
    gpu = get_gpu_manager()
    status = gpu.detect_all()
    
    # GPU toggle
    gpu_available = gpu.any_gpu_available
    
    if gpu_available:
        # Show toggle
        use_gpu = st.toggle(
            "🚀 Use GPU",
            value=st.session_state.use_gpu,
            help=f"GPU detected: {status.best_device_name}",
            key="gpu_toggle"
        )
        
        # Update state and GPU manager
        if use_gpu != st.session_state.use_gpu:
            st.session_state.use_gpu = use_gpu
            gpu.use_gpu = use_gpu
            # Clear model cache to reload with new device
            clear_model_cache()
            st.rerun()
        
        # Show device info
        if use_gpu:
            st.caption(f"✅ {status.best_device_name}")
        else:
            st.caption("⚡ Using CPU (GPU disabled)")
    else:
        st.caption("💻 CPU only (no GPU detected)")
        st.session_state.use_gpu = False
    
    # Expandable GPU details
    with st.expander("GPU Details", expanded=False):
        if status.cuda_available:
            for d in status.cuda_devices:
                st.text(f"🎮 CUDA {d.device_id}: {d.name}")
                st.text(f"   Memory: {d.memory_available_gb:.1f}/{d.memory_total_gb:.1f} GB")
        elif status.mps_available:
            st.text(f"🍎 {status.mps_info.name}")
        else:
            st.text("No GPU detected")
        
        if status.torch_available:
            st.text(f"PyTorch: {status.torch_version}")
        if status.paddle_available:
            st.text(f"Paddle: {status.paddle_version}")
            st.text(f"Paddle GPU: {'✓' if status.paddle_gpu else '✗'}")


def render_sidebar():
    """Render sidebar with navigation and settings."""
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/car-roof-box.png", width=80)
        st.title("VIN OCR System")
        
        st.divider()
        
        # Navigation
        page = st.radio(
            "Navigation",
            ["🔍 Recognition", "📊 Batch Evaluation", "🎯 Training", "📈 Results Dashboard", "🔧 System Health"],
            label_visibility="collapsed"
        )
        
        st.divider()
        
        # GPU Toggle
        st.subheader("⚡ Compute Device")
        render_gpu_toggle()
        
        st.divider()
        
        # Cache Management
        st.subheader("🔄 Cache & Refresh")
        
        col_cache1, col_cache2 = st.columns(2)
        with col_cache1:
            if st.button("🗑️ Clear Cache", help="Clear all cached models and data", use_container_width=True):
                clear_model_cache()
                st.cache_data.clear()
                st.cache_resource.clear()
                st.success("✓ All caches cleared!")
                st.rerun()
        
        with col_cache2:
            if st.button("� Full Refresh", help="Clear everything and refresh the entire page", use_container_width=True):
                clear_model_cache()
                st.cache_data.clear()
                st.cache_resource.clear()
                # Clear session state
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
        
        # Show last pipeline error if any
        if st.session_state.get('pipeline_load_error'):
            with st.expander("⚠️ Last Load Error", expanded=False):
                st.code(st.session_state.pipeline_load_error, language="text")
        
        st.divider()
        
        # Quick Stats
        st.subheader("📊 Session Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Images Processed", len(st.session_state.results_history))
        with col2:
            if st.session_state.results_history:
                avg_conf = sum(r.get('confidence', 0) for r in st.session_state.results_history) / len(st.session_state.results_history)
                st.metric("Avg Confidence", f"{avg_conf:.1%}")
            else:
                st.metric("Avg Confidence", "N/A")
        
        return page


def render_recognition_page():
    """Render the single image recognition page."""
    st.markdown('<h1 class="main-header">🔍 VIN Recognition</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload an image to extract the Vehicle Identification Number</p>', unsafe_allow_html=True)
    
    # Model Selection at top of recognition page
    st.markdown("### 🤖 Select OCR Model")
    available_models = get_available_models()
    model_options = [m for m, available in available_models.items()]
    
    col_model1, col_model2 = st.columns([3, 1])
    with col_model1:
        selected_model = st.selectbox(
            "Choose the OCR model for recognition",
            model_options,
            help="Select which model to use for VIN extraction",
            key="recognition_model_selector"
        )
        st.session_state.current_model = selected_model
    
    with col_model2:
        # Show model status badge
        if available_models.get(selected_model, False):
            st.success(f"✅ Ready")
        else:
            st.warning(f"⚠️ Setup needed")
    
    st.markdown("---")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a VIN image",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Upload an image containing a VIN plate or sticker",
            key="vin_image_uploader"
        )
        
        # Track filename changes to reset ground truth when new file uploaded
        if uploaded_file:
            current_filename = uploaded_file.name
            if st.session_state.get('_last_uploaded_filename') != current_filename:
                # New file uploaded - reset ground truth to auto-detected value
                st.session_state['_last_uploaded_filename'] = current_filename
                auto_gt = extract_vin_from_filename(current_filename)
                st.session_state['gt_vin_input'] = auto_gt if auto_gt else ""
        
        if uploaded_file:
            if PIL_AVAILABLE:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_container_width=True)
            else:
                st.info("Image preview not available (PIL not installed)")
            
            # Ground Truth Section
            st.markdown("---")
            st.subheader("🎯 Ground Truth Comparison")
            
            # Get auto-extracted VIN from filename (already set in session state above)
            auto_gt = extract_vin_from_filename(uploaded_file.name)
            
            # Use session state value (auto-updated when file changes)
            default_gt = st.session_state.get('gt_vin_input', auto_gt if auto_gt else "")
            
            ground_truth = st.text_input(
                "Expected VIN (Ground Truth)",
                value=default_gt,
                max_chars=17,
                help="Enter the correct VIN to compare against recognition result. Auto-detected from filename when possible.",
                key="gt_vin_input"
            ).upper().strip()
            
            if auto_gt:
                st.caption(f"💡 Auto-detected from filename: `{auto_gt}`")
    
    with col2:
        st.subheader("📋 Recognition Result")
        
        if uploaded_file:
            # Save to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            
            model_name = st.session_state.current_model
            
            with st.spinner(f"Processing with {model_name}..."):
                result = recognize_with_model(tmp_path, model_name)
            
            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except Exception:
                pass  # Ignore cleanup errors
            
            # Display results
            if result['error']:
                error_type = result.get('error_type', 'unknown')
                
                # Show appropriate error message based on error type
                if error_type == 'file_validation':
                    st.error(f"📁 File Error: {result['error']}")
                elif error_type == 'model_load':
                    st.error(f"🤖 Model Load Error: {result['error']}")
                    st.info("💡 Tip: Try clicking 'Clear Model Cache & Reload' in the sidebar, then retry.")
                elif error_type in ('file_not_found', 'permission_error', 'io_error'):
                    st.error(f"📂 {error_type.replace('_', ' ').title()}: {result['error']}")
                else:
                    st.error(f"❌ Recognition Error: {result['error']}")
                
                # Show full traceback in expander for debugging
                if result.get('error_traceback'):
                    with st.expander("🔍 Full Error Details (for debugging)"):
                        st.code(result['error_traceback'], language="python")
            else:
                # VIN Display
                vin = result['vin'] or "No VIN detected"
                st.markdown(f'<div class="vin-display">{vin}</div>', unsafe_allow_html=True)
                
                # Ground Truth Comparison
                ground_truth = st.session_state.get('gt_vin_input', '').upper().strip()
                if ground_truth and len(ground_truth) == 17:
                    is_exact_match = vin == ground_truth
                    
                    # Calculate character-level accuracy
                    chars_correct = sum(1 for a, b in zip(vin[:17], ground_truth[:17]) if a == b)
                    char_accuracy = chars_correct / 17 * 100
                    
                    # Visual comparison
                    st.markdown("#### 🔍 Ground Truth Comparison")
                    
                    if is_exact_match:
                        st.success(f"✅ **EXACT MATCH** - Recognition is correct!")
                    else:
                        st.error(f"❌ **MISMATCH** - {chars_correct}/17 characters correct ({char_accuracy:.1f}%)")
                        
                        # Show character-by-character comparison
                        st.markdown("**Character-by-character comparison:**")
                        
                        # Build comparison visualization
                        comparison_html = '<div style="font-family: monospace; font-size: 14px; line-height: 1.8;">'
                        comparison_html += '<div><strong>Pred:</strong> '
                        for i, (pred_char, gt_char) in enumerate(zip(vin.ljust(17)[:17], ground_truth[:17])):
                            if pred_char == gt_char:
                                comparison_html += f'<span style="color: green; font-weight: bold;">{pred_char}</span>'
                            else:
                                comparison_html += f'<span style="color: red; background-color: #ffcccc; font-weight: bold;">{pred_char}</span>'
                        comparison_html += '</div>'
                        comparison_html += f'<div><strong>GT:  </strong> {ground_truth}</div>'
                        comparison_html += '</div>'
                        
                        st.markdown(comparison_html, unsafe_allow_html=True)
                        
                        # Show which positions are wrong
                        wrong_positions = [i+1 for i, (a, b) in enumerate(zip(vin[:17], ground_truth[:17])) if a != b]
                        if wrong_positions:
                            st.caption(f"⚠️ Errors at positions: {', '.join(map(str, wrong_positions))}")
                    
                    # Store comparison result
                    result['ground_truth'] = ground_truth
                    result['exact_match'] = is_exact_match
                    result['char_accuracy'] = char_accuracy
                
                elif ground_truth:
                    st.warning(f"⚠️ Ground truth must be exactly 17 characters (got {len(ground_truth)})")
                
                st.markdown("---")
                
                # Metrics
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Confidence", f"{result['confidence']:.1%}")
                with col_b:
                    st.metric("Length", f"{len(result['vin'])}/17")
                with col_c:
                    st.metric("Time", f"{result['processing_time']:.2f}s")
                
                # Validation
                if len(result['vin']) == 17:
                    st.success("✓ Valid VIN length (17 characters)")
                else:
                    st.warning(f"⚠ Invalid VIN length ({len(result['vin'])} characters)")
                
                # Raw OCR output
                with st.expander("🔍 Raw OCR Output"):
                    st.code(result.get('raw_ocr', 'N/A'))
                
                # Corrections applied
                if result.get('corrections'):
                    with st.expander("🔧 Corrections Applied"):
                        for corr in result['corrections']:
                            st.write(f"- {corr}")
                
                # Save to history
                result['timestamp'] = datetime.now().isoformat()
                result['filename'] = uploaded_file.name
                st.session_state.results_history.append(result)
        else:
            st.info("👆 Upload an image to get started")


def render_batch_evaluation_page():
    """Render the batch evaluation page with file upload."""
    st.markdown('<h1 class="main-header">📊 Batch Evaluation</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload multiple VIN images to evaluate model performance</p>', unsafe_allow_html=True)
    
    # File Upload Section
    st.subheader("📤 Upload VIN Images")
    uploaded_files = st.file_uploader(
        "Choose VIN images from your device",
        type=['jpg', 'jpeg', 'png'],
        accept_multiple_files=True,
        help="Select multiple images to evaluate. Ground truth is extracted from filenames."
    )
    
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} images uploaded")
        
        # Preview uploaded images (show first 5)
        with st.expander("👁️ Preview Uploaded Images", expanded=False):
            preview_cols = st.columns(min(5, len(uploaded_files)))
            for i, (col, uploaded_file) in enumerate(zip(preview_cols, uploaded_files[:5])):
                with col:
                    try:
                        if PIL_AVAILABLE:
                            img = Image.open(uploaded_file)
                            st.image(img, caption=uploaded_file.name[:20], use_container_width=True)
                            uploaded_file.seek(0)  # Reset file pointer
                    except Exception:
                        st.caption(uploaded_file.name[:15])
            if len(uploaded_files) > 5:
                st.caption(f"... and {len(uploaded_files) - 5} more images")
        
        st.divider()
        
        # Model Selection Section
        st.subheader("🤖 Select Model for Evaluation")
        
        # Get all available models and categorize them
        available_models = get_available_models()
        finetuned_models = get_finetuned_models()
        
        # Build categorized model list
        base_models = []
        finetuned_list = []
        onnx_models = []
        model_categories = {}
        
        for model_name, is_available in available_models.items():
            if not is_available:
                continue
            if model_name.startswith("🎯"):
                finetuned_list.append(model_name)
                model_categories[model_name] = "Fine-tuned"
            elif model_name.startswith("📦"):
                onnx_models.append(model_name)
                model_categories[model_name] = "ONNX"
            else:
                base_models.append(model_name)
                model_categories[model_name] = "Base"
        
        # Combine all models
        all_models = base_models + finetuned_list + onnx_models
        
        # Display model counts
        col_counts = st.columns(3)
        with col_counts[0]:
            st.metric("Base Models", len(base_models))
        with col_counts[1]:
            st.metric("Fine-tuned", len(finetuned_list))
        with col_counts[2]:
            st.metric("ONNX Models", len(onnx_models))
        
        if all_models:
            # Default to VIN Pipeline if available
            default_idx = 0
            for i, m in enumerate(all_models):
                if "VIN Pipeline" in m:
                    default_idx = i
                    break
            
            selected_model = st.selectbox(
                "Choose ONE model for evaluation",
                options=all_models,
                index=default_idx,
                help="Select a single model to evaluate on all uploaded images",
                key="batch_eval_model"
            )
            
            # Show model category info
            category = model_categories.get(selected_model, "Unknown")
            if category == "Base":
                st.info(f"📦 **{selected_model}** - Pretrained base model (default weights)")
            elif category == "Fine-tuned":
                st.info(f"🎯 **{selected_model}** - Custom fine-tuned model")
            elif category == "ONNX":
                st.info(f"📦 **{selected_model}** - Production ONNX model")
        else:
            st.warning("⚠️ No models available. Please check model installation.")
            selected_model = None
        
        st.divider()
        
        # Options Section
        st.subheader("⚙️ Options")
        col_opt1, col_opt2 = st.columns(2)
        with col_opt1:
            save_results = st.checkbox(
                "💾 Save results to dashboard", 
                value=True,
                help="Save evaluation results for viewing in Results Dashboard"
            )
        with col_opt2:
            show_details = st.checkbox(
                "📋 Show detailed results",
                value=True,
                help="Show per-image results table"
            )
        
        st.divider()
        
        # Run Evaluation Button
        if st.button("🚀 Run Batch Evaluation", type="primary", disabled=selected_model is None, use_container_width=True):
            process_uploaded_batch(uploaded_files, selected_model, save_results=save_results, show_details=show_details)
    
    else:
        # Instructions when no files uploaded
        st.info("� **Click 'Browse files' to select VIN images from your device**")
        
        with st.expander("ℹ️ How Batch Evaluation Works"):
            st.markdown("""
            ### Steps:
            1. **Upload Images** - Select multiple VIN images from your device
            2. **Select Model** - Choose which OCR model to evaluate
            3. **Run Evaluation** - Process all images and see results
            
            ### Ground Truth:
            - Ground truth VINs are **automatically extracted from filenames**
            - Supported filename patterns:
              - `1-VIN -SAL1A2A40SA606662.jpg`
              - `VIN_train_1234-WVWZZZ3CZWE123456.png`
              - `WVWZZZ3CZWE123456.jpg`
            
            ### Metrics Calculated:
            - **Exact Match Accuracy** - % of perfectly matched VINs
            - **Character Accuracy** - % of correctly predicted characters
            - **Average Confidence** - Model's confidence score
            - **Processing Time** - Time per image
            """)


def run_batch_evaluation(image_folder: str, labels_file: str, max_images: int, models: List[str], save_results: bool = True):
    """
    Run batch evaluation on a folder of images with real-time metrics display.
    
    Args:
        image_folder: Path to folder containing VIN images
        labels_file: Optional path to labels file
        max_images: Maximum number of images to process
        models: List of model names to evaluate
        save_results: Whether to save results to the results directory
    """
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Real-time metrics display
    metrics_container = st.container()
    
    # Find images
    folder = Path(image_folder)
    image_files = list(folder.glob("*.jpg")) + list(folder.glob("*.png")) + list(folder.glob("*.jpeg"))
    image_files = image_files[:max_images]
    
    if not image_files:
        st.error("No images found in folder")
        return
    
    st.info(f"Found {len(image_files)} images")
    
    results_by_model = {model: [] for model in models}
    
    # Real-time metrics tracking
    running_metrics = {model: {'correct': 0, 'total': 0, 'char_correct': 0, 'total_chars': 0, 'total_time': 0.0} for model in models}
    
    for i, img_path in enumerate(image_files):
        progress_bar.progress((i + 1) / len(image_files))
        status_text.text(f"Processing {img_path.name} ({i+1}/{len(image_files)})...")
        
        for model in models:
            result = recognize_with_model(str(img_path), model)
            result['filename'] = img_path.name
            
            # Try to extract ground truth from filename (using canonical function)
            gt_vin = extract_vin_from_filename(img_path.name)
            result['ground_truth'] = gt_vin
            result['exact_match'] = result['vin'] == gt_vin if gt_vin else None
            
            results_by_model[model].append(result)
            
            # Update running metrics
            if gt_vin:
                running_metrics[model]['total'] += 1
                running_metrics[model]['total_chars'] += 17
                if result['exact_match']:
                    running_metrics[model]['correct'] += 1
                # Character accuracy
                if result['vin'] and len(result['vin']) >= 17:
                    running_metrics[model]['char_correct'] += sum(
                        1 for a, b in zip(result['vin'][:17], gt_vin[:17]) if a == b
                    )
            running_metrics[model]['total_time'] += result.get('processing_time', 0)
        
        # Update real-time metrics display every 5 images
        if (i + 1) % 5 == 0 or (i + 1) == len(image_files):
            with metrics_container:
                st.markdown("##### 📊 Real-time Metrics")
                cols = st.columns(len(models))
                for j, model in enumerate(models):
                    m = running_metrics[model]
                    with cols[j]:
                        model_short = model[:20] + "..." if len(model) > 20 else model
                        exact_acc = m['correct'] / m['total'] * 100 if m['total'] > 0 else 0
                        char_acc = m['char_correct'] / m['total_chars'] * 100 if m['total_chars'] > 0 else 0
                        avg_time = m['total_time'] / (i + 1)
                        st.metric(model_short, f"{exact_acc:.1f}%", f"Char: {char_acc:.1f}%")
    
    status_text.text("✅ Evaluation complete!")
    
    # Display results
    display_evaluation_results(results_by_model, save_results=save_results)


def extract_vin_from_filename(filename: str) -> Optional[str]:
    """
    Extract VIN from filename pattern.
    
    Uses the canonical implementation from vin_utils.py (Single Source of Truth)
    with a fallback for when the import is unavailable.
    
    Supported filename patterns:
    - "1-VIN -SAL1A2A40SA606662.jpg"
    - "7-VIN_-_SAL109F97TA467227.jpg"
    - "VIN_train_1234-WVWZZZ3CZWE123456.png"
    - "WVWZZZ3CZWE123456.jpg"
    - And many more variants
    
    Args:
        filename: Image filename or path
        
    Returns:
        Extracted VIN (17 characters, uppercase) or None if not found
    """
    # Use canonical implementation if available (preferred)
    if VIN_UTILS_AVAILABLE and _extract_vin_from_filename is not None:
        return _extract_vin_from_filename(filename)
    
    # Fallback: Simple extraction (less comprehensive)
    # This fallback is only used if vin_utils import fails
    import re
    name = Path(filename).stem.upper()
    
    # Try to find any valid 17-character VIN (excluding I, O, Q)
    fallback_pattern = re.compile(r'\b([A-HJ-NPR-Z0-9]{17})\b')
    match = fallback_pattern.search(name)
    if match:
        return match.group(1)
    
    # Last resort: split and look for 17-char segments
    parts = name.replace('-', '_').split('_')
    for part in parts:
        cleaned = ''.join(c for c in part if c.isalnum())
        if len(cleaned) == 17 and not any(c in cleaned for c in 'IOQ'):
            return cleaned
    
    return None


def display_evaluation_results(results_by_model: Dict[str, List[Dict]], save_results: bool = False):
    """
    Display evaluation results with charts and optionally save to results directory.
    
    Args:
        results_by_model: Dictionary mapping model names to list of results
        save_results: Whether to save results to the results directory for dashboard
    """
    st.subheader("📈 Evaluation Results")
    
    # Summary metrics
    summary_data = []
    for model, results in results_by_model.items():
        valid_results = [r for r in results if r.get('ground_truth')]
        if valid_results:
            exact_matches = sum(1 for r in valid_results if r.get('exact_match'))
            char_correct = sum(
                sum(1 for a, b in zip(r['vin'][:17], r['ground_truth'][:17]) if a == b)
                for r in valid_results
            )
            total_chars = len(valid_results) * 17
            
            exact_match_pct = exact_matches / len(valid_results) if valid_results else 0
            char_acc_pct = char_correct / total_chars if total_chars else 0
            avg_conf = sum(r['confidence'] for r in results) / len(results)
            avg_time = sum(r['processing_time'] for r in results) / len(results)
            
            # Calculate F1 scores (per-character basis)
            # True positives = characters that match
            # For simplicity, use character accuracy as proxy for F1
            f1_micro = char_acc_pct  # Simplified approximation
            f1_macro = char_acc_pct  # Simplified approximation
            
            summary_data.append({
                'Model': model,
                'Total Images': len(results),
                'With Ground Truth': len(valid_results),
                'Exact Matches': exact_matches,
                'Exact Match Acc': exact_match_pct,
                'Char Acc': char_acc_pct,
                'F1 Micro': f1_micro,
                'F1 Macro': f1_macro,
                'Avg Confidence': avg_conf,
                'Avg Time': avg_time
            })
        else:
            summary_data.append({
                'Model': model,
                'Total Images': len(results),
                'With Ground Truth': 0,
                'Exact Matches': 0,
                'Exact Match Acc': 0,
                'Char Acc': 0,
                'F1 Micro': 0,
                'F1 Macro': 0,
                'Avg Confidence': sum(r['confidence'] for r in results) / len(results) if results else 0,
                'Avg Time': sum(r['processing_time'] for r in results) / len(results) if results else 0
            })
    
    # Display summary table with formatted values
    df_summary = pd.DataFrame(summary_data)
    
    # Create display version with formatted percentages
    df_display = df_summary.copy()
    df_display['Exact Match Acc'] = df_display['Exact Match Acc'].apply(lambda x: f"{x:.2%}" if isinstance(x, (int, float)) else x)
    df_display['Char Acc'] = df_display['Char Acc'].apply(lambda x: f"{x:.2%}" if isinstance(x, (int, float)) else x)
    df_display['F1 Micro'] = df_display['F1 Micro'].apply(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x)
    df_display['F1 Macro'] = df_display['F1 Macro'].apply(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x)
    df_display['Avg Confidence'] = df_display['Avg Confidence'].apply(lambda x: f"{x:.2%}" if isinstance(x, (int, float)) else x)
    df_display['Avg Time'] = df_display['Avg Time'].apply(lambda x: f"{x:.3f}s" if isinstance(x, (int, float)) else x)
    
    st.dataframe(df_display, use_container_width=True, hide_index=True)
    
    # Charts
    if PLOTLY_AVAILABLE and len(summary_data) > 1:
        col1, col2 = st.columns(2)
        
        with col1:
            # Use numeric values for plotting
            fig = px.bar(
                df_summary,
                x='Model',
                y='Char Acc',
                title='Character Accuracy by Model',
                color='Model',
                labels={'Char Acc': 'Character Accuracy'}
            )
            fig.update_layout(yaxis_tickformat='.1%')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(
                df_summary,
                x='Model',
                y='Avg Time',
                title='Processing Time by Model',
                color='Model',
                labels={'Avg Time': 'Avg Time (seconds)'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Save results to dashboard if requested
    if save_results and summary_data:
        save_evaluation_to_dashboard(df_summary, results_by_model)
    
    # Detailed results expander
    with st.expander("📋 Detailed Results"):
        for model, results in results_by_model.items():
            st.subheader(model)
            df = pd.DataFrame([
                {
                    'Image': r['filename'],
                    'Predicted': r['vin'],
                    'Ground Truth': r.get('ground_truth', 'N/A'),
                    'Match': '✓' if r.get('exact_match') else '✗' if r.get('ground_truth') else '-',
                    'Confidence': f"{r['confidence']:.1%}",
                    'Time': f"{r['processing_time']:.3f}s"
                }
                for r in results[:50]  # Limit display
            ])
            st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Export option
    if st.button("📥 Export Results as CSV"):
        all_results = []
        for model, results in results_by_model.items():
            for r in results:
                all_results.append({
                    'Model': model,
                    'Image': r['filename'],
                    'Predicted VIN': r['vin'],
                    'Ground Truth': r.get('ground_truth', ''),
                    'Exact Match': r.get('exact_match', ''),
                    'Confidence': r['confidence'],
                    'Processing Time': r['processing_time']
                })
        
        df_export = pd.DataFrame(all_results)
        csv = df_export.to_csv(index=False)
        st.download_button(
            "Download CSV",
            csv,
            "evaluation_results.csv",
            "text/csv"
        )


def save_evaluation_to_dashboard(df_summary: pd.DataFrame, results_by_model: Dict[str, List[Dict]]):
    """
    Save evaluation results to the results directory for the dashboard.
    
    This creates/updates the files that the Results Dashboard reads:
    - model_comparison.csv - Summary metrics per model
    - sample_results.csv - Per-image results
    - multi_model_evaluation.json - Full evaluation details
    """
    from datetime import datetime
    
    results_dir = ProjectPaths.RESULTS
    results_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().isoformat()
    
    try:
        # Save model comparison CSV (for dashboard charts)
        comparison_file = results_dir / "model_comparison.csv"
        df_summary.to_csv(comparison_file, index=False)
        
        # Save sample results CSV
        all_results = []
        for model, results in results_by_model.items():
            for r in results:
                all_results.append({
                    'Model': model,
                    'Image': r.get('filename', ''),
                    'Predicted VIN': r.get('vin', ''),
                    'Ground Truth': r.get('ground_truth', ''),
                    'Exact Match': r.get('exact_match', False),
                    'Confidence': r.get('confidence', 0),
                    'Processing Time': r.get('processing_time', 0)
                })
        
        if all_results:
            sample_file = results_dir / "sample_results.csv"
            pd.DataFrame(all_results).to_csv(sample_file, index=False)
        
        # Save full evaluation JSON
        eval_data = {
            'timestamp': timestamp,
            'summary': df_summary.to_dict(orient='records'),
            'total_images': sum(len(r) for r in results_by_model.values()),
            'models_evaluated': list(results_by_model.keys()),
        }
        
        eval_file = results_dir / "multi_model_evaluation.json"
        with open(eval_file, 'w') as f:
            json.dump(eval_data, f, indent=2, default=str)
        
        st.success(f"✅ Results saved to `{results_dir}`")
        
    except Exception as e:
        st.warning(f"⚠️ Could not save results: {e}")


def process_uploaded_batch(uploaded_files, model_name: str, save_results: bool = True, show_details: bool = True):
    """
    Process batch of uploaded files with the specified model.
    
    Args:
        uploaded_files: List of uploaded Streamlit file objects
        model_name: Name of the model to use for inference
        save_results: Whether to save results to the results directory
        show_details: Whether to show detailed per-image results table
    """
    results = []
    
    st.info(f"🤖 Processing with model: **{model_name}**")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Real-time metrics display
    metrics_placeholder = st.empty()
    
    # Real-time metrics
    correct_count = 0
    total_with_gt = 0
    char_correct = 0
    total_chars = 0
    total_time = 0.0
    
    for i, uploaded_file in enumerate(uploaded_files):
        progress_bar.progress((i + 1) / len(uploaded_files))
        status_text.text(f"Processing {uploaded_file.name} ({i+1}/{len(uploaded_files)})...")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
        
        result = recognize_with_model(tmp_path, model_name)
        result['filename'] = uploaded_file.name
        
        # Extract ground truth from filename
        gt_vin = extract_vin_from_filename(uploaded_file.name)
        result['ground_truth'] = gt_vin
        result['exact_match'] = result['vin'] == gt_vin if gt_vin else None
        
        # Update metrics
        if gt_vin:
            total_with_gt += 1
            total_chars += 17
            if result['exact_match']:
                correct_count += 1
            if result['vin'] and len(result['vin']) >= 17:
                char_correct += sum(1 for a, b in zip(result['vin'][:17], gt_vin[:17]) if a == b)
        
        total_time += result.get('processing_time', 0)
        results.append(result)
        
        os.unlink(tmp_path)
        
        # Update real-time metrics every 3 images
        if (i + 1) % 3 == 0 or (i + 1) == len(uploaded_files):
            with metrics_placeholder.container():
                if total_with_gt > 0:
                    exact_acc = correct_count / total_with_gt * 100
                    char_acc = char_correct / total_chars * 100 if total_chars > 0 else 0
                    cols = st.columns(4)
                    cols[0].metric("Exact Match", f"{exact_acc:.1f}%")
                    cols[1].metric("Char Accuracy", f"{char_acc:.1f}%")
                    cols[2].metric("Processed", f"{i+1}/{len(uploaded_files)}")
                    cols[3].metric("Avg Time", f"{total_time/(i+1):.2f}s")
    
    progress_bar.progress(1.0)
    status_text.text("✅ Processing complete!")
    
    # Quick Summary
    st.divider()
    st.subheader("✅ Evaluation Complete")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Images Processed", len(results))
    with col2:
        if total_with_gt > 0:
            exact_acc = correct_count / total_with_gt * 100
            st.metric("Quick Accuracy", f"{exact_acc:.1f}%")
        else:
            st.metric("Quick Accuracy", "N/A")
    with col3:
        st.metric("Total Time", f"{total_time:.1f}s")
    
    # Save results to dashboard
    if save_results:
        try:
            results_dir = ProjectPaths.RESULTS
            results_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            exact_acc = correct_count / total_with_gt * 100 if total_with_gt > 0 else None
            char_acc = char_correct / total_chars * 100 if total_chars > 0 else None
            
            # Save detailed results
            eval_data = {
                'model': model_name,
                'timestamp': timestamp,
                'total_images': len(results),
                'images_with_gt': total_with_gt,
                'exact_matches': correct_count,
                'exact_accuracy': exact_acc,
                'char_accuracy': char_acc,
                'avg_confidence': sum(r['confidence'] for r in results) / len(results) if results else 0,
                'total_time': total_time,
                'results': [
                    {
                        'filename': r['filename'],
                        'predicted': r['vin'],
                        'ground_truth': r.get('ground_truth'),
                        'exact_match': r.get('exact_match'),
                        'confidence': r['confidence'],
                        'time': r['processing_time']
                    }
                    for r in results
                ]
            }
            
            eval_file = results_dir / f"batch_evaluation_{timestamp}.json"
            with open(eval_file, 'w') as f:
                json.dump(eval_data, f, indent=2, default=str)
            
            st.success(f"✅ Results saved! Go to **📈 Results Dashboard** to view full metrics including F1 scores, precision, recall, and more.")
            
            # Direct link hint
            st.info("💡 **Tip:** Navigate to the **Results Dashboard** page from the sidebar to see comprehensive industry metrics and visualizations.")
            
        except Exception as e:
            st.warning(f"⚠️ Could not save results: {e}")
    
    # Optional: Show quick preview of results (condensed)
    if show_details:
        with st.expander("👁️ Quick Preview (First 10 Results)", expanded=False):
            df = pd.DataFrame([
                {
                    'Image': r['filename'][:30] + '...' if len(r['filename']) > 30 else r['filename'],
                    'Predicted': r['vin'],
                    'Match': '✅' if r.get('exact_match') else ('❌' if r.get('exact_match') is False else '—'),
                }
                for r in results[:10]
            ])
            st.dataframe(df, use_container_width=True)
            if len(results) > 10:
                st.caption(f"... and {len(results) - 10} more. See full results in **Results Dashboard**.")


def render_training_page():
    """Render the training/fine-tuning page."""
    st.markdown('<h1 class="main-header">🎯 Model Training</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Train or fine-tune models on your VIN dataset</p>', unsafe_allow_html=True)
    
    # Hardware Detection Info (collapsible)
    with st.expander("🖥️ **Hardware Detection**", expanded=False):
        try:
            from .training_components import TrainingUI
            hw_info = TrainingUI.get_hardware_info()
            
            if "error" in hw_info:
                st.warning(f"Hardware detection unavailable: {hw_info['error']}")
            else:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**System**")
                    st.write(f"• Platform: {hw_info.get('platform', 'Unknown')}")
                    st.write(f"• CPU Cores: {hw_info.get('cpu_cores', 'N/A')}")
                    st.write(f"• Python: {hw_info.get('python_version', 'N/A')}")
                
                with col2:
                    st.markdown("**GPU/Accelerator**")
                    gpu_info = hw_info.get('gpu', {})
                    if gpu_info.get('available', False):
                        gpus = gpu_info.get('devices', [])
                        for gpu in gpus[:2]:  # Show first 2 GPUs
                            st.write(f"• {gpu.get('name', 'Unknown')}")
                            st.write(f"  Memory: {gpu.get('memory_gb', 'N/A')} GB")
                    else:
                        st.write("• No GPU detected")
                        st.write("• CPU training only")
                
                with col3:
                    st.markdown("**Features**")
                    libs = hw_info.get('libraries', {})
                    cuda = hw_info.get('cuda', {})
                    mps = hw_info.get('mps', {})
                    
                    if cuda.get('available', False):
                        st.write("✅ CUDA available")
                        st.write("✅ 4-bit/8-bit quantization")
                    elif mps.get('available', False):
                        st.write("✅ MPS available (Apple)")
                        st.write("❌ Quantization (CUDA only)")
                    else:
                        st.write("❌ No GPU acceleration")
                    
                    if libs.get('peft', False):
                        st.write("✅ LoRA fine-tuning")
                    else:
                        st.write("❌ LoRA unavailable")
        except Exception as e:
            st.caption(f"Hardware detection unavailable: {e}")
    
    # Training mode selection
    st.info("""
    **Training Modes:**
    - **Fine-Tuning**: Adapt a pre-trained model to VIN data (500-5,000 images, hours)
    - **Train from Scratch**: Build a new model from random weights (50,000+ images, days)
    - **Hyperparameter Tuning**: Use Optuna to find optimal training parameters
    """)
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔧 PaddleOCR Fine-Tuning", 
        "🏗️ PaddleOCR from Scratch",
        "🤖 DeepSeek Fine-Tuning",
        "🆕 DeepSeek from Scratch",
        "⚙️ Hyperparameter Tuning"
    ])
    
    with tab1:
        render_paddleocr_finetuning()
    
    with tab2:
        render_paddleocr_scratch()
    
    with tab3:
        render_deepseek_finetuning()
    
    with tab4:
        render_deepseek_scratch()
    
    with tab5:
        render_hyperparameter_tuning()
    
    # Render training progress once at the end (outside tabs)
    render_training_progress()


def render_device_selector(key_prefix: str = "device") -> str:
    """Render device selection widget and return selected device."""
    col1, col2 = st.columns([3, 1])
    
    with col1:
        device_options = ["CPU"]
        
        # Check for GPU availability
        try:
            import paddle
            if paddle.device.is_compiled_with_cuda():
                device_options.append("GPU (CUDA - Paddle)")
        except ImportError:
            pass
        
        try:
            import torch
            if torch.cuda.is_available():
                if "GPU (CUDA" not in str(device_options):
                    device_options.append("GPU (CUDA)")
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device_options.append("GPU (MPS - Apple Silicon)")
        except ImportError:
            pass
        
        selection = st.selectbox(
            "🖥️ Compute Device",
            options=device_options,
            key=f"{key_prefix}_select",
            help="Select CPU or GPU for training. GPU significantly speeds up training.",
        )
    
    with col2:
        # Show device info
        if "CUDA" in selection:
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_name = torch.cuda.get_device_name(0)
                    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    st.caption(f"🎮 {gpu_name[:20]}...")
                    st.caption(f"💾 {gpu_mem:.1f} GB VRAM")
            except:
                st.caption("🎮 GPU Detected")
        elif "MPS" in selection:
            st.caption("🍎 Apple Silicon")
        else:
            st.caption("💻 CPU Mode")
    
    # Convert to device string
    if "CUDA" in selection:
        return "cuda"
    elif "MPS" in selection:
        return "mps"
    else:
        return "cpu"


def check_hf_model_cached(model_id: str, cache_dir: Optional[str] = None) -> bool:
    """Check if a HuggingFace model is already cached locally."""
    model_path = model_id.replace("/", "--")
    cache_root = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "huggingface" / "hub"
    expected_dir = cache_root / f"models--{model_path}"
    return expected_dir.exists()


def generate_labels_from_split_folders(
    train_dir: str,
    val_dir: str,
    test_dir: str,
    output_dir: str
) -> Dict[str, str]:
    """Generate PaddleOCR label files from separate train/val/test image folders."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    train_images = load_images_with_labels(train_dir)
    val_images = load_images_with_labels(val_dir)
    test_images = load_images_with_labels(test_dir)

    train_labels = create_paddleocr_labels(train_images, output_path, "train", copy_images=False)
    val_labels = create_paddleocr_labels(val_images, output_path, "val", copy_images=False)
    test_labels = create_paddleocr_labels(test_images, output_path, "test", copy_images=False)

    return {
        "train_labels": train_labels,
        "val_labels": val_labels,
        "test_labels": test_labels,
        "data_dir": str(output_path),
        "train_count": len(train_images),
        "val_count": len(val_images),
        "test_count": len(test_images),
    }


def render_model_console_output(model_type: str, key_suffix: str = ""):
    """
    Render real-time console output and synchronized progress for a specific model type.
    
    Only displays output if the specified model is currently training.
    
    Args:
        model_type: The model type to show console for (e.g., "paddleocr_finetune", "deepseek_finetune")
        key_suffix: Unique suffix for Streamlit widget keys
    """
    if not TRAINING_UI_AVAILABLE:
        return
    
    runner = get_global_runner()
    tracker = get_global_tracker()
    state = tracker.get_state()
    
    # Only show if training is active AND it's this model
    current_model = runner.get_current_model_type() if hasattr(runner, 'get_current_model_type') else None
    
    if not state.is_running or current_model != model_type:
        # Not training this model - show nothing or a message
        if state.is_running and current_model and current_model != model_type:
            st.info(f"ℹ️ Another model ({current_model.replace('_', ' ').title()}) is currently training. Console output is shown on that model's tab.")
        return
    
    st.markdown("---")
    st.markdown("### 🖥️ Real-Time Training Console & Progress")
    
    # Status indicator
    status = "🟢 **Training in Progress**" if not state.is_paused else "🟡 **Paused**"
    model_display = model_type.replace("_", " ").title()
    status += f" | **Model:** {model_display}"
    if state.history:
        last_update = state.history[-1].timestamp
        status += f" _(last update: {last_update[-12:]})_"
    st.markdown(status)
    
    # Get console log path for this model
    console_log_path = runner.get_current_console_log() if hasattr(runner, 'get_current_console_log') else None
    
    # Try to get the output directory for this model to read progress file
    output_dir = None
    if console_log_path:
        output_dir = Path(console_log_path).parent
    
    # Read progress file for detailed metrics if available
    progress_data = None
    dataset_info = None
    detailed_metrics = None
    
    if output_dir:
        progress_file = output_dir / "training_progress.json"
        if progress_file.exists():
            try:
                import json
                with open(progress_file) as f:
                    progress_data = json.load(f)
                    dataset_info = progress_data.get('dataset_info', {})
                    detailed_metrics = progress_data.get('detailed_metrics', {})
                    
                    # Sync state from progress file if available (more accurate)
                    if progress_data.get('current_epoch'):
                        state.current_epoch = progress_data.get('current_epoch', state.current_epoch)
                    if progress_data.get('total_epochs'):
                        state.total_epochs = progress_data.get('total_epochs', state.total_epochs)
                    if progress_data.get('current_batch'):
                        state.current_batch = progress_data.get('current_batch', state.current_batch)
                    if progress_data.get('total_batches'):
                        state.total_batches = progress_data.get('total_batches', state.total_batches)
                    if progress_data.get('train_loss') is not None:
                        state.current_loss = progress_data.get('train_loss', state.current_loss)
                    if progress_data.get('best_accuracy') is not None:
                        state.best_accuracy = progress_data.get('best_accuracy', state.best_accuracy)
            except Exception as e:
                pass  # Silently ignore progress file read errors
    
    # Show dataset info if available
    if dataset_info and (dataset_info.get('train_samples') or dataset_info.get('val_samples')):
        col_ds1, col_ds2, col_ds3 = st.columns(3)
        with col_ds1:
            st.caption(f"🏋️ Training: {dataset_info.get('train_samples', 0)} samples")
        with col_ds2:
            st.caption(f"🧪 Validation: {dataset_info.get('val_samples', 0)} samples")
        with col_ds3:
            st.caption(f"📦 Batch Size: {dataset_info.get('batch_size', '-')}")
    
    # Progress bar - synchronized with progress file
    if state.total_epochs > 0:
        total_steps = state.total_epochs * max(state.total_batches, 1)
        safe_epoch = max(state.current_epoch, 0)
        safe_batch = max(state.current_batch, 0)
        current_step = max(0, (safe_epoch - 1)) * max(state.total_batches, 1) + safe_batch
        progress = max(0.0, min(current_step / total_steps, 1.0)) if total_steps > 0 else 0.0
        st.progress(progress, text=f"Epoch {state.current_epoch}/{state.total_epochs}, Batch {state.current_batch}/{state.total_batches}")
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📉 Train Loss", f"{state.current_loss:.4f}")
    with col2:
        st.metric("🎯 Best Accuracy", f"{state.best_accuracy:.2%}")
    with col3:
        st.metric("⏱️ Elapsed", tracker.format_elapsed_time())
    with col4:
        st.metric("⏳ ETA", tracker.format_remaining_time())
    
    # Detailed Metrics Section (from progress file)
    if detailed_metrics and any(detailed_metrics.values()):
        with st.expander("📈 Detailed Validation Metrics", expanded=True):
            col_m1, col_m2, col_m3 = st.columns(3)
            
            # Image-Level Metrics
            with col_m1:
                img_metrics = detailed_metrics.get('image_level', {})
                if img_metrics:
                    correct = img_metrics.get('correct', 0)
                    total = img_metrics.get('total', 0)
                    acc = img_metrics.get('accuracy', 0)
                    st.markdown("**📊 Image-Level**")
                    st.metric("Exact Match", f"{correct}/{total}", f"{acc:.1f}%")
            
            # Character-Level Metrics
            with col_m2:
                char_metrics = detailed_metrics.get('char_level', {})
                if char_metrics:
                    char_acc = char_metrics.get('accuracy', 0)
                    f1_micro = char_metrics.get('f1_micro', 0)
                    f1_macro = char_metrics.get('f1_macro', 0)
                    st.markdown("**📝 Character-Level**")
                    st.metric("Char Accuracy", f"{char_acc:.1f}%")
                    st.caption(f"F1-micro: {f1_micro:.4f} | F1-macro: {f1_macro:.4f}")
            
            # Industry Metrics
            with col_m3:
                ind_metrics = detailed_metrics.get('industry', {})
                if ind_metrics:
                    cer = ind_metrics.get('cer', 100)
                    ned = ind_metrics.get('ned', 1)
                    st.markdown("**🏭 Industry Metrics**")
                    st.metric("CER", f"{cer:.1f}%", delta=f"-{100-cer:.1f}%" if cer < 100 else None, delta_color="normal")
                    st.caption(f"NED: {ned:.4f}")
    
    # Loss History Chart
    if progress_data and progress_data.get('train_losses'):
        losses = progress_data.get('train_losses', [])
        if losses and len(losses) > 0:
            with st.expander("📉 Loss History", expanded=False):
                import pandas as pd
                df = pd.DataFrame({
                    'Epoch': list(range(1, len(losses) + 1)),
                    'Train Loss': losses
                })
                st.line_chart(df.set_index('Epoch'))
    
    # Console output display
    st.markdown("#### 🖥️ Console Output")
    
    if console_log_path and Path(console_log_path).exists():
        try:
            with open(console_log_path, 'r') as f:
                lines = f.readlines()
                # Show last 30 lines for real-time view
                recent_lines = lines[-30:] if len(lines) > 30 else lines
                console_text = "".join(recent_lines)
                
                # Display in a prominent scrollable code block
                st.code(console_text, language="text", line_numbers=False)
                
                # Show total line count
                st.caption(f"📜 Showing last {len(recent_lines)} of {len(lines)} lines | Log: `{Path(console_log_path).name}`")
        except Exception as e:
            st.warning(f"⚠️ Waiting for console output... ({e})")
    else:
        st.info("⏳ Waiting for training output...")
    
    # Controls row
    col_ctrl1, col_ctrl2, col_ctrl3, col_ctrl4 = st.columns([2, 1, 1, 1])
    
    with col_ctrl1:
        refresh_rate = st.selectbox(
            "🔄 Auto-refresh rate (seconds)", 
            options=[1, 2, 3, 5, 10],
            index=1,  # Default 2 seconds
            key=f"console_refresh_rate_{key_suffix}",
            help="How often to refresh the console output and progress"
        )
    with col_ctrl2:
        auto_refresh = st.checkbox("Auto-refresh", value=True, key=f"auto_refresh_{key_suffix}")
    with col_ctrl3:
        if st.button("🔃 Refresh Now", key=f"manual_refresh_{key_suffix}"):
            st.rerun()
    with col_ctrl4:
        if st.button("⏹️ Stop Training", type="secondary", key=f"stop_training_{key_suffix}"):
            runner.stop()
            st.warning("⏹️ Stop requested...")
            st.rerun()
    
    # Auto-refresh logic
    if auto_refresh:
        import time as time_module
        refresh_key = f'last_refresh_time_{key_suffix}'
        if refresh_key not in st.session_state:
            st.session_state[refresh_key] = time_module.time()
        
        current_time = time_module.time()
        if current_time - st.session_state[refresh_key] >= float(refresh_rate):
            st.session_state[refresh_key] = current_time
            st.rerun()
        
        # Show countdown
        time_until_refresh = float(refresh_rate) - (current_time - st.session_state[refresh_key])
        st.caption(f"⏱️ Next auto-refresh in {time_until_refresh:.1f}s")


def render_training_progress():
    """
    Render a summary of training progress at the bottom of the training page.
    
    This provides a quick overview and directs users to the model-specific tab
    for full console output and detailed metrics.
    """
    if not TRAINING_UI_AVAILABLE:
        return
    
    runner = get_global_runner()
    tracker = get_global_tracker()
    state = tracker.get_state()
    
    # Don't show if nothing is happening
    if not state.is_running and not state.history:
        return
    
    st.markdown("---")
    st.markdown("### 📊 Training Status Summary")
    
    # Get current model info
    current_model = runner.get_current_model_type() if hasattr(runner, 'get_current_model_type') else None
    console_log_path = runner.get_current_console_log() if hasattr(runner, 'get_current_console_log') else None
    
    # Try to read progress from the current training's output directory
    output_dir = None
    progress_data = None
    if console_log_path:
        output_dir = Path(console_log_path).parent
        progress_file = output_dir / "training_progress.json"
        if progress_file.exists():
            try:
                import json
                with open(progress_file) as f:
                    progress_data = json.load(f)
                    # Sync state from progress file
                    if progress_data.get('current_epoch'):
                        state.current_epoch = progress_data.get('current_epoch', state.current_epoch)
                    if progress_data.get('total_epochs'):
                        state.total_epochs = progress_data.get('total_epochs', state.total_epochs)
                    if progress_data.get('current_batch'):
                        state.current_batch = progress_data.get('current_batch', state.current_batch)
                    if progress_data.get('total_batches'):
                        state.total_batches = progress_data.get('total_batches', state.total_batches)
                    if progress_data.get('train_loss') is not None:
                        state.current_loss = progress_data.get('train_loss', state.current_loss)
                    if progress_data.get('best_accuracy') is not None:
                        state.best_accuracy = progress_data.get('best_accuracy', state.best_accuracy)
            except:
                pass
    
    # Status indicator
    if state.is_running:
        status = "🟢 **Training in Progress**" if not state.is_paused else "� **Paused**"
        if state.history:
            last_update = state.history[-1].timestamp
            status += f" _(last update: {last_update[-12:]})_"
        
        if current_model:
            model_display = current_model.replace("_", " ").title()
            status += f" | **Model:** {model_display}"
            st.markdown(status)
            
            # Guide user to the correct tab
            st.info(f"💡 **Tip:** Full console output and detailed progress are shown on the **{model_display}** tab above.")
        else:
            st.markdown(status)
    elif state.error:
        st.markdown(f"🔴 **Error:** {state.error}")
    else:
        st.markdown("✅ **Training Complete**")
    
    # Progress bar - synchronized
    if state.total_epochs > 0:
        total_steps = state.total_epochs * max(state.total_batches, 1)
        safe_epoch = max(state.current_epoch, 0)
        safe_batch = max(state.current_batch, 0)
        current_step = max(0, (safe_epoch - 1)) * max(state.total_batches, 1) + safe_batch
        progress = max(0.0, min(current_step / total_steps, 1.0)) if total_steps > 0 else 0.0
        st.progress(progress, text=f"Epoch {state.current_epoch}/{state.total_epochs}, Batch {state.current_batch}/{state.total_batches}")
    
    # Compact metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("� Loss", f"{state.current_loss:.4f}")
    with col2:
        st.metric("🎯 Best Acc", f"{state.best_accuracy:.2%}")
    with col3:
        st.metric("⏱️ Elapsed", tracker.format_elapsed_time())
    with col4:
        st.metric("⏳ ETA", tracker.format_remaining_time())
    
    # Stop button only when running
    if state.is_running:
        if st.button("⏹️ Stop Training", type="secondary", key="stop_training_global"):
            runner.stop()
            st.warning("⏹️ Stop requested...")
            st.rerun()
    
    # Show last few log entries in expander (for debugging)
    if state.history:
        with st.expander("📜 Recent Training Updates", expanded=False):
            for update in reversed(state.history[-10:]):
                timestamp = update.timestamp[-8:] if update.timestamp else "??:??:??"
                msg = f"[{timestamp}] E{update.epoch}/B{update.batch}"
                if update.loss > 0:
                    msg += f" Loss:{update.loss:.4f}"
                if update.message:
                    message = update.message[:80] + "..." if len(update.message) > 80 else update.message
                    msg += f"\n    → {message}"
                st.text(msg)


def render_paddleocr_finetuning():
    """Render PaddleOCR fine-tuning interface."""
    st.subheader("🔧 PaddleOCR Fine-Tuning")
    st.caption("Adapt a pre-trained model to VIN data (recommended for most users)")

    if not PADDLE_AVAILABLE:
        st.error(
            "❌ **PaddlePaddle not installed.** Training requires PaddlePaddle. "
            "Install with: `pip install paddlepaddle` (CPU) or `pip install paddlepaddle-gpu` (GPU)"
        )
    elif not PADDLE_TRAINING_AVAILABLE:
        st.warning(
            "⚠️ PaddleOCR training module not fully loaded. "
            "Check console for import errors."
        )
    
    # Model Selection
    st.markdown("### 🤖 Base Model Selection")
    col_model1, col_model2 = st.columns(2)
    with col_model1:
        base_model = st.selectbox(
            "PaddleOCR Base Model",
            ["PP-OCRv4 (Latest)", "PP-OCRv3 (Stable)"],
            help="Choose which pre-trained PaddleOCR model to fine-tune",
            key="ft_base_model"
        )
    with col_model2:
        model_architecture = st.selectbox(
            "Recognition Architecture",
            ["PP-OCRv5 (State-of-the-art)", "SVTR_LCNet", "SVTR_Tiny", "CRNN"],
            help="PP-OCRv5: Latest 2024 model. SVTR_LCNet: Good accuracy. CRNN: Classic.",
            key="ft_architecture"
        )
    
    st.markdown("---")
    
    # Dataset Configuration Section
    st.markdown("### 📁 Dataset Configuration")

    dataset_mode = st.radio(
        "Dataset Input Mode",
        ["Use existing labels", "Provide split folders", "Split from single folder"],
        help="Choose how to provide your training/validation/test data",
        horizontal=True,
        key="ft_dataset_mode",
    )

    # Use centralized path defaults
    data_dir = st.session_state.get("ft_data_dir_path", _DEFAULT_PATHS['finetune_data'])
    train_labels = st.session_state.get("ft_train_labels_path", _DEFAULT_PATHS['train_labels'])
    val_labels = st.session_state.get("ft_val_labels_path", _DEFAULT_PATHS['val_labels'])
    test_labels = st.session_state.get("ft_test_labels_path", _DEFAULT_PATHS['test_labels'])

    if dataset_mode == "Use existing labels":
        col_data1, col_data2, col_data3 = st.columns(3)
        with col_data1:
            data_dir = st.text_input(
                "Data Directory (contains images)",
                data_dir,
                key="ft_data_dir",
                help="Directory containing training/validation/test images",
            )
            train_labels = st.text_input(
                "Training Labels File",
                train_labels,
                key="ft_train_labels",
                help="Label file format: image_path\tVIN_LABEL",
            )
        with col_data2:
            val_labels = st.text_input(
                "Validation Labels File",
                val_labels,
                key="ft_val_labels",
                help="Label file for validation during training",
            )
        with col_data3:
            test_labels = st.text_input(
                "Test Labels File",
                test_labels,
                key="ft_test_labels",
                help="Label file for final evaluation after training",
            )

    elif dataset_mode == "Provide split folders":
        if not DATASET_PREP_AVAILABLE:
            st.error("Dataset preparation utilities are unavailable. Please install required dependencies.")
        col_split1, col_split2, col_split3 = st.columns(3)
        with col_split1:
            train_dir = st.text_input(
                "Training Images Folder",
                _DEFAULT_PATHS['train_dir'],
                key="ft_train_dir",
                help="Folder containing training images",
            )
        with col_split2:
            val_dir = st.text_input(
                "Validation Images Folder",
                _DEFAULT_PATHS['val_dir'],
                key="ft_val_dir",
                help="Folder containing validation images",
            )
        with col_split3:
            test_dir = st.text_input(
                "Test Images Folder",
                _DEFAULT_PATHS['test_dir'],
                key="ft_test_dir",
                help="Folder containing test images",
            )

        output_dir_labels = st.text_input(
            "Label Output Directory",
            data_dir,
            key="ft_labels_output",
            help="Where to save generated label files",
        )

        if DATASET_PREP_AVAILABLE and st.button("🧾 Generate Labels from Folders", key="ft_gen_labels"):
            try:
                result = generate_labels_from_split_folders(train_dir, val_dir, test_dir, output_dir_labels)
                train_labels = result["train_labels"]
                val_labels = result["val_labels"]
                test_labels = result["test_labels"]
                data_dir = result["data_dir"]
                st.session_state.ft_train_labels_path = train_labels
                st.session_state.ft_val_labels_path = val_labels
                st.session_state.ft_test_labels_path = test_labels
                st.session_state.ft_data_dir_path = data_dir
                st.success("✅ Generated label files successfully")
                st.caption(
                    f"Train: {result['train_count']} | Val: {result['val_count']} | Test: {result['test_count']}"
                )
            except Exception as e:
                st.error(f"❌ Failed to generate labels: {e}")

    else:
        if not DATASET_PREP_AVAILABLE:
            st.error("Dataset preparation utilities are unavailable. Please install required dependencies.")
        col_split_a, col_split_b = st.columns([2, 1])
        with col_split_a:
            single_folder = st.text_input(
                "Images Folder (will be split)",
                str(ProjectPaths.ROOT / "raw_images"),
                key="ft_single_folder",
                help="Folder containing all images. VIN will be extracted from filenames.",
            )
            output_dir_labels = st.text_input(
                "Output Directory for Split Dataset",
                data_dir,
                key="ft_split_output",
                help="Where to store split images and label files",
            )
        with col_split_b:
            train_ratio = st.slider("Train %", 0.5, 0.9, 0.7, step=0.05, key="ft_train_ratio")
            val_ratio = st.slider("Val %", 0.05, 0.3, 0.15, step=0.05, key="ft_val_ratio")
            test_ratio = st.slider("Test %", 0.05, 0.3, 0.15, step=0.05, key="ft_test_ratio")

        total_ratio = train_ratio + val_ratio + test_ratio
        if abs(total_ratio - 1.0) > 0.001:
            st.error("Train/Val/Test ratios must sum to 1.0")

        if DATASET_PREP_AVAILABLE and st.button("✂️ Split Images + Create Labels", key="ft_split_images"):
            try:
                stats = prepare_dataset(
                    single_folder,
                    output_dir_labels,
                    train_ratio=train_ratio,
                    val_ratio=val_ratio,
                    test_ratio=test_ratio,
                )
                train_labels = stats["train_label_file"]
                val_labels = stats["val_label_file"]
                test_labels = stats["test_label_file"]
                data_dir = stats["output_dir"]
                st.session_state.ft_train_labels_path = train_labels
                st.session_state.ft_val_labels_path = val_labels
                st.session_state.ft_test_labels_path = test_labels
                st.session_state.ft_data_dir_path = data_dir
                st.success("✅ Dataset split and labels created")
                st.caption(
                    f"Train: {stats['train_images']} | Val: {stats['val_images']} | Test: {stats['test_images']}"
                )
            except Exception as e:
                st.error(f"❌ Dataset split failed: {e}")
    
    # Helper to resolve paths using centralized path resolver
    def resolve_label_path(path_str: str) -> Optional[Path]:
        """
        Resolve a path using ProjectPaths.resolve().
        
        This ensures all paths are properly resolved relative to project root.
        """
        if not path_str:
            return None
        resolved = ProjectPaths.resolve(path_str)
        if resolved.exists():
            return resolved
        return None
    
    # Dataset Statistics
    train_count = 0
    val_count = 0
    test_count = 0
    
    train_labels_path = resolve_label_path(train_labels)
    val_labels_path = resolve_label_path(val_labels)
    test_labels_path = resolve_label_path(test_labels)
    
    if train_labels_path and train_labels_path.exists():
        with open(train_labels_path) as f:
            train_count = sum(1 for line in f if line.strip())
    
    if val_labels_path and val_labels_path.exists():
        with open(val_labels_path) as f:
            val_count = sum(1 for line in f if line.strip())
            
    if test_labels_path and test_labels_path.exists():
        with open(test_labels_path) as f:
            test_count = sum(1 for line in f if line.strip())
    
    total_count = train_count + val_count + test_count
    
    # Display dataset stats
    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
    
    with col_stat1:
        st.metric("🏋️ Training", train_count, 
                  help=f"{train_count/total_count*100:.1f}%" if total_count > 0 else "N/A")
    with col_stat2:
        st.metric("🧪 Validation", val_count,
                  help=f"{val_count/total_count*100:.1f}%" if total_count > 0 else "N/A")
    with col_stat3:
        st.metric("� Test", test_count,
                  help=f"{test_count/total_count*100:.1f}%" if total_count > 0 else "N/A")
    with col_stat4:
        st.metric("📊 Total", total_count)
    
    # Show split percentages as progress bar
    if total_count > 0:
        train_pct = train_count / total_count
        val_pct = val_count / total_count
        test_pct = test_count / total_count
        st.caption(f"📈 Split: Train {train_pct*100:.0f}% | Val {val_pct*100:.0f}% | Test {test_pct*100:.0f}%")
    
    # Validation messages
    if train_count == 0:
        st.error("❌ No training data found. Please check the labels file path.")
    elif train_count < 100:
        st.warning(f"⚠️ Only {train_count} training samples. Recommend 500+ for good results.")
    elif test_count == 0:
        st.warning(f"✅ {train_count} train + {val_count} val images. ⚠️ No test set for final evaluation.")
    else:
        st.success(f"✅ Dataset ready: {train_count} train + {val_count} val + {test_count} test images")
    
    st.markdown("---")
    
    # Training Parameters Section
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### ⚙️ Training Parameters")
        epochs = st.slider("Number of Epochs", 1, 50, 10, key="ft_epochs",
                          help="More epochs = longer training but potentially better results")
        batch_size = st.selectbox("Batch Size", [4, 8, 16, 32], index=1, key="ft_batch",
                                  help="Lower = less memory, higher = faster (if GPU can handle it)")
        learning_rate = st.select_slider(
            "Learning Rate",
            options=[0.00001, 0.00005, 0.0001, 0.0005, 0.001],
            value=0.0005,
            key="ft_lr",
            help="Start with 0.0005 for fine-tuning"
        )
    
    with col2:
        st.markdown("### 🖥️ Hardware & Export")
        device = render_device_selector("ft")
        use_amp = st.checkbox("Mixed Precision (AMP)", value=True, key="ft_amp",
                             help="Faster training with lower memory on compatible GPUs")
        export_onnx = st.checkbox("Export to ONNX after training", value=False, key="ft_onnx",
                                  help="Export model to ONNX format for deployment")
        
        st.markdown("### 📂 Output")
        output_dir = st.text_input("Output Directory", _DEFAULT_PATHS['output_finetune'], key="ft_output")
    
    # Training controls
    st.markdown("---")
    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        start_disabled = not PADDLE_TRAINING_AVAILABLE
        if TRAINING_UI_AVAILABLE:
            runner = get_global_runner()
            start_disabled = start_disabled or runner.is_running
        
        if st.button("🚀 Start Fine-Tuning", type="primary", key="paddle_finetune_btn", disabled=start_disabled):
            if not PADDLE_TRAINING_AVAILABLE:
                st.error("❌ PaddlePaddle not available. Install with: pip install paddlepaddle-gpu")
            elif TRAINING_UI_AVAILABLE:
                config = {
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "device": device,
                    # Resolve all paths to absolute using ProjectPaths
                    "train_data_dir": str(ProjectPaths.resolve(data_dir)),
                    "train_labels": str(ProjectPaths.resolve(train_labels)),
                    "val_data_dir": str(ProjectPaths.resolve(data_dir)),
                    "val_labels": str(ProjectPaths.resolve(val_labels)),
                    "test_labels": str(ProjectPaths.resolve(test_labels)),
                    "output_dir": str(ProjectPaths.resolve(output_dir)),
                    "base_model": base_model,
                    "architecture": model_architecture,
                }
                try:
                    runner = get_global_runner()
                    runner.start_paddleocr_finetuning(config)
                    st.success(f"✅ Training started with {train_count} train + {val_count} val images!")
                    st.info(f"📝 After training, evaluate on {test_count} test images.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to start training: {e}")
            else:
                st.warning("Training UI module not available. Use the command below.")
    
    with col_b:
        if st.button("📊 View Training Logs", key="paddle_logs"):
            log_path = Path(output_dir) / "train.log"
            if log_path.exists():
                with open(log_path) as f:
                    st.code(f.read()[-5000:])
            else:
                st.info("No training logs found yet")
    
    with col_c:
        # Show command for terminal use
        with st.expander("📋 Terminal Command"):
            onnx_flag = " --export-onnx" if export_onnx else ""
            gpu_flag = "" if device != "cpu" else " --no-gpu"
            cmd = f"""python -m src.vin_ocr.training.finetune_paddleocr \\
    --config configs/vin_finetune_config.yml \\
    --epochs {epochs} \\
    --batch-size {batch_size} \\
    --lr {learning_rate}{gpu_flag}{onnx_flag}"""
            st.code(cmd, language="bash")
    
    # Test Set Evaluation Section (after training)
    st.markdown("---")
    st.markdown("### 📝 Test Set Evaluation")
    st.caption("Evaluate trained model on held-out test data (run after training completes)")
    
    col_eval1, col_eval2 = st.columns([2, 1])
    
    with col_eval1:
        # Default to latest model - most training runs create this
        default_model = f"{output_dir}/latest"
        model_path = st.text_input(
            "Trained Model Path",
            default_model,
            key="ft_eval_model",
            help="Path to the trained model (without extension). Examples: output/vin_rec_finetune/latest, output/vin_rec_finetune/best_accuracy, output/vin_rec_finetune/epoch_85"
        )
    
    with col_eval2:
        if st.button("🧪 Evaluate on Test Set", key="ft_eval_test_btn"):
            # Check if model file exists (either as directory or .pdparams file)
            model_dir = Path(model_path)
            model_params_file = Path(f"{model_path}.pdparams")
            
            # Resolve relative paths against project root
            if not model_dir.is_absolute():
                model_dir = _project_root / model_path
                model_params_file = _project_root / f"{model_path}.pdparams"
            
            # Model can be either a directory or a .pdparams file prefix
            model_exists = model_dir.exists() or model_params_file.exists()
            
            if not model_exists:
                st.error(f"❌ Model not found: {model_path}")
                st.info("💡 Available models: Check `output/vin_rec_finetune/` for `latest.pdparams`, `best_accuracy.pdparams`, or `epoch_XX.pdparams`")
            elif not test_labels_path.exists():
                st.error(f"❌ Test labels not found: {test_labels}")
            else:
                st.info(f"🔄 Evaluating on {test_count} test images...")
                try:
                    # Run evaluation on test set
                    from src.vin_ocr.training.finetune_paddleocr import PaddleOCRFineTuner
                    from paddleocr import PaddleOCR
                    
                    # Determine the model directory for PaddleOCR
                    # If user provided a file prefix (e.g., output/vin_rec_finetune/latest),
                    # we use the parent directory and specify rec_char_dict_path
                    if model_params_file.exists():
                        # User provided a file prefix - use parent dir
                        rec_model_dir = str(model_params_file.parent)
                        # PaddleOCR expects the directory containing model files
                        st.info(f"📂 Using model from: {rec_model_dir}")
                    else:
                        # User provided a directory
                        rec_model_dir = str(model_dir)
                    
                    # Initialize OCR with trained model
                    ocr = PaddleOCR(
                        rec_model_dir=rec_model_dir,
                        use_angle_cls=False,
                        lang='en',
                        show_log=False
                    )
                    
                    # Load test samples
                    test_samples = []
                    with open(test_labels_path) as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                parts = line.split(None, 1)  # Split on whitespace
                                if len(parts) == 2:
                                    img_file, label = parts
                                    test_samples.append((img_file, label))
                    
                    # Evaluate
                    correct = 0
                    results = []
                    
                    # Resolve data_path - check both cwd and project root
                    data_path = Path(data_dir)
                    if not data_path.exists():
                        data_path = _project_root / data_dir
                    
                    progress_bar = st.progress(0)
                    for i, (img_file, label) in enumerate(test_samples):
                        img_path = data_path / img_file
                        if img_path.exists():
                            result = ocr.ocr(str(img_path), cls=False)
                            predicted = ""
                            if result and result[0]:
                                # Get text from recognition
                                for line in result[0]:
                                    if line and len(line) > 1 and line[1]:
                                        predicted = line[1][0]
                                        break
                            
                            is_match = predicted.upper() == label.upper()
                            if is_match:
                                correct += 1
                            results.append({
                                "Image": img_file,
                                "Ground Truth": label,
                                "Predicted": predicted,
                                "Match": "✅" if is_match else "❌"
                            })
                        
                        progress_bar.progress((i + 1) / len(test_samples))
                    
                    accuracy = correct / len(test_samples) * 100 if test_samples else 0
                    
                    # Show results
                    st.markdown("---")
                    col_r1, col_r2, col_r3 = st.columns(3)
                    with col_r1:
                        st.metric("🎯 Test Accuracy", f"{accuracy:.1f}%")
                    with col_r2:
                        st.metric("✅ Correct", correct)
                    with col_r3:
                        st.metric("❌ Incorrect", len(test_samples) - correct)
                    
                    # Show detailed results
                    import pandas as pd
                    df = pd.DataFrame(results)
                    st.dataframe(df, use_container_width=True)
                    
                    # Download results
                    csv = df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Test Results",
                        csv,
                        "test_evaluation_results.csv",
                        "text/csv"
                    )
                    
                except Exception as e:
                    st.error(f"❌ Evaluation failed: {e}")
                    import traceback
                    st.code(traceback.format_exc())
    
    # Model-specific console output for PaddleOCR fine-tuning
    render_model_console_output("paddleocr_finetune", "paddle_ft")


def render_deepseek_finetuning():
    """Render DeepSeek fine-tuning interface."""
    st.subheader("🤖 DeepSeek-OCR Fine-Tuning")
    st.caption("Adapt the pre-trained DeepSeek-OCR model to VIN data")
    
    # Check dependencies first
    deps_ok = True
    try:
        import torch
        import transformers
        import peft
        deps_installed = True
    except ImportError:
        deps_installed = False
    
    if not deps_installed:
        st.error("❌ Required dependencies not installed: torch, transformers, peft")
        st.code("pip install torch transformers peft", language="bash")
        return

    # Check model cache status
    deepseek_model_id = "deepseek-ai/DeepSeek-OCR"
    model_cached = check_hf_model_cached(deepseek_model_id)
    if not model_cached:
        st.warning(
            "⚠️ DeepSeek-OCR model is not cached locally. "
            "The first fine-tuning run will download the model (this can take several minutes)."
        )
        st.caption("Model cache location: ~/.cache/huggingface/hub")
    
    # Check for MPS/CUDA
    device_info = "CPU only"
    if torch.cuda.is_available():
        device_info = f"CUDA: {torch.cuda.get_device_name(0)}"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device_info = "MPS (Apple Silicon)"
    
    st.info(f"🖥️ Available compute: **{device_info}**")
    
    # Dataset info section
    st.markdown("---")
    st.markdown("### 📁 Dataset Information")

    dataset_mode = st.radio(
        "Dataset Input Mode",
        ["Use existing labels", "Provide split folders", "Split from single folder"],
        help="Choose how to provide your training/validation/test data",
        horizontal=True,
        key="dsft_dataset_mode",
    )

    # Use centralized path defaults
    data_dir = st.session_state.get("dsft_data_dir_path", _DEFAULT_PATHS['finetune_data'])
    train_data = st.session_state.get("dsft_train_labels_path", _DEFAULT_PATHS['train_labels'])
    val_data = st.session_state.get("dsft_val_labels_path", _DEFAULT_PATHS['val_labels'])

    if dataset_mode == "Use existing labels":
        train_data = st.text_input("Training Labels File", train_data, key="dsft_train")
        val_data = st.text_input("Validation Labels File", val_data, key="dsft_val")
        data_dir = st.text_input("Data Directory (images)", data_dir, key="dsft_data_dir")

    elif dataset_mode == "Provide split folders":
        if not DATASET_PREP_AVAILABLE:
            st.error("Dataset preparation utilities are unavailable. Please install required dependencies.")
        col_ds1, col_ds2, col_ds3 = st.columns(3)
        with col_ds1:
            train_dir = st.text_input("Training Images Folder", _DEFAULT_PATHS['train_dir'], key="dsft_train_dir")
        with col_ds2:
            val_dir = st.text_input("Validation Images Folder", _DEFAULT_PATHS['val_dir'], key="dsft_val_dir")
        with col_ds3:
            test_dir = st.text_input("Test Images Folder", _DEFAULT_PATHS['test_dir'], key="dsft_test_dir")

        output_dir_labels = st.text_input(
            "Label Output Directory",
            data_dir,
            key="dsft_labels_output",
            help="Where to save generated label files",
        )

        if DATASET_PREP_AVAILABLE and st.button("🧾 Generate Labels from Folders", key="dsft_gen_labels"):
            try:
                result = generate_labels_from_split_folders(train_dir, val_dir, test_dir, output_dir_labels)
                train_data = result["train_labels"]
                val_data = result["val_labels"]
                data_dir = result["data_dir"]
                st.session_state.dsft_train_labels_path = train_data
                st.session_state.dsft_val_labels_path = val_data
                st.session_state.dsft_data_dir_path = data_dir
                st.success("✅ Generated label files successfully")
                st.caption(
                    f"Train: {result['train_count']} | Val: {result['val_count']} | Test: {result['test_count']}"
                )
            except Exception as e:
                st.error(f"❌ Failed to generate labels: {e}")

    else:
        if not DATASET_PREP_AVAILABLE:
            st.error("Dataset preparation utilities are unavailable. Please install required dependencies.")
        col_split_a, col_split_b = st.columns([2, 1])
        with col_split_a:
            single_folder = st.text_input(
                "Images Folder (will be split)",
                "./raw_images",
                key="dsft_single_folder",
                help="Folder containing all images. VIN will be extracted from filenames.",
            )
            output_dir_labels = st.text_input(
                "Output Directory for Split Dataset",
                data_dir,
                key="dsft_split_output",
                help="Where to store split images and label files",
            )
        with col_split_b:
            train_ratio = st.slider("Train %", 0.5, 0.9, 0.7, step=0.05, key="dsft_train_ratio")
            val_ratio = st.slider("Val %", 0.05, 0.3, 0.15, step=0.05, key="dsft_val_ratio")
            test_ratio = st.slider("Test %", 0.05, 0.3, 0.15, step=0.05, key="dsft_test_ratio")

        total_ratio = train_ratio + val_ratio + test_ratio
        if abs(total_ratio - 1.0) > 0.001:
            st.error("Train/Val/Test ratios must sum to 1.0")

        if DATASET_PREP_AVAILABLE and st.button("✂️ Split Images + Create Labels", key="dsft_split_images"):
            try:
                stats = prepare_dataset(
                    single_folder,
                    output_dir_labels,
                    train_ratio=train_ratio,
                    val_ratio=val_ratio,
                    test_ratio=test_ratio,
                )
                train_data = stats["train_label_file"]
                val_data = stats["val_label_file"]
                data_dir = stats["output_dir"]
                st.session_state.dsft_train_labels_path = train_data
                st.session_state.dsft_val_labels_path = val_data
                st.session_state.dsft_data_dir_path = data_dir
                st.success("✅ Dataset split and labels created")
                st.caption(
                    f"Train: {stats['train_images']} | Val: {stats['val_images']} | Test: {stats['test_images']}"
                )
            except Exception as e:
                st.error(f"❌ Dataset split failed: {e}")
    
    # Show dataset stats
    col_info1, col_info2, col_info3 = st.columns(3)
    
    train_count = 0
    val_count = 0
    test_count = 0
    
    if Path(train_data).exists():
        with open(train_data) as f:
            train_count = sum(1 for _ in f)
    
    if Path(val_data).exists():
        with open(val_data) as f:
            val_count = sum(1 for _ in f)
    
    test_labels_path = Path(data_dir) / "test_labels.txt"
    if test_labels_path.exists():
        with open(test_labels_path) as f:
            test_count = sum(1 for _ in f)
    
    with col_info1:
        st.metric("Training Samples", train_count)
    with col_info2:
        st.metric("Validation Samples", val_count)
    with col_info3:
        st.metric("Total", train_count + val_count + test_count)
    
    if train_count == 0:
        st.warning("⚠️ No training data found. Please check the path.")
    elif train_count < 100:
        st.warning(f"⚠️ Only {train_count} samples. Recommend 500+ for good results.")
    else:
        if test_count == 0:
            st.success(f"✓ Dataset ready with {train_count} training samples")
            st.caption("⚠️ No test set detected yet. Provide test images for evaluation.")
        else:
            st.success(f"✓ Dataset ready: {train_count} train + {val_count} val + {test_count} test")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### ⚙️ Training Parameters")
        epochs = st.slider("Number of Epochs", 1, 20, 5, key="dsft_epochs",
                          help="More epochs = longer training but potentially better results")
        batch_size = st.selectbox("Batch Size", [1, 2, 4, 8], index=1, key="dsft_batch",
                                  help="Lower = less memory, higher = faster training")
        learning_rate = st.select_slider(
            "Learning Rate",
            options=[0.00001, 0.00002, 0.00005, 0.0001],
            value=0.00002,
            key="dsft_lr",
            help="Start with 2e-5, increase if not converging"
        )
        
        gradient_accumulation = st.slider("Gradient Accumulation Steps", 1, 16, 4, key="dsft_grad_accum",
                                          help="Effective batch = batch_size × grad_accum")
        
        st.caption(f"📊 Effective batch size: {batch_size * gradient_accumulation}")
    
    with col2:
        st.markdown("### 🎛️ LoRA Configuration")
        use_lora = st.checkbox("Use LoRA (recommended)", value=True, key="dsft_lora",
                               help="Memory-efficient fine-tuning, works on 8GB+ GPU")
        
        if use_lora:
            lora_r = st.slider("LoRA Rank (r)", 4, 64, 16, key="dsft_lora_r",
                              help="Higher = more capacity but more memory")
            lora_alpha = st.slider("LoRA Alpha", 8, 128, 32, key="dsft_lora_alpha",
                                   help="Scaling factor, typically 2×rank")
            lora_dropout = st.slider("LoRA Dropout", 0.0, 0.2, 0.05, step=0.01, key="dsft_lora_drop")
        else:
            st.warning("⚠️ Full fine-tuning requires 40GB+ VRAM")
            lora_r, lora_alpha, lora_dropout = 16, 32, 0.05
        
        st.markdown("### 💾 Output")
        output_dir = st.text_input("Output Directory", _DEFAULT_PATHS['output_deepseek'], key="dsft_output")
    
    # Training controls
    st.markdown("---")
    st.markdown("### 🚀 Training Controls")
    
    col_a, col_b, col_c, col_d = st.columns(4)
    
    with col_a:
        start_disabled = False
        runner = None
        if TRAINING_UI_AVAILABLE:
            runner = get_global_runner()
            start_disabled = runner.is_running
        
        start_btn = st.button(
            "🚀 Start Training" if not start_disabled else "⏳ Training...",
            type="primary",
            key="dsft_train_btn",
            disabled=start_disabled or train_count == 0,
            use_container_width=True
        )
        
        if start_btn and not start_disabled:
            if TRAINING_UI_AVAILABLE:
                config = {
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "use_lora": use_lora,
                    "lora_r": lora_r,
                    "lora_alpha": lora_alpha,
                    "lora_dropout": lora_dropout,
                    "gradient_accumulation_steps": gradient_accumulation,
                    "train_data_path": train_data,
                    "val_data_path": val_data,
                    "data_dir": data_dir,
                    "output_dir": output_dir,
                }
                try:
                    runner.start_deepseek_finetuning(config)
                    st.success("✅ Training started! Monitor progress below.")
                    time.sleep(1)
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Failed to start training: {e}")
                    import traceback
                    with st.expander("Error Details"):
                        st.code(traceback.format_exc())
            else:
                st.warning("Training UI module not available. Use the terminal command.")
    
    with col_b:
        stop_disabled = not (TRAINING_UI_AVAILABLE and runner and runner.is_running)
        if st.button("⏹️ Stop Training", key="dsft_stop_btn", disabled=stop_disabled, use_container_width=True):
            if runner:
                runner.stop()
                st.info("Training stopped.")
                st.rerun()
    
    with col_c:
        if st.button("📊 View Logs", key="deepseek_logs", use_container_width=True):
            log_path = Path(output_dir) / "train.log"
            if log_path.exists():
                with open(log_path) as f:
                    st.code(f.read()[-5000:])
            else:
                # Check for HuggingFace trainer logs
                trainer_log = Path(output_dir) / "trainer_state.json"
                if trainer_log.exists():
                    with open(trainer_log) as f:
                        st.json(json.load(f))
                else:
                    st.info("No training logs found yet")
    
    with col_d:
        with st.popover("📋 CLI Command"):
            cmd = f"""python -m src.vin_ocr.training.finetune_deepseek \\
    --epochs {epochs} \\
    --batch-size {batch_size} \\
    --lr {learning_rate} \\
    --output {output_dir} \\
    {'--lora' if use_lora else '--full'}"""
            st.code(cmd, language="bash")
            st.caption("Copy this to run in terminal")
    
    # Show checkpoints if any exist
    checkpoint_dir = Path(output_dir)
    if checkpoint_dir.exists():
        checkpoints = list(checkpoint_dir.glob("checkpoint-*"))
        if checkpoints:
            st.markdown("### 📦 Saved Checkpoints")
            for cp in sorted(checkpoints, reverse=True)[:5]:
                col_cp1, col_cp2 = st.columns([3, 1])
                with col_cp1:
                    st.text(cp.name)
                with col_cp2:
                    if st.button("Load", key=f"load_{cp.name}"):
                        st.info(f"To resume from {cp.name}, use --resume flag in CLI")
    
    # Model-specific console output for DeepSeek fine-tuning
    render_model_console_output("deepseek_finetune", "deepseek_ft")


def render_paddleocr_scratch():
    """Render PaddleOCR train from scratch interface."""
    st.subheader("🏗️ PaddleOCR Train from Scratch")
    st.caption("Build a new model from random weights (requires large dataset)")
    
    if not PADDLE_AVAILABLE:
        st.error(
            "❌ **PaddlePaddle not installed.** Training requires PaddlePaddle. "
            "Install with: `pip install paddlepaddle` (CPU) or `pip install paddlepaddle-gpu` (GPU)"
        )
    elif not PADDLE_TRAINING_AVAILABLE:
        st.warning(
            "⚠️ PaddleOCR training module not fully loaded. "
            "Check console for import errors."
        )
    
    st.warning("""
    ⚠️ **Training from Scratch Requirements:**
    - 10,000+ labeled VIN images recommended (more is better)
    - 24GB+ GPU memory recommended for large batches
    - Training time: Hours to days depending on dataset size
    - Consider fine-tuning instead if you have < 5,000 images
    """)
    
    # Model Architecture Selection
    st.markdown("### 🏗️ Model Architecture Selection")
    col_arch1, col_arch2, col_arch3 = st.columns(3)
    with col_arch1:
        architecture = st.selectbox(
            "Recognition Model",
            ["PP-OCRv5", "SVTR_LCNet", "SVTR_Tiny", "CRNN"],
            help="PP-OCRv5: State-of-the-art (2024). SVTR_LCNet: Good accuracy. CRNN: Classic.",
            key="ps_arch_model"
        )
    with col_arch2:
        backbone = st.selectbox(
            "Backbone Network",
            ["PPLCNetV3", "MobileNetV3", "ResNet"],
            help="Feature extraction backbone",
            key="ps_backbone_net"
        )
    with col_arch3:
        head_type = st.selectbox(
            "Recognition Head",
            ["CTC", "Attention", "NRTR"],
            help="Text recognition decoder type",
            key="ps_head_type"
        )
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Dataset Configuration")
        # Use centralized path defaults for consistency
        train_dir = st.text_input("Training Data Directory", _DEFAULT_PATHS['finetune_data'], key="ps_train")
        train_labels = st.text_input("Training Labels File", _DEFAULT_PATHS['train_labels'], key="ps_train_labels")
        val_dir = st.text_input("Validation Data Directory", _DEFAULT_PATHS['finetune_data'], key="ps_val")
        val_labels = st.text_input("Validation Labels File", _DEFAULT_PATHS['val_labels'], key="ps_val_labels")
        
        # Show dataset counts
        train_count = 0
        val_count = 0
        train_labels_path = ProjectPaths.resolve(train_labels)
        val_labels_path = ProjectPaths.resolve(val_labels)
        
        if train_labels_path.exists():
            with open(train_labels_path) as f:
                train_count = sum(1 for line in f if line.strip())
        if val_labels_path.exists():
            with open(val_labels_path) as f:
                val_count = sum(1 for line in f if line.strip())
        
        if train_count > 0:
            st.success(f"✅ Found {train_count} training + {val_count} validation images")
        else:
            st.error("❌ No training data found! Check your labels file path.")
        
        st.markdown("#### Training Parameters")
        epochs = st.slider("Number of Epochs", 10, 200, 100, key="ps_epochs")
        batch_size = st.selectbox("Batch Size", [8, 16, 32, 64, 128], index=1, key="ps_batch")
        learning_rate = st.select_slider(
            "Learning Rate",
            options=[0.0001, 0.0005, 0.001, 0.005, 0.01],
            value=0.001,
            key="ps_lr"
        )
    
    with col2:
        st.markdown("#### Architecture")
        architecture = st.selectbox(
            "Model Architecture",
            ["PP-OCRv5", "SVTR_LCNet", "SVTR_Tiny", "CRNN"],
            help="PP-OCRv5: State-of-the-art (2024). SVTR_LCNet: Good accuracy. CRNN: Classic.",
            key="ps_arch"
        )
        
        backbone = st.selectbox(
            "Backbone",
            ["PPLCNetV3", "MobileNetV3", "ResNet"],
            key="ps_backbone"
        )
        
        st.markdown("#### Hardware & Export")
        device = render_device_selector("ps")
        use_amp = st.checkbox("Mixed Precision (AMP)", value=True, key="ps_amp")
        export_onnx = st.checkbox("Export to ONNX after training", value=False, key="ps_onnx")
        
        st.markdown("#### Output")
        output_dir = st.text_input("Output Directory", _DEFAULT_PATHS['output_scratch'], key="ps_output")
    
    # Training controls
    st.markdown("---")
    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        start_disabled = not PADDLE_TRAINING_AVAILABLE
        if TRAINING_UI_AVAILABLE:
            runner = get_global_runner()
            start_disabled = start_disabled or runner.is_running
        
        if st.button("🏗️ Start Training from Scratch", type="primary", key="ps_train_btn", disabled=start_disabled):
            if not PADDLE_TRAINING_AVAILABLE:
                st.error("❌ PaddlePaddle not available. Install with: pip install paddlepaddle-gpu")
            elif TRAINING_UI_AVAILABLE:
                config = {
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "device": device,
                    "train_data_dir": train_dir,
                    "train_labels": train_labels,
                    "val_data_dir": val_dir,
                    "val_labels": val_labels,
                    "architecture": architecture,
                    "backbone": backbone,
                    "head_type": head_type,
                    "output_dir": output_dir,
                }
                try:
                    runner = get_global_runner()
                    runner.start_paddleocr_scratch(config)
                    st.success("✅ Training started! Progress will appear below.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to start training: {e}")
            else:
                st.warning("⚠️ This will take a LONG time. Use the command in terminal.")
    
    with col_b:
        if st.button("📊 View Training Logs", key="ps_logs"):
            log_path = Path(output_dir) / "train.log"
            if log_path.exists():
                with open(log_path) as f:
                    st.code(f.read()[-5000:])
            else:
                st.info("No training logs found yet")
    
    with col_c:
        with st.expander("📋 Terminal Command"):
            onnx_flag = " --export-onnx" if export_onnx else ""
            cmd = f"""python -m src.vin_ocr.training.train_from_scratch --model paddleocr \\
    --epochs {epochs} \\
    --batch-size {batch_size} \\
    --lr {learning_rate} \\
    --output-dir {output_dir}{onnx_flag}"""
            st.code(cmd, language="bash")
    
    # Model-specific console output for PaddleOCR scratch training
    render_model_console_output("paddleocr_scratch", "paddle_scratch")


def render_deepseek_scratch():
    """Render DeepSeek train from scratch interface."""
    st.subheader("🆕 Vision-Language Model from Scratch")
    st.caption("Build a custom vision-language model for VIN recognition")
    
    st.error("""
    ⚠️ **Training from Scratch Requirements:**
    - 100,000+ labeled images recommended
    - 48GB+ GPU memory (A100 or better)
    - Training time: Weeks
    - This creates a SMALLER model than DeepSeek-OCR
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Dataset Configuration")
        train_data = st.text_input("Training Labels File", "./data/train/labels.txt", key="dss_train")
        val_data = st.text_input("Validation Labels File", "./data/val/labels.txt", key="dss_val")
        data_dir = st.text_input("Data Directory", "./data", key="dss_data_dir")
        
        st.markdown("#### Training Parameters")
        epochs = st.slider("Number of Epochs", 10, 100, 50, key="dss_epochs")
        batch_size = st.selectbox("Batch Size", [2, 4, 8, 16], index=1, key="dss_batch")
        grad_accum = st.slider("Gradient Accumulation Steps", 1, 16, 8, key="dss_grad_accum")
        learning_rate = st.select_slider(
            "Learning Rate",
            options=[0.00005, 0.0001, 0.0002, 0.0005],
            value=0.0001,
            key="dss_lr"
        )
    
    with col2:
        st.markdown("#### Model Configuration")
        image_size = st.selectbox("Image Size", [256, 384, 512], index=1, key="dss_img_size")
        
        st.markdown("#### Hardware & Export")
        device = render_device_selector("dss")
        use_bf16 = st.checkbox("BFloat16 (recommended for Ampere+ GPUs)", value=True, key="dss_bf16")
        use_fp16 = st.checkbox("Float16 (for older GPUs)", value=False, key="dss_fp16")
        export_onnx = st.checkbox("Export to ONNX after training", value=False, key="dss_onnx")
        
        st.markdown("#### Output")
        output_dir = st.text_input("Output Directory", "./output/deepseek_scratch_train", key="dss_output")
    
    # Training controls
    st.markdown("---")
    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        start_disabled = False
        if TRAINING_UI_AVAILABLE:
            runner = get_global_runner()
            start_disabled = runner.is_running
        
        if st.button("🆕 Start Training from Scratch", type="primary", key="dss_train_btn", disabled=start_disabled):
            if TRAINING_UI_AVAILABLE:
                config = {
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "device": device,
                    "train_data_dir": data_dir,
                    "train_labels": train_data,
                    "val_labels": val_data,
                    "output_dir": output_dir,
                }
                try:
                    runner = get_global_runner()
                    runner.start_deepseek_scratch(config)
                    st.success("✅ Training started! Progress will appear below.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to start training: {e}")
            else:
                st.warning("⚠️ This will take a VERY LONG time. Use the command in terminal.")
    
    with col_b:
        if st.button("📊 View Training Logs", key="dss_logs"):
            log_path = Path(output_dir) / "train.log"
            if log_path.exists():
                with open(log_path) as f:
                    st.code(f.read()[-5000:])
            else:
                st.info("No training logs found yet")
    
    with col_c:
        with st.expander("📋 Terminal Command"):
            onnx_flag = " --export-onnx" if export_onnx else ""
            cmd = f"""python -m src.vin_ocr.training.train_from_scratch --model deepseek \\
    --epochs {epochs} \\
    --batch-size {batch_size} \\
    --lr {learning_rate} \\
    --output-dir {output_dir}{onnx_flag}"""
            st.code(cmd, language="bash")
    
    # Model-specific console output for DeepSeek scratch training
    render_model_console_output("deepseek_scratch", "deepseek_scratch")


def render_hyperparameter_tuning():
    """Render the Optuna hyperparameter tuning page."""
    st.subheader("⚙️ Hyperparameter Tuning with Optuna")
    st.caption("Automatically find optimal training parameters using Bayesian optimization")
    
    st.info("""
    **How it works:**
    - Optuna uses Bayesian optimization to efficiently search the hyperparameter space
    - Each trial trains a model with different parameters and measures performance
    - Early stopping (pruning) discards unpromising trials to save time
    - Results are saved and can be resumed if interrupted
    """)
    
    # Model selection
    col1, col2 = st.columns(2)
    
    with col1:
        model_type = st.selectbox(
            "Model Type",
            options=["paddleocr", "deepseek"],
            index=0,
            help="Which model type to tune",
            key="optuna_model_type"
        )
    
    with col2:
        n_trials = st.number_input(
            "Number of Trials",
            min_value=5,
            max_value=500,
            value=50,
            step=5,
            help="Number of hyperparameter combinations to try",
            key="optuna_n_trials"
        )
    
    # Data paths
    st.subheader("📂 Data Configuration")
    
    col1, col2 = st.columns(2)
    with col1:
        train_data_dir = st.text_input(
            "Training Data Directory",
            value="./finetune_data",
            help="Directory containing training images",
            key="optuna_train_data_dir"
        )
        train_labels = st.text_input(
            "Training Labels File",
            value="./finetune_data/train_labels.txt",
            help="Path to training labels file",
            key="optuna_train_labels"
        )
    
    with col2:
        val_data_dir = st.text_input(
            "Validation Data Directory",
            value="./finetune_data",
            help="Directory containing validation images",
            key="optuna_val_data_dir"
        )
        val_labels = st.text_input(
            "Validation Labels File",
            value="./finetune_data/val_labels.txt",
            help="Path to validation labels file",
            key="optuna_val_labels"
        )
    
    # Output settings
    st.subheader("📁 Output Settings")
    
    col1, col2 = st.columns(2)
    with col1:
        output_dir = st.text_input(
            "Output Directory",
            value="./output/hyperparameter_tuning",
            help="Directory to save tuning results",
            key="optuna_output_dir"
        )
        study_name = st.text_input(
            "Study Name",
            value="vin_ocr_tuning",
            help="Name for the Optuna study (for resuming)",
            key="optuna_study_name"
        )
    
    with col2:
        timeout_hours = st.number_input(
            "Timeout (hours)",
            min_value=0,
            max_value=168,
            value=0,
            help="Maximum time to run (0 = no limit)",
            key="optuna_timeout"
        )
        
        # Device selection
        device = render_device_selector("optuna_device")
    
    # Advanced options
    with st.expander("🔧 Advanced Options"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            enable_pruning = st.checkbox(
                "Enable Pruning",
                value=True,
                help="Stop unpromising trials early",
                key="optuna_pruning"
            )
        
        with col2:
            n_startup_trials = st.number_input(
                "Startup Trials",
                min_value=1,
                max_value=50,
                value=10,
                help="Random trials before TPE",
                key="optuna_startup"
            )
        
        with col3:
            early_stopping = st.number_input(
                "Early Stop Trials",
                min_value=5,
                max_value=100,
                value=10,
                help="Stop if no improvement",
                key="optuna_early_stop"
            )
        
        use_storage = st.checkbox(
            "Save to Database (for resuming)",
            value=False,
            help="Save study to SQLite database",
            key="optuna_use_storage"
        )
    
    # Search space info
    with st.expander("📊 Search Space Information"):
        if model_type == "paddleocr":
            st.markdown("""
            **PaddleOCR Search Space:**
            - Architecture: PP-OCRv5, SVTR_LCNet, SVTR_Tiny, CRNN
            - Learning Rate: 1e-5 to 1e-2 (log scale)
            - Batch Size: 4, 8, 16, 32
            - Epochs: 5 to 50
            - Optimizer: Adam, AdamW, SGD
            - Weight Decay: 1e-6 to 1e-2
            - Warmup Ratio: 0.0 to 0.2
            - Label Smoothing: 0.0 to 0.2
            - Image Size: Various height/width combinations
            """)
        else:
            st.markdown("""
            **DeepSeek Search Space:**
            - Learning Rate: 1e-6 to 1e-4 (log scale)
            - Batch Size: 1, 2, 4
            - Epochs: 1 to 10
            - LoRA Rank: 4, 8, 16, 32
            - LoRA Alpha: 8, 16, 32, 64
            - LoRA Dropout: 0.0 to 0.2
            - Gradient Accumulation: 4, 8, 16, 32
            - Weight Decay: 0.0 to 0.1
            """)
    
    # Start button
    st.markdown("---")
    
    from src.vin_ocr.web.training_components import get_global_runner
    runner = get_global_runner()
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        start_disabled = runner.is_running
        
        if st.button(
            "🚀 Start Hyperparameter Tuning",
            disabled=start_disabled,
            use_container_width=True,
            key="start_optuna_btn"
        ):
            # Resolve paths - check both cwd and project root
            def resolve_path(path_str: str) -> Path:
                p = Path(path_str)
                if p.exists():
                    return p
                # Try relative to project root
                p_from_root = _project_root / path_str
                if p_from_root.exists():
                    return p_from_root
                return p  # Return original for error message
            
            train_data_dir_resolved = resolve_path(train_data_dir)
            train_labels_resolved = resolve_path(train_labels)
            val_data_dir_resolved = resolve_path(val_data_dir)
            val_labels_resolved = resolve_path(val_labels)
            
            # Validate paths
            if not train_data_dir_resolved.exists():
                st.error(f"Training data directory not found: {train_data_dir}")
                st.info(f"💡 Expected at: `{_project_root / train_data_dir}`")
            elif not train_labels_resolved.exists():
                st.error(f"Training labels file not found: {train_labels}")
                st.info(f"💡 Expected at: `{_project_root / train_labels}`")
            elif not val_data_dir_resolved.exists():
                st.error(f"Validation data directory not found: {val_data_dir}")
                st.info(f"💡 Expected at: `{_project_root / val_data_dir}`")
            elif not val_labels_resolved.exists():
                st.error(f"Validation labels file not found: {val_labels}")
                st.info(f"💡 Expected at: `{_project_root / val_labels}`")
            else:
                # Resolve output_dir to absolute path
                output_dir_resolved = Path(output_dir)
                if not output_dir_resolved.is_absolute():
                    output_dir_resolved = _project_root / output_dir
                
                # Show resolved paths for transparency
                with st.expander("📍 Resolved Paths (click to verify)", expanded=False):
                    st.code(f"""Training Data: {train_data_dir_resolved}
Training Labels: {train_labels_resolved}
Validation Data: {val_data_dir_resolved}
Validation Labels: {val_labels_resolved}
Output Directory: {output_dir_resolved}""")
                
                # Build config with resolved paths
                config = {
                    "model_type": model_type,
                    "n_trials": n_trials,
                    "train_data_dir": str(train_data_dir_resolved),
                    "train_labels": str(train_labels_resolved),
                    "val_data_dir": str(val_data_dir_resolved),
                    "val_labels": str(val_labels_resolved),
                    "output_dir": str(output_dir_resolved),
                    "device": device,
                    "study_name": study_name,
                }
                
                if timeout_hours > 0:
                    config["timeout"] = timeout_hours * 3600
                
                if use_storage:
                    storage_path = output_dir_resolved / f"{study_name}.db"
                    config["storage"] = f"sqlite:///{storage_path}"
                
                try:
                    runner.start_hyperparameter_tuning(config)
                    st.success("✅ Hyperparameter tuning started!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to start tuning: {e}")
    
    with col2:
        if st.button(
            "⏹️ Stop",
            disabled=not runner.is_running,
            use_container_width=True,
            key="stop_optuna_btn"
        ):
            runner.stop()
            st.warning("Tuning stopped")
            st.rerun()
    
    with col3:
        # Show results button - resolve output_dir for checking
        output_dir_check = Path(output_dir)
        if not output_dir_check.is_absolute():
            output_dir_check = _project_root / output_dir
        
        results_file = output_dir_check / "optimization_results.json"
        if results_file.exists():
            if st.button("📊 View Results", use_container_width=True, key="view_optuna_results"):
                import json
                with open(results_file) as f:
                    results = json.load(f)
                
                st.success(f"**Best Value:** {results.get('best_value', 'N/A'):.4f}")
                st.json(results.get('best_params', {}))
    
    # CLI command - show with resolved paths
    with st.expander("💻 CLI Command"):
        # Resolve paths for CLI display
        cli_train_data = _project_root / train_data_dir if not Path(train_data_dir).is_absolute() else train_data_dir
        cli_train_labels = _project_root / train_labels if not Path(train_labels).is_absolute() else train_labels
        cli_val_labels = _project_root / val_labels if not Path(val_labels).is_absolute() else val_labels
        cli_output_dir = _project_root / output_dir if not Path(output_dir).is_absolute() else output_dir
        
        timeout_flag = f" --timeout {int(timeout_hours * 3600)}" if timeout_hours > 0 else ""
        storage_flag = f" --storage sqlite:///{cli_output_dir}/{study_name}.db" if use_storage else ""
        
        cmd = f"""python -m src.vin_ocr.training.hyperparameter_tuning.optuna_tuning \\
    --model {model_type} \\
    --n-trials {n_trials} \\
    --train-data {cli_train_data} \\
    --train-labels {cli_train_labels} \\
    --val-labels {cli_val_labels} \\
    --output {cli_output_dir} \\
    --device {device} \\
    --study-name {study_name}{timeout_flag}{storage_flag}"""
        st.code(cmd, language="bash")
        st.caption("💡 This command uses absolute paths resolved from the project root.")
    
    # Live console output for hyperparameter tuning
    # The runner sets model_type to "{model_type}_tuning" (e.g., "paddleocr_tuning" or "deepseek_tuning")
    render_model_console_output(f"{model_type}_tuning", "optuna")


def render_results_dashboard():
    """
    Render a comprehensive results dashboard with professional metrics visualization.
    
    Features:
    - Model comparison metrics
    - Detailed per-model analysis
    - Character-level confusion analysis
    - Training history and progress
    - Export capabilities
    """
    st.markdown('<h1 class="main-header">📈 Results Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Comprehensive analysis of VIN recognition performance</p>', unsafe_allow_html=True)
    
    # Results directory - use project root for correct path resolution
    results_dir = _project_root / "results"
    
    # Dashboard tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Model Comparison",
        "🔍 Detailed Metrics", 
        "📝 Sample Results",
        "🕐 Session History",
        "📁 Saved Reports"
    ])
    
    # ==========================================================================
    # TAB 1: Model Comparison Overview
    # ==========================================================================
    with tab1:
        st.subheader("🏆 Model Performance Comparison")
        
        # Load model comparison data
        comparison_file = results_dir / "model_comparison.csv"
        multi_eval_file = results_dir / "multi_model_evaluation.json"
        
        if comparison_file.exists():
            df_comparison = pd.read_csv(comparison_file)
            
            # Key metrics cards
            st.markdown("#### Key Performance Indicators")
            cols = st.columns(4)
            
            # Find best model for each metric
            if len(df_comparison) > 0:
                best_exact = df_comparison.loc[df_comparison['Exact Match Acc'].idxmax()]
                best_char = df_comparison.loc[df_comparison['Char Acc'].idxmax()]
                best_f1 = df_comparison.loc[df_comparison['F1 Micro'].idxmax()]
                fastest = df_comparison.loc[df_comparison['Avg Time'].idxmin()]
                
                with cols[0]:
                    st.metric(
                        "🎯 Best Exact Match",
                        f"{best_exact['Exact Match Acc']:.1%}",
                        delta=best_exact['Model'][:20],
                        delta_color="off"
                    )
                with cols[1]:
                    st.metric(
                        "📝 Best Char Accuracy",
                        f"{best_char['Char Acc']:.1%}",
                        delta=best_char['Model'][:20],
                        delta_color="off"
                    )
                with cols[2]:
                    st.metric(
                        "⚖️ Best F1 Score",
                        f"{best_f1['F1 Micro']:.3f}",
                        delta=best_f1['Model'][:20],
                        delta_color="off"
                    )
                with cols[3]:
                    st.metric(
                        "⚡ Fastest Model",
                        f"{fastest['Avg Time']:.2f}s",
                        delta=fastest['Model'][:20],
                        delta_color="off"
                    )
            
            st.markdown("---")
            
            # Comparison table with formatting
            st.markdown("#### 📋 Full Comparison Table")
            
            # Format the dataframe for display
            df_display = df_comparison.copy()
            df_display['Exact Match Acc'] = df_display['Exact Match Acc'].apply(lambda x: f"{x:.2%}")
            df_display['Char Acc'] = df_display['Char Acc'].apply(lambda x: f"{x:.2%}")
            df_display['F1 Micro'] = df_display['F1 Micro'].apply(lambda x: f"{x:.4f}")
            df_display['F1 Macro'] = df_display['F1 Macro'].apply(lambda x: f"{x:.4f}")
            df_display['Avg Confidence'] = df_display['Avg Confidence'].apply(lambda x: f"{x:.2%}")
            df_display['Avg Time'] = df_display['Avg Time'].apply(lambda x: f"{x:.2f}s")
            
            st.dataframe(df_display, use_container_width=True, hide_index=True)
            
            # Charts
            if PLOTLY_AVAILABLE and len(df_comparison) > 1:
                st.markdown("#### 📈 Visual Comparison")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Accuracy comparison
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        name='Exact Match',
                        x=df_comparison['Model'],
                        y=df_comparison['Exact Match Acc'] * 100,
                        marker_color='#2E86AB'
                    ))
                    fig.add_trace(go.Bar(
                        name='Character Accuracy',
                        x=df_comparison['Model'],
                        y=df_comparison['Char Acc'] * 100,
                        marker_color='#A23B72'
                    ))
                    fig.update_layout(
                        title='Accuracy Metrics by Model',
                        yaxis_title='Accuracy (%)',
                        barmode='group',
                        legend=dict(orientation="h", yanchor="bottom", y=1.02)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # F1 Score comparison
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        name='F1 Micro',
                        x=df_comparison['Model'],
                        y=df_comparison['F1 Micro'],
                        marker_color='#F18F01'
                    ))
                    fig.add_trace(go.Bar(
                        name='F1 Macro',
                        x=df_comparison['Model'],
                        y=df_comparison['F1 Macro'],
                        marker_color='#C73E1D'
                    ))
                    fig.update_layout(
                        title='F1 Scores by Model',
                        yaxis_title='F1 Score',
                        barmode='group',
                        legend=dict(orientation="h", yanchor="bottom", y=1.02)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Speed vs Accuracy trade-off
                st.markdown("#### ⚡ Speed vs Accuracy Trade-off")
                fig = px.scatter(
                    df_comparison,
                    x='Avg Time',
                    y='Char Acc',
                    size='F1 Micro',
                    color='Model',
                    hover_data=['Exact Match Acc', 'Avg Confidence'],
                    title='Processing Time vs Character Accuracy'
                )
                fig.update_layout(
                    xaxis_title='Average Processing Time (seconds)',
                    yaxis_title='Character Accuracy',
                    yaxis_tickformat='.1%'
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("📁 No model comparison data found. Run a batch evaluation to generate comparison metrics.")
    
    # ==========================================================================
    # TAB 2: Detailed Metrics
    # ==========================================================================
    with tab2:
        st.subheader("🔍 Detailed Performance Metrics")
        
        detailed_file = results_dir / "detailed_metrics.json"
        experiment_file = results_dir / "experiment_summary.json"
        
        if detailed_file.exists():
            with open(detailed_file) as f:
                detailed = json.load(f)
            
            # Metrics Summary Header
            summary = detailed.get('metrics_summary', {})
            st.markdown(f"""
            **Evaluation Details**
            - 📅 Date: {summary.get('evaluation_date', 'N/A')}
            - 📊 Sample Size: {summary.get('sample_size', 'N/A')} images
            - 📝 {summary.get('description', '')}
            """)
            
            st.markdown("---")
            
            # Baseline vs Pipeline comparison
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📦 Baseline (Raw OCR)")
                baseline = detailed.get('baseline_metrics', {})
                if baseline:
                    metrics_data = []
                    for key, value in baseline.items():
                        if key != 'description' and isinstance(value, dict):
                            metrics_data.append({
                                'Metric': key.replace('_', ' ').title(),
                                'Value': f"{value.get('value', 0):.2%}" if value.get('value', 0) <= 1 else str(value.get('value', 0)),
                                'Description': value.get('description', '')[:50]
                            })
                    if metrics_data:
                        st.dataframe(pd.DataFrame(metrics_data), use_container_width=True, hide_index=True)
            
            with col2:
                st.markdown("### 🚀 Pipeline (With Processing)")
                pipeline = detailed.get('pipeline_metrics', {})
                if pipeline:
                    metrics_data = []
                    for key, value in pipeline.items():
                        if key != 'description' and isinstance(value, dict):
                            val = value.get('value', 0)
                            if isinstance(val, (int, float)):
                                if val <= 1:
                                    display_val = f"{val:.2%}"
                                else:
                                    unit = value.get('unit', '')
                                    display_val = f"{val:,.0f} {unit}" if unit else str(val)
                            else:
                                display_val = str(val)
                            metrics_data.append({
                                'Metric': key.replace('_', ' ').title(),
                                'Value': display_val,
                                'Description': value.get('description', '')[:50]
                            })
                    if metrics_data:
                        st.dataframe(pd.DataFrame(metrics_data), use_container_width=True, hide_index=True)
            
            # Improvement Analysis
            improvement = detailed.get('improvement_analysis', {})
            if improvement:
                st.markdown("---")
                st.markdown("### 📈 Improvement Analysis")
                
                cols = st.columns(len(improvement))
                for i, (metric, data) in enumerate(improvement.items()):
                    with cols[i]:
                        st.metric(
                            metric.replace('_', ' ').title(),
                            data.get('relative', 'N/A'),
                            delta=data.get('absolute', ''),
                            delta_color="normal"
                        )
                        st.caption(f"From {data.get('from', 0):.0%} → {data.get('to', 0):.0%}")
            
            # Character Confusion Analysis
            confusion = detailed.get('character_level_analysis', {})
            if confusion:
                st.markdown("---")
                st.markdown("### 🔤 Character-Level Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**✅ Most Accurate Characters**")
                    accurate = confusion.get('most_accurate_characters', [])
                    if accurate:
                        st.code(' '.join(accurate), language=None)
                
                with col2:
                    st.markdown("**⚠️ Common Confusions**")
                    confused = confusion.get('most_confused_characters', {})
                    for char, desc in confused.items():
                        st.markdown(f"- **{char}**: {desc}")
            
            # Metric Definitions (collapsible)
            definitions = detailed.get('metric_definitions', {})
            if definitions:
                with st.expander("📚 Metric Definitions"):
                    for metric, info in definitions.items():
                        st.markdown(f"**{metric.replace('_', ' ').title()}**")
                        st.code(info.get('formula', ''), language=None)
                        st.caption(info.get('interpretation', ''))
                        st.markdown("---")
        
        # Load experiment summary for additional context
        if experiment_file.exists():
            with open(experiment_file) as f:
                experiment = json.load(f)
            
            with st.expander("🔬 Experiment Configuration"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**OCR Configuration**")
                    config = experiment.get('config', {})
                    for key, value in config.items():
                        st.text(f"{key}: {value}")
                
                with col2:
                    st.markdown("**Dataset Info**")
                    dataset = experiment.get('dataset', {})
                    for key, value in dataset.items():
                        st.text(f"{key}: {value}")
            
            # Recommendations
            recommendations = experiment.get('recommendations', [])
            if recommendations:
                st.markdown("### 💡 Recommendations")
                for rec in recommendations:
                    st.markdown(f"- {rec}")
        
        if not detailed_file.exists() and not experiment_file.exists():
            st.info("📁 No detailed metrics found. Run an evaluation experiment to generate detailed metrics.")
    
    # ==========================================================================
    # TAB 3: Sample Results
    # ==========================================================================
    with tab3:
        st.subheader("📝 Sample Recognition Results")
        
        sample_file = results_dir / "sample_results.csv"
        multi_eval_file = results_dir / "multi_model_evaluation.json"
        
        if sample_file.exists():
            df_samples = pd.read_csv(sample_file)
            
            # Check if model information is available
            has_model_info = 'model_name' in df_samples.columns
            
            # Model filter (if multiple models in results)
            if has_model_info and df_samples['model_name'].nunique() > 1:
                st.markdown("### 🤖 Models Used in Evaluation")
                model_names = df_samples['model_name'].unique().tolist()
                
                # Show model summary
                model_summary = df_samples.groupby(['model_name', 'model_type']).agg({
                    'exact_match': ['count', 'sum', 'mean'],
                    'confidence': 'mean'
                }).round(4)
                model_summary.columns = ['Total', 'Correct', 'Accuracy', 'Avg Confidence']
                st.dataframe(model_summary, use_container_width=True)
                
                # Model selector
                selected_model = st.selectbox(
                    "Filter by Model",
                    ["All Models"] + model_names,
                    key="model_filter"
                )
                
                if selected_model != "All Models":
                    df_samples = df_samples[df_samples['model_name'] == selected_model]
                    st.info(f"Showing results for: **{selected_model}**")
                
                st.markdown("---")
            elif has_model_info:
                # Single model - show which one
                model_name = df_samples['model_name'].iloc[0] if len(df_samples) > 0 else "Unknown"
                model_type = df_samples['model_type'].iloc[0] if 'model_type' in df_samples.columns and len(df_samples) > 0 else "Unknown"
                st.info(f"📊 Results from: **{model_name}** (Type: `{model_type}`)")
            
            # Summary stats
            total = len(df_samples)
            exact_matches = df_samples['exact_match'].sum() if 'exact_match' in df_samples.columns else 0
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Samples", total)
            with col2:
                st.metric("Exact Matches", int(exact_matches))
            with col3:
                st.metric("Accuracy", f"{exact_matches/total:.1%}" if total > 0 else "N/A")
            with col4:
                avg_conf = df_samples['confidence'].mean() if 'confidence' in df_samples.columns else 0
                st.metric("Avg Confidence", f"{avg_conf:.1%}")
            
            st.markdown("---")
            
            # Filter options
            col1, col2 = st.columns([1, 3])
            with col1:
                filter_option = st.selectbox(
                    "Filter Results",
                    ["All", "Correct Only", "Errors Only"],
                    key="sample_filter"
                )
            
            # Apply filter
            if filter_option == "Correct Only":
                df_filtered = df_samples[df_samples['exact_match'] == True]
            elif filter_option == "Errors Only":
                df_filtered = df_samples[df_samples['exact_match'] == False]
            else:
                df_filtered = df_samples
            
            # Display table with better formatting
            st.markdown(f"**Showing {len(df_filtered)} of {total} results**")
            
            # Format for display - include model info if available
            # Try new format first (from multi-model evaluation)
            display_cols_new = ['model_name', 'image', 'ground_truth', 'prediction', 'confidence', 'exact_match', 'chars_correct']
            # Fallback to old format
            display_cols_old = ['file', 'ground_truth', 'corrected_vin', 'confidence', 'exact_match', 'char_matches', 'corrections_applied']
            
            # Determine which columns exist
            if 'image' in df_filtered.columns:
                display_cols = display_cols_new
            else:
                display_cols = display_cols_old
            
            available_cols = [c for c in display_cols if c in df_filtered.columns]
            
            df_display = df_filtered[available_cols].copy()
            if 'confidence' in df_display.columns:
                df_display['confidence'] = df_display['confidence'].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
            if 'exact_match' in df_display.columns:
                df_display['exact_match'] = df_display['exact_match'].apply(lambda x: "✅" if x else "❌")
            
            # Rename columns for display
            column_rename = {
                'model_name': 'Model',
                'model_type': 'Type',
                'file': 'Image',
                'image': 'Image',
                'ground_truth': 'Ground Truth',
                'prediction': 'Predicted',
                'corrected_vin': 'Predicted',
                'confidence': 'Confidence',
                'exact_match': 'Match',
                'chars_correct': 'Chars',
                'char_matches': 'Chars',
                'corrections_applied': 'Corrections'
            }
            df_display = df_display.rename(columns=column_rename)
            
            st.dataframe(df_display, use_container_width=True, hide_index=True)
            
            # Error analysis
            errors_df = df_samples[df_samples['exact_match'] == False] if 'exact_match' in df_samples.columns else pd.DataFrame()
            if len(errors_df) > 0:
                with st.expander(f"🔍 Error Analysis ({len(errors_df)} errors)"):
                    st.markdown("**Common Error Patterns:**")
                    
                    # Analyze corrections
                    if 'corrections_applied' in errors_df.columns:
                        corrections = errors_df['corrections_applied'].value_counts()
                        for correction, count in corrections.head(5).items():
                            st.markdown(f"- {correction}: {count} occurrences")
        
        # Multi-model evaluation results
        if multi_eval_file.exists():
            with st.expander("📊 Multi-Model Evaluation Details"):
                with open(multi_eval_file) as f:
                    multi_eval = json.load(f)
                
                models = multi_eval.get('models', {})
                
                for model_name, model_data in models.items():
                    st.markdown(f"#### {model_data.get('model_name', model_name)}")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Image-Level Metrics**")
                        img_metrics = model_data.get('image_level', {})
                        st.text(f"Total Images: {img_metrics.get('total_images', 0)}")
                        st.text(f"Exact Matches: {img_metrics.get('exact_matches', 0)}")
                        st.text(f"Accuracy: {img_metrics.get('exact_match_accuracy', 0):.2%}")
                    
                    with col2:
                        st.markdown("**Character-Level Metrics**")
                        char_metrics = model_data.get('character_level', {})
                        st.text(f"Character Accuracy: {char_metrics.get('character_accuracy', 0):.2%}")
                        st.text(f"F1 Micro: {char_metrics.get('f1_micro', 0):.4f}")
                        st.text(f"F1 Macro: {char_metrics.get('f1_macro', 0):.4f}")
                    
                    # Sample predictions
                    samples = model_data.get('sample_results', [])[:5]
                    if samples:
                        st.markdown("**Sample Predictions:**")
                        for sample in samples:
                            match_icon = "✅" if sample.get('exact_match') else "❌"
                            st.text(f"{match_icon} GT: {sample.get('ground_truth', '')} | Pred: {sample.get('prediction', '')}")
                    
                    st.markdown("---")
        
        if not sample_file.exists() and not multi_eval_file.exists():
            st.info("📁 No sample results found. Process some images to generate sample results.")
    
    # ==========================================================================
    # TAB 4: Session History
    # ==========================================================================
    with tab4:
        st.subheader("🕐 Current Session History")
        
        if st.session_state.results_history:
            # Session summary
            history = st.session_state.results_history
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Images Processed", len(history))
            with col2:
                avg_conf = sum(r.get('confidence', 0) for r in history) / len(history)
                st.metric("Avg Confidence", f"{avg_conf:.1%}")
            with col3:
                exact_matches = sum(1 for r in history if r.get('exact_match', False))
                st.metric("Exact Matches", exact_matches)
            with col4:
                avg_time = sum(r.get('processing_time', 0) for r in history) / len(history)
                st.metric("Avg Time", f"{avg_time:.2f}s")
            
            st.markdown("---")
            
            # Model breakdown
            models_used = {}
            for r in history:
                model = r.get('model', 'Unknown')
                if model not in models_used:
                    models_used[model] = {'count': 0, 'matches': 0, 'total_conf': 0}
                models_used[model]['count'] += 1
                models_used[model]['total_conf'] += r.get('confidence', 0)
                if r.get('exact_match'):
                    models_used[model]['matches'] += 1
            
            if len(models_used) > 1:
                st.markdown("#### By Model")
                model_stats = []
                for model, stats in models_used.items():
                    model_stats.append({
                        'Model': model,
                        'Count': stats['count'],
                        'Matches': stats['matches'],
                        'Avg Confidence': f"{stats['total_conf']/stats['count']:.1%}"
                    })
                st.dataframe(pd.DataFrame(model_stats), use_container_width=True, hide_index=True)
            
            # Full history table
            st.markdown("#### Recognition History")
            
            df = pd.DataFrame([
                {
                    'Time': r.get('timestamp', 'N/A')[:19] if r.get('timestamp') else 'N/A',
                    'Model': r.get('model', 'N/A')[:25],
                    'Image': r.get('filename', 'N/A')[:30],
                    'VIN': r.get('vin', 'N/A'),
                    'Ground Truth': r.get('ground_truth', '-'),
                    'Match': '✅' if r.get('exact_match') else '❌' if r.get('ground_truth') else '-',
                    'Confidence': f"{r.get('confidence', 0):.1%}",
                    'Time (s)': f"{r.get('processing_time', 0):.2f}"
                }
                for r in reversed(history[-100:])  # Most recent first, limit 100
            ])
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            # Export and clear options
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Export to CSV
                export_data = pd.DataFrame([
                    {
                        'timestamp': r.get('timestamp', ''),
                        'model': r.get('model', ''),
                        'filename': r.get('filename', ''),
                        'predicted_vin': r.get('vin', ''),
                        'ground_truth': r.get('ground_truth', ''),
                        'exact_match': r.get('exact_match', ''),
                        'confidence': r.get('confidence', 0),
                        'processing_time': r.get('processing_time', 0),
                        'error': r.get('error', '')
                    }
                    for r in history
                ])
                csv = export_data.to_csv(index=False)
                st.download_button(
                    "📥 Export to CSV",
                    csv,
                    f"session_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv",
                    use_container_width=True
                )
            
            with col2:
                # Export to JSON
                json_data = json.dumps(history, indent=2, default=str)
                st.download_button(
                    "📥 Export to JSON",
                    json_data,
                    f"session_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    "application/json",
                    use_container_width=True
                )
            
            with col3:
                if st.button("🗑️ Clear History", use_container_width=True):
                    st.session_state.results_history = []
                    st.rerun()
        else:
            st.info("📭 No recognition results in this session yet. Process some images to build history.")
    
    # ==========================================================================
    # TAB 5: Saved Reports
    # ==========================================================================
    with tab5:
        st.subheader("📁 Saved Result Files & Training Outputs")
        
        # Define all directories to scan for results - use project root
        output_dir = _project_root / "output"
        
        # Directory selector
        st.markdown("#### 📂 Select Directory to Browse")
        
        available_dirs = []
        
        # Add results directory if it exists
        if results_dir.exists():
            available_dirs.append(("📊 Results (./results)", results_dir))
        
        # Add output directory and its subdirectories
        if output_dir.exists():
            available_dirs.append(("🏋️ All Training Outputs (./output)", output_dir))
            
            # Get training run subdirectories
            for subdir in sorted(output_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
                if subdir.is_dir():
                    # Count files in subdirectory
                    file_count = len(list(subdir.glob("*")))
                    # Get last modified time
                    try:
                        mtime = datetime.fromtimestamp(subdir.stat().st_mtime).strftime('%m/%d %H:%M')
                    except:
                        mtime = "N/A"
                    available_dirs.append((f"  └─ {subdir.name} ({file_count} files, {mtime})", subdir))
        
        if not available_dirs:
            st.warning("No result directories found. Run some evaluations or training to generate results.")
        else:
            # Directory selector
            selected_dir_name = st.selectbox(
                "Browse Directory",
                [d[0] for d in available_dirs],
                key="report_dir_select"
            )
            
            # Find the selected directory path
            selected_dir = None
            for name, path in available_dirs:
                if name == selected_dir_name:
                    selected_dir = path
                    break
            
            if selected_dir and selected_dir.exists():
                st.markdown("---")
                
                # Scan for files in selected directory
                json_files = list(selected_dir.glob("*.json"))
                csv_files = list(selected_dir.glob("*.csv"))
                log_files = list(selected_dir.glob("*.log"))
                model_files = list(selected_dir.glob("*.pdparams")) + list(selected_dir.glob("*.pt")) + list(selected_dir.glob("*.pth"))
                subdirs = [d for d in selected_dir.iterdir() if d.is_dir()]
                
                # Summary metrics
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("📄 JSON", len(json_files))
                with col2:
                    st.metric("📊 CSV", len(csv_files))
                with col3:
                    st.metric("📝 Logs", len(log_files))
                with col4:
                    st.metric("🧠 Models", len(model_files))
                with col5:
                    st.metric("📂 Folders", len(subdirs))
                
                st.markdown("---")
                
                # Combine all viewable files
                all_files = json_files + csv_files + log_files
                
                if all_files:
                    # Build file info with metadata
                    file_info = []
                    for f in sorted(all_files, key=lambda x: x.stat().st_mtime, reverse=True):
                        try:
                            stat = f.stat()
                            size_kb = stat.st_size / 1024
                            size_str = f"{size_kb:.1f} KB" if size_kb < 1024 else f"{size_kb/1024:.1f} MB"
                            
                            # Determine file type and icon
                            if f.suffix == '.json':
                                icon = "📄"
                                ftype = "JSON"
                            elif f.suffix == '.csv':
                                icon = "📊"
                                ftype = "CSV"
                            elif f.suffix == '.log':
                                icon = "📝"
                                ftype = "Log"
                            else:
                                icon = "📁"
                                ftype = f.suffix.upper()[1:]
                            
                            file_info.append({
                                'icon': icon,
                                'name': f.name,
                                'path': f,
                                'type': ftype,
                                'size': size_str,
                                'modified': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                            })
                        except Exception:
                            continue
                    
                    if file_info:
                        # Display file table
                        st.markdown("#### 📋 Available Files")
                        df_files = pd.DataFrame(file_info)
                        st.dataframe(
                            df_files[['icon', 'name', 'type', 'size', 'modified']].rename(columns={
                                'icon': '',
                                'name': 'File Name',
                                'type': 'Type',
                                'size': 'Size',
                                'modified': 'Last Modified'
                            }),
                            use_container_width=True,
                            hide_index=True
                        )
                        
                        # File viewer
                        st.markdown("#### 🔍 File Viewer")
                        
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            selected_file = st.selectbox(
                                "Select a file to view",
                                [f['name'] for f in file_info],
                                key="output_file_viewer_select"
                            )
                        
                        if selected_file:
                            # Find the file path
                            file_path = None
                            for fi in file_info:
                                if fi['name'] == selected_file:
                                    file_path = fi['path']
                                    break
                            
                            if file_path and file_path.exists():
                                with col2:
                                    # Download button
                                    with open(file_path, 'rb') as f:
                                        st.download_button(
                                            "📥 Download",
                                            f.read(),
                                            selected_file,
                                            key="download_output_file"
                                        )
                                
                                # Display content based on file type
                                try:
                                    if selected_file.endswith('.json'):
                                        with open(file_path) as f:
                                            data = json.load(f)
                                        
                                        # Pretty display for known formats
                                        if 'training_progress' in str(file_path) or 'epoch' in data:
                                            st.success("🏋️ Training Progress Data")
                                            
                                            # Extract key metrics if available
                                            if isinstance(data, dict):
                                                key_metrics = ['epoch', 'loss', 'accuracy', 'val_loss', 'val_accuracy', 'best_accuracy']
                                                metrics_found = {k: v for k, v in data.items() if k in key_metrics}
                                                if metrics_found:
                                                    cols = st.columns(len(metrics_found))
                                                    for i, (k, v) in enumerate(metrics_found.items()):
                                                        with cols[i]:
                                                            if isinstance(v, float):
                                                                st.metric(k.replace('_', ' ').title(), f"{v:.4f}")
                                                            else:
                                                                st.metric(k.replace('_', ' ').title(), str(v))
                                        
                                        elif 'metrics_summary' in data:
                                            st.success("📊 Detailed Metrics Report")
                                        elif 'experiment' in data:
                                            st.success("🔬 Experiment Summary")
                                        elif 'models' in data:
                                            st.success("📈 Multi-Model Evaluation")
                                        elif 'dataset_info' in data:
                                            st.success("📦 Dataset Information")
                                        
                                        # Show JSON with syntax highlighting
                                        with st.expander("View Raw JSON", expanded=True):
                                            st.json(data)
                                    
                                    elif selected_file.endswith('.csv'):
                                        df = pd.read_csv(file_path)
                                        st.dataframe(df, use_container_width=True)
                                        st.caption(f"📊 {len(df)} rows × {len(df.columns)} columns")
                                        
                                        # Show column info
                                        with st.expander("Column Information"):
                                            col_info = pd.DataFrame({
                                                'Column': df.columns,
                                                'Type': df.dtypes.astype(str),
                                                'Non-Null': df.count(),
                                                'Sample': [str(df[c].iloc[0])[:30] if len(df) > 0 else '' for c in df.columns]
                                            })
                                            st.dataframe(col_info, use_container_width=True, hide_index=True)
                                    
                                    elif selected_file.endswith('.log'):
                                        with open(file_path, 'r') as f:
                                            log_content = f.read()
                                        
                                        # Show log stats
                                        lines = log_content.split('\n')
                                        st.caption(f"� {len(lines)} lines | {len(log_content)} characters")
                                        
                                        # Show last N lines option
                                        show_lines = st.slider(
                                            "Lines to show (from end)",
                                            min_value=10,
                                            max_value=min(500, len(lines)),
                                            value=min(100, len(lines)),
                                            key="log_lines_slider"
                                        )
                                        
                                        # Display log content
                                        display_content = '\n'.join(lines[-show_lines:])
                                        st.code(display_content, language="text")
                                        
                                        # Search in log
                                        search_term = st.text_input("🔍 Search in log", key="log_search")
                                        if search_term:
                                            matching_lines = [l for l in lines if search_term.lower() in l.lower()]
                                            if matching_lines:
                                                st.success(f"Found {len(matching_lines)} matching lines")
                                                st.code('\n'.join(matching_lines[:50]), language="text")
                                            else:
                                                st.warning("No matches found")
                                
                                except Exception as e:
                                    st.error(f"Error reading file: {e}")
                                    # Try to show raw content
                                    try:
                                        with open(file_path, 'r') as f:
                                            st.code(f.read()[:10000], language="text")
                                    except:
                                        st.error("Could not read file content")
                
                # Show model files separately
                if model_files:
                    with st.expander(f"🧠 Model Checkpoints ({len(model_files)})"):
                        for mf in sorted(model_files, key=lambda x: x.stat().st_mtime, reverse=True)[:10]:
                            size_mb = mf.stat().st_size / (1024 * 1024)
                            mtime = datetime.fromtimestamp(mf.stat().st_mtime).strftime('%Y-%m-%d %H:%M')
                            st.markdown(f"- **{mf.name}** ({size_mb:.1f} MB) - {mtime}")
                
                # Show subdirectories
                if subdirs:
                    with st.expander(f"📂 Subdirectories ({len(subdirs)})"):
                        for subdir in sorted(subdirs, key=lambda x: x.name):
                            files_in_subdir = list(subdir.glob("*"))
                            json_count = len(list(subdir.glob("*.json")))
                            model_count = len(list(subdir.glob("*.pd*"))) + len(list(subdir.glob("*.pt*")))
                            st.markdown(f"**{subdir.name}/** - {len(files_in_subdir)} items ({json_count} JSON, {model_count} models)")


# =============================================================================
# SYSTEM HEALTH PAGE
# =============================================================================

def render_system_health_page():
    """Render the system health check page."""
    st.markdown('<h1 class="main-header">🔧 System Health</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Check environment, dependencies, and model cache status</p>', unsafe_allow_html=True)
    
    # Python Environment
    st.subheader("🐍 Python Environment")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Python Version", f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    with col2:
        st.metric("Platform", sys.platform)
    
    st.divider()
    
    # GPU Status
    st.subheader("🖥️ GPU / Compute Device")
    
    if GPU_UTILS_AVAILABLE:
        gpu = get_gpu_manager()
        status = gpu.detect_all(force_refresh=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if status.cuda_available:
                st.success(f"✓ CUDA Available")
                for d in status.cuda_devices:
                    st.caption(f"  {d.name} ({d.memory_total_gb}GB)")
            else:
                st.warning("✗ CUDA not available")
        
        with col2:
            if status.mps_available:
                st.success(f"✓ MPS (Apple Silicon)")
            else:
                st.info("✗ MPS not available")
        
        with col3:
            st.metric("Best Device", status.best_device_name)
            st.caption(f"Using GPU: {'Yes' if st.session_state.get('use_gpu', False) else 'No'}")
        
        # PyTorch / Paddle info
        col1, col2 = st.columns(2)
        with col1:
            if status.torch_available:
                st.success(f"✓ PyTorch: {status.torch_version}")
            else:
                st.warning("✗ PyTorch not installed")
        with col2:
            if status.paddle_available:
                gpu_icon = "✓" if status.paddle_gpu else "✗"
                st.success(f"✓ Paddle: {status.paddle_version} (GPU: {gpu_icon})")
            else:
                st.warning("✗ PaddlePaddle not installed")
        
        if status.errors:
            with st.expander("⚠️ GPU Detection Warnings"):
                for err in status.errors:
                    st.text(err)
    else:
        st.warning("GPU utilities not available")
    
    st.divider()
    
    # PaddlePaddle & PaddleOCR
    st.subheader("🏓 PaddlePaddle & PaddleOCR")
    
    paddle_status = {"installed": False, "version": "N/A"}
    paddleocr_status = {"installed": False, "version": "N/A"}
    
    try:
        import paddle
        paddle_status = {"installed": True, "version": paddle.__version__}
    except ImportError:
        pass
    
    try:
        import paddleocr
        paddleocr_status = {"installed": True, "version": getattr(paddleocr, '__version__', 'unknown')}
    except ImportError:
        pass
    
    col1, col2 = st.columns(2)
    with col1:
        if paddle_status["installed"]:
            st.success(f"✓ PaddlePaddle: {paddle_status['version']}")
        else:
            st.error("✗ PaddlePaddle not installed")
    with col2:
        if paddleocr_status["installed"]:
            st.success(f"✓ PaddleOCR: {paddleocr_status['version']}")
        else:
            st.error("✗ PaddleOCR not installed")
    
    st.divider()
    
    # Model Cache Health
    st.subheader("📦 Model Cache")
    cache_health = check_model_cache_health()
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**PaddleOCR Cache:**")
        st.code(cache_health['cache_paths']['paddleocr'])
        if cache_health['paddleocr_cache_exists']:
            st.success("✓ Directory exists")
        else:
            st.warning("⚠ Directory does not exist (will be created on first model download)")
        if cache_health['paddleocr_cache_writable']:
            st.success("✓ Writable")
        else:
            st.error("✗ Not writable - model downloads will fail!")
    
    with col2:
        st.write("**PaddleX Cache:**")
        st.code(cache_health['cache_paths']['paddlex'])
        if cache_health['paddlex_cache_exists']:
            st.success("✓ Directory exists")
        else:
            st.warning("⚠ Directory does not exist (will be created on first model download)")
        if cache_health['paddlex_cache_writable']:
            st.success("✓ Writable")
        else:
            st.error("✗ Not writable - model downloads will fail!")
    
    st.divider()
    
    # Module Availability
    st.subheader("📚 Module Availability")
    
    modules = {
        "VIN Pipeline": VIN_PIPELINE_AVAILABLE,
        "Multi-Model Evaluator": EVALUATOR_AVAILABLE,
        "Training UI": TRAINING_UI_AVAILABLE,
        "Plotly (Charts)": PLOTLY_AVAILABLE,
        "PIL (Images)": PIL_AVAILABLE,
    }
    
    cols = st.columns(3)
    for i, (module, available) in enumerate(modules.items()):
        with cols[i % 3]:
            if available:
                st.success(f"✓ {module}")
            else:
                st.warning(f"⚠ {module}")
    
    st.divider()
    
    # Last Pipeline Error
    st.subheader("🚨 Last Pipeline Error")
    if st.session_state.get('pipeline_load_error'):
        st.error("Pipeline encountered an error on last load attempt:")
        st.code(st.session_state.pipeline_load_error, language="text")
        
        if st.button("🔄 Clear Error & Retry Pipeline Load"):
            clear_model_cache()
            # Attempt to reload
            use_gpu = st.session_state.get('use_gpu', False)
            pipeline = load_vin_pipeline(_use_gpu=use_gpu)
            if pipeline is not None:
                st.success("✓ Pipeline loaded successfully!")
            else:
                st.error("✗ Pipeline still failing. Check the error above.")
            st.rerun()
    else:
        st.success("✓ No pipeline errors recorded")
    
    st.divider()
    
    # DeepSeek-OCR Section
    st.subheader("🤖 DeepSeek-OCR Model")
    
    # Check DeepSeek availability
    deepseek_available = False
    deepseek_error = None
    try:
        from src.vin_ocr.providers.ocr_providers import DeepSeekOCRProvider
        provider = DeepSeekOCRProvider()
        deepseek_available = provider.is_available
    except Exception as e:
        deepseek_error = str(e)
    
    col1, col2 = st.columns(2)
    with col1:
        if deepseek_available:
            st.success("✓ DeepSeek-OCR dependencies installed")
        else:
            st.warning("⚠️ DeepSeek-OCR dependencies not ready")
            if deepseek_error:
                st.caption(f"Error: {deepseek_error[:100]}...")
    
    with col2:
        st.write("**Required packages:**")
        st.caption("transformers>=4.46.0, torch, einops, addict, easydict")
    
    # Download button
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📥 Download/Initialize DeepSeek Model", help="This will download ~6-7GB model (3B params in BF16) on first run"):
            with st.spinner("Downloading DeepSeek-OCR model (~6-7GB, this may take several minutes)..."):
                try:
                    from src.vin_ocr.providers.ocr_providers import DeepSeekOCRProvider
                    use_gpu = st.session_state.get('use_gpu', False)
                    device = "cuda" if use_gpu else "cpu"
                    if GPU_UTILS_AVAILABLE:
                        gpu = get_gpu_manager()
                        device = gpu.get_device_string(for_torch=True) if use_gpu else "cpu"
                    
                    provider = DeepSeekOCRProvider(device=device)
                    provider.initialize()
                    st.success("✓ DeepSeek-OCR model downloaded and initialized!")
                except Exception as e:
                    st.error(f"Failed to initialize DeepSeek-OCR: {e}")
                    import traceback
                    with st.expander("Error Details"):
                        st.code(traceback.format_exc())
    
    with col2:
        if st.button("📦 Install DeepSeek Dependencies"):
            st.info("Run this command in your terminal:")
            st.code("pip install 'transformers>=4.46.0,<5.0' torch einops addict easydict", language="bash")
    
    st.divider()
    
    # Actions
    st.subheader("🛠️ Actions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🗑️ Clear All Model Caches"):
            clear_model_cache()
            st.success("Model caches cleared!")
            st.rerun()
    
    with col2:
        if st.button("🔄 Test Pipeline Load"):
            with st.spinner("Loading pipeline..."):
                use_gpu = st.session_state.get('use_gpu', False)
                pipeline = load_vin_pipeline(_use_gpu=use_gpu)
            if pipeline is not None:
                st.success("✓ Pipeline loaded successfully!")
            else:
                st.error(f"✗ Pipeline failed to load")
    
    with col3:
        if st.button("📋 Copy Debug Info"):
            # Get GPU info
            gpu_info = "N/A"
            if GPU_UTILS_AVAILABLE:
                gpu = get_gpu_manager()
                gpu_info = gpu.get_status_dict()
            
            debug_info = f"""
VIN OCR System - Debug Info
============================
Python: {sys.version}
Platform: {sys.platform}
PaddlePaddle: {paddle_status}
PaddleOCR: {paddleocr_status}
GPU: {gpu_info}
Cache Health: {cache_health}
Modules: {modules}
Pipeline Error: {st.session_state.get('pipeline_load_error', 'None')}
"""
            st.code(debug_info)
            st.info("Copy the above text for bug reports")


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    """Main application entry point."""
    init_session_state()
    
    # Render sidebar and get selected page
    page = render_sidebar()
    
    # Render selected page
    if page == "🔍 Recognition":
        render_recognition_page()
    elif page == "📊 Batch Evaluation":
        render_batch_evaluation_page()
    elif page == "🎯 Training":
        render_training_page()
    elif page == "📈 Results Dashboard":
        render_results_dashboard()
    elif page == "🔧 System Health":
        render_system_health_page()


if __name__ == "__main__":
    main()
