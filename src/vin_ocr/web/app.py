#!/usr/bin/env python3
"""
VIN OCR Web UI - Simplified Streamlit Application
=================================================

A streamlined web interface for VIN recognition with clear workflow:
1. 📁 Data Management - Single source of truth for all images
2. 🎯 Training - Fine-tuning and hyperparameter optimization  
3. 🔍 Inference - Test your trained models
4. 📊 Results Dashboard - View metrics and export
5. 🔧 System Health - Monitor system status

Usage:
    streamlit run src/vin_ocr/web/app_simple.py --server.port 8501

Author: VIN OCR Pipeline
Date: February 2026
"""

import os
import sys
import json
import shutil
import random
import tempfile
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import streamlit as st
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Project root
_project_root = Path(__file__).parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Optional imports
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Project imports
try:
    from src.vin_ocr.core.vin_utils import extract_vin_from_filename, is_valid_vin
    VIN_UTILS_AVAILABLE = True
except ImportError:
    VIN_UTILS_AVAILABLE = False
    def extract_vin_from_filename(filename):
        import re
        patterns = [
            r'[A-HJ-NPR-Z0-9]{17}',
        ]
        name = Path(filename).stem.upper()
        for pattern in patterns:
            match = re.search(pattern, name)
            if match:
                return match.group(0)
        return None
    
    def is_valid_vin(vin):
        if not vin or len(vin) != 17:
            return False
        invalid_chars = set('IOQ')
        return all(c not in invalid_chars for c in vin.upper())

# DeepSeek OCR Provider imports (existing implementation)
try:
    from src.vin_ocr.providers.ocr_providers import DeepSeekOCRProvider, DeepSeekOCRConfig
    DEEPSEEK_AVAILABLE = True
except ImportError as e:
    DEEPSEEK_AVAILABLE = False
    logger.warning(f"DeepSeek OCR provider not available: {e}")

try:
    from src.vin_ocr.inference.paddle_inference import VINInference
    from src.vin_ocr.inference.onnx_inference import ONNXVINRecognizer
    INFERENCE_AVAILABLE = True
except ImportError:
    INFERENCE_AVAILABLE = False
    VINInference = None
    ONNXVINRecognizer = None

try:
    from src.vin_ocr.training.finetune_paddleocr import VINFineTuner, TRAINING_AVAILABLE
except ImportError:
    TRAINING_AVAILABLE = False
    VINFineTuner = None

try:
    from src.vin_ocr.training.hyperparameter_tuning import HyperparameterTuner
    HYPERPARAMETER_TUNING_AVAILABLE = True
except ImportError:
    HYPERPARAMETER_TUNING_AVAILABLE = False
    HyperparameterTuner = None

# Import training components for subprocess-based training with console output
try:
    from src.vin_ocr.web.training_components import (
        get_global_runner,
        get_global_tracker,
        TrainingRunner,
        ProgressTracker,
    )
    TRAINING_COMPONENTS_AVAILABLE = True
except ImportError:
    TRAINING_COMPONENTS_AVAILABLE = False
    get_global_runner = None
    get_global_tracker = None

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Centralized configuration."""
    ROOT = _project_root
    DATA_DIR = ROOT / "data"
    DATASET_DIR = ROOT / "dataset"
    OUTPUT_DIR = ROOT / "output"
    ONNX_DIR = OUTPUT_DIR / "onnx"
    RESULTS_DIR = ROOT / "results"
    
    # Image extensions
    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    
    @classmethod
    def ensure_dirs(cls):
        """Create necessary directories."""
        for d in [cls.DATA_DIR, cls.DATASET_DIR, cls.OUTPUT_DIR, cls.ONNX_DIR, cls.RESULTS_DIR]:
            d.mkdir(parents=True, exist_ok=True)


# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="VIN OCR System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: bold;
        color: #1E88E5;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 1rem;
        color: white;
    }
    .vin-display {
        font-family: 'Courier New', monospace;
        font-size: 1.8rem;
        font-weight: bold;
        letter-spacing: 0.15rem;
        background: linear-gradient(135deg, #e0f7fa 0%, #b2ebf2 100%);
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        border: 2px solid #00acc1;
    }
    .split-badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: bold;
    }
    .split-train { background-color: #c8e6c9; color: #2e7d32; }
    .split-val { background-color: #fff3e0; color: #ef6c00; }
    .split-test { background-color: #e3f2fd; color: #1565c0; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# SESSION STATE & PERSISTENCE
# =============================================================================

# Persistent storage file
PERSISTENT_STATE_FILE = Config.ROOT / ".streamlit_state.json"


def save_persistent_state():
    """Save current session state to disk for persistence across restarts."""
    try:
        state_to_save = {
            'image_pool': st.session_state.get('image_pool', []),
            'train_images': st.session_state.get('train_images', []),
            'val_images': st.session_state.get('val_images', []),
            'test_images': st.session_state.get('test_images', []),
            'label_files': st.session_state.get('label_files', {}),
            'inference_results': st.session_state.get('inference_results', []),
            'evaluation_results': st.session_state.get('evaluation_results', {}),
            'use_gpu': st.session_state.get('use_gpu', True),
            'last_saved': datetime.now().isoformat(),
        }
        
        with open(PERSISTENT_STATE_FILE, 'w') as f:
            json.dump(state_to_save, f, indent=2)
        
        logger.info(f"✅ State saved to {PERSISTENT_STATE_FILE}")
        return True
    except Exception as e:
        logger.warning(f"⚠️ Could not save persistent state: {e}")
        return False


def load_persistent_state() -> Dict:
    """Load persistent state from disk."""
    try:
        if PERSISTENT_STATE_FILE.exists():
            with open(PERSISTENT_STATE_FILE, 'r') as f:
                state = json.load(f)
            logger.info(f"✅ State loaded from {PERSISTENT_STATE_FILE}")
            return state
    except Exception as e:
        logger.warning(f"⚠️ Could not load persistent state: {e}")
    return {}


def init_session_state():
    """Initialize session state variables with persistence support."""
    
    # Load any persisted state first
    persisted = load_persistent_state()
    
    # Auto-detect GPU availability for default setting
    # Check for NVIDIA CUDA (PaddlePaddle)
    cuda_available = False
    try:
        import paddle
        cuda_available = paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0
    except:
        pass
    
    # Check for Apple Silicon MPS (PyTorch only - PaddlePaddle doesn't support MPS)
    mps_available = False
    try:
        import torch
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            mps_available = True
    except:
        pass
    
    # Note: For PaddleOCR training, only CUDA is supported, not MPS
    # MPS can be used for PyTorch-based models like DeepSeek
    gpu_available = cuda_available  # PaddleOCR only supports CUDA
    
    defaults = {
        # Data management
        'image_pool': [],  # List of {'path': str, 'filename': str, 'vin': str, 'split': str}
        'train_images': [],
        'val_images': [],
        'test_images': [],
        'label_files': {},  # {'train_labels': path, 'val_labels': path, 'test_labels': path}
        
        # Training state
        'training_running': False,
        'training_progress': 0,
        'training_logs': [],
        
        # Results
        'inference_results': [],
        'evaluation_results': {},
        
        # GPU info cache
        'cuda_available': cuda_available,
        'mps_available': mps_available,
        
        # Settings - auto-enable GPU if available
        'use_gpu': gpu_available,
    }
    
    for key, default in defaults.items():
        if key not in st.session_state:
            # Use persisted value if available, otherwise use default
            st.session_state[key] = persisted.get(key, default)
    
    # Validate persisted image paths still exist
    if st.session_state.image_pool:
        valid_pool = []
        for img in st.session_state.image_pool:
            if Path(img.get('path', '')).exists():
                valid_pool.append(img)
        st.session_state.image_pool = valid_pool
        
        # Rebuild split lists from pool
        st.session_state.train_images = [img for img in valid_pool if img.get('split') == 'train']
        st.session_state.val_images = [img for img in valid_pool if img.get('split') == 'val']
        st.session_state.test_images = [img for img in valid_pool if img.get('split') == 'test']


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_image_files(directory: Path) -> List[Path]:
    """Get all image files from a directory."""
    images = []
    if directory.exists():
        for ext in Config.IMAGE_EXTENSIONS:
            images.extend(directory.glob(f"*{ext}"))
            images.extend(directory.glob(f"*{ext.upper()}"))
    return sorted(images)


def load_image_pool_from_directory(directory: Path) -> List[Dict]:
    """Load images from a directory into the pool format."""
    pool = []
    images = get_image_files(directory)
    
    for img_path in images:
        vin = extract_vin_from_filename(img_path.name)
        pool.append({
            'path': str(img_path),
            'filename': img_path.name,
            'vin': vin or '',
            'split': 'unassigned',
            'source': 'directory'
        })
    
    return pool


def save_uploaded_images(uploaded_files, target_dir: Path) -> List[Dict]:
    """Save uploaded files and return pool entries."""
    target_dir.mkdir(parents=True, exist_ok=True)
    pool = []
    
    for uploaded_file in uploaded_files:
        # Save file
        save_path = target_dir / uploaded_file.name
        with open(save_path, 'wb') as f:
            f.write(uploaded_file.getvalue())
        
        # Extract VIN from filename
        vin = extract_vin_from_filename(uploaded_file.name)
        
        pool.append({
            'path': str(save_path),
            'filename': uploaded_file.name,
            'vin': vin or '',
            'split': 'unassigned',
            'source': 'upload'
        })
    
    return pool


def split_images(
    image_pool: List[Dict],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    shuffle: bool = True
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split images into train/val/test sets with NO overlap.
    
    Args:
        image_pool: List of image entries
        train_ratio: Fraction for training (e.g., 0.7)
        val_ratio: Fraction for validation (e.g., 0.15)
        test_ratio: Fraction for testing (e.g., 0.15)
        shuffle: Whether to shuffle before splitting
    
    Returns:
        Tuple of (train_images, val_images, test_images)
    """
    # Validate ratios
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 0.01:
        # Normalize
        train_ratio /= total
        val_ratio /= total
        test_ratio /= total
    
    # Make a copy and optionally shuffle
    pool = image_pool.copy()
    if shuffle:
        random.shuffle(pool)
    
    n = len(pool)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    # Split with no overlap
    train_images = pool[:n_train]
    val_images = pool[n_train:n_train + n_val]
    test_images = pool[n_train + n_val:]
    
    # Update split labels
    for img in train_images:
        img['split'] = 'train'
    for img in val_images:
        img['split'] = 'val'
    for img in test_images:
        img['split'] = 'test'
    
    return train_images, val_images, test_images


def create_label_files(
    train_images: List[Dict],
    val_images: List[Dict],
    output_dir: Path
) -> Dict[str, str]:
    """
    Create PaddleOCR format label files.
    
    Format: image_path\tvin_label
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_labels_path = output_dir / "train_labels.txt"
    val_labels_path = output_dir / "val_labels.txt"
    
    # Write train labels
    with open(train_labels_path, 'w') as f:
        for img in train_images:
            if img['vin']:
                f.write(f"{img['path']}\t{img['vin']}\n")
    
    # Write val labels
    with open(val_labels_path, 'w') as f:
        for img in val_images:
            if img['vin']:
                f.write(f"{img['path']}\t{img['vin']}\n")
    
    return {
        'train_labels': str(train_labels_path),
        'val_labels': str(val_labels_path),
        'train_count': len([i for i in train_images if i['vin']]),
        'val_count': len([i for i in val_images if i['vin']]),
    }


def get_available_models() -> Dict[str, Dict[str, str]]:
    """
    Get all available models for inference including base and fine-tuned models.
    
    Returns dict with model display name -> {"path": str, "type": str}
    Types: "base", "finetuned", "onnx"
    """
    models = {}
    
    # ==========================================================================
    # 1. Base/Pretrained Models (always available if PaddleOCR installed)
    # ==========================================================================
    try:
        import paddle
        # PP-OCRv5 - Latest (2024)
        models["🏆 PP-OCRv5 (Base - General OCR)"] = {"path": "PP-OCRv5", "type": "base"}
        # PP-OCRv4 - Production ready
        models["🥈 PP-OCRv4 (Base - General OCR)"] = {"path": "PP-OCRv4", "type": "base"}
        # PP-OCRv3 - Stable
        models["🥉 PP-OCRv3 (Base - General OCR)"] = {"path": "PP-OCRv3", "type": "base"}
    except ImportError:
        pass
    
    # ==========================================================================
    # 2. Fine-tuned PaddleOCR Models (from output directory)
    # ==========================================================================
    output_dir = Config.OUTPUT_DIR
    if output_dir.exists():
        for model_dir in sorted(output_dir.iterdir(), reverse=True):
            if not model_dir.is_dir():
                continue
            
            # Check for inference directory (exported model)
            inference_dir = model_dir / "inference"
            if inference_dir.exists() and (inference_dir / "inference.pdiparams").exists():
                models[f"🎯 {model_dir.name} (Fine-tuned)"] = {
                    "path": str(inference_dir), 
                    "type": "finetuned"
                }
    
    # ==========================================================================
    # 3. ONNX Models
    # ==========================================================================
    onnx_dir = Config.ONNX_DIR
    if onnx_dir.exists():
        for onnx_file in onnx_dir.glob("*.onnx"):
            models[f"📦 {onnx_file.stem} (ONNX)"] = {
                "path": str(onnx_file),
                "type": "onnx"
            }
    
    return models


def get_model_path(models: Dict, selected_name: str) -> str:
    """Extract model path from models dict."""
    model_info = models.get(selected_name, {})
    if isinstance(model_info, dict):
        return model_info.get("path", "")
    return model_info  # Backwards compatibility


def get_model_type(models: Dict, selected_name: str) -> str:
    """Extract model type from models dict."""
    model_info = models.get(selected_name, {})
    if isinstance(model_info, dict):
        return model_info.get("type", "finetuned")
    # Infer from name for backwards compatibility
    if "📦" in selected_name or "ONNX" in selected_name:
        return "onnx"
    elif "Base" in selected_name or "PP-OCR" in selected_name:
        return "base"
    return "finetuned"


def run_inference(model_path: str, image_path: str, model_type: str = "finetuned") -> Dict[str, Any]:
    """Run inference on a single image with any model type.
    
    Args:
        model_path: Path to model or base model name (PP-OCRv5/v4/v3)
        image_path: Path to image file
        model_type: One of "base", "finetuned", "onnx"
    
    Returns:
        Dict with 'vin', 'confidence', 'raw_text', 'error' keys
    """
    result = {'vin': '', 'confidence': 0.0, 'raw_text': '', 'error': None}
    
    try:
        # ==================================================================
        # ONNX Models
        # ==================================================================
        if model_type == "onnx" or (isinstance(model_path, str) and model_path.endswith('.onnx')):
            if not INFERENCE_AVAILABLE or ONNXVINRecognizer is None:
                result['error'] = "ONNX inference not available"
                return result
            recognizer = ONNXVINRecognizer(model_path)
            res = recognizer.recognize(image_path)
            result['vin'] = res.get('vin', '')
            result['confidence'] = res.get('confidence', 0.0)
            result['raw_text'] = res.get('raw_text', '')
        
        # ==================================================================
        # Base PaddleOCR Models (PP-OCRv3/v4/v5)
        # ==================================================================
        elif model_type == "base" or model_path in ["PP-OCRv5", "PP-OCRv4", "PP-OCRv3"]:
            try:
                from paddleocr import PaddleOCR
                import paddle
                
                # Set device
                use_gpu = st.session_state.get('use_gpu', False)
                if use_gpu:
                    try:
                        paddle.device.set_device('gpu')
                    except:
                        paddle.device.set_device('cpu')
                else:
                    paddle.device.set_device('cpu')
                
                # Initialize PaddleOCR with specific version
                if model_path == "PP-OCRv5":
                    ocr = PaddleOCR(lang='en', ocr_version='PP-OCRv5', show_log=False)
                elif model_path == "PP-OCRv4":
                    ocr = PaddleOCR(lang='en', ocr_version='PP-OCRv4', show_log=False)
                else:  # PP-OCRv3
                    ocr = PaddleOCR(lang='en', ocr_version='PP-OCRv3', show_log=False)
                
                # Run OCR
                ocr_result = ocr.ocr(image_path, cls=True)
                
                # Extract text and confidence
                texts = []
                confidences = []
                
                if ocr_result and ocr_result[0]:
                    for line in ocr_result[0]:
                        if line and len(line) >= 2:
                            text = line[1][0] if isinstance(line[1], (list, tuple)) else str(line[1])
                            conf = line[1][1] if isinstance(line[1], (list, tuple)) and len(line[1]) > 1 else 0.5
                            texts.append(text)
                            confidences.append(conf)
                
                # Combine all detected text
                raw_text = ' '.join(texts)
                result['raw_text'] = raw_text
                
                # Extract VIN from text (17 alphanumeric characters)
                import re
                # Look for VIN pattern (17 chars, no I, O, Q)
                vin_pattern = r'[A-HJ-NPR-Z0-9]{17}'
                matches = re.findall(vin_pattern, raw_text.upper().replace(' ', ''))
                
                if matches:
                    result['vin'] = matches[0]
                else:
                    # Fallback: take first 17 alphanumeric chars
                    cleaned = ''.join(c for c in raw_text.upper() if c.isalnum())
                    result['vin'] = cleaned[:17] if len(cleaned) >= 17 else cleaned
                
                result['confidence'] = sum(confidences) / len(confidences) if confidences else 0.0
                
            except ImportError:
                result['error'] = "PaddleOCR not installed"
            except Exception as e:
                result['error'] = f"PaddleOCR error: {str(e)}"
        
        # ==================================================================
        # Fine-tuned PaddleOCR Models (from inference directory)
        # ==================================================================
        else:
            if not INFERENCE_AVAILABLE or VINInference is None:
                result['error'] = "Paddle inference module not available"
                return result
            engine = VINInference(model_path, use_gpu=st.session_state.get('use_gpu', False))
            res = engine.recognize(image_path)
            result['vin'] = res.get('vin', '')
            result['confidence'] = res.get('confidence', 0.0)
            result['raw_text'] = res.get('raw_text', '')
            
    except Exception as e:
        result['error'] = str(e)
    
    return result


# =============================================================================
# PAGE: DATA MANAGEMENT
# =============================================================================

def render_data_management_page():
    """Render the data management page - Single Source of Truth for all images."""
    st.markdown('<h1 class="main-header">📁 Data Management</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Single source of truth for all VIN images. Upload, organize, and split your dataset.</p>', unsafe_allow_html=True)
    
    Config.ensure_dirs()
    
    # Tabs for different data sources
    tab_upload, tab_existing, tab_split = st.tabs([
        "📤 Upload Images",
        "📂 Load from Directory", 
        "✂️ Split Dataset"
    ])
    
    # =========================================================================
    # TAB: Upload Images
    # =========================================================================
    with tab_upload:
        st.subheader("Upload VIN Images")
        st.info("""
        **📌 Filename Convention:** Include the VIN in the filename for automatic labeling.
        
        Examples:
        - `1-VIN-SAL1A2A40SA606662.jpg` → VIN: SAL1A2A40SA606662
        - `WVWZZZ3CZWE123456.png` → VIN: WVWZZZ3CZWE123456
        """)
        
        uploaded_files = st.file_uploader(
            "Select VIN images",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            accept_multiple_files=True,
            help="Upload multiple VIN images. Ground truth is extracted from filenames."
        )
        
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} files selected")
            
            # Preview
            with st.expander("👁️ Preview Uploaded Images", expanded=False):
                cols = st.columns(min(5, len(uploaded_files)))
                for i, (col, f) in enumerate(zip(cols, uploaded_files[:5])):
                    with col:
                        if PIL_AVAILABLE:
                            img = Image.open(f)
                            st.image(img, caption=f.name[:15] + "...", use_container_width=True)
                            f.seek(0)
                
                if len(uploaded_files) > 5:
                    st.caption(f"... and {len(uploaded_files) - 5} more")
            
            # Add to pool button
            col1, col2 = st.columns([1, 3])
            with col1:
                if st.button("➕ Add to Image Pool", type="primary", use_container_width=True):
                    # Save files and add to pool
                    upload_dir = Config.DATA_DIR / "uploads" / datetime.now().strftime("%Y%m%d_%H%M%S")
                    new_entries = save_uploaded_images(uploaded_files, upload_dir)
                    st.session_state.image_pool.extend(new_entries)
                    save_persistent_state()  # Save to disk
                    st.success(f"✅ Added {len(new_entries)} images to pool")
                    st.rerun()
            
            with col2:
                # Show VIN extraction preview
                sample_vins = []
                for f in uploaded_files[:3]:
                    vin = extract_vin_from_filename(f.name)
                    sample_vins.append(f"{f.name[:25]}... → {vin or 'No VIN found'}")
                st.caption("**VIN extraction preview:**\n" + "\n".join(sample_vins))
    
    # =========================================================================
    # TAB: Load from Directory
    # =========================================================================
    with tab_existing:
        st.subheader("Load Images from Directory")
        
        # Default directories
        default_dirs = [
            Config.DATASET_DIR,
            Config.DATASET_DIR / "train",
            Config.DATASET_DIR / "test",
            Config.DATA_DIR,
        ]
        
        existing_dirs = [str(d) for d in default_dirs if d.exists()]
        
        col1, col2 = st.columns([3, 1])
        with col1:
            if existing_dirs:
                selected_dir = st.selectbox(
                    "Select a directory",
                    options=existing_dirs + ["Custom path..."],
                    help="Choose a directory containing VIN images"
                )
                
                if selected_dir == "Custom path...":
                    selected_dir = st.text_input("Enter custom path:", value=str(Config.DATASET_DIR))
            else:
                selected_dir = st.text_input(
                    "Enter directory path:",
                    value=str(Config.DATASET_DIR),
                    help="Enter the path to a directory containing VIN images"
                )
        
        with col2:
            include_subdirs = st.checkbox("Include subdirectories", value=False)
        
        if selected_dir and selected_dir != "Custom path...":
            dir_path = Path(selected_dir)
            
            if dir_path.exists():
                # Count images
                if include_subdirs:
                    images = list(dir_path.rglob("*.jpg")) + list(dir_path.rglob("*.png"))
                else:
                    images = get_image_files(dir_path)
                
                st.info(f"📊 Found **{len(images)}** images in `{dir_path.name}`")
                
                if images and st.button("📥 Load into Image Pool", type="primary"):
                    new_entries = load_image_pool_from_directory(dir_path)
                    
                    # Avoid duplicates
                    existing_paths = {img['path'] for img in st.session_state.image_pool}
                    new_entries = [e for e in new_entries if e['path'] not in existing_paths]
                    
                    st.session_state.image_pool.extend(new_entries)
                    save_persistent_state()  # Save to disk
                    st.success(f"✅ Loaded {len(new_entries)} new images into pool")
                    st.rerun()
            else:
                st.warning(f"⚠️ Directory not found: {dir_path}")
    
    # =========================================================================
    # TAB: Split Dataset
    # =========================================================================
    with tab_split:
        st.subheader("Configure Train/Val/Test Split")
        
        pool = st.session_state.image_pool
        
        if not pool:
            st.warning("⚠️ No images in pool. Upload images or load from a directory first.")
            return
        
        # Current pool stats
        st.markdown(f"### 📊 Image Pool: **{len(pool)}** images")
        
        # Count images with VINs
        with_vin = len([img for img in pool if img['vin']])
        without_vin = len(pool) - with_vin
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Images", len(pool))
        with col2:
            st.metric("With VIN Label", with_vin, help="Images with VIN extracted from filename")
        with col3:
            st.metric("No VIN Label", without_vin, delta=f"-{without_vin}" if without_vin > 0 else None, delta_color="inverse")
        
        if without_vin > 0:
            st.warning(f"⚠️ {without_vin} images have no VIN label (will be excluded from training)")
        
        st.markdown("---")
        
        # Split configuration
        st.markdown("### ✂️ Configure Split Ratios")
        st.caption("Ratios must sum to 100%. Images are randomly shuffled and split with **NO overlap**.")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            train_pct = st.slider(
                "🏋️ Training %",
                min_value=0,
                max_value=100,
                value=70,
                step=5,
                help="Percentage of images for training"
            )
        
        with col2:
            val_pct = st.slider(
                "🧪 Validation %",
                min_value=0,
                max_value=100,
                value=15,
                step=5,
                help="Percentage of images for validation during training"
            )
        
        with col3:
            test_pct = st.slider(
                "📝 Test %",
                min_value=0,
                max_value=100,
                value=15,
                step=5,
                help="Percentage of images for final testing/evaluation"
            )
        
        # Validation
        total_pct = train_pct + val_pct + test_pct
        
        if total_pct != 100:
            st.error(f"❌ Ratios sum to {total_pct}%, must equal 100%")
        else:
            st.success(f"✅ Split: {train_pct}% train / {val_pct}% val / {test_pct}% test")
            
            # Calculate actual counts
            n_total = with_vin  # Only count images with VINs
            n_train = int(n_total * train_pct / 100)
            n_val = int(n_total * val_pct / 100)
            n_test = n_total - n_train - n_val
            
            st.markdown(f"""
            **Expected split (images with VINs only):**
            - 🏋️ Training: **{n_train}** images
            - 🧪 Validation: **{n_val}** images  
            - 📝 Test: **{n_test}** images
            """)
            
            # Shuffle option
            shuffle = st.checkbox("🔀 Shuffle before splitting", value=True, help="Randomize image order before splitting")
            
            # Apply split button
            if st.button("✂️ Apply Split", type="primary", use_container_width=True):
                # Filter to only images with VINs
                labeled_pool = [img for img in pool if img['vin']]
                
                # Perform split
                train, val, test = split_images(
                    labeled_pool,
                    train_pct / 100,
                    val_pct / 100,
                    test_pct / 100,
                    shuffle=shuffle
                )
                
                # Update session state
                st.session_state.train_images = train
                st.session_state.val_images = val
                st.session_state.test_images = test
                
                # Create label files
                label_dir = Config.DATA_DIR / "labels"
                labels = create_label_files(train, val, label_dir)
                
                st.success(f"""
                ✅ **Split Complete!**
                - Training: {len(train)} images → `{labels['train_labels']}`
                - Validation: {len(val)} images → `{labels['val_labels']}`
                - Test: {len(test)} images (for evaluation)
                """)
                
                st.session_state['label_files'] = labels
                
                # Save state persistently
                save_persistent_state()
                
                st.rerun()
        
        # Show current split status
        st.markdown("---")
        st.markdown("### 📋 Current Split Status")
        
        train_count = len(st.session_state.train_images)
        val_count = len(st.session_state.val_images)
        test_count = len(st.session_state.test_images)
        
        if train_count + val_count + test_count > 0:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🏋️ Training", train_count)
            with col2:
                st.metric("🧪 Validation", val_count)
            with col3:
                st.metric("📝 Test", test_count)
            
            # Show sample from each split
            with st.expander("👁️ Preview Split", expanded=False):
                tab1, tab2, tab3 = st.tabs(["Train", "Val", "Test"])
                
                with tab1:
                    if st.session_state.train_images:
                        df = pd.DataFrame(st.session_state.train_images[:10])
                        if not df.empty:
                            df = df.assign(
                                filename=df["filename"].astype(str),
                                vin=df["vin"].astype(str),
                            )
                        st.dataframe(df[['filename', 'vin']], use_container_width=True)
                
                with tab2:
                    if st.session_state.val_images:
                        df = pd.DataFrame(st.session_state.val_images[:10])
                        if not df.empty:
                            df = df.assign(
                                filename=df["filename"].astype(str),
                                vin=df["vin"].astype(str),
                            )
                        st.dataframe(df[['filename', 'vin']], use_container_width=True)
                
                with tab3:
                    if st.session_state.test_images:
                        df = pd.DataFrame(st.session_state.test_images[:10])
                        if not df.empty:
                            df = df.assign(
                                filename=df["filename"].astype(str),
                                vin=df["vin"].astype(str),
                            )
                        st.dataframe(df[['filename', 'vin']], use_container_width=True)
        else:
            st.info("No split configured yet. Use the sliders above and click 'Apply Split'.")
    
    # Clear pool button
    st.markdown("---")
    if st.button("🗑️ Clear Image Pool", help="Remove all images from the pool"):
        st.session_state.image_pool = []
        st.session_state.train_images = []
        st.session_state.val_images = []
        st.session_state.test_images = []
        st.session_state.label_files = {}
        save_persistent_state()  # Save cleared state
        st.success("✅ Image pool cleared")
        st.rerun()


# =============================================================================
# PAGE: TRAINING
# =============================================================================

def render_training_page():
    """Render the training page."""
    st.markdown('<h1 class="main-header">🎯 Model Training</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Train or fine-tune VIN recognition models on your dataset.</p>', unsafe_allow_html=True)
    
    # Check prerequisites - data first
    if not st.session_state.train_images:
        st.warning("⚠️ No training data configured. Go to **📁 Data Management** to set up your dataset first.")
        if st.button("Go to Data Management →"):
            st.session_state.current_page = "📁 Data Management"
            st.rerun()
        return
    
    # Check if training components are available
    if not TRAINING_COMPONENTS_AVAILABLE:
        st.error("""
        ❌ **Training components not available.**
        
        This usually means the training_components module could not be imported.
        """)
        return
    
    # Get global runner and tracker
    runner = get_global_runner()
    tracker = get_global_tracker()
    state = tracker.get_state()
    
    # Show current dataset status
    st.markdown("### 📊 Dataset Status")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🏋️ Training Images", len(st.session_state.train_images))
    with col2:
        st.metric("🧪 Validation Images", len(st.session_state.val_images))
    with col3:
        label_files = st.session_state.get('label_files', {})
        st.metric("📄 Labels Generated", "✅ Yes" if label_files else "❌ No")
    
    st.markdown("---")
    
    # Training tabs
    tab_finetune, tab_hyperopt, tab_console = st.tabs([
        "🔧 Fine-Tuning",
        "⚙️ Hyperparameter Tuning",
        "🖥️ Training Console"
    ])
    
    # =========================================================================
    # TAB: Fine-Tuning
    # =========================================================================
    with tab_finetune:
        st.subheader("Model Fine-Tuning")
        
        st.info("""
        **Fine-tuning** adapts a pre-trained model to VIN-specific data.
        - Faster than training from scratch
        - Requires 500-5,000 images typically
        - Training time: 1-4 hours on GPU
        """)
        
        # Model Selection
        st.markdown("### 🤖 Select Base Model to Fine-tune")
        
        model_options = {
            "🏆 PP-OCRv5 (Recommended)": {
                "key": "PP-OCRv5",
                "type": "paddleocr",
                "description": "Latest 2024 PaddleOCR model with best accuracy. Uses PPHGNetV2 backbone.",
                "accuracy": "⭐⭐⭐⭐⭐",
                "speed": "⚡⚡⚡⚡",
                "size": "~15MB"
            },
            "🥈 PP-OCRv4": {
                "key": "PP-OCRv4",
                "type": "paddleocr",
                "description": "Production-ready 2023 model. Uses SVTR_LCNet architecture.",
                "accuracy": "⭐⭐⭐⭐",
                "speed": "⚡⚡⚡⚡⚡",
                "size": "~12MB"
            },
            "🥉 PP-OCRv3": {
                "key": "PP-OCRv3",
                "type": "paddleocr",
                "description": "Stable 2022 model. Good balance of speed and accuracy.",
                "accuracy": "⭐⭐⭐⭐",
                "speed": "⚡⚡⚡⚡⚡",
                "size": "~10MB"
            },
            "🤖 DeepSeek-OCR": {
                "key": "deepseek",
                "type": "deepseek",
                "description": "Vision-language model. Best for complex/degraded images.",
                "accuracy": "⭐⭐⭐⭐⭐",
                "speed": "⚡⚡",
                "size": "~1.5GB"
            },
        }
        
        selected_model = st.selectbox(
            "Select Model",
            options=list(model_options.keys()),
            index=0,
            help="Select the pre-trained model to fine-tune on your VIN data"
        )
        
        model_info = model_options[selected_model]
        
        # Show model info card
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"**Accuracy**")
            st.caption(model_info['accuracy'])
        with col2:
            st.markdown(f"**Speed**")
            st.caption(model_info['speed'])
        with col3:
            st.markdown(f"**Size**")
            st.caption(model_info['size'])
        with col4:
            st.markdown(f"**Type**")
            st.caption(model_info['type'].upper())
        
        st.caption(f"📝 {model_info['description']}")
        
        st.markdown("---")
        
        # Model naming section
        st.markdown("### 📝 Model Naming")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            model_name = st.text_input(
                "Model Name",
                value=f"vin_{model_info['key'].lower()}_{datetime.now().strftime('%Y%m%d')}",
                help="Give your model a memorable name to find it easily in Results Dashboard"
            )
        with col2:
            st.caption("**Naming Tips:**")
            st.caption("• Use descriptive names")
            st.caption("• Include dataset info")
            st.caption("• Example: `vin_v5_1000imgs`")
        
        output_dir = Config.OUTPUT_DIR / model_name
        st.caption(f"📁 Output: `{output_dir}`")
        
        st.markdown("---")
        
        # Training configuration
        with st.expander("⚙️ Training Configuration", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                epochs = st.number_input(
                    "Training Epochs",
                    min_value=1,
                    max_value=500,
                    value=50,
                    help="Number of complete passes through the training data"
                )
                
                batch_size = st.selectbox(
                    "Batch Size",
                    options=[4, 8, 16, 32, 64],
                    index=2,
                    help="Number of images per training step. Lower if running out of memory."
                )
            
            with col2:
                learning_rate = st.select_slider(
                    "Learning Rate",
                    options=[1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
                    value=1e-4,
                    format_func=lambda x: f"{x:.0e}",
                    help="Learning rate for optimizer"
                )
                
                use_gpu = st.checkbox(
                    "🚀 Use GPU",
                    value=st.session_state.use_gpu,
                    help="Enable GPU acceleration if available"
                )
                st.session_state.use_gpu = use_gpu
        
        # Get label files from session state
        label_files = st.session_state.get('label_files', {})
        
        # Start training button
        col_btn1, col_btn2 = st.columns(2)
        
        is_training = runner.is_running
        
        with col_btn1:
            if st.button("� Start Fine-Tuning", type="primary", use_container_width=True, disabled=is_training):
                if not label_files:
                    st.error("❌ Label files not found. Go to Data Management and apply the split first.")
                elif model_info['type'] == 'deepseek':
                    # Use DeepSeek training
                    config = {
                        'epochs': epochs,
                        'batch_size': batch_size,
                        'learning_rate': learning_rate,
                        'output_dir': str(output_dir),
                        'train_data_path': label_files.get('train_labels'),
                        'val_data_path': label_files.get('val_labels'),
                        'data_dir': str(Config.DATA_DIR),
                        'use_lora': True,
                    }
                    try:
                        runner.start_deepseek_finetuning(config)
                        st.success(f"✅ DeepSeek fine-tuning started!")
                        st.info("📊 View progress in the **Training Console** tab.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Failed to start training: {e}")
                else:
                    # Use PaddleOCR training via TrainingRunner
                    config = {
                        'epochs': epochs,
                        'batch_size': batch_size,
                        'learning_rate': learning_rate,
                        'output_dir': str(output_dir),
                        'train_data_dir': str(Config.DATA_DIR),
                        'train_labels': label_files.get('train_labels'),
                        'val_data_dir': str(Config.DATA_DIR),
                        'val_labels': label_files.get('val_labels'),
                        'device': 'cuda' if use_gpu else 'cpu',
                        'architecture': model_info['key'],
                    }
                    
                    try:
                        runner.start_paddleocr_finetuning(config)
                        st.success(f"✅ Fine-tuning started with **{selected_model}**!")
                        st.info("📊 View progress in the **Training Console** tab.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Failed to start training: {e}")
        
        with col_btn2:
            if st.button("🛑 Stop Training", use_container_width=True, disabled=not is_training):
                runner.stop()
                st.warning("⚠️ Training stopped by user")
                st.rerun()
        
        # Show brief status
        if is_training:
            st.success(f"🟢 **Training in progress** - View details in **Training Console** tab")
    
    # =========================================================================
    # TAB: Hyperparameter Tuning
    # =========================================================================
    with tab_hyperopt:
        st.subheader("Hyperparameter Tuning with Optuna")
        
        st.info("""
        **Hyperparameter tuning** automatically finds optimal training parameters using Optuna.
        - Tests multiple configurations automatically
        - Uses Bayesian optimization for efficiency
        - Requires more time but can significantly improve results
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            n_trials = st.number_input(
                "Number of Trials",
                min_value=5,
                max_value=100,
                value=20,
                help="Number of different parameter combinations to try"
            )
            
            tuning_model = st.selectbox(
                "Model to Tune",
                options=["paddleocr", "deepseek"],
                index=0,
                help="Model type for hyperparameter tuning"
            )
        
        with col2:
            st.markdown("**Search Space:**")
            st.caption("""
            - Learning rate: 1e-5 to 1e-3
            - Batch size: 4, 8, 16, 32
            - Optimizer: Adam, SGD, AdamW
            """)
        
        is_tuning = runner.is_running
        
        if st.button("� Start Hyperparameter Search", type="primary", use_container_width=True, disabled=is_tuning):
            label_files = st.session_state.get('label_files', {})
            if not label_files:
                st.error("❌ Label files not found. Go to Data Management and apply the split first.")
            else:
                config = {
                    'model_type': tuning_model,
                    'n_trials': n_trials,
                    'train_data_dir': str(Config.DATA_DIR),
                    'train_labels': label_files.get('train_labels'),
                    'val_data_dir': str(Config.DATA_DIR),
                    'val_labels': label_files.get('val_labels'),
                    'output_dir': str(Config.OUTPUT_DIR / 'hyperparameter_tuning'),
                    'device': 'cuda' if st.session_state.use_gpu else 'cpu',
                }
                
                try:
                    runner.start_hyperparameter_tuning(config)
                    st.success("✅ Hyperparameter tuning started!")
                    st.info("📊 View progress in the **Training Console** tab.")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Failed to start tuning: {e}")
    
    # =========================================================================
    # TAB: Training Console
    # =========================================================================
    with tab_console:
        render_training_console(runner, tracker)


def render_training_console(runner, tracker):
    """Render the real-time training console output."""
    st.subheader("🖥️ Real-Time Training Console")
    
    state = tracker.get_state()
    
    # Check if training is running
    if not state.is_running:
        # Check for completion by looking at history
        has_completed = state.history and not state.error and state.current_epoch > 0
        if has_completed:
            st.success("✅ Training completed!")
            if state.history:
                st.info(f"Last message: {state.history[-1].message[:200] if state.history else 'N/A'}")
        elif state.error:
            st.error(f"❌ Training error: {state.error}")
        else:
            st.info("ℹ️ No training in progress. Start training from the Fine-Tuning or Hyperparameter Tuning tab.")
        return
    
    # Training is active
    st.markdown("---")
    
    # Status indicator
    model_type = runner.get_current_model_type() or "Unknown"
    status = "🟢 **Training in Progress**"
    status += f" | **Model:** {model_type}"
    if state.history:
        last_update = state.history[-1].timestamp
        status += f" _(last update: {last_update[-12:]})_"
    st.markdown(status)
    
    # Progress bar
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
    
    # Get console log path
    console_log_path = runner.get_current_console_log()
    
    # Console output display
    st.markdown("#### 📝 Live Console Output")
    
    if console_log_path and Path(console_log_path).exists():
        try:
            with open(console_log_path, 'r') as f:
                lines = f.readlines()
                # Show last 40 lines for real-time view
                recent_lines = lines[-40:] if len(lines) > 40 else lines
                console_text = "".join(recent_lines)
                
                # Display in a scrollable code block
                st.code(console_text, language="text", line_numbers=False)
                
                # Show total line count
                st.caption(f"📜 Showing last {len(recent_lines)} of {len(lines)} lines | Log: `{Path(console_log_path).name}`")
        except Exception as e:
            st.warning(f"⚠️ Waiting for console output... ({e})")
    else:
        st.info("⏳ Waiting for training output...")
    
    # Controls row
    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([2, 1, 1])
    
    with col_ctrl1:
        refresh_rate = st.selectbox(
            "🔄 Auto-refresh rate (seconds)", 
            options=[1, 2, 3, 5, 10],
            index=1,
            key="console_refresh_rate",
            help="How often to refresh the console output and progress"
        )
    with col_ctrl2:
        auto_refresh = st.checkbox("Auto-refresh", value=True, key="auto_refresh_console")
    with col_ctrl3:
        if st.button("🔃 Refresh Now", key="manual_refresh"):
            st.rerun()
    
    # Stop button
    if st.button("⏹️ Stop Training", type="secondary", key="stop_training_console"):
        runner.stop()
        st.warning("⏹️ Stop requested...")
        st.rerun()
    
    # Auto-refresh logic
    if auto_refresh:
        import time
        time.sleep(refresh_rate)
        st.rerun()


# =============================================================================
# PAGE: INFERENCE
# =============================================================================

def render_inference_page():
    """Render the inference/testing page."""
    st.markdown('<h1 class="main-header">🔍 Inference & Testing</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Test your trained models on new images or evaluate on the test set.</p>', unsafe_allow_html=True)
    
    # Get available models
    models = get_available_models()
    
    if not models:
        st.warning("⚠️ No trained models found. Train a model first in the Training page.")
        return
    
    # Tabs
    tab_single, tab_batch, tab_evaluate = st.tabs([
        "🖼️ Single Image",
        "📦 Batch Inference",
        "📊 Evaluate Test Set"
    ])
    
    # =========================================================================
    # TAB: Single Image
    # =========================================================================
    with tab_single:
        st.subheader("Test on a Single Image")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Model selection
            selected_model = st.selectbox(
                "Select Model",
                options=list(models.keys()),
                help="Choose a trained model for inference"
            )
            model_path = get_model_path(models, selected_model)
            model_type = get_model_type(models, selected_model)
            
            # Image upload
            uploaded_file = st.file_uploader(
                "Upload a VIN image",
                type=['jpg', 'jpeg', 'png'],
                key="single_inference_upload"
            )
            
            if uploaded_file and PIL_AVAILABLE:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_container_width=True)
        
        with col2:
            st.subheader("Result")
            
            if uploaded_file:
                # Save to temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name
                
                # Run inference
                with st.spinner("Processing..."):
                    result = run_inference(model_path, tmp_path, model_type)
                
                # Clean up
                try:
                    os.unlink(tmp_path)
                except:
                    pass
                
                # Display result
                if result['error']:
                    st.error(f"❌ Error: {result['error']}")
                else:
                    st.markdown(f'<div class="vin-display">{result["vin"] or "No VIN detected"}</div>', unsafe_allow_html=True)
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Confidence", f"{result['confidence']:.1%}")
                    with col_b:
                        st.metric("Length", f"{len(result['vin'])}/17")
                    
                    # Ground truth comparison
                    gt_vin = extract_vin_from_filename(uploaded_file.name)
                    if gt_vin:
                        st.markdown("---")
                        st.markdown("**Ground Truth Comparison:**")
                        if result['vin'] == gt_vin:
                            st.success(f"✅ MATCH - Predicted VIN matches ground truth: `{gt_vin}`")
                        else:
                            st.error(f"❌ MISMATCH")
                            st.caption(f"Predicted: `{result['vin']}`")
                            st.caption(f"Expected:  `{gt_vin}`")
            else:
                st.info("👆 Upload an image to test")
    
    # =========================================================================
    # TAB: Batch Inference
    # =========================================================================
    with tab_batch:
        st.subheader("Batch Inference")
        
        # Model selection
        selected_model = st.selectbox(
            "Select Model",
            options=list(models.keys()),
            key="batch_model_select"
        )
        model_path = get_model_path(models, selected_model)
        model_type = get_model_type(models, selected_model)
        
        # Upload multiple files
        uploaded_files = st.file_uploader(
            "Upload VIN images",
            type=['jpg', 'jpeg', 'png'],
            accept_multiple_files=True,
            key="batch_inference_upload"
        )
        
        if uploaded_files:
            st.info(f"📊 {len(uploaded_files)} images ready for processing")
            
            if st.button("🚀 Run Batch Inference", type="primary"):
                results = []
                progress = st.progress(0)
                status = st.empty()
                
                for i, f in enumerate(uploaded_files):
                    status.text(f"Processing {i+1}/{len(uploaded_files)}: {f.name}")
                    
                    # Save to temp
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                        tmp.write(f.getvalue())
                        tmp_path = tmp.name
                    
                    # Run inference
                    result = run_inference(model_path, tmp_path, model_type)
                    result['filename'] = f.name
                    result['ground_truth'] = extract_vin_from_filename(f.name) or ''
                    result['correct'] = result['vin'] == result['ground_truth'] if result['ground_truth'] else None
                    results.append(result)
                    
                    # Cleanup
                    try:
                        os.unlink(tmp_path)
                    except:
                        pass
                    
                    progress.progress((i + 1) / len(uploaded_files))
                
                status.text("✅ Complete!")
                
                # Show results
                df = pd.DataFrame(results)
                st.dataframe(df[['filename', 'vin', 'ground_truth', 'correct', 'confidence']], use_container_width=True)
                
                # Summary metrics
                correct = df['correct'].sum()
                total_with_gt = df['correct'].notna().sum()
                
                if total_with_gt > 0:
                    accuracy = correct / total_with_gt * 100
                    st.success(f"**Accuracy:** {accuracy:.1f}% ({correct}/{total_with_gt} correct)")
                
                # Save results
                st.session_state.inference_results = results
    
    # =========================================================================
    # TAB: Evaluate Test Set
    # =========================================================================
    with tab_evaluate:
        st.subheader("Evaluate on Test Set")
        
        test_images = st.session_state.test_images
        
        if not test_images:
            st.warning("⚠️ No test set configured. Go to **📁 Data Management** and apply a split first.")
            return
        
        st.info(f"📊 Test set contains **{len(test_images)}** images")
        
        # Model selection
        selected_model = st.selectbox(
            "Select Model",
            options=list(models.keys()),
            key="eval_model_select"
        )
        model_path = get_model_path(models, selected_model)
        model_type = get_model_type(models, selected_model)
        
        if st.button("📊 Run Evaluation", type="primary", use_container_width=True):
            results = []
            progress = st.progress(0)
            status = st.empty()
            
            for i, img in enumerate(test_images):
                status.text(f"Evaluating {i+1}/{len(test_images)}: {img['filename']}")
                
                result = run_inference(model_path, img['path'], model_type)
                result['filename'] = img['filename']
                result['ground_truth'] = img['vin']
                result['correct'] = result['vin'] == result['ground_truth']
                
                # Character-level accuracy
                if result['ground_truth']:
                    chars_correct = sum(1 for a, b in zip(result['vin'][:17], result['ground_truth'][:17]) if a == b)
                    result['char_accuracy'] = chars_correct / 17 * 100
                else:
                    result['char_accuracy'] = 0
                
                results.append(result)
                progress.progress((i + 1) / len(test_images))
            
            status.text("✅ Evaluation complete!")
            
            # Calculate metrics
            df = pd.DataFrame(results)
            
            exact_match = df['correct'].mean() * 100
            char_accuracy = df['char_accuracy'].mean()
            avg_confidence = df['confidence'].mean() * 100
            
            st.markdown("### 📈 Evaluation Results")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Exact Match Accuracy", f"{exact_match:.1f}%")
            with col2:
                st.metric("Character Accuracy", f"{char_accuracy:.1f}%")
            with col3:
                st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
            
            # Save to session state
            st.session_state.evaluation_results = {
                'model': selected_model,
                'exact_match': exact_match,
                'char_accuracy': char_accuracy,
                'avg_confidence': avg_confidence,
                'results': results,
                'timestamp': datetime.now().isoformat()
            }
            
            # Show detailed results
            with st.expander("📋 Detailed Results"):
                st.dataframe(
                    df[['filename', 'vin', 'ground_truth', 'correct', 'char_accuracy', 'confidence']],
                    use_container_width=True
                )
            
            # Export option
            if st.button("💾 Export Results to CSV"):
                csv_path = Config.RESULTS_DIR / f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                df.to_csv(csv_path, index=False)
                st.success(f"✅ Results exported to `{csv_path}`")


# =============================================================================
# PAGE: RESULTS DASHBOARD
# =============================================================================

def get_training_runs() -> List[Dict]:
    """Get all training runs with their metrics from output directory."""
    runs = []
    output_dir = Config.OUTPUT_DIR
    
    if not output_dir.exists():
        return runs
    
    for run_dir in sorted(output_dir.iterdir(), reverse=True):
        if not run_dir.is_dir():
            continue
        
        # Look for training metrics file
        metrics_file = run_dir / "training_metrics.json"
        progress_file = run_dir / "training_progress.json"
        
        run_info = {
            'name': run_dir.name,
            'path': str(run_dir),
            'timestamp': None,
            'has_metrics': False,
            'has_inference': (run_dir / "inference").exists(),
        }
        
        # Parse timestamp from directory name
        # Format: paddleocr_finetune_20260202_142552 or finetune_20260203_161646
        import re
        ts_match = re.search(r'(\d{8})_(\d{6})', run_dir.name)
        if ts_match:
            date_str = ts_match.group(1)
            time_str = ts_match.group(2)
            try:
                run_info['timestamp'] = datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
            except:
                pass
        
        # Load metrics if available
        if metrics_file.exists():
            try:
                with open(metrics_file, 'r') as f:
                    run_info['metrics'] = json.load(f)
                    run_info['has_metrics'] = True
            except:
                pass
        elif progress_file.exists():
            try:
                with open(progress_file, 'r') as f:
                    run_info['progress'] = json.load(f)
            except:
                pass
        
        runs.append(run_info)
    
    return runs


def render_results_dashboard():
    """Render the results dashboard."""
    st.markdown('<h1 class="main-header">📊 Results Dashboard</h1>', unsafe_allow_html=True)
    
    # Get all training runs
    runs = get_training_runs()
    runs_with_metrics = [r for r in runs if r.get('has_metrics')]
    
    if not runs_with_metrics:
        st.info("📭 No training results found. Train a model in the **🎯 Training** page first.")
        if runs:
            with st.expander("📁 Available Output Directories"):
                for run in runs[:10]:
                    st.caption(f"• {run['name']}")
        return
    
    # Sidebar-style run selector
    col_select, col_info = st.columns([2, 3])
    with col_select:
        run_names = [r['name'] for r in runs_with_metrics]
        selected_run_name = st.selectbox("Select Model", options=run_names, index=0)
    
    selected_run = next((r for r in runs_with_metrics if r['name'] == selected_run_name), None)
    if not selected_run:
        return
    
    metrics = selected_run.get('metrics', {})
    metadata = metrics.get('metadata', {})
    training_config = metrics.get('training_config', {})
    dataset_info = metrics.get('dataset_info', {})
    eval_metrics = metrics.get('evaluation_metrics', {})
    training_results = metrics.get('training_results', {})
    image_level = eval_metrics.get('image_level', {})
    char_level = eval_metrics.get('character_level', {})
    
    with col_info:
        ts = selected_run.get('timestamp')
        ts_str = ts.strftime("%b %d, %Y %H:%M") if ts else "N/A"
        train_n = dataset_info.get('train_samples', 0)
        val_n = dataset_info.get('val_samples', 0)
        st.caption(f"📅 {ts_str} • 🏋️ {train_n} train / {val_n} val images")
    
    # Tabs
    tab_overview, tab_detailed, tab_compare, tab_export = st.tabs([
        "Overview", "Detailed", "Compare", "Export"
    ])
    
    # =========================================================================
    # TAB: Overview - Compact Professional Layout
    # =========================================================================
    with tab_overview:
        # Key Metrics - Compact Cards Style
        st.markdown("##### 🎯 Performance Summary")
        
        correct = image_level.get('exact_match_count', 0)
        failed = image_level.get('incorrect_predictions', 0)
        img_acc = image_level.get('exact_match_accuracy', 0) * 100
        char_acc = char_level.get('character_accuracy', 0) * 100
        f1_micro = char_level.get('f1_micro', 0)
        f1_macro = char_level.get('f1_macro', 0)
        total_chars = char_level.get('total_characters', 0)
        avg_conf = eval_metrics.get('avg_confidence', 0) * 100
        
        # Two-row compact metrics
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("Correct", correct)
        col2.metric("Failed", failed)
        col3.metric("Img Acc", f"{img_acc:.1f}%")
        col4.metric("Char Acc", f"{char_acc:.1f}%")
        col5.metric("F1-μ", f"{f1_micro:.3f}")
        col6.metric("F1-M", f"{f1_macro:.3f}")
        
        st.caption(f"Total: {total_chars:,} characters • Avg Confidence: {avg_conf:.1f}%")
        
        st.markdown("---")
        
        # Training Config - Compact Table
        st.markdown("##### ⚙️ Training Configuration")
        
        # Get device info from metadata or training_config
        device_info = metadata.get('device', {})
        if isinstance(device_info, dict):
            device_str = device_info.get('device_name', device_info.get('device', 'N/A'))
        else:
            device_str = training_config.get('device', 'N/A')
        if device_str:
            device_str = device_str.upper() if device_str.lower() in ['cpu', 'gpu'] else device_str
        
        config_data = {
            "Epochs": training_config.get('epochs', '-'),
            "Batch": training_config.get('batch_size', '-'),
            "LR": f"{training_config.get('learning_rate', 0):.0e}",
            "Optimizer": training_config.get('optimizer', '-'),
            "Device": f"⚡{device_str}" if device_str and device_str != 'N/A' else '-',
            "Time": f"{training_results.get('total_time_hours', 0):.1f}h",
        }
        
        cols = st.columns(len(config_data))
        for i, (k, v) in enumerate(config_data.items()):
            cols[i].caption(f"**{k}**")
            cols[i].caption(v)
        
        st.markdown("---")
        
        # Quick Sample Results
        st.markdown("##### 📋 Sample Predictions")
        samples = eval_metrics.get('sample_results', [])[:8]
        if samples:
            sample_df = pd.DataFrame([{
                'Ground Truth': s.get('ground_truth', ''),
                'Prediction': s.get('prediction', ''),
                '✓': '✅' if s.get('exact_match') else '❌',
                'Char%': f"{s.get('char_accuracy', 0)*100:.0f}%",
                'Conf': f"{s.get('confidence', 0)*100:.0f}%",
            } for s in samples])
            st.dataframe(sample_df, use_container_width=True, hide_index=True, height=300)
    
    # =========================================================================
    # TAB: Detailed Metrics
    # =========================================================================
    with tab_detailed:
        st.markdown(f"##### 🔍 {selected_run_name}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Position Accuracy
            st.markdown("###### Position Accuracy")
            position_acc = eval_metrics.get('position_accuracy', {})
            if position_acc:
                pos_data = []
                for i in range(1, 18):
                    acc = position_acc.get(f'position_{i}', 0) * 100
                    pos_data.append({'Pos': i, 'Acc%': f"{acc:.0f}", 'Bar': '█' * int(acc/10)})
                
                if PLOTLY_AVAILABLE:
                    import plotly.graph_objects as go
                    positions = [f"P{i}" for i in range(1, 18)]
                    accs = [position_acc.get(f'position_{i}', 0) * 100 for i in range(1, 18)]
                    colors = ['#28a745' if a >= 80 else '#ffc107' if a >= 50 else '#dc3545' for a in accs]
                    fig = go.Figure(data=[go.Bar(x=positions, y=accs, marker_color=colors)])
                    fig.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20),
                                      yaxis_range=[0, 100], title_text="Accuracy by Position")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.dataframe(pd.DataFrame(pos_data), hide_index=True, height=200)
        
        with col2:
            # Edit Distance
            st.markdown("###### Edit Distance Distribution")
            edit_dist = eval_metrics.get('edit_distance_distribution', {})
            if edit_dist:
                dist_data = [
                    {"Distance": "0 (Perfect)", "Count": edit_dist.get('zero_edit', 0)},
                    {"Distance": "1 char off", "Count": edit_dist.get('one_edit', 0)},
                    {"Distance": "2 chars off", "Count": edit_dist.get('two_edit', 0)},
                    {"Distance": "3+ chars off", "Count": edit_dist.get('three_plus_edit', 0)},
                ]
                st.dataframe(pd.DataFrame(dist_data), hide_index=True, use_container_width=True)
                st.caption(f"Mean: {edit_dist.get('mean', 0):.1f} | Range: {edit_dist.get('min', 0)}-{edit_dist.get('max', 0)}")
        
        # Character Confusions
        st.markdown("###### Top Character Confusions")
        confusions = eval_metrics.get('top_confusions', [])[:10]
        if confusions:
            conf_df = pd.DataFrame(confusions)
            conf_df.columns = ['Predicted', 'Actual', 'Count']
            st.dataframe(conf_df, hide_index=True, use_container_width=True, height=200)
    
    # =========================================================================
    # TAB: Compare Models
    # =========================================================================
    with tab_compare:
        if len(runs_with_metrics) < 2:
            st.info("Train more models to compare results.")
            return
        
        compare_runs = st.multiselect("Select models to compare", 
                                       options=[r['name'] for r in runs_with_metrics],
                                       default=[runs_with_metrics[0]['name']])
        
        if compare_runs:
            comparison = []
            for run_name in compare_runs:
                run = next((r for r in runs_with_metrics if r['name'] == run_name), None)
                if run and run.get('metrics'):
                    m = run['metrics']
                    em = m.get('evaluation_metrics', {})
                    il = em.get('image_level', {})
                    cl = em.get('character_level', {})
                    cfg = m.get('training_config', {})
                    ds = m.get('dataset_info', {})
                    
                    # Device info
                    device = cfg.get('device', 'N/A')
                    device_name = cfg.get('device_name', '')
                    device_str = f"{device.upper()}"
                    if device_name:
                        device_str = f"{device_name[:15]}"
                    
                    comparison.append({
                        'Model': run_name[:25],
                        'Device': device_str,
                        'Train': ds.get('train_samples', 0),
                        'Epochs': cfg.get('epochs', 0),
                        'Img Acc%': f"{il.get('exact_match_accuracy', 0)*100:.1f}",
                        'Char Acc%': f"{cl.get('character_accuracy', 0)*100:.1f}",
                        'F1-μ': f"{cl.get('f1_micro', 0):.3f}",
                        'F1-M': f"{cl.get('f1_macro', 0):.3f}",
                    })
            
            st.dataframe(pd.DataFrame(comparison), hide_index=True, use_container_width=True)
    
    # =========================================================================
    # TAB: Export
    # =========================================================================
    with tab_export:
        st.markdown(f"##### Export: {selected_run_name}")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📊 Export CSV", use_container_width=True):
                samples = eval_metrics.get('sample_results', [])
                if samples:
                    df = pd.DataFrame(samples)
                    csv_path = Config.RESULTS_DIR / f"{selected_run_name}_results.csv"
                    csv_path.parent.mkdir(parents=True, exist_ok=True)
                    df.to_csv(csv_path, index=False)
                    st.success(f"✅ Saved to `{csv_path.name}`")
        
        with col2:
            if st.button("📄 Export JSON", use_container_width=True):
                if metrics:
                    json_path = Config.RESULTS_DIR / f"{selected_run_name}_full.json"
                    json_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(json_path, 'w') as f:
                        json.dump(metrics, f, indent=2, default=str)
                    st.success(f"✅ Saved to `{json_path.name}`")
        
        st.caption(f"📁 Source: `output/{selected_run_name}/training_metrics.json`")


# =============================================================================
# PAGE: SYSTEM HEALTH
# =============================================================================

def render_system_health():
    """Render system health page."""
    st.markdown('<h1 class="main-header">🔧 System Health</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Monitor system status, dependencies, and resources.</p>', unsafe_allow_html=True)
    
    # System Info
    st.markdown("### 💻 System Information")
    
    col1, col2, col3 = st.columns(3)
    
    import platform
    
    with col1:
        st.markdown("**Platform**")
        st.text(f"OS: {platform.system()} {platform.release()}")
        st.text(f"Python: {platform.python_version()}")
        st.text(f"Architecture: {platform.machine()}")
    
    with col2:
        st.markdown("**Dependencies**")
        deps = {
            "PaddlePaddle": False,
            "PaddleOCR": False,
            "ONNX Runtime": False,
            "Streamlit": True,
            "PIL/Pillow": PIL_AVAILABLE,
            "Plotly": PLOTLY_AVAILABLE,
        }
        
        try:
            import paddle
            deps["PaddlePaddle"] = True
        except ImportError:
            pass
        
        try:
            import importlib.util
            deps["PaddleOCR"] = importlib.util.find_spec("paddleocr") is not None
        except Exception:
            # Avoid hard import because PaddleOCR can raise RuntimeError on init
            deps["PaddleOCR"] = False
        
        try:
            import onnxruntime
            deps["ONNX Runtime"] = True
        except ImportError:
            pass
        
        for name, available in deps.items():
            status = "✅" if available else "❌"
            st.text(f"{status} {name}")
    
    with col3:
        st.markdown("**GPU Status**")
        
        # Check CUDA (NVIDIA GPU for PaddlePaddle)
        cuda_available = False
        cuda_compiled = False
        
        try:
            import paddle
            cuda_compiled = paddle.device.is_compiled_with_cuda()
            if cuda_compiled:
                st.text("⚠️ Paddle CUDA: Compiled")
                try:
                    gpu_count = paddle.device.cuda.device_count()
                    if gpu_count > 0:
                        cuda_available = True
                        st.text(f"✅ NVIDIA GPUs: {gpu_count}")
                    else:
                        st.text("❌ NVIDIA GPUs: 0")
                except:
                    st.text("❌ NVIDIA GPUs: 0")
            else:
                st.text("❌ Paddle CUDA: Not compiled")
        except ImportError:
            st.text("❌ Paddle: Not installed")
        except Exception as e:
            st.text(f"⚠️ Paddle: {str(e)[:25]}")
        
        # Check MPS (Apple Silicon - PyTorch only)
        mps_available = False
        try:
            import torch
            st.text(f"PyTorch: {torch.__version__}")
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                st.text("✅ Apple MPS: Available")
                mps_available = True
            else:
                st.text("❌ Apple MPS: Not available")
        except ImportError:
            st.text("❌ PyTorch: Not installed")
        except:
            pass
        
        # Show compatibility info
        st.markdown("---")
        st.markdown("**Compatibility:**")
        if cuda_available:
            st.success("✅ PaddleOCR → NVIDIA GPU")
        else:
            st.warning("⚠️ PaddleOCR → CPU only")
        
        if mps_available:
            st.success("✅ DeepSeek → Apple M3 GPU")
        else:
            st.info("ℹ️ DeepSeek → CPU")
    
    st.markdown("---")
    
    # Model Status
    st.markdown("### 🤖 Available Models")
    
    models = get_available_models()
    
    if models:
        for name, path in models.items():
            col1, col2 = st.columns([1, 3])
            with col1:
                st.text(name)
            with col2:
                st.caption(path)
    else:
        st.info("No trained models found")
    
    st.markdown("---")
    
    # Data Status
    st.markdown("### 📊 Data Status")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Image Pool", len(st.session_state.image_pool))
    with col2:
        st.metric("Training Set", len(st.session_state.train_images))
    with col3:
        st.metric("Validation Set", len(st.session_state.val_images))
    with col4:
        st.metric("Test Set", len(st.session_state.test_images))
    
    st.markdown("---")
    
    # Cache Management
    st.markdown("### 🗑️ Cache Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Clear Streamlit Cache", use_container_width=True):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("✅ Cache cleared")
    
    with col2:
        if st.button("Reset Session State", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            # Also delete persistent state file
            if PERSISTENT_STATE_FILE.exists():
                PERSISTENT_STATE_FILE.unlink()
            st.success("✅ Session state reset")
            st.rerun()
    
    st.markdown("---")
    
    # Persistent State Info
    st.markdown("### 💾 Persistent State")
    
    if PERSISTENT_STATE_FILE.exists():
        try:
            state = load_persistent_state()
            last_saved = state.get('last_saved', 'Unknown')
            st.success(f"✅ State file exists | Last saved: {last_saved}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.caption(f"Images in pool: {len(state.get('image_pool', []))}")
                st.caption(f"Training images: {len(state.get('train_images', []))}")
            with col2:
                st.caption(f"Validation images: {len(state.get('val_images', []))}")
                st.caption(f"Test images: {len(state.get('test_images', []))}")
            
            st.caption(f"📁 State file: `{PERSISTENT_STATE_FILE}`")
        except Exception as e:
            st.warning(f"⚠️ Error reading state file: {e}")
    else:
        st.info("ℹ️ No persistent state file. Data will be saved when you configure a dataset.")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 Save State Now", use_container_width=True):
            if save_persistent_state():
                st.success("✅ State saved!")
            else:
                st.error("❌ Failed to save state")
    
    with col2:
        if st.button("📂 Reload State", use_container_width=True):
            persisted = load_persistent_state()
            if persisted:
                for key, value in persisted.items():
                    if key != 'last_saved':
                        st.session_state[key] = value
                st.success("✅ State reloaded!")
                st.rerun()
            else:
                st.info("No saved state to reload")
    
    st.markdown("---")
    
    # =========================================================================
    # DEEPSEEK-OCR MODEL SETUP
    # =========================================================================
    render_deepseek_setup_section()


def render_deepseek_setup_section():
    """Render the DeepSeek-OCR download and setup section using existing DeepSeekOCRProvider."""
    st.markdown("### 🧠 DeepSeek-OCR Model Setup")
    st.markdown("Download and configure DeepSeek-OCR for advanced VIN recognition with vision-language capabilities.")
    
    if not DEEPSEEK_AVAILABLE:
        st.warning("⚠️ DeepSeek OCR provider not available. Install required dependencies:")
        st.code("pip install 'transformers>=4.46.0' torch einops addict easydict", language="bash")
        return
    
    # Check DeepSeek availability and status
    deepseek_ready = False
    deepseek_error = None
    deepseek_device_info = {}
    
    try:
        provider = DeepSeekOCRProvider()
        deepseek_ready = provider.is_available
        if deepseek_ready:
            deepseek_device_info = provider.device_info
    except Exception as e:
        deepseek_error = str(e)
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📥 Model Status", "📋 Requirements", "⚙️ Fine-tuning"])
    
    with tab1:
        render_deepseek_status_tab(deepseek_ready, deepseek_error, deepseek_device_info)
    
    with tab2:
        render_deepseek_requirements_tab()
    
    with tab3:
        render_deepseek_finetune_tab()


def render_deepseek_status_tab(deepseek_ready: bool, deepseek_error: Optional[str], device_info: dict):
    """Render DeepSeek model status and download tab."""
    
    # Status display
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Dependencies Status**")
        if deepseek_ready:
            st.success("✅ All dependencies installed")
        else:
            st.warning("⚠️ Dependencies not ready")
            if deepseek_error:
                st.caption(f"Error: {deepseek_error[:100]}...")
    
    with col2:
        st.markdown("**Required Packages**")
        st.caption("• transformers>=4.46.0")
        st.caption("• torch>=2.0")
        st.caption("• einops, addict, easydict")
    
    st.markdown("---")
    
    # Model info
    st.markdown("**DeepSeek-OCR Model**")
    st.markdown("""
    - **Model**: DeepSeek-OCR (~6-7GB, 3B parameters)
    - **Source**: HuggingFace Hub (`deepseek-ai/DeepSeek-OCR`)
    - **Precision**: BFloat16 (GPU) / Float32 (CPU)
    """)
    
    # Show model status
    if device_info and device_info.get("device") != "not_initialized":
        st.success(f"✅ Model loaded on: {device_info.get('device', 'unknown')}")
        if device_info.get("gpu_name"):
            st.caption(f"GPU: {device_info['gpu_name']} ({device_info.get('gpu_memory_gb', 0):.1f}GB)")
    else:
        st.info("ℹ️ Model not yet downloaded/initialized")
    
    st.markdown("---")
    
    # Action buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # ACTUAL DOWNLOAD BUTTON
        if st.button("📥 Download & Initialize Model", 
                     type="primary", 
                     use_container_width=True,
                     help="Downloads ~6-7GB model from HuggingFace (requires disk space)"):
            with st.spinner("Downloading DeepSeek-OCR model (~6-7GB, this may take several minutes)..."):
                try:
                    # Determine device
                    use_gpu = st.session_state.get('use_gpu', False)
                    device = "cpu"
                    
                    try:
                        import torch
                        if use_gpu and torch.cuda.is_available():
                            device = "cuda"
                        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                            device = "mps"
                    except ImportError:
                        pass
                    
                    # Initialize provider (this downloads the model)
                    config = DeepSeekOCRConfig(device=device)
                    provider = DeepSeekOCRProvider(config=config)
                    provider.initialize()
                    
                    st.success("✅ DeepSeek-OCR model downloaded and initialized!")
                    st.balloons()
                    
                    # Show device info
                    info = provider.device_info
                    st.info(f"Model loaded on: {info.get('device', 'unknown')}")
                    
                except Exception as e:
                    st.error(f"Failed to initialize DeepSeek-OCR: {e}")
                    import traceback
                    with st.expander("Error Details"):
                        st.code(traceback.format_exc())
    
    with col2:
        # VERIFY READINESS (no download)
        if st.button("🔍 Verify Readiness", 
                     use_container_width=True,
                     help="Check if system is ready (does NOT download)"):
            with st.spinner("Checking system readiness..."):
                issues = []
                checks_passed = 0
                total_checks = 4
                
                # Check 1: Dependencies
                try:
                    import transformers
                    from packaging import version
                    if version.parse(transformers.__version__) >= version.parse("4.46.0"):
                        st.success("✅ transformers >= 4.46.0")
                        checks_passed += 1
                    else:
                        issues.append(f"transformers version {transformers.__version__} < 4.46.0")
                except ImportError:
                    issues.append("transformers not installed")
                
                # Check 2: PyTorch
                try:
                    import torch
                    st.success(f"✅ PyTorch {torch.__version__}")
                    checks_passed += 1
                except ImportError:
                    issues.append("PyTorch not installed")
                
                # Check 3: GPU/CUDA
                try:
                    import torch
                    if torch.cuda.is_available():
                        gpu_name = torch.cuda.get_device_name(0)
                        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                        st.success(f"✅ CUDA: {gpu_name} ({vram:.1f}GB)")
                        checks_passed += 1
                    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                        st.success("✅ Apple MPS available")
                        checks_passed += 1
                    else:
                        issues.append("No GPU detected (CPU inference will be slow)")
                except:
                    issues.append("Could not check GPU")
                
                # Check 4: Disk space
                try:
                    import shutil
                    project_root = Path(__file__).parent.parent.parent.parent
                    _, _, free = shutil.disk_usage(project_root)
                    free_gb = free / (1024**3)
                    if free_gb >= 15:
                        st.success(f"✅ Disk space: {free_gb:.1f}GB free")
                        checks_passed += 1
                    else:
                        issues.append(f"Low disk space: {free_gb:.1f}GB (need ~15GB for model)")
                except:
                    issues.append("Could not check disk space")
                
                # Summary
                st.markdown("---")
                if checks_passed == total_checks:
                    st.success(f"✅ All {total_checks} checks passed! Ready to download.")
                else:
                    st.warning(f"⚠️ {checks_passed}/{total_checks} checks passed")
                    for issue in issues:
                        st.error(f"• {issue}")
    
    with col3:
        if st.button("📦 Install Dependencies", use_container_width=True):
            st.info("Run this command in your terminal:")
            st.code("pip install 'transformers>=4.46.0' torch einops addict easydict", language="bash")


def render_deepseek_requirements_tab():
    """Render DeepSeek system requirements tab."""
    
    st.markdown("### System Requirements")
    
    # Check system capabilities
    cuda_available = False
    cuda_version = ""
    gpu_name = ""
    gpu_vram = 0.0
    mps_available = False
    
    try:
        import torch
        if torch.cuda.is_available():
            cuda_available = True
            cuda_version = torch.version.cuda or "Unknown"
            gpu_name = torch.cuda.get_device_name(0)
            gpu_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            mps_available = True
    except ImportError:
        pass
    
    # Display system info
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**GPU Status**")
        if cuda_available:
            st.success(f"✅ CUDA {cuda_version}")
            st.text(f"GPU: {gpu_name}")
            st.text(f"VRAM: {gpu_vram:.1f} GB")
        elif mps_available:
            st.success("✅ Apple MPS Available")
            st.caption("PyTorch can use Apple Silicon GPU")
        else:
            st.warning("⚠️ No GPU detected")
            st.caption("Will use CPU (slower)")
    
    with col2:
        st.markdown("**Disk Space**")
        try:
            import shutil
            project_root = Path(__file__).parent.parent.parent.parent
            total, used, free = shutil.disk_usage(project_root)
            free_gb = free / (1024**3)
            if free_gb >= 20:
                st.success(f"✅ {free_gb:.1f} GB available")
            else:
                st.warning(f"⚠️ {free_gb:.1f} GB available")
                st.caption("Recommend 20GB+ for model")
        except Exception:
            st.info("Could not check disk space")
    
    # Requirements table
    st.markdown("---")
    with st.expander("📋 Full Requirements", expanded=True):
        st.markdown("""
        | Component | Minimum | Recommended |
        |-----------|---------|-------------|
        | **GPU** | 8GB VRAM (LoRA) | 24GB VRAM (RTX 3090) |
        | **CUDA** | 11.8+ | 12.0+ |
        | **RAM** | 16GB | 32GB |
        | **Disk** | 15GB | 30GB |
        | **Python** | 3.9+ | 3.10 |
        
        **Note:** Apple Silicon (MPS) is supported for inference. Full fine-tuning requires NVIDIA GPU.
        """)


def render_deepseek_finetune_tab():
    """Render DeepSeek fine-tuning options tab."""
    
    st.markdown("### Fine-tuning Methods")
    st.markdown("DeepSeek-OCR supports multiple fine-tuning approaches.")
    
    # Fine-tuning methods
    methods = {
        "lora": {
            "name": "LoRA",
            "description": "Low-Rank Adaptation - Efficient fine-tuning with minimal VRAM",
            "vram_gb": 12,
            "params_pct": "0.1%"
        },
        "qlora": {
            "name": "QLoRA",
            "description": "Quantized LoRA - 4-bit quantization for extreme memory efficiency",
            "vram_gb": 8,
            "params_pct": "0.1%"
        },
        "full": {
            "name": "Full Fine-tuning",
            "description": "Train all parameters - Best results but requires most VRAM",
            "vram_gb": 24,
            "params_pct": "100%"
        }
    }
    
    for key, info in methods.items():
        with st.expander(f"🔧 {info['name']}", expanded=key == "lora"):
            st.markdown(f"**{info['description']}**")
            st.markdown(f"- VRAM Required: **{info['vram_gb']} GB**")
            st.markdown(f"- Trainable Parameters: **{info['params_pct']}**")
    
    st.markdown("---")
    
    # Training tips
    st.markdown("### 💡 Training Tips for RTX 3090")
    st.markdown("""
    **Recommended Settings:**
    - **LoRA**: batch_size=4, gradient_accumulation=8
    - **QLoRA**: batch_size=8, load_in_4bit=True
    - Learning rate: 1e-4 to 5e-5
    - LoRA rank: 8-32
    - Epochs: 3-10
    
    **To start fine-tuning**, go to the **🎯 Training** page and select DeepSeek-OCR.
    """)


# =============================================================================
# SIDEBAR
# =============================================================================

def render_sidebar() -> str:
    """Render sidebar and return selected page."""
    with st.sidebar:
        st.markdown("## 🚗 VIN OCR")
        st.caption("Simplified Pipeline")
        
        st.divider()
        
        # Navigation
        page = st.radio(
            "Navigation",
            [
                "📁 Data Management",
                "🎯 Training",
                "🔍 Inference",
                "📊 Results Dashboard",
                "🔧 System Health"
            ],
            label_visibility="collapsed"
        )
        
        st.divider()
        
        # Quick Stats
        st.markdown("### 📈 Quick Stats")
        st.caption(f"📷 Images: {len(st.session_state.image_pool)}")
        st.caption(f"🏋️ Train: {len(st.session_state.train_images)}")
        st.caption(f"🧪 Val: {len(st.session_state.val_images)}")
        st.caption(f"📝 Test: {len(st.session_state.test_images)}")
        
        st.divider()
        
        # GPU Toggle with status info
        st.markdown("### ⚡ Compute Device")
        
        # Detect GPU availability
        cuda_available = False
        mps_available = False
        
        try:
            import paddle
            if paddle.device.is_compiled_with_cuda():
                try:
                    gpu_count = paddle.device.cuda.device_count()
                    if gpu_count > 0:
                        cuda_available = True
                except:
                    pass
        except:
            pass
        
        # Check MPS (Apple Silicon)
        try:
            import torch
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                mps_available = True
        except:
            pass
        
        # Determine GPU status message
        if cuda_available:
            gpu_info = "NVIDIA CUDA"
            gpu_available = True
        elif mps_available:
            gpu_info = "Apple M3 (MPS)"
            gpu_available = False  # PaddleOCR doesn't support MPS
        else:
            gpu_info = "None detected"
            gpu_available = False
        
        # For PaddleOCR, only CUDA is supported
        use_gpu = st.checkbox(
            "🚀 Use GPU (CUDA)",
            value=st.session_state.use_gpu,
            help="PaddleOCR requires NVIDIA CUDA GPU",
            disabled=not cuda_available
        )
        st.session_state.use_gpu = use_gpu
        
        # Show current device status
        if cuda_available and use_gpu:
            st.success(f"✅ Using NVIDIA GPU")
        elif mps_available:
            st.warning(f"🍎 Apple M3 detected but PaddleOCR doesn't support MPS. Using CPU.")
            st.caption("PyTorch models (DeepSeek) can use MPS.")
        else:
            st.info("💻 Using CPU")
        
        return page


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main application entry point."""
    init_session_state()
    Config.ensure_dirs()
    
    # Render sidebar and get selected page
    page = render_sidebar()
    
    # Render selected page
    if page == "📁 Data Management":
        render_data_management_page()
    elif page == "🎯 Training":
        render_training_page()
    elif page == "🔍 Inference":
        render_inference_page()
    elif page == "📊 Results Dashboard":
        render_results_dashboard()
    elif page == "🔧 System Health":
        render_system_health()


if __name__ == "__main__":
    main()
