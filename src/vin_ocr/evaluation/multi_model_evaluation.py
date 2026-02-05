#!/usr/bin/env python3
"""
Multi-Model VIN Recognition Evaluation
=======================================

IMPORTANT: This evaluation compares RECOGNITION performance only using 
pre-trained/default model weights. It does NOT evaluate fine-tuned model 
performance. For fine-tuned model evaluation, use the dedicated fine-tuned 
evaluation scripts or export models to ONNX format.

This script evaluates multiple OCR models/approaches on all available VIN images
and produces comprehensive comparison metrics including F1 Micro and F1 Macro.

Models evaluated (Recognition Only - Default Weights):
1. PaddleOCR PP-OCRv4 (default pretrained)
2. PaddleOCR PP-OCRv3 (default pretrained)
3. VIN Pipeline (PaddleOCR + VIN-specific post-processing)
4. Fine-tuned VIN Model (if available - uses trained weights)
5. DeepSeek-OCR (if available)
6. ONNX Exported Models (if available - for production deployment)

Note: For proper fine-tuned model evaluation, models should be exported to 
ONNX format for consistent, production-ready inference.

Author: JLR VIN Project
"""

import os
import sys
import json
import time
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
import re

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import shared utilities (Single Source of Truth for VIN extraction)
from src.vin_ocr.core.vin_utils import extract_vin_from_filename

import numpy as np


@dataclass
class ModelResult:
    """Result from a single model on a single image."""
    model_name: str
    image_path: str
    ground_truth: str
    prediction: str
    confidence: float
    processing_time: float
    exact_match: bool
    chars_correct: int
    match_pattern: str


@dataclass
class ModelMetrics:
    """Aggregated metrics for a model across all images."""
    model_name: str
    total_images: int
    exact_matches: int
    incorrect_predictions: int
    exact_match_accuracy: float
    total_characters: int
    correct_characters: int
    character_accuracy: float
    f1_micro: float
    f1_macro: float
    micro_precision: float
    micro_recall: float
    avg_confidence: float
    avg_processing_time: float
    per_class_metrics: Dict[str, Dict[str, float]]
    sample_results: List[Dict]


class VINCharValidator:
    """VIN character validation and post-processing."""
    
    VIN_CHARS = set("0123456789ABCDEFGHJKLMNPRSTUVWXYZ")
    INVALID_CHARS = {'I', 'O', 'Q'}  # Not allowed in VINs
    
    # Character replacement map for common OCR errors
    CHAR_MAP = {
        'I': '1', 'O': '0', 'Q': '0',
        'i': '1', 'o': '0', 'q': '0',
        'l': '1', 'L': '1',
        ' ': '', '-': '', '.': '',
    }
    
    @classmethod
    def clean_vin(cls, raw_text: str) -> str:
        """Clean and normalize VIN text."""
        if not raw_text:
            return ""
        
        # Uppercase
        text = raw_text.upper()
        
        # Replace common errors
        for old, new in cls.CHAR_MAP.items():
            text = text.replace(old, new)
        
        # Keep only valid VIN characters
        text = ''.join(c for c in text if c in cls.VIN_CHARS)
        
        # Truncate/pad to 17 characters
        if len(text) > 17:
            text = text[:17]
        
        return text
    
    @classmethod
    def extract_vin_from_text(cls, text: str) -> str:
        """Extract best 17-character VIN from longer text."""
        if not text:
            return ""
        
        cleaned = cls.clean_vin(text)
        
        # If already 17 chars, return
        if len(cleaned) == 17:
            return cleaned
        
        # Try to find 17 consecutive valid chars
        if len(cleaned) >= 17:
            return cleaned[:17]
        
        return cleaned


class MultiModelEvaluator:
    """
    Evaluates multiple OCR models on VIN images.
    
    IMPORTANT: This evaluator tests RECOGNITION performance using default/pretrained
    model weights. For fine-tuned model evaluation, export models to ONNX format
    and use the ONNX inference path for production-ready results.
    
    Evaluation Types:
    - Recognition Only: Tests raw OCR capability with default weights
    - Fine-tuned Evaluation: Requires ONNX export for consistent results
    """
    
    # Model type descriptions for documentation
    MODEL_TYPE_DESCRIPTIONS = {
        # Recognition Only Models (Default/Pretrained Weights)
        'paddleocr': 'PaddleOCR base model (Recognition Only) - Default PP-OCR weights, no fine-tuning',
        'vin_pipeline': 'VIN Pipeline (Recognition Only) - PaddleOCR + VIN-specific pre/post processing',
        'deepseek': 'DeepSeek-OCR (Recognition Only) - Vision-Language Model with default weights',
        
        # Fine-tuned Models (Custom Trained - Paddle Format)
        'finetuned': 'Fine-tuned PaddleOCR (Paddle format) - Custom trained on VIN dataset',
        'finetuned_deepseek': 'Fine-tuned DeepSeek (PyTorch format) - Custom trained VLM on VIN dataset',
        
        # Production Models (ONNX Export - Recommended for Evaluation)
        'onnx': 'ONNX Model - Exported model for cross-platform deployment',
        'finetuned_onnx': 'Fine-tuned PaddleOCR (ONNX) - Production-ready exported PaddleOCR model',
        'finetuned_deepseek_onnx': 'Fine-tuned DeepSeek (ONNX) - Production-ready exported VLM model',
    }
    
    # Evaluation mode descriptions
    EVALUATION_MODES = {
        'recognition': 'Tests raw OCR recognition using default/pretrained weights',
        'finetuned': 'Evaluates custom fine-tuned models (PaddleOCR and DeepSeek)',
        'production': 'Tests ONNX exported models for production deployment validation',
    }
    
    # Supported ONNX model prefixes for auto-discovery
    ONNX_MODEL_PREFIXES = {
        'paddleocr': 'finetuned_onnx',
        'deepseek': 'finetuned_deepseek_onnx',
        'vin': 'finetuned_onnx',
    }
    
    def __init__(self, output_dir: str = "results", evaluation_mode: str = "recognition"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.models = {}
        self.results = []
        self.evaluation_mode = evaluation_mode
        self.onnx_runtime_available = self._check_onnx_runtime()
    
    def _check_onnx_runtime(self) -> bool:
        """Check if ONNX Runtime is available."""
        try:
            import onnxruntime
            return True
        except ImportError:
            return False
    
    def _get_model_description(self, model_type: str) -> str:
        """Get description for a model type."""
        return self.MODEL_TYPE_DESCRIPTIONS.get(model_type, f'Unknown model type: {model_type}')
        
    def load_models(self):
        """Load all available models."""
        print("=" * 60)
        print("Loading OCR Models...")
        print("=" * 60)
        
        # Model 1: PaddleOCR PP-OCRv4 (default) - Updated API
        try:
            from paddleocr import PaddleOCR
            self.models['paddleocr_v4'] = {
                'name': 'PaddleOCR PP-OCRv4',
                'engine': PaddleOCR(
                    use_textline_orientation=True,
                    lang='en',
                    text_det_thresh=0.3,
                    text_det_box_thresh=0.5,
                ),
                'type': 'paddleocr'
            }
            print("  ✓ PaddleOCR PP-OCRv4 loaded")
        except Exception as e:
            print(f"  ✗ PaddleOCR PP-OCRv4 failed: {e}")
        
        # Model 2: PaddleOCR PP-OCRv3 - Updated API
        try:
            from paddleocr import PaddleOCR
            self.models['paddleocr_v3'] = {
                'name': 'PaddleOCR PP-OCRv3',
                'engine': PaddleOCR(
                    use_textline_orientation=True,
                    lang='en',
                    ocr_version='PP-OCRv3',
                ),
                'type': 'paddleocr'
            }
            print("  ✓ PaddleOCR PP-OCRv3 loaded")
        except Exception as e:
            print(f"  ✗ PaddleOCR PP-OCRv3 failed: {e}")
        
        # Model 3: VINOCRPipeline (includes post-processing)
        try:
            from src.vin_ocr.pipeline.vin_pipeline import VINOCRPipeline
            self.models['vin_pipeline'] = {
                'name': 'VIN Pipeline (with post-processing)',
                'engine': VINOCRPipeline(),
                'type': 'vin_pipeline'
            }
            print("  ✓ VIN Pipeline loaded")
        except Exception as e:
            print(f"  ✗ VIN Pipeline failed: {e}")
        
        # Model 4: Fine-tuned model (if available)
        finetuned_path = project_root / "output" / "vin_rec_finetune" / "best_accuracy.pdparams"
        if finetuned_path.exists():
            try:
                # We'll use the finetuned model through finetune_paddleocr's inference
                self.models['finetuned'] = {
                    'name': 'Fine-tuned VIN Model',
                    'engine': None,  # Will load separately
                    'type': 'finetuned',
                    'path': str(finetuned_path)
                }
                print("  ✓ Fine-tuned model found")
            except Exception as e:
                print(f"  ✗ Fine-tuned model failed: {e}")
        else:
            print("  ⚠ Fine-tuned model not found (train first)")
        
        # Model 5: DeepSeek-OCR (if available)
        try:
            from src.vin_ocr.providers.ocr_providers import DeepSeekOCRProvider, DeepSeekOCRConfig
            
            # Check if dependencies are available
            deepseek_provider = DeepSeekOCRProvider()
            if deepseek_provider.is_available:
                self.models['deepseek'] = {
                    'name': 'DeepSeek-OCR',
                    'engine': deepseek_provider,
                    'type': 'deepseek'
                }
                print("  ✓ DeepSeek-OCR loaded")
            else:
                print("  ⚠ DeepSeek-OCR dependencies not installed (transformers>=4.46.0, torch)")
        except ImportError as e:
            print(f"  ⚠ DeepSeek-OCR not available: {e}")
        except Exception as e:
            print(f"  ✗ DeepSeek-OCR failed: {e}")
        
        # Model 6: Fine-tuned DeepSeek (if available - for HPC/CUDA environments)
        self._load_finetuned_deepseek()
        
        # Model 7: ONNX Exported Models (for production evaluation)
        self._load_onnx_models()
        
        print(f"\n  Total models loaded: {len(self.models)}")
        print(f"  Evaluation mode: {self.evaluation_mode} - {self.EVALUATION_MODES.get(self.evaluation_mode, 'Unknown')}")
        print("=" * 60)
    
    def _load_finetuned_deepseek(self):
        """Load fine-tuned DeepSeek model if available (requires HPC/CUDA)."""
        # Search for fine-tuned DeepSeek models
        deepseek_search_paths = [
            project_root / "output" / "deepseek_finetune",
            project_root / "output" / "deepseek_scratch",
            project_root / "models" / "deepseek_finetuned",
            project_root / "models" / "deepseek",
        ]
        
        for search_path in deepseek_search_paths:
            if not search_path.exists():
                continue
            
            # Look for PyTorch checkpoints
            for checkpoint in search_path.glob("**/pytorch_model.bin"):
                try:
                    model_dir = checkpoint.parent
                    model_name = model_dir.name
                    
                    # Check if we have the config
                    config_path = model_dir / "config.json"
                    if not config_path.exists():
                        continue
                    
                    self.models[f'deepseek_finetuned_{model_name}'] = {
                        'name': f'Fine-tuned DeepSeek: {model_name}',
                        'engine': None,  # Lazy load due to memory requirements
                        'type': 'deepseek_finetuned',
                        'path': str(model_dir),
                        'requires_gpu': True,
                    }
                    print(f"  ✓ Fine-tuned DeepSeek found: {model_name} (requires GPU)")
                    
                except Exception as e:
                    print(f"  ✗ Fine-tuned DeepSeek failed ({checkpoint}): {e}")
            
            # Also check for safetensors format
            for checkpoint in search_path.glob("**/model.safetensors"):
                try:
                    model_dir = checkpoint.parent
                    model_name = model_dir.name
                    
                    if f'deepseek_finetuned_{model_name}' in self.models:
                        continue  # Already loaded
                    
                    self.models[f'deepseek_finetuned_{model_name}'] = {
                        'name': f'Fine-tuned DeepSeek: {model_name}',
                        'engine': None,
                        'type': 'deepseek_finetuned',
                        'path': str(model_dir),
                        'requires_gpu': True,
                    }
                    print(f"  ✓ Fine-tuned DeepSeek found: {model_name} (requires GPU)")
                    
                except Exception as e:
                    print(f"  ✗ Fine-tuned DeepSeek failed ({checkpoint}): {e}")
        
        # Note about HPC requirements
        if not any(k.startswith('deepseek_finetuned') for k in self.models):
            print("  ⚠ No fine-tuned DeepSeek models found")
            print("    → Fine-tune on HPC (RTX 3090 24GB): python -m src.vin_ocr.training.finetune_deepseek")
            print("    → Export to ONNX for portable inference: python -m src.vin_ocr.training.export_deepseek_onnx")
    
    def _load_onnx_models(self):
        """Load ONNX exported models for production-ready evaluation."""
        if not self.onnx_runtime_available:
            print("  ⚠ ONNX Runtime not installed (pip install onnxruntime or onnxruntime-gpu)")
            print("    ONNX models provide production-ready inference for fine-tuned models")
            print("    Supports both PaddleOCR and DeepSeek fine-tuned exports")
            return
        
        # Search for ONNX models in standard locations
        onnx_search_paths = [
            project_root / "output" / "onnx",
            project_root / "output" / "onnx" / "paddleocr",
            project_root / "output" / "onnx" / "deepseek",
            project_root / "output" / "vin_rec_finetune" / "onnx",
            project_root / "output" / "deepseek_finetune" / "onnx",
            project_root / "models" / "onnx",
            project_root / "output",
        ]
        
        onnx_models_found = []
        for search_path in onnx_search_paths:
            if search_path.exists():
                for onnx_file in search_path.glob("**/*.onnx"):
                    onnx_models_found.append(onnx_file)
        
        if not onnx_models_found:
            print("  ⚠ No ONNX models found. Export fine-tuned models for production evaluation:")
            print("    PaddleOCR: python -m src.vin_ocr.training.export_onnx --model-path output/vin_rec_finetune/best_accuracy")
            print("    DeepSeek:  python -m src.vin_ocr.training.export_deepseek_onnx --model-path output/deepseek_finetune/best")
            return
        
        # Determine available execution providers
        import onnxruntime as ort
        available_providers = ort.get_available_providers()
        
        # Prefer GPU if available
        if 'CUDAExecutionProvider' in available_providers:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            print("  ✓ CUDA available for ONNX inference")
        elif 'CoreMLExecutionProvider' in available_providers:
            providers = ['CoreMLExecutionProvider', 'CPUExecutionProvider']
            print("  ✓ CoreML available for ONNX inference (Apple Silicon)")
        else:
            providers = ['CPUExecutionProvider']
        
        # Load each ONNX model
        for onnx_path in onnx_models_found:
            try:
                # Create inference session
                session = ort.InferenceSession(str(onnx_path), providers=providers)
                
                model_name = onnx_path.stem
                model_key = f"onnx_{model_name}"
                
                # Determine model type based on path/name
                if 'deepseek' in str(onnx_path).lower() or 'deepseek' in model_name.lower():
                    model_type = 'finetuned_deepseek_onnx'
                    display_name = f'ONNX DeepSeek: {model_name}'
                else:
                    model_type = 'finetuned_onnx'
                    display_name = f'ONNX PaddleOCR: {model_name}'
                
                self.models[model_key] = {
                    'name': display_name,
                    'engine': session,
                    'type': model_type,
                    'path': str(onnx_path),
                    'input_name': session.get_inputs()[0].name,
                    'output_name': session.get_outputs()[0].name,
                    'providers': session.get_providers(),
                }
                print(f"  ✓ ONNX model loaded: {model_name} ({model_type})")
                
            except Exception as e:
                print(f"  ✗ ONNX model failed ({onnx_path.name}): {e}")
    
    def load_dataset(self, custom_folder: Optional[str] = None, labels_file: Optional[str] = None) -> List[Tuple[str, str]]:
        """
        Load VIN images with their ground truth labels.
        
        Args:
            custom_folder: Path to custom image folder (VIN extracted from filename)
            labels_file: Path to labels file with format: image_path\\tVIN
        
        Returns:
            List of (image_path, ground_truth_vin) tuples
        """
        print("\nLoading dataset...")
        
        dataset = []
        
        # Option 1: Load from labels file
        if labels_file:
            labels_path = Path(labels_file)
            if labels_path.exists():
                with open(labels_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if '\t' in line:
                            img_path, vin = line.split('\t', 1)
                            if Path(img_path).exists() and len(vin) == 17:
                                dataset.append((img_path, vin))
                print(f"  Loaded {len(dataset)} images from labels file: {labels_file}")
                return dataset
            else:
                print(f"  WARNING: Labels file not found: {labels_file}")
        
        # Option 2: Load from custom folder
        if custom_folder:
            custom_path = Path(custom_folder)
            if custom_path.exists():
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                    for img_path in custom_path.glob(ext):
                        filename = img_path.stem
                        vin = extract_vin_from_filename(filename)
                        if vin and len(vin) == 17:
                            dataset.append((str(img_path), vin))
                print(f"  Found {len(dataset)} images in custom folder: {custom_folder}")
                return dataset
            else:
                print(f"  WARNING: Custom folder not found: {custom_folder}")
        
        # Default: Load from standard dataset locations
        # Check dagshub_data/test/images
        test_dir = project_root / "dagshub_data" / "test" / "images"
        if test_dir.exists():
            for img_path in test_dir.glob("*.jpg"):
                # Extract VIN from filename (format: VIN_...-VIN_-_VIN_.jpg)
                filename = img_path.stem
                # Try to extract VIN from filename
                vin = extract_vin_from_filename(filename)
                if vin and len(vin) == 17:
                    dataset.append((str(img_path), vin))
        
        # Check dagshub_data/train/images  
        train_dir = project_root / "dagshub_data" / "train" / "images"
        if train_dir.exists():
            for img_path in train_dir.glob("*.jpg"):
                filename = img_path.stem
                vin = extract_vin_from_filename(filename)
                if vin and len(vin) == 17:
                    dataset.append((str(img_path), vin))
        
        # Check original dataset folder
        orig_test = project_root / "dataset" / "test"
        if orig_test.exists():
            for img_path in orig_test.glob("*.jpg"):
                filename = img_path.stem
                vin = extract_vin_from_filename(filename)
                if vin and len(vin) == 17:
                    dataset.append((str(img_path), vin))
        
        print(f"  Found {len(dataset)} images with ground truth")
        return dataset
    
    # Note: VIN extraction uses vin_utils.extract_vin_from_filename (Single Source of Truth)
    # Supported formats: "1-VIN -SAL1A2A40SA606662.jpg", "7-VIN_-_SAL109F97TA467227.jpg", etc.
    
    def run_paddleocr(self, engine, image_path: str) -> Tuple[str, float]:
        """Run PaddleOCR on an image using the new predict() API."""
        try:
            # Use predict() instead of deprecated ocr()
            result = engine.predict(image_path)
            
            if not result:
                return "", 0.0
            
            # Handle new PaddleOCR result format
            texts = []
            confidences = []
            
            # New API returns list of results
            for item in result:
                # Try to extract rec_texts and rec_scores from result
                if hasattr(item, 'rec_texts'):
                    rec_texts = item.rec_texts if item.rec_texts else []
                    rec_scores = item.rec_scores if hasattr(item, 'rec_scores') and item.rec_scores else [0.5] * len(rec_texts)
                    texts.extend(rec_texts)
                    confidences.extend(rec_scores)
                elif isinstance(item, dict):
                    if 'rec_texts' in item:
                        texts.extend(item['rec_texts'])
                        confidences.extend(item.get('rec_scores', [0.5] * len(item['rec_texts'])))
                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                    # Legacy format: [[box, (text, conf)], ...]
                    for line in item:
                        if line and len(line) >= 2:
                            text = line[1][0] if isinstance(line[1], tuple) else str(line[1])
                            conf = line[1][1] if isinstance(line[1], tuple) else 0.5
                            texts.append(text)
                            confidences.append(conf)
            
            if not texts:
                return "", 0.0
            
            combined_text = ' '.join(texts)
            avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
            
            # Clean and extract VIN
            vin = VINCharValidator.extract_vin_from_text(combined_text)
            
            return vin, avg_conf
            
        except Exception as e:
            print(f"    Error processing {image_path}: {e}")
            return "", 0.0
    
    def run_vin_pipeline(self, engine, image_path: str) -> Tuple[str, float]:
        """Run VIN Pipeline on an image."""
        try:
            result = engine.recognize(image_path)
            vin = result.get('vin', '') or ''
            conf = result.get('confidence', 0.0) or 0.0
            return vin[:17], conf
        except Exception as e:
            print(f"    Error with VIN pipeline on {image_path}: {e}")
            return "", 0.0
    
    def run_deepseek(self, engine, image_path: str) -> Tuple[str, float]:
        """Run DeepSeek-OCR on an image."""
        try:
            # Initialize the model if not already initialized
            if not engine._initialized:
                # Check if we already tried and failed
                if hasattr(engine, '_init_failed') and engine._init_failed:
                    return "", 0.0
                    
                print("    Initializing DeepSeek-OCR model (this may take a moment)...")
                try:
                    engine.initialize()
                except Exception as init_error:
                    print(f"    DeepSeek-OCR initialization failed: {init_error}")
                    engine._init_failed = True
                    return "", 0.0
            
            # Run OCR
            result = engine.recognize(image_path)
            
            # Extract text and confidence
            raw_text = result.text if result else ''
            confidence = result.confidence if result else 0.0
            
            # Clean and extract VIN from result
            vin = VINCharValidator.extract_vin_from_text(raw_text)
            
            return vin[:17] if vin else '', confidence
        except Exception as e:
            print(f"    Error with DeepSeek-OCR on {image_path}: {e}")
            return "", 0.0
    
    def run_deepseek_finetuned(self, engine, image_path: str) -> Tuple[str, float]:
        """
        Run fine-tuned DeepSeek-OCR on an image.
        
        This is for DeepSeek models fine-tuned on VIN data using HPC/CUDA.
        The fine-tuned model should have better VIN-specific recognition.
        
        Note: This method expects a transformers-based model loaded from
        a fine-tuned checkpoint (e.g., models/deepseek_finetuned/).
        """
        try:
            # Initialize the model if not already initialized
            if hasattr(engine, '_initialized') and not engine._initialized:
                if hasattr(engine, '_init_failed') and engine._init_failed:
                    return "", 0.0
                    
                print("    Initializing fine-tuned DeepSeek model (this may take a moment)...")
                try:
                    engine.initialize()
                except Exception as init_error:
                    print(f"    Fine-tuned DeepSeek initialization failed: {init_error}")
                    engine._init_failed = True
                    return "", 0.0
            
            # Run inference - fine-tuned model may have different interface
            if hasattr(engine, 'recognize_vin'):
                # Custom VIN-specific method if available
                result = engine.recognize_vin(image_path)
            elif hasattr(engine, 'recognize'):
                # Standard recognize interface
                result = engine.recognize(image_path)
            else:
                # Direct model call for transformers models
                result = engine(image_path)
            
            # Handle different result formats
            if isinstance(result, dict):
                raw_text = result.get('text', result.get('vin', ''))
                confidence = result.get('confidence', 0.0)
            elif hasattr(result, 'text'):
                raw_text = result.text
                confidence = result.confidence if hasattr(result, 'confidence') else 0.0
            else:
                raw_text = str(result) if result else ''
                confidence = 0.0
            
            # Clean and extract VIN from result
            vin = VINCharValidator.extract_vin_from_text(raw_text)
            
            return vin[:17] if vin else '', confidence
        except Exception as e:
            print(f"    Error with fine-tuned DeepSeek on {image_path}: {e}")
            return "", 0.0
    
    def run_deepseek_onnx(self, model_info: Dict, image_path: str) -> Tuple[str, float]:
        """
        Run ONNX-exported DeepSeek model inference on an image.
        
        This is for DeepSeek models exported to ONNX format after fine-tuning.
        ONNX provides faster inference and doesn't require PyTorch/transformers.
        
        Expected model_info structure:
        {
            'engine': onnxruntime.InferenceSession,
            'input_name': str,
            'output_name': str,
            'processor': optional image processor config
        }
        """
        try:
            import cv2
            import numpy as np
            
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                return "", 0.0
            
            # Convert BGR to RGB (transformers models expect RGB)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Get model input requirements
            session = model_info['engine']
            input_info = session.get_inputs()[0]
            input_name = input_info.name
            input_shape = input_info.shape
            
            # Preprocess image for vision transformer
            # Typical ViT input: [batch, channels, height, width] = [1, 3, 384, 384] or similar
            if len(input_shape) == 4:
                # Get expected dimensions
                if isinstance(input_shape[2], int) and isinstance(input_shape[3], int):
                    target_h, target_w = input_shape[2], input_shape[3]
                else:
                    target_h, target_w = 384, 384  # Default ViT size
                
                # Resize image
                resized = cv2.resize(image_rgb, (target_w, target_h))
                
                # Normalize (ImageNet normalization for transformers)
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                normalized = (resized.astype(np.float32) / 255.0 - mean) / std
                
                # Transpose to NCHW format
                input_data = np.transpose(normalized, (2, 0, 1))  # HWC -> CHW
                input_data = np.expand_dims(input_data, axis=0)   # Add batch
                input_data = input_data.astype(np.float32)
            else:
                # Fallback for unexpected input format
                print(f"    Unexpected input shape: {input_shape}")
                return "", 0.0
            
            # Run inference
            output_names = [o.name for o in session.get_outputs()]
            outputs = session.run(output_names, {input_name: input_data})
            
            # Decode output
            # For VLM models, output is typically token IDs that need decoding
            output = outputs[0]
            
            # If output is logits, decode them
            if len(output.shape) >= 2:
                if output.shape[-1] > 100:  # Likely vocabulary logits
                    # Use argmax to get token IDs
                    pred_indices = np.argmax(output, axis=-1)
                    
                    # Simple ASCII-based decoding for VIN characters
                    # Fine-tuned model should output VIN-like text
                    decoded_chars = []
                    for idx in pred_indices.flatten():
                        if 32 <= idx < 127:  # Printable ASCII
                            decoded_chars.append(chr(idx))
                    raw_text = ''.join(decoded_chars)
                else:
                    # Direct character indices
                    char_set = "0123456789ABCDEFGHJKLMNPRSTUVWXYZ"
                    decoded = [char_set[int(i)] for i in output.flatten() if 0 <= i < len(char_set)]
                    raw_text = ''.join(decoded)
            else:
                raw_text = str(output)
            
            # Extract VIN from decoded text
            vin = VINCharValidator.extract_vin_from_text(raw_text)
            
            # Calculate confidence from output probabilities
            if len(outputs) > 0 and hasattr(outputs[0], 'shape'):
                confidence = float(np.mean(np.max(outputs[0], axis=-1))) if outputs[0].size > 0 else 0.0
            else:
                confidence = 0.5  # Default confidence
            
            return vin[:17] if vin else '', confidence
            
        except Exception as e:
            print(f"    Error with DeepSeek ONNX model on {image_path}: {e}")
            return "", 0.0
    
    def run_onnx(self, model_info: Dict, image_path: str) -> Tuple[str, float]:
        """
        Run ONNX model inference on an image.
        
        This is the recommended method for evaluating fine-tuned models
        as ONNX provides consistent, production-ready inference.
        """
        try:
            import cv2
            import numpy as np
            
            # Load and preprocess image
            image = cv2.imread(image_path)
            if image is None:
                return "", 0.0
            
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Resize to expected input size (typical OCR input: 32x320 or similar)
            # TODO: Get actual input shape from model
            session = model_info['engine']
            input_shape = session.get_inputs()[0].shape
            
            # Handle dynamic shapes
            if isinstance(input_shape[2], int) and isinstance(input_shape[3], int):
                target_h, target_w = input_shape[2], input_shape[3]
            else:
                target_h, target_w = 32, 320  # Default OCR input size
            
            # Resize maintaining aspect ratio
            h, w = gray.shape[:2]
            ratio = target_h / h
            new_w = int(w * ratio)
            if new_w > target_w:
                new_w = target_w
            
            resized = cv2.resize(gray, (new_w, target_h))
            
            # Pad to target width
            if new_w < target_w:
                padded = np.zeros((target_h, target_w), dtype=np.uint8)
                padded[:, :new_w] = resized
                resized = padded
            
            # Normalize and add batch/channel dimensions
            input_data = resized.astype(np.float32) / 255.0
            input_data = np.expand_dims(input_data, axis=0)  # Add batch
            input_data = np.expand_dims(input_data, axis=0)  # Add channel
            
            # Run inference
            input_name = model_info['input_name']
            output_name = model_info['output_name']
            
            outputs = session.run([output_name], {input_name: input_data})
            
            # Decode output (CTC decoding)
            output = outputs[0]
            pred_indices = np.argmax(output, axis=2)[0]
            
            # Simple CTC decode (remove blanks and duplicates)
            # TODO: Use proper character set from model config
            char_set = "0123456789ABCDEFGHJKLMNPRSTUVWXYZ"  # VIN valid chars
            blank_idx = len(char_set)
            
            decoded = []
            prev_idx = blank_idx
            for idx in pred_indices:
                if idx != blank_idx and idx != prev_idx:
                    if idx < len(char_set):
                        decoded.append(char_set[idx])
                prev_idx = idx
            
            vin = ''.join(decoded)[:17]
            confidence = float(np.max(output)) if len(output) > 0 else 0.0
            
            return vin, confidence
            
        except Exception as e:
            print(f"    Error with ONNX model on {image_path}: {e}")
            return "", 0.0
    
    def evaluate_model(self, model_key: str, model_info: Dict, dataset: List[Tuple[str, str]]) -> ModelMetrics:
        """Evaluate a single model on the entire dataset."""
        print(f"\n  Evaluating: {model_info['name']}...")
        
        predictions = []
        ground_truths = []
        confidences = []
        processing_times = []
        sample_results = []
        
        for i, (img_path, gt_vin) in enumerate(dataset):
            start_time = time.time()
            
            if model_info['type'] == 'paddleocr':
                pred_vin, conf = self.run_paddleocr(model_info['engine'], img_path)
            elif model_info['type'] == 'vin_pipeline':
                pred_vin, conf = self.run_vin_pipeline(model_info['engine'], img_path)
            elif model_info['type'] == 'deepseek':
                pred_vin, conf = self.run_deepseek(model_info['engine'], img_path)
            elif model_info['type'] == 'deepseek_finetuned':
                pred_vin, conf = self.run_deepseek_finetuned(model_info['engine'], img_path)
            elif model_info['type'] == 'deepseek_finetuned_onnx':
                pred_vin, conf = self.run_deepseek_onnx(model_info, img_path)
            elif model_info['type'] == 'onnx':
                pred_vin, conf = self.run_onnx(model_info, img_path)
            elif model_info['type'] == 'finetuned_onnx':
                pred_vin, conf = self.run_onnx(model_info, img_path)
            elif model_info['type'] == 'finetuned':
                # For fine-tuned PaddleOCR models (native format)
                pred_vin, conf = self.run_paddleocr(model_info['engine'], img_path)
            else:
                pred_vin, conf = "", 0.0
            
            proc_time = time.time() - start_time
            
            # Ensure 17 chars for comparison
            pred_vin = (pred_vin + '_' * 17)[:17]
            
            predictions.append(pred_vin)
            ground_truths.append(gt_vin)
            confidences.append(conf)
            processing_times.append(proc_time)
            
            # Character-by-character match
            chars_correct = sum(1 for j in range(17) if j < len(pred_vin) and pred_vin[j] == gt_vin[j])
            match_pattern = ''.join(['✓' if j < len(pred_vin) and pred_vin[j] == gt_vin[j] else '✗' for j in range(17)])
            
            sample_results.append({
                'image': Path(img_path).name,
                'ground_truth': gt_vin,
                'prediction': pred_vin,
                'exact_match': pred_vin == gt_vin,
                'chars_correct': chars_correct,
                'char_accuracy': chars_correct / 17,
                'match_pattern': match_pattern,
                'confidence': conf,
                'processing_time': proc_time,
                # Track which model produced this result
                'model_name': model_info['name'],
                'model_type': model_info['type'],
                'model_key': model_key,
            })
            
            if (i + 1) % 50 == 0:
                print(f"    Processed {i + 1}/{len(dataset)} images...")
        
        # Calculate metrics
        metrics = self._calculate_metrics(
            model_info['name'],
            predictions,
            ground_truths,
            confidences,
            processing_times,
            sample_results
        )
        
        return metrics
    
    def _calculate_metrics(
        self,
        model_name: str,
        predictions: List[str],
        ground_truths: List[str],
        confidences: List[float],
        processing_times: List[float],
        sample_results: List[Dict]
    ) -> ModelMetrics:
        """Calculate comprehensive metrics including F1 micro/macro."""
        
        n_samples = len(predictions)
        
        # Image-level metrics
        exact_matches = sum(1 for p, g in zip(predictions, ground_truths) if p == g)
        incorrect_predictions = n_samples - exact_matches
        exact_match_accuracy = exact_matches / n_samples if n_samples > 0 else 0
        
        # Character-level metrics
        char_correct = 0
        char_total = 0
        
        vin_chars = set("0123456789ABCDEFGHJKLMNPRSTUVWXYZ")
        class_tp = defaultdict(int)
        class_fp = defaultdict(int)
        class_fn = defaultdict(int)
        
        for pred, gt in zip(predictions, ground_truths):
            for i in range(17):
                pred_char = pred[i] if i < len(pred) else ''
                gt_char = gt[i] if i < len(gt) else ''
                
                if gt_char:
                    char_total += 1
                    if pred_char == gt_char:
                        char_correct += 1
                        class_tp[gt_char] += 1
                    else:
                        class_fn[gt_char] += 1
                        if pred_char and pred_char in vin_chars:
                            class_fp[pred_char] += 1
        
        char_accuracy = char_correct / char_total if char_total > 0 else 0
        
        # F1 Micro
        total_tp = sum(class_tp.values())
        total_fp = sum(class_fp.values())
        total_fn = sum(class_fn.values())
        
        micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1_micro = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0
        
        # F1 Macro
        class_f1_scores = []
        per_class_metrics = {}
        
        for char in sorted(vin_chars):
            tp = class_tp[char]
            fp = class_fp[char]
            fn = class_fn[char]
            
            if (tp + fn) > 0:
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                class_f1_scores.append(f1)
                per_class_metrics[char] = {
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'support': tp + fn
                }
        
        f1_macro = sum(class_f1_scores) / len(class_f1_scores) if class_f1_scores else 0
        
        return ModelMetrics(
            model_name=model_name,
            total_images=n_samples,
            exact_matches=exact_matches,
            incorrect_predictions=incorrect_predictions,
            exact_match_accuracy=exact_match_accuracy,
            total_characters=char_total,
            correct_characters=char_correct,
            character_accuracy=char_accuracy,
            f1_micro=f1_micro,
            f1_macro=f1_macro,
            micro_precision=micro_precision,
            micro_recall=micro_recall,
            avg_confidence=sum(confidences) / len(confidences) if confidences else 0,
            avg_processing_time=sum(processing_times) / len(processing_times) if processing_times else 0,
            per_class_metrics=per_class_metrics,
            sample_results=sample_results
        )
    
    def run_evaluation(self, max_images: Optional[int] = None, custom_image_folder: Optional[str] = None, labels_file: Optional[str] = None):
        """
        Run full multi-model evaluation.
        
        Args:
            max_images: Maximum number of images to evaluate
            custom_image_folder: Path to custom image folder
            labels_file: Path to labels file (format: image_path\\tVIN)
        """
        print("\n" + "=" * 60)
        print("MULTI-MODEL VIN RECOGNITION EVALUATION")
        print("=" * 60)
        
        # Load models
        self.load_models()
        
        if not self.models:
            print("ERROR: No models loaded!")
            return
        
        # Load dataset
        dataset = self.load_dataset(custom_folder=custom_image_folder, labels_file=labels_file)
        
        if not dataset:
            print("ERROR: No images found!")
            return
        
        if max_images:
            dataset = dataset[:max_images]
            print(f"  Limited to {max_images} images for testing")
        
        # Evaluate each model
        all_metrics = {}
        
        for model_key, model_info in self.models.items():
            if model_info['type'] == 'finetuned':
                print(f"\n  Skipping {model_info['name']} (requires separate inference)")
                continue
                
            metrics = self.evaluate_model(model_key, model_info, dataset)
            all_metrics[model_key] = metrics
        
        # Print comparison
        self._print_comparison(all_metrics)
        
        # Save results
        self._save_results(all_metrics)
        
        return all_metrics
    
    def _print_comparison(self, all_metrics: Dict[str, ModelMetrics]):
        """Print side-by-side comparison of all models."""
        print("\n" + "=" * 80)
        print("MODEL COMPARISON RESULTS")
        print("=" * 80)
        
        # Header
        print(f"\n{'Metric':<30}", end="")
        for metrics in all_metrics.values():
            print(f"{metrics.model_name[:20]:<22}", end="")
        print()
        print("-" * 80)
        
        # Image-level metrics
        print(f"{'Total Images':<30}", end="")
        for m in all_metrics.values():
            print(f"{m.total_images:<22}", end="")
        print()
        
        print(f"{'Exact Matches':<30}", end="")
        for m in all_metrics.values():
            print(f"{m.exact_matches:<22}", end="")
        print()
        
        print(f"{'Exact Match Accuracy':<30}", end="")
        for m in all_metrics.values():
            print(f"{m.exact_match_accuracy:.2%}".ljust(22), end="")
        print()
        
        print("-" * 80)
        
        # Character-level metrics
        print(f"{'Total Characters':<30}", end="")
        for m in all_metrics.values():
            print(f"{m.total_characters:<22}", end="")
        print()
        
        print(f"{'Character Accuracy':<30}", end="")
        for m in all_metrics.values():
            print(f"{m.character_accuracy:.2%}".ljust(22), end="")
        print()
        
        print(f"{'★ F1 Micro':<30}", end="")
        for m in all_metrics.values():
            print(f"{m.f1_micro:.4f}".ljust(22), end="")
        print()
        
        print(f"{'★ F1 Macro':<30}", end="")
        for m in all_metrics.values():
            print(f"{m.f1_macro:.4f}".ljust(22), end="")
        print()
        
        print(f"{'Micro Precision':<30}", end="")
        for m in all_metrics.values():
            print(f"{m.micro_precision:.4f}".ljust(22), end="")
        print()
        
        print(f"{'Micro Recall':<30}", end="")
        for m in all_metrics.values():
            print(f"{m.micro_recall:.4f}".ljust(22), end="")
        print()
        
        print("-" * 80)
        
        # Performance metrics
        print(f"{'Avg Confidence':<30}", end="")
        for m in all_metrics.values():
            print(f"{m.avg_confidence:.2%}".ljust(22), end="")
        print()
        
        print(f"{'Avg Processing Time (s)':<30}", end="")
        for m in all_metrics.values():
            print(f"{m.avg_processing_time:.3f}s".ljust(22), end="")
        print()
        
        print("=" * 80)
        
        # Best model
        if all_metrics:
            best_f1_micro = max(all_metrics.values(), key=lambda m: m.f1_micro)
            best_f1_macro = max(all_metrics.values(), key=lambda m: m.f1_macro)
            best_exact = max(all_metrics.values(), key=lambda m: m.exact_match_accuracy)
            
            print(f"\n🏆 BEST MODELS:")
            print(f"   Best F1 Micro:      {best_f1_micro.model_name} ({best_f1_micro.f1_micro:.4f})")
            print(f"   Best F1 Macro:      {best_f1_macro.model_name} ({best_f1_macro.f1_macro:.4f})")
            print(f"   Best Exact Match:   {best_exact.model_name} ({best_exact.exact_match_accuracy:.2%})")
        
        # Sample results from best model
        if all_metrics:
            best = max(all_metrics.values(), key=lambda m: m.f1_micro)
            print(f"\n📋 SAMPLE RESULTS ({best.model_name}):")
            for i, s in enumerate(best.sample_results[:10]):
                status = "✓ EXACT" if s['exact_match'] else f"✗ {s['chars_correct']}/17"
                print(f"   {i+1}. GT:   {s['ground_truth']}")
                print(f"      Pred: {s['prediction']}")
                print(f"      {s['match_pattern']} [{status}]")
                print()
    
    def _save_results(self, all_metrics: Dict[str, ModelMetrics]):
        """Save results to JSON file."""
        from datetime import datetime
        
        # Build model registry with detailed information
        model_registry = {}
        for model_key, model_info in self.models.items():
            model_registry[model_key] = {
                'name': model_info['name'],
                'type': model_info['type'],
                'description': self._get_model_description(model_info['type']),
                'path': model_info.get('path', 'N/A'),
            }
        
        results = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'evaluation_type': 'multi_model_comparison',
                'total_models_evaluated': len(all_metrics),
            },
            'model_registry': model_registry,
            'models': {}
        }
        
        for model_key, metrics in all_metrics.items():
            results['models'][model_key] = {
                'model_name': metrics.model_name,
                'image_level': {
                    'total_images': metrics.total_images,
                    'exact_matches': metrics.exact_matches,
                    'incorrect_predictions': metrics.incorrect_predictions,
                    'exact_match_accuracy': metrics.exact_match_accuracy,
                },
                'character_level': {
                    'total_characters': metrics.total_characters,
                    'correct_characters': metrics.correct_characters,
                    'character_accuracy': metrics.character_accuracy,
                    'f1_micro': metrics.f1_micro,
                    'f1_macro': metrics.f1_macro,
                    'micro_precision': metrics.micro_precision,
                    'micro_recall': metrics.micro_recall,
                },
                'performance': {
                    'avg_confidence': metrics.avg_confidence,
                    'avg_processing_time': metrics.avg_processing_time,
                },
                'per_class_metrics': metrics.per_class_metrics,
                'sample_results': metrics.sample_results,
            }
        
        output_path = self.output_dir / 'multi_model_evaluation.json'
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n📁 Results saved to: {output_path}")
        
        # Also save CSV summary
        csv_path = self.output_dir / 'model_comparison.csv'
        with open(csv_path, 'w') as f:
            f.write("Model,Model Type,Exact Match Acc,Char Acc,F1 Micro,F1 Macro,Avg Confidence,Avg Time\n")
            for model_key, metrics in all_metrics.items():
                model_type = self.models.get(model_key, {}).get('type', 'unknown')
                f.write(f"{metrics.model_name},{model_type},{metrics.exact_match_accuracy:.4f},"
                       f"{metrics.character_accuracy:.4f},{metrics.f1_micro:.4f},"
                       f"{metrics.f1_macro:.4f},{metrics.avg_confidence:.4f},"
                       f"{metrics.avg_processing_time:.4f}\n")
        
        print(f"📁 CSV saved to: {csv_path}")
        
        # Save combined sample results CSV with model information
        sample_results_path = self.output_dir / 'sample_results.csv'
        with open(sample_results_path, 'w') as f:
            f.write("model_name,model_type,image,ground_truth,prediction,exact_match,chars_correct,char_accuracy,confidence,processing_time\n")
            for model_key, metrics in all_metrics.items():
                for sample in metrics.sample_results:
                    f.write(f"{sample.get('model_name', metrics.model_name)},"
                           f"{sample.get('model_type', 'unknown')},"
                           f"{sample['image']},"
                           f"{sample['ground_truth']},"
                           f"{sample['prediction']},"
                           f"{sample['exact_match']},"
                           f"{sample['chars_correct']},"
                           f"{sample['char_accuracy']:.4f},"
                           f"{sample['confidence']:.4f},"
                           f"{sample['processing_time']:.4f}\n")
        
        print(f"📁 Sample results saved to: {sample_results_path}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Multi-Model VIN Recognition Evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Evaluation Modes:
  recognition  - Tests raw OCR recognition using default/pretrained weights (default)
  finetuned    - Evaluates custom fine-tuned models (requires trained models)
  production   - Tests ONNX exported models for production deployment validation

Note: For proper fine-tuned model evaluation, export models to ONNX format:
  python -m src.vin_ocr.training.export_onnx --model-path output/vin_rec_finetune/best_accuracy

Examples:
  # Standard recognition evaluation
  python -m src.vin_ocr.evaluation.multi_model_evaluation

  # Evaluate with ONNX models for production
  python -m src.vin_ocr.evaluation.multi_model_evaluation --mode production

  # Evaluate on custom dataset
  python -m src.vin_ocr.evaluation.multi_model_evaluation --labels-file data/test_labels.txt
        """
    )
    parser.add_argument('--max-images', type=int, default=None,
                       help='Maximum number of images to evaluate (default: all)')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Output directory for results')
    parser.add_argument('--image-folder', type=str, default=None,
                       help='Custom image folder path (images should have VIN in filename)')
    parser.add_argument('--labels-file', type=str, default=None,
                       help='Labels file with format: image_path\\tVIN (one per line)')
    parser.add_argument('--mode', type=str, default='recognition',
                       choices=['recognition', 'finetuned', 'production'],
                       help='Evaluation mode (default: recognition)')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("VIN OCR Multi-Model Evaluation")
    print("="*70)
    print(f"Mode: {args.mode.upper()}")
    if args.mode == 'recognition':
        print("  → Testing raw OCR recognition with default/pretrained weights")
        print("  → This does NOT evaluate fine-tuned model performance")
    elif args.mode == 'finetuned':
        print("  → Evaluating fine-tuned models")
        print("  → For best results, export models to ONNX format first")
    elif args.mode == 'production':
        print("  → Testing ONNX models for production deployment")
        print("  → Requires: pip install onnxruntime")
    print("="*70 + "\n")
    
    evaluator = MultiModelEvaluator(output_dir=args.output_dir, evaluation_mode=args.mode)
    evaluator.run_evaluation(
        max_images=args.max_images,
        custom_image_folder=args.image_folder,
        labels_file=args.labels_file
    )


if __name__ == '__main__':
    main()