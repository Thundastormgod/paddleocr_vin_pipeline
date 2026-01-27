#!/usr/bin/env python3
"""
Multi-Model VIN Recognition Evaluation
=======================================

This script evaluates multiple OCR models/approaches on all available VIN images
and produces comprehensive comparison metrics including F1 Micro and F1 Macro.

Models evaluated:
1. PaddleOCR PP-OCRv4 (default pretrained)
2. Fine-tuned VIN Model (if available)
3. PaddleOCR with VIN-specific post-processing

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
from vin_utils import extract_vin_from_filename

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
    """Evaluates multiple OCR models on VIN images."""
    
    def __init__(self, output_dir: str = "results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.models = {}
        self.results = []
        
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
            from vin_pipeline import VINOCRPipeline
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
            from ocr_providers import DeepSeekOCRProvider, DeepSeekOCRConfig
            
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
        
        print(f"\n  Total models loaded: {len(self.models)}")
        print("=" * 60)
    
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
                'processing_time': proc_time
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
        
        results = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'evaluation_type': 'multi_model_comparison',
            },
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
            f.write("Model,Exact Match Acc,Char Acc,F1 Micro,F1 Macro,Avg Confidence,Avg Time\n")
            for metrics in all_metrics.values():
                f.write(f"{metrics.model_name},{metrics.exact_match_accuracy:.4f},"
                       f"{metrics.character_accuracy:.4f},{metrics.f1_micro:.4f},"
                       f"{metrics.f1_macro:.4f},{metrics.avg_confidence:.4f},"
                       f"{metrics.avg_processing_time:.4f}\n")
        
        print(f"📁 CSV saved to: {csv_path}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Multi-Model VIN Recognition Evaluation')
    parser.add_argument('--max-images', type=int, default=None,
                       help='Maximum number of images to evaluate (default: all)')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Output directory for results')
    parser.add_argument('--image-folder', type=str, default=None,
                       help='Custom image folder path (images should have VIN in filename)')
    parser.add_argument('--labels-file', type=str, default=None,
                       help='Labels file with format: image_path\\tVIN (one per line)')
    
    args = parser.parse_args()
    
    evaluator = MultiModelEvaluator(output_dir=args.output_dir)
    evaluator.run_evaluation(
        max_images=args.max_images,
        custom_image_folder=args.image_folder,
        labels_file=args.labels_file
    )


if __name__ == '__main__':
    main()
