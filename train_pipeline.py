#!/usr/bin/env python3
"""
Training Pipeline for PaddleOCR VIN Recognition
================================================

This module provides TWO training approaches:

1. RULE-BASED LEARNING (fast, no GPU required):
   - Runs pretrained OCR on training images
   - Learns character confusion patterns
   - Generates deterministic correction rules
   - Good for quick improvements with limited data

2. NEURAL NETWORK FINE-TUNING (industry-standard):
   - Fine-tunes PP-OCRv4 recognition model weights
   - Requires PaddlePaddle with GPU support
   - Recommended for 1000+ training images
   - Achieves highest accuracy for VIN recognition

Usage:
    # Rule-based learning (quick)
    python train_pipeline.py --dataset-dir dataset --method rules
    
    # Neural network fine-tuning (recommended for 11k+ images)
    python train_pipeline.py --dataset-dir dataset --method finetune --epochs 100
    
    # Or use dedicated fine-tuning script directly:
    python finetune_paddleocr.py --config configs/vin_finetune_config.yml

Author: JRL-VIN Project
Date: January 2026
"""

import argparse
import json
import logging
import os
import sys
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

# Import from shared utilities (Single Source of Truth)
from vin_utils import VIN_LENGTH, VIN_VALID_CHARS
from config import get_config

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class VINTrainingPipeline:
    """Training pipeline for VIN OCR."""
    
    # Use shared constants from vin_utils
    VIN_CHARSET = VIN_VALID_CHARS
    
    def __init__(
        self,
        dataset_dir: str,
        output_dir: str = "training_output",
        batch_size: int = 8,
        epochs: int = 50,
        learning_rate: float = 0.0001,
        use_gpu: bool = True,
        augment_data: bool = True
    ):
        self.dataset_dir = Path(dataset_dir)
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.use_gpu = use_gpu
        self.augment_data = augment_data
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir = self.output_dir / "checkpoints"
        self.logs_dir = self.output_dir / "logs"
        self.checkpoints_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        
        self._validate_dataset()
        self.pipeline = None
        self.device = None
    
    def _validate_dataset(self):
        """Validate dataset structure."""
        if not (self.dataset_dir / "train_labels.txt").exists():
            raise FileNotFoundError("Missing train_labels.txt. Run prepare_dataset.py first.")
        
        self.train_count = self._count_samples("train_labels.txt")
        self.val_count = self._count_samples("val_labels.txt")
        
        print(f"Dataset: Train={self.train_count}, Val={self.val_count}")
        
        if self.train_count == 0:
            raise ValueError("No training samples!")
    
    def _count_samples(self, label_file: str) -> int:
        path = self.dataset_dir / label_file
        if not path.exists():
            return 0
        with open(path) as f:
            return sum(1 for line in f if line.strip())
    
    def _load_paddle(self):
        try:
            import paddle
            self.paddle = paddle
            if self.use_gpu and paddle.device.is_compiled_with_cuda():
                paddle.device.set_device('gpu:0')
                self.device = 'gpu'
            else:
                paddle.device.set_device('cpu')
                self.device = 'cpu'
            print(f"Using {self.device.upper()}")
            return True
        except ImportError:
            print("PaddlePaddle not installed")
            return False
    
    def _init_ocr_pipeline(self):
        if self.pipeline is None:
            sys.path.insert(0, str(Path(__file__).parent))
            from vin_pipeline import VINOCRPipeline
            self.pipeline = VINOCRPipeline()
    
    def load_dataset(self, split: str = "train") -> Tuple[List[str], List[str]]:
        label_file = self.dataset_dir / f"{split}_labels.txt"
        if not label_file.exists():
            return [], []
        
        paths, labels = [], []
        with open(label_file) as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    img = parts[0]
                    vin = parts[1]
                    if not os.path.isabs(img):
                        img = str(self.dataset_dir / img)
                    if os.path.exists(img):
                        paths.append(img)
                        labels.append(vin)
        return paths, labels
    
    def augment_image(self, image):
        import cv2
        import numpy as np
        
        aug = image.copy()
        if np.random.random() > 0.5:
            alpha = np.random.uniform(0.8, 1.2)
            beta = np.random.randint(-20, 20)
            aug = cv2.convertScaleAbs(aug, alpha=alpha, beta=beta)
        if np.random.random() > 0.5:
            angle = np.random.uniform(-3, 3)
            h, w = aug.shape[:2]
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            aug = cv2.warpAffine(aug, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        if np.random.random() > 0.7:
            ksize = np.random.choice([3, 5])
            aug = cv2.GaussianBlur(aug, (ksize, ksize), 0)
        return aug
    
    def create_augmented_dataset(self, multiplier: int = 5) -> Path:
        import cv2
        
        print(f"\nAugmenting dataset (x{multiplier})...")
        aug_dir = self.output_dir / "augmented_dataset"
        aug_dir.mkdir(exist_ok=True)
        (aug_dir / "train").mkdir(exist_ok=True)
        
        paths, labels = self.load_dataset("train")
        aug_labels = []
        
        for img_path, vin in zip(paths, labels):
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            base = Path(img_path).stem
            ext = Path(img_path).suffix
            
            orig_name = f"{base}_orig{ext}"
            cv2.imwrite(str(aug_dir / "train" / orig_name), img)
            aug_labels.append((f"train/{orig_name}", vin))
            
            for i in range(multiplier):
                aug_name = f"{base}_aug{i}{ext}"
                cv2.imwrite(str(aug_dir / "train" / aug_name), self.augment_image(img))
                aug_labels.append((f"train/{aug_name}", vin))
        
        with open(aug_dir / "train_labels.txt", 'w') as f:
            for p, v in aug_labels:
                f.write(f"{p}\t{v}\n")
        
        # Copy val if exists
        if (self.dataset_dir / "val_labels.txt").exists():
            (aug_dir / "val").mkdir(exist_ok=True)
            val_labels = []
            vp, vl = self.load_dataset("val")
            for p, v in zip(vp, vl):
                shutil.copy(p, aug_dir / "val" / Path(p).name)
                val_labels.append((f"val/{Path(p).name}", v))
            with open(aug_dir / "val_labels.txt", 'w') as f:
                for p, v in val_labels:
                    f.write(f"{p}\t{v}\n")
        
        print(f"  Orig: {len(paths)}, Aug: {len(aug_labels)}")
        return aug_dir
    
    def train_rule_learning(self, verbose: bool = True) -> Dict:
        """
        Learn correction rules from OCR errors.
        
        This is NOT neural network training. It:
        1. Runs pretrained OCR on all training images
        2. Compares predictions to ground truth
        3. Builds character confusion matrix
        4. Generates correction rules from most common confusions
        
        Returns:
            Dict with baseline accuracy, learned rules, and validation results
        """
        print("\n" + "=" * 50)
        print("Rule-Based Correction Learning")
        print("(Note: This does NOT fine-tune model weights)")
        print("=" * 50)
        
        self._init_ocr_pipeline()
        train_paths, train_labels = self.load_dataset("train")
        val_paths, val_labels = self.load_dataset("val")
        
        print(f"Train: {len(train_labels)}, Val: {len(val_labels)}")
        
        if not train_labels:
            return {"error": "No training data"}
        
        print("\nCollecting OCR predictions...")
        results = []
        for i, (p, gt) in enumerate(zip(train_paths, train_labels)):
            try:
                r = self.pipeline.recognize(p)
                pred = r.get('vin', '')
                results.append({
                    'image': Path(p).name,
                    'gt': gt, 'pred': pred,
                    'correct': gt == pred
                })
                if verbose:
                    s = "OK" if gt == pred else "X"
                    print(f"  [{i+1}/{len(train_paths)}] {s} GT:{gt} P:{pred}")
            except Exception as e:
                print(f"  Error: {e}")
        
        correct = sum(1 for r in results if r['correct'])
        baseline = correct / len(results) if results else 0
        print(f"\nBaseline: {baseline*100:.2f}%")
        
        print("Building corrections...")
        rules = self._build_correction_rules(results)
        print(f"  {len(rules)} rules")
        
        model = {
            "type": "rule_based",  # NOT transfer learning
            "timestamp": datetime.now().isoformat(),
            "baseline": baseline,
            "samples": len(results),
            "rules": rules,
            "note": "Rules learned from OCR error patterns, no model weights modified"
        }
        
        model_file = self.checkpoints_dir / "model.json"
        with open(model_file, 'w') as f:
            json.dump(model, f, indent=2)
        print(f"Saved: {model_file}")
        
        if val_paths:
            print("\nValidation...")
            val_res = self._evaluate(val_paths, val_labels, rules)
            model["val"] = val_res
            print(f"  Orig: {val_res['orig']*100:.2f}%, Corr: {val_res['corr']*100:.2f}%")
            with open(model_file, 'w') as f:
                json.dump(model, f, indent=2)
        
        return model
    
    def _build_correction_rules(self, results: List[Dict]) -> Dict:
        rules = {"I": "1", "O": "0", "Q": "0", "l": "1", "o": "0", "S": "5", "B": "8", "G": "6", "Z": "2"}
        mappings = {}
        for r in results:
            if r['correct']:
                continue
            for g, p in zip(r['gt'], r['pred']):
                if g != p:
                    if p not in mappings:
                        mappings[p] = {}
                    mappings[p][g] = mappings[p].get(g, 0) + 1
        for pc, corr in mappings.items():
            if corr:
                best = max(corr, key=corr.get)
                if best in self.VIN_CHARSET:
                    rules[pc] = best
        return rules
    
    def _evaluate(self, paths, labels, rules) -> Dict:
        results = []
        for p, gt in zip(paths, labels):
            try:
                r = self.pipeline.recognize(p)
                pred = r.get('vin', '')
                corr = self._apply_rules(pred, rules)
                results.append({'orig': gt == pred, 'corr': gt == corr})
            except Exception as e:
                # Log error but continue evaluation - one bad image shouldn't stop evaluation
                logger.warning(f"Evaluation failed for {p}: {type(e).__name__}: {e}")
                continue
        if not results:
            return {"orig": 0, "corr": 0}
        return {
            "orig": sum(r['orig'] for r in results) / len(results),
            "corr": sum(r['corr'] for r in results) / len(results)
        }
    
    def _apply_rules(self, vin: str, rules: Dict) -> str:
        out = ""
        for c in vin:
            if c in rules:
                out += rules[c]
            elif c.upper() in self.VIN_CHARSET:
                out += c.upper()
            else:
                out += c.upper()
        return out[:17]
    
    def train(self, method: str = "rules", verbose: bool = True) -> Dict:
        """
        Run the training pipeline.
        
        Args:
            method: Training method:
                - 'rules': Rule-based learning (fast, no neural network training)
                - 'finetune': Full neural network fine-tuning (recommended for 11k+ images)
                - 'transfer': Alias for 'rules' (backward compat)
                - 'full': Alias for 'finetune'
            verbose: Print progress
            
        Returns:
            Dict with training results
        """
        if not self._load_paddle():
            return {"error": "PaddlePaddle unavailable"}
        
        orig_dir = self.dataset_dir
        
        # Data augmentation for small datasets (only for rule-based)
        if self.augment_data and self.train_count < 50 and method in ('rules', 'transfer'):
            mult = max(5, 50 // max(self.train_count, 1))
            self.dataset_dir = self.create_augmented_dataset(mult)
        
        try:
            if method in ("rules", "transfer"):
                return self.train_rule_learning(verbose)
            elif method in ("finetune", "full"):
                return self._full_finetuning()
            else:
                print(f"Unknown method: {method}")
                print("Available methods: rules, finetune")
                return {"error": f"Unknown method: {method}"}
        finally:
            self.dataset_dir = orig_dir
    
    def _full_finetuning(self) -> Dict:
        """
        Full neural network fine-tuning using PaddlePaddle.
        
        This method:
        1. Prepares data in PaddleOCR format
        2. Calls the finetune_paddleocr.py script
        3. Trains the recognition model weights
        4. Exports the fine-tuned model for inference
        
        Returns:
            Dict with training results and model path
        """
        print("\n" + "=" * 60)
        print("NEURAL NETWORK FINE-TUNING")
        print("Training PP-OCRv4 recognition model weights")
        print("=" * 60)
        
        # Check if finetune script exists
        finetune_script = Path(__file__).parent / "finetune_paddleocr.py"
        if not finetune_script.exists():
            return {
                "error": "finetune_paddleocr.py not found",
                "hint": "Ensure finetune_paddleocr.py is in the project root"
            }
        
        # Prepare fine-tuning data
        finetune_data_dir = self.output_dir / "finetune_data"
        
        print(f"\n1. Preparing fine-tuning data...")
        print(f"   Source: {self.dataset_dir}")
        print(f"   Output: {finetune_data_dir}")
        
        # Create finetune data directory structure
        finetune_data_dir.mkdir(parents=True, exist_ok=True)
        train_dir = finetune_data_dir / "train"
        val_dir = finetune_data_dir / "val"
        train_dir.mkdir(exist_ok=True)
        val_dir.mkdir(exist_ok=True)
        
        # Copy and organize data
        train_paths, train_labels = self.load_dataset("train")
        val_paths, val_labels = self.load_dataset("val")
        
        print(f"   Train samples: {len(train_paths)}")
        print(f"   Val samples: {len(val_paths)}")
        
        # Create label files in PaddleOCR format
        with open(finetune_data_dir / "train_labels.txt", 'w') as f:
            for i, (src_path, label) in enumerate(zip(train_paths, train_labels)):
                ext = Path(src_path).suffix
                dest_name = f"train_{i:06d}{ext}"
                dest_path = train_dir / dest_name
                
                # Copy image
                shutil.copy(src_path, dest_path)
                
                # Write label
                f.write(f"train/{dest_name}\t{label}\n")
        
        with open(finetune_data_dir / "val_labels.txt", 'w') as f:
            for i, (src_path, label) in enumerate(zip(val_paths, val_labels)):
                ext = Path(src_path).suffix
                dest_name = f"val_{i:06d}{ext}"
                dest_path = val_dir / dest_name
                
                # Copy image
                shutil.copy(src_path, dest_path)
                
                # Write label
                f.write(f"val/{dest_name}\t{label}\n")
        
        print(f"   Data prepared successfully")
        
        # Update config for this training run
        config_path = Path(__file__).parent / "configs" / "vin_finetune_config.yml"
        
        if not config_path.exists():
            return {
                "error": f"Config file not found: {config_path}",
                "hint": "Create configs/vin_finetune_config.yml"
            }
        
        # Create runtime config with updated paths
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Update config
        config['Global']['epoch_num'] = self.epochs
        config['Global']['save_model_dir'] = str(self.output_dir / "model")
        config['Global']['use_gpu'] = self.use_gpu
        config['Train']['dataset']['data_dir'] = str(finetune_data_dir)
        config['Train']['dataset']['label_file_list'] = [str(finetune_data_dir / "train_labels.txt")]
        config['Train']['loader']['batch_size_per_card'] = self.batch_size
        config['Eval']['dataset']['data_dir'] = str(finetune_data_dir)
        config['Eval']['dataset']['label_file_list'] = [str(finetune_data_dir / "val_labels.txt")]
        config['Optimizer']['lr']['learning_rate'] = self.learning_rate
        
        runtime_config_path = self.output_dir / "runtime_config.yml"
        with open(runtime_config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"\n2. Starting neural network training...")
        print(f"   Epochs: {self.epochs}")
        print(f"   Batch size: {self.batch_size}")
        print(f"   Learning rate: {self.learning_rate}")
        print(f"   GPU: {'Enabled' if self.use_gpu else 'Disabled'}")
        print(f"   Config: {runtime_config_path}")
        
        # Run fine-tuning
        cmd = [
            sys.executable,
            str(finetune_script),
            "--config", str(runtime_config_path),
        ]
        
        if not self.use_gpu:
            cmd.append("--cpu")
        
        print(f"\n   Running: {' '.join(cmd)}")
        print("=" * 60)
        
        try:
            # Run training subprocess with real-time output streaming
            # This allows users to see progress during long training runs
            process = subprocess.Popen(
                cmd,
                cwd=str(Path(__file__).parent),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,  # Line buffered
                universal_newlines=True
            )
            
            # Stream output in real-time
            output_lines = []
            for line in process.stdout:
                print(line, end='')  # Print immediately
                output_lines.append(line)
            
            # Wait for process to complete
            return_code = process.wait()
            
            if return_code != 0:
                raise subprocess.CalledProcessError(return_code, cmd)
            
            print("=" * 60)
            print("Training completed successfully!")
            
            # Check for output model
            model_dir = self.output_dir / "model"
            best_model = model_dir / "best_accuracy.pdparams"
            inference_dir = model_dir / "inference"
            
            return {
                "method": "finetune",
                "status": "success",
                "epochs": self.epochs,
                "train_samples": len(train_paths),
                "val_samples": len(val_paths),
                "model_dir": str(model_dir),
                "best_model": str(best_model) if best_model.exists() else None,
                "inference_model": str(inference_dir) if inference_dir.exists() else None,
                "config": str(runtime_config_path),
            }
            
        except subprocess.CalledProcessError as e:
            print(f"\nTraining failed with error code {e.returncode}")
            return {
                "method": "finetune",
                "status": "failed",
                "error": str(e),
                "returncode": e.returncode,
            }
        except Exception as e:
            print(f"\nTraining failed: {e}")
            return {
                "method": "finetune",
                "status": "failed",
                "error": str(e),
            }


def main():
    parser = argparse.ArgumentParser(
        description='VIN OCR Training Pipeline - Rule-based learning and Neural Network Fine-tuning'
    )
    parser.add_argument('--dataset-dir', required=True, 
                        help='Directory with train_labels.txt and images')
    parser.add_argument('--output-dir', default='training_output',
                        help='Output directory for trained models/rules')
    parser.add_argument('--method', choices=['rules', 'finetune', 'transfer', 'full'], default='rules',
                        help='Training method: rules (fast rule-based), '
                             'finetune (neural network training for large datasets). '
                             'transfer/full are aliases for backward compat.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Training epochs (for finetune method)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size (for finetune method)')
    parser.add_argument('--learning-rate', type=float, default=0.0001,
                        help='Learning rate (for finetune method)')
    parser.add_argument('--no-gpu', action='store_true',
                        help='Disable GPU acceleration')
    parser.add_argument('--no-augment', action='store_true',
                        help='Disable data augmentation (rule-based only)')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress verbose output')
    args = parser.parse_args()
    
    pipe = VINTrainingPipeline(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        use_gpu=not args.no_gpu,
        augment_data=not args.no_augment
    )
    
    res = pipe.train(method=args.method, verbose=not args.quiet)
    
    print("\n" + "=" * 50)
    print("Training Complete!")
    
    if "error" in res:
        print(f"Error: {res['error']}")
        if "hint" in res:
            print(f"Hint: {res['hint']}")
    elif res.get("method") == "finetune":
        print(f"Method: Neural Network Fine-tuning")
        print(f"Status: {res.get('status', 'unknown')}")
        print(f"Train samples: {res.get('train_samples', 0)}")
        print(f"Val samples: {res.get('val_samples', 0)}")
        if res.get('model_dir'):
            print(f"Model directory: {res['model_dir']}")
        if res.get('inference_model'):
            print(f"Inference model: {res['inference_model']}")
    else:
        print(f"Method: Rule-based Learning")
        print(f"Baseline accuracy: {res.get('baseline', 0)*100:.2f}%")
        if 'val' in res:
            print(f"Corrected accuracy: {res['val'].get('corr', 0)*100:.2f}%")


if __name__ == '__main__':
    main()
