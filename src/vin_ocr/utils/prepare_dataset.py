#!/usr/bin/env python3
"""
Dataset Preparation for PaddleOCR VIN Training
===============================================

Prepares the dataset for training by:
1. Splitting images into train/val/test sets
2. Creating PaddleOCR-compatible label files
3. Validating the dataset structure

Usage:
    python prepare_dataset.py --data-dir data --output-dir dataset
    python prepare_dataset.py --data-dir data --output-dir dataset --train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15

Author: JRL-VIN Project
Date: January 2026
"""

import argparse
import json
import logging
import os
import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Import from shared utilities (Single Source of Truth)
from src.vin_ocr.core.vin_utils import extract_vin_from_filename, VIN_LENGTH, VIN_VALID_CHARS
from config import get_config

logger = logging.getLogger(__name__)


def load_images_with_labels(data_dir: str) -> Dict[str, str]:
    """Load all images and extract VIN labels from filenames."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    data_path = Path(data_dir)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    images = {}
    failed = []
    
    for img_file in data_path.iterdir():
        if img_file.suffix.lower() in image_extensions:
            vin = extract_vin_from_filename(img_file.name)
            if vin:
                images[str(img_file)] = vin
            else:
                failed.append(img_file.name)
    
    if failed:
        print(f"Warning: Could not extract VIN from {len(failed)} files:")
        for f in failed[:5]:
            print(f"  - {f}")
        if len(failed) > 5:
            print(f"  ... and {len(failed) - 5} more")
    
    return images


def create_splits(
    images: Dict[str, str],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
    """Split images into train/val/test sets."""
    
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.01, \
        "Ratios must sum to 1.0"
    
    # Shuffle paths
    paths = list(images.keys())
    random.seed(seed)
    random.shuffle(paths)
    
    # Calculate split indices
    n = len(paths)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    train_paths = paths[:n_train]
    val_paths = paths[n_train:n_train + n_val]
    test_paths = paths[n_train + n_val:]
    
    train = {p: images[p] for p in train_paths}
    val = {p: images[p] for p in val_paths}
    test = {p: images[p] for p in test_paths}
    
    return train, val, test


def create_paddleocr_labels(
    images: Dict[str, str],
    output_dir: Path,
    split_name: str,
    copy_images: bool = True
) -> str:
    """
    Create PaddleOCR-compatible label file.
    
    PaddleOCR label format (for recognition):
    image_path\tlabel
    
    Returns path to label file.
    """
    split_dir = output_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    
    label_file = output_dir / f"{split_name}_labels.txt"
    
    with open(label_file, 'w', encoding='utf-8') as f:
        for img_path, vin in images.items():
            src = Path(img_path)
            
            if copy_images:
                # Copy image to split directory
                dst = split_dir / src.name
                shutil.copy2(src, dst)
                # Use relative path in label file
                rel_path = f"{split_name}/{src.name}"
            else:
                rel_path = str(src)
            
            # PaddleOCR format: path\tlabel
            f.write(f"{rel_path}\t{vin}\n")
    
    return str(label_file)


def create_dataset_config(output_dir: Path, train_file: str, val_file: str, test_file: str):
    """Create dataset configuration file."""
    config = {
        "dataset_name": "vin_plates",
        "description": "Vehicle Identification Number (VIN) plate images",
        "train_label_file": train_file,
        "val_label_file": val_file,
        "test_label_file": test_file,
        "charset": "0123456789ABCDEFGHJKLMNPRSTUVWXYZ",
        "max_text_length": 17,
        "image_shape": [3, 32, 320],
        "created_at": "2026-01-23"
    }
    
    config_file = output_dir / "dataset_config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    return str(config_file)


def prepare_dataset(
    data_dir: str,
    output_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    copy_images: bool = True
) -> Dict:
    """
    Main function to prepare dataset for training.
    
    Returns dict with dataset statistics.
    """
    print("=" * 60)
    print("Preparing Dataset for PaddleOCR Training")
    print("=" * 60)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load images
    print(f"\n[1/4] Loading images from: {data_dir}")
    images = load_images_with_labels(data_dir)
    print(f"      Found {len(images)} images with valid VIN labels")
    
    if len(images) == 0:
        raise ValueError("No images with valid VIN labels found!")
    
    # Step 2: Create splits
    print(f"\n[2/4] Creating splits (train={train_ratio:.0%}, val={val_ratio:.0%}, test={test_ratio:.0%})")
    train, val, test = create_splits(images, train_ratio, val_ratio, test_ratio, seed)
    print(f"      Train: {len(train)} images")
    print(f"      Val:   {len(val)} images")
    print(f"      Test:  {len(test)} images")
    
    # Step 3: Create label files
    print(f"\n[3/4] Creating PaddleOCR label files in: {output_dir}")
    train_file = create_paddleocr_labels(train, output_path, "train", copy_images)
    val_file = create_paddleocr_labels(val, output_path, "val", copy_images)
    test_file = create_paddleocr_labels(test, output_path, "test", copy_images)
    print(f"      Created: {train_file}")
    print(f"      Created: {val_file}")
    print(f"      Created: {test_file}")
    
    # Step 4: Create config
    print(f"\n[4/4] Creating dataset configuration")
    config_file = create_dataset_config(output_path, train_file, val_file, test_file)
    print(f"      Created: {config_file}")
    
    # Summary
    stats = {
        "total_images": len(images),
        "train_images": len(train),
        "val_images": len(val),
        "test_images": len(test),
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "seed": seed,
        "output_dir": str(output_path),
        "train_label_file": train_file,
        "val_label_file": val_file,
        "test_label_file": test_file,
        "config_file": config_file
    }
    
    # Save stats
    stats_file = output_path / "dataset_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Dataset Preparation Complete!")
    print("=" * 60)
    print(f"\nOutput directory: {output_path}")
    print(f"Total images: {len(images)}")
    print(f"  - Train: {len(train)} ({len(train)/len(images)*100:.1f}%)")
    print(f"  - Val:   {len(val)} ({len(val)/len(images)*100:.1f}%)")
    print(f"  - Test:  {len(test)} ({len(test)/len(images)*100:.1f}%)")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Prepare dataset for PaddleOCR VIN training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python prepare_dataset.py --data-dir data --output-dir dataset
    python prepare_dataset.py --data-dir data --output-dir dataset --train-ratio 0.8
        """
    )
    parser.add_argument('--data-dir', required=True, help='Directory containing VIN images')
    parser.add_argument('--output-dir', default='dataset', help='Output directory for prepared dataset')
    parser.add_argument('--train-ratio', type=float, default=0.7, help='Training set ratio (default: 0.7)')
    parser.add_argument('--val-ratio', type=float, default=0.15, help='Validation set ratio (default: 0.15)')
    parser.add_argument('--test-ratio', type=float, default=0.15, help='Test set ratio (default: 0.15)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--no-copy', action='store_true', help='Do not copy images (use original paths)')
    
    args = parser.parse_args()
    
    prepare_dataset(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        copy_images=not args.no_copy
    )


if __name__ == '__main__':
    main()
