#!/usr/bin/env python3
"""
Dataset Preparation Script
==========================

This script properly splits VIN images into train, validation, and test sets.

It ensures:
1. No data leakage between splits
2. Stratified splitting by unique VINs (same VIN stays in same split)
3. Proper label file format for training
4. Dataset statistics and validation

Usage:
    python scripts/prepare_dataset.py --data-dir ./finetune_data --output-dir ./finetune_data

Author: VIN OCR Pipeline
"""

import os
import sys
import re
import random
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import shutil

# Add project root to path for imports
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Try to import canonical VIN utilities (Single Source of Truth)
try:
    from src.vin_ocr.core.vin_utils import extract_vin_from_filename as _canonical_extract_vin
    _VIN_UTILS_AVAILABLE = True
except ImportError:
    _VIN_UTILS_AVAILABLE = False
    _canonical_extract_vin = None


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
    if _VIN_UTILS_AVAILABLE and _canonical_extract_vin is not None:
        return _canonical_extract_vin(filename)
    
    # Fallback: extraction logic (less comprehensive)
    # This fallback is only used if vin_utils import fails
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


def load_existing_labels(label_file: str) -> Dict[str, str]:
    """Load existing label file into dict {filename: vin}."""
    labels = {}
    if not Path(label_file).exists():
        return labels
    
    with open(label_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Handle both tab and space-separated formats
            if '\t' in line:
                parts = line.split('\t', 1)
            else:
                parts = line.split(None, 1)
            
            if len(parts) >= 2:
                labels[parts[0].strip()] = parts[1].strip()
    
    return labels


def get_all_images(data_dir: str) -> List[str]:
    """Get all image files in directory."""
    data_path = Path(data_dir)
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP'}
    
    images = []
    for ext in extensions:
        images.extend(data_path.glob(f'*{ext}'))
    
    return [img.name for img in images]


def group_by_vin(images: List[str], labels: Dict[str, str]) -> Dict[str, List[str]]:
    """Group images by their VIN to prevent data leakage."""
    vin_to_images = defaultdict(list)
    
    for img in images:
        # Get VIN from labels or filename
        vin = labels.get(img) or extract_vin_from_filename(img)
        if vin:
            vin_to_images[vin].append(img)
        else:
            print(f"⚠️ Could not extract VIN for: {img}")
    
    return vin_to_images


def stratified_split(
    vin_to_images: Dict[str, List[str]],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    Split data by VIN to prevent data leakage.
    
    Returns:
        train_samples: List of (filename, vin) tuples
        val_samples: List of (filename, vin) tuples  
        test_samples: List of (filename, vin) tuples
    """
    random.seed(seed)
    
    # Get all unique VINs
    vins = list(vin_to_images.keys())
    random.shuffle(vins)
    
    # Calculate split indices
    n_vins = len(vins)
    n_train = int(n_vins * train_ratio)
    n_val = int(n_vins * val_ratio)
    
    train_vins = set(vins[:n_train])
    val_vins = set(vins[n_train:n_train + n_val])
    test_vins = set(vins[n_train + n_val:])
    
    # Collect samples
    train_samples = []
    val_samples = []
    test_samples = []
    
    for vin, images in vin_to_images.items():
        for img in images:
            sample = (img, vin)
            if vin in train_vins:
                train_samples.append(sample)
            elif vin in val_vins:
                val_samples.append(sample)
            else:
                test_samples.append(sample)
    
    return train_samples, val_samples, test_samples


def write_labels_file(samples: List[Tuple[str, str]], output_path: str):
    """Write samples to label file in PaddleOCR format."""
    with open(output_path, 'w') as f:
        for filename, vin in samples:
            # Use tab separator (PaddleOCR standard)
            f.write(f"{filename}\t{vin}\n")
    print(f"  ✓ Wrote {len(samples)} samples to {output_path}")


def validate_split(
    train_samples: List[Tuple[str, str]],
    val_samples: List[Tuple[str, str]],
    test_samples: List[Tuple[str, str]]
):
    """Validate there's no data leakage between splits."""
    train_vins = set(vin for _, vin in train_samples)
    val_vins = set(vin for _, vin in val_samples)
    test_vins = set(vin for _, vin in test_samples)
    
    # Check for overlaps
    train_val_overlap = train_vins & val_vins
    train_test_overlap = train_vins & test_vins
    val_test_overlap = val_vins & test_vins
    
    if train_val_overlap:
        print(f"❌ LEAK: {len(train_val_overlap)} VINs in both train and val!")
    if train_test_overlap:
        print(f"❌ LEAK: {len(train_test_overlap)} VINs in both train and test!")
    if val_test_overlap:
        print(f"❌ LEAK: {len(val_test_overlap)} VINs in both val and test!")
    
    if not (train_val_overlap or train_test_overlap or val_test_overlap):
        print("✅ No data leakage detected!")
    
    return not (train_val_overlap or train_test_overlap or val_test_overlap)


def print_statistics(
    train_samples: List[Tuple[str, str]],
    val_samples: List[Tuple[str, str]],
    test_samples: List[Tuple[str, str]]
):
    """Print dataset statistics."""
    total = len(train_samples) + len(val_samples) + len(test_samples)
    
    train_vins = len(set(vin for _, vin in train_samples))
    val_vins = len(set(vin for _, vin in val_samples))
    test_vins = len(set(vin for _, vin in test_samples))
    
    print("\n" + "=" * 60)
    print("📊 DATASET STATISTICS")
    print("=" * 60)
    print(f"\n{'Split':<15} {'Images':>10} {'Unique VINs':>15} {'Percentage':>12}")
    print("-" * 52)
    print(f"{'Training':<15} {len(train_samples):>10} {train_vins:>15} {len(train_samples)/total*100:>11.1f}%")
    print(f"{'Validation':<15} {len(val_samples):>10} {val_vins:>15} {len(val_samples)/total*100:>11.1f}%")
    print(f"{'Test':<15} {len(test_samples):>10} {test_vins:>15} {len(test_samples)/total*100:>11.1f}%")
    print("-" * 52)
    print(f"{'TOTAL':<15} {total:>10} {train_vins + val_vins + test_vins:>15} {'100.0%':>12}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Prepare VIN dataset with proper train/val/test splits')
    parser.add_argument('--data-dir', default='./finetune_data', help='Directory containing images')
    parser.add_argument('--output-dir', default='./finetune_data', help='Output directory for label files')
    parser.add_argument('--train-ratio', type=float, default=0.80, help='Training split ratio (default: 0.80)')
    parser.add_argument('--val-ratio', type=float, default=0.10, help='Validation split ratio (default: 0.10)')
    parser.add_argument('--test-ratio', type=float, default=0.10, help='Test split ratio (default: 0.10)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--existing-labels', help='Existing labels file to use (optional)')
    
    args = parser.parse_args()
    
    # Validate ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 0.01:
        print(f"❌ Error: Ratios must sum to 1.0 (got {total_ratio})")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("🔧 VIN DATASET PREPARATION")
    print("=" * 60)
    print(f"\n📁 Data directory: {args.data_dir}")
    print(f"📂 Output directory: {args.output_dir}")
    print(f"🎲 Random seed: {args.seed}")
    print(f"📊 Split ratios: Train={args.train_ratio:.0%}, Val={args.val_ratio:.0%}, Test={args.test_ratio:.0%}")
    
    # Load existing labels if provided
    labels = {}
    if args.existing_labels:
        print(f"\n📜 Loading existing labels from: {args.existing_labels}")
        labels = load_existing_labels(args.existing_labels)
        print(f"  ✓ Loaded {len(labels)} existing labels")
    else:
        # Try to load from default locations
        for label_file in ['train_labels.txt', 'val_labels.txt', 'labels.txt']:
            label_path = Path(args.data_dir) / label_file
            if label_path.exists():
                file_labels = load_existing_labels(str(label_path))
                labels.update(file_labels)
                print(f"  ✓ Loaded {len(file_labels)} labels from {label_file}")
    
    # Get all images
    print(f"\n🔍 Scanning for images in {args.data_dir}...")
    images = get_all_images(args.data_dir)
    print(f"  ✓ Found {len(images)} images")
    
    if len(images) == 0:
        print("❌ No images found!")
        sys.exit(1)
    
    # Group by VIN
    print("\n🔗 Grouping images by VIN...")
    vin_to_images = group_by_vin(images, labels)
    print(f"  ✓ Found {len(vin_to_images)} unique VINs")
    
    # Perform stratified split
    print("\n✂️ Splitting dataset...")
    train_samples, val_samples, test_samples = stratified_split(
        vin_to_images,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )
    
    # Validate split
    print("\n🔒 Validating data integrity...")
    is_valid = validate_split(train_samples, val_samples, test_samples)
    
    if not is_valid:
        print("❌ Data validation failed!")
        sys.exit(1)
    
    # Print statistics
    print_statistics(train_samples, val_samples, test_samples)
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Write label files
    print("\n📝 Writing label files...")
    write_labels_file(train_samples, output_path / 'train_labels.txt')
    write_labels_file(val_samples, output_path / 'val_labels.txt')
    write_labels_file(test_samples, output_path / 'test_labels.txt')
    
    # Show sample entries
    print("\n📋 Sample entries from each split:")
    print("\n  TRAINING (first 3):")
    for filename, vin in train_samples[:3]:
        print(f"    {filename[:50]}... → {vin}")
    
    print("\n  VALIDATION (first 3):")
    for filename, vin in val_samples[:3]:
        print(f"    {filename[:50]}... → {vin}")
    
    print("\n  TEST (first 3):")
    for filename, vin in test_samples[:3]:
        print(f"    {filename[:50]}... → {vin}")
    
    print("\n" + "=" * 60)
    print("✅ DATASET PREPARATION COMPLETE!")
    print("=" * 60)
    print(f"\nLabel files created:")
    print(f"  • {output_path / 'train_labels.txt'} ({len(train_samples)} samples)")
    print(f"  • {output_path / 'val_labels.txt'} ({len(val_samples)} samples)")
    print(f"  • {output_path / 'test_labels.txt'} ({len(test_samples)} samples)")
    print("\nNext steps:")
    print("  1. Review the label files")
    print("  2. Start training with: python -m src.vin_ocr.training.finetune_paddleocr")
    print("  3. Evaluate on test set after training")
    print()


if __name__ == '__main__':
    main()
