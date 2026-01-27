#!/usr/bin/env python3
"""
Data Preparation for PaddleOCR Fine-Tuning
==========================================

Converts the VIN dataset into PaddleOCR training format:
- Organizes images into train/val directories
- Creates label files in PaddleOCR format: "image_path\tlabel"
- Validates VIN labels and image integrity
- Optionally crops VIN regions if bounding boxes are available

Usage:
    python scripts/prepare_finetune_data.py \
        --input-dir ./data \
        --output-dir ./finetune_data \
        --train-ratio 0.9 \
        --val-ratio 0.1

Author: JRL-VIN Project
Date: January 2026
"""

import argparse
import os
import sys
import shutil
import random
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from vin_utils import extract_vin_from_filename, VIN_LENGTH, VIN_VALID_CHARS

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# VIN VALIDATION
# =============================================================================

def is_valid_vin_label(vin: str) -> bool:
    """Check if VIN label is valid for training."""
    if not vin or len(vin) != VIN_LENGTH:
        return False
    # Check all characters are in valid VIN charset
    return all(c in VIN_VALID_CHARS for c in vin.upper())


def validate_image(image_path: str) -> Tuple[bool, Optional[Tuple[int, int]]]:
    """
    Validate image can be loaded and get dimensions.
    
    Returns:
        (is_valid, (height, width)) or (False, None)
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return False, None
        h, w = img.shape[:2]
        if h < 10 or w < 10:
            return False, None
        return True, (h, w)
    except Exception:
        return False, None


# =============================================================================
# IMAGE PREPROCESSING FOR TRAINING
# =============================================================================

def preprocess_image_for_training(
    image: np.ndarray,
    target_height: int = 48,
    max_width: int = 320,
    pad_value: int = 0
) -> np.ndarray:
    """
    Preprocess image for PaddleOCR recognition training.
    
    - Resize to target height while maintaining aspect ratio
    - Pad or crop to max width
    - Apply CLAHE for contrast enhancement (optional)
    
    Args:
        image: Input BGR image
        target_height: Target height (default 48 for PP-OCRv4)
        max_width: Maximum width (default 320)
        pad_value: Padding value (0 = black)
        
    Returns:
        Preprocessed image
    """
    h, w = image.shape[:2]
    
    # Calculate new width maintaining aspect ratio
    ratio = target_height / h
    new_width = int(w * ratio)
    
    # Resize
    resized = cv2.resize(image, (new_width, target_height), interpolation=cv2.INTER_LINEAR)
    
    # Pad or crop to max_width
    if new_width < max_width:
        # Pad on the right
        padded = np.full((target_height, max_width, 3), pad_value, dtype=np.uint8)
        padded[:, :new_width, :] = resized
        return padded
    elif new_width > max_width:
        # Center crop
        start = (new_width - max_width) // 2
        return resized[:, start:start + max_width, :]
    else:
        return resized


def apply_clahe_enhancement(image: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """Apply CLAHE contrast enhancement for better training on engraved plates."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Convert back to 3-channel
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


# =============================================================================
# DATA PREPARATION
# =============================================================================

def collect_samples(
    input_dir: str,
    extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
) -> List[Tuple[str, str]]:
    """
    Collect image paths and extract VIN labels from filenames.
    
    Returns:
        List of (image_path, vin_label) tuples
    """
    samples = []
    input_path = Path(input_dir)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    # Recursively find all images
    for ext in extensions:
        for img_file in input_path.rglob(f'*{ext}'):
            vin = extract_vin_from_filename(img_file.name)
            if vin and is_valid_vin_label(vin):
                samples.append((str(img_file), vin.upper()))
    
    # Also check for lowercase extensions
    for ext in extensions:
        for img_file in input_path.rglob(f'*{ext.upper()}'):
            vin = extract_vin_from_filename(img_file.name)
            if vin and is_valid_vin_label(vin):
                sample = (str(img_file), vin.upper())
                if sample not in samples:
                    samples.append(sample)
    
    return samples


def process_single_image(
    args: Tuple[str, str, str, bool, bool]
) -> Optional[Tuple[str, str]]:
    """
    Process a single image for training.
    
    Args:
        args: (source_path, dest_path, vin_label, apply_preprocess, apply_clahe)
        
    Returns:
        (relative_path, vin_label) or None if failed
    """
    source_path, dest_path, vin_label, apply_preprocess, apply_clahe = args
    
    try:
        # Load image
        img = cv2.imread(source_path)
        if img is None:
            return None
        
        # Optional CLAHE enhancement
        if apply_clahe:
            img = apply_clahe_enhancement(img)
        
        # Optional preprocessing
        if apply_preprocess:
            img = preprocess_image_for_training(img)
        
        # Save processed image
        cv2.imwrite(dest_path, img)
        
        # Return relative path for label file
        rel_path = Path(dest_path).name
        return (rel_path, vin_label)
        
    except Exception as e:
        logger.warning(f"Failed to process {source_path}: {e}")
        return None


def prepare_dataset(
    input_dir: str,
    output_dir: str,
    train_ratio: float = 0.9,
    val_ratio: float = 0.1,
    seed: int = 42,
    apply_preprocess: bool = False,
    apply_clahe: bool = True,
    num_workers: int = 8,
    validate_images: bool = True
) -> Dict[str, int]:
    """
    Prepare dataset for PaddleOCR fine-tuning.
    
    Args:
        input_dir: Directory containing source images
        output_dir: Output directory for prepared data
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        seed: Random seed for reproducibility
        apply_preprocess: Resize/pad images to training dimensions
        apply_clahe: Apply CLAHE contrast enhancement
        num_workers: Number of parallel workers
        validate_images: Validate images can be loaded
        
    Returns:
        Dict with statistics
    """
    assert abs(train_ratio + val_ratio - 1.0) < 0.01, "Ratios must sum to 1.0"
    
    output_path = Path(output_dir)
    train_dir = output_path / "train"
    val_dir = output_path / "val"
    
    # Create directories
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Collecting samples from {input_dir}...")
    samples = collect_samples(input_dir)
    logger.info(f"Found {len(samples)} valid samples")
    
    if len(samples) == 0:
        raise ValueError("No valid samples found!")
    
    # Validate images if requested
    if validate_images:
        logger.info("Validating images...")
        valid_samples = []
        for img_path, vin in samples:
            is_valid, _ = validate_image(img_path)
            if is_valid:
                valid_samples.append((img_path, vin))
            else:
                logger.warning(f"Invalid image: {img_path}")
        samples = valid_samples
        logger.info(f"{len(samples)} valid images after validation")
    
    # Shuffle and split
    random.seed(seed)
    random.shuffle(samples)
    
    n_train = int(len(samples) * train_ratio)
    train_samples = samples[:n_train]
    val_samples = samples[n_train:]
    
    logger.info(f"Split: {len(train_samples)} train, {len(val_samples)} val")
    
    # Process training samples
    logger.info("Processing training samples...")
    train_tasks = []
    for i, (src_path, vin) in enumerate(train_samples):
        ext = Path(src_path).suffix
        dest_path = str(train_dir / f"train_{i:06d}{ext}")
        train_tasks.append((src_path, dest_path, vin, apply_preprocess, apply_clahe))
    
    train_results = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_single_image, task): task for task in train_tasks}
        for future in as_completed(futures):
            result = future.result()
            if result:
                train_results.append(result)
    
    # Process validation samples
    logger.info("Processing validation samples...")
    val_tasks = []
    for i, (src_path, vin) in enumerate(val_samples):
        ext = Path(src_path).suffix
        dest_path = str(val_dir / f"val_{i:06d}{ext}")
        val_tasks.append((src_path, dest_path, vin, apply_preprocess, apply_clahe))
    
    val_results = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_single_image, task): task for task in val_tasks}
        for future in as_completed(futures):
            result = future.result()
            if result:
                val_results.append(result)
    
    # Write label files
    logger.info("Writing label files...")
    
    with open(output_path / "train_labels.txt", 'w') as f:
        for rel_path, vin in train_results:
            f.write(f"train/{rel_path}\t{vin}\n")
    
    with open(output_path / "val_labels.txt", 'w') as f:
        for rel_path, vin in val_results:
            f.write(f"val/{rel_path}\t{vin}\n")
    
    # Write combined labels for reference
    with open(output_path / "all_labels.txt", 'w') as f:
        for rel_path, vin in train_results:
            f.write(f"train/{rel_path}\t{vin}\n")
        for rel_path, vin in val_results:
            f.write(f"val/{rel_path}\t{vin}\n")
    
    stats = {
        'total_source': len(samples),
        'train_processed': len(train_results),
        'val_processed': len(val_results),
        'train_failed': len(train_samples) - len(train_results),
        'val_failed': len(val_samples) - len(val_results),
    }
    
    logger.info(f"Dataset preparation complete!")
    logger.info(f"  Train: {stats['train_processed']} samples")
    logger.info(f"  Val: {stats['val_processed']} samples")
    logger.info(f"  Output: {output_dir}")
    
    # Write stats
    import json
    with open(output_path / "dataset_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    return stats


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Prepare VIN dataset for PaddleOCR fine-tuning"
    )
    parser.add_argument(
        '--input-dir', '-i',
        required=True,
        help='Input directory containing VIN images'
    )
    parser.add_argument(
        '--output-dir', '-o',
        default='./finetune_data',
        help='Output directory for prepared data'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.9,
        help='Training split ratio (default: 0.9)'
    )
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.1,
        help='Validation split ratio (default: 0.1)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--preprocess',
        action='store_true',
        help='Apply preprocessing (resize/pad to training dimensions)'
    )
    parser.add_argument(
        '--no-clahe',
        action='store_true',
        help='Disable CLAHE contrast enhancement'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=8,
        help='Number of parallel workers'
    )
    parser.add_argument(
        '--skip-validation',
        action='store_true',
        help='Skip image validation (faster but may include corrupt images)'
    )
    
    args = parser.parse_args()
    
    stats = prepare_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        apply_preprocess=args.preprocess,
        apply_clahe=not args.no_clahe,
        num_workers=args.workers,
        validate_images=not args.skip_validation
    )
    
    print("\n" + "=" * 50)
    print("Dataset Preparation Complete")
    print("=" * 50)
    print(f"Train samples: {stats['train_processed']}")
    print(f"Val samples:   {stats['val_processed']}")
    print(f"\nNext steps:")
    print(f"  1. Review prepared data in: {args.output_dir}")
    print(f"  2. Run fine-tuning:")
    print(f"     python finetune_paddleocr.py --config configs/vin_finetune_config.yml")


if __name__ == '__main__':
    main()
