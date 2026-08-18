#!/usr/bin/env python3
"""
VIN OCR Pipeline Evaluation Script
===================================

Evaluates the PaddleOCR VIN pipeline against a dataset with ground truth.
Calculates industry-standard metrics and supports train/val/test splits.

Usage:
    # Evaluate on full dataset
    python -m src.vin_ocr.evaluation.evaluate --data-dir /path/to/images
    
    # Evaluate with specific split
    python -m src.vin_ocr.evaluation.evaluate --data-dir /path/to/images --split test
    
    # Create splits and evaluate
    python -m src.vin_ocr.evaluation.evaluate --data-dir /path/to/images --create-splits --split test
    
    # Export results
    python -m src.vin_ocr.evaluation.evaluate --data-dir /path/to/images --output results.json

Author: JRL-VIN Project
Date: January 2026
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.vin_ocr.pipeline.vin_pipeline import VINOCRPipeline, validate_vin

# Import from shared utilities (Single Source of Truth)
from src.vin_ocr.core.vin_utils import (
    extract_vin_from_filename,
    VIN_LENGTH,
    VIN_VALID_CHARS,
    levenshtein_distance,
)
from config import get_config

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class PredictionResult:
    """Single prediction result."""
    image_path: str
    ground_truth: str
    prediction: str
    confidence: float
    raw_ocr: str
    exact_match: bool
    edit_distance: int
    processing_time_ms: float
    error: Optional[str] = None


@dataclass 
class EvaluationMetrics:
    """Aggregated evaluation metrics."""
    # Dataset info
    total_samples: int = 0
    split_name: str = "all"
    
    # Primary metrics
    exact_match_rate: float = 0.0
    exact_match_count: int = 0
    
    # Character-level metrics
    character_precision: float = 0.0
    character_recall: float = 0.0
    character_f1: float = 0.0
    
    # Error metrics
    character_error_rate: float = 0.0  # CER
    normalized_edit_distance: float = 0.0  # NED
    mean_edit_distance: float = 0.0
    
    # Per-position accuracy (17 positions)
    position_accuracy: List[float] = field(default_factory=list)
    
    # Confidence stats
    mean_confidence: float = 0.0
    confidence_when_correct: float = 0.0
    confidence_when_wrong: float = 0.0
    
    # Processing stats
    total_processing_time_ms: float = 0.0
    mean_processing_time_ms: float = 0.0
    
    # Error counts
    failed_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class DatasetSplit:
    """Dataset split information."""
    name: str
    image_paths: List[str]
    ground_truths: Dict[str, str]  # path -> VIN


# =============================================================================
# GROUND TRUTH LOADING (uses shared extract_vin_from_filename)
# =============================================================================

def load_ground_truth(data_dir: str) -> Dict[str, str]:
    """
    Load ground truth VINs from image filenames.
    
    Args:
        data_dir: Directory containing images
        
    Returns:
        Dict mapping image path to ground truth VIN
    """
    ground_truth = {}
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    for img_file in data_path.iterdir():
        if img_file.suffix.lower() in image_extensions:
            vin = extract_vin_from_filename(img_file.name)
            if vin:
                ground_truth[str(img_file)] = vin
            else:
                print(f"Warning: Could not extract VIN from: {img_file.name}")
    
    return ground_truth


# =============================================================================
# DATASET SPLITS
# =============================================================================

def create_splits(
    ground_truth: Dict[str, str],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    output_dir: Optional[str] = None
) -> Dict[str, DatasetSplit]:
    """
    Create train/validation/test splits, GROUPED BY VIN.

    Delegates to utils.prepare_dataset.create_splits - the single splitting
    implementation in this repository - which shuffles unique VINs (not image
    paths) and verifies the result with assert_no_vin_leakage().

    This function previously shuffled image PATHS: a physical plate that
    appears in several images (multiple shots, crops, augmented copies) was
    scattered across train and test, so the scoring path measured plates the
    model had trained on. Reproduced directly before the fix: 5 VINs x 4
    images put the same VIN in train, val and test simultaneously.

    Args:
        ground_truth: Dict of image_path -> VIN
        train_ratio: Proportion of unique VINs for training
        val_ratio: Proportion of unique VINs for validation
        test_ratio: Proportion of unique VINs for testing
        seed: Random seed for reproducibility
        output_dir: Directory to save split files

    Returns:
        Dict with 'train', 'val', 'test' and 'all' DatasetSplit objects.
        Ratios apply to unique VINs, not image counts, so image counts per
        split vary with how many images each VIN has.

    Raises:
        ValueError: If a VIN would appear in more than one split (raised by
            assert_no_vin_leakage inside the canonical splitter).
    """
    # Imported here, not at module level: prepare_dataset imports the repo's
    # root config.py, which this module must not require for plain imports.
    from ..utils.prepare_dataset import create_splits as create_grouped_splits

    train_gt, val_gt, test_gt = create_grouped_splits(
        ground_truth,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    splits = {
        'train': DatasetSplit(
            name='train',
            image_paths=list(train_gt.keys()),
            ground_truths=train_gt
        ),
        'val': DatasetSplit(
            name='val',
            image_paths=list(val_gt.keys()),
            ground_truths=val_gt
        ),
        'test': DatasetSplit(
            name='test',
            image_paths=list(test_gt.keys()),
            ground_truths=test_gt
        ),
        'all': DatasetSplit(
            name='all',
            image_paths=list(ground_truth.keys()),
            ground_truths=ground_truth
        )
    }
    
    # Save split files if output_dir specified
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for split_name, split in splits.items():
            if split_name == 'all':
                continue
            split_file = output_path / f"{split_name}_split.txt"
            with open(split_file, 'w') as f:
                for path in split.image_paths:
                    f.write(f"{Path(path).name}\t{split.ground_truths[path]}\n")
            print(f"Saved {split_name} split: {len(split.image_paths)} samples -> {split_file}")
    
    return splits


def load_splits(split_dir: str, data_dir: str) -> Dict[str, DatasetSplit]:
    """
    Load existing splits from files.
    
    Args:
        split_dir: Directory containing split files
        data_dir: Directory containing images
        
    Returns:
        Dict with DatasetSplit objects
    """
    splits = {}
    split_path = Path(split_dir)
    data_path = Path(data_dir)
    
    for split_name in ['train', 'val', 'test']:
        split_file = split_path / f"{split_name}_split.txt"
        if split_file.exists():
            image_paths = []
            ground_truths = {}
            
            with open(split_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if '\t' in line:
                        filename, vin = line.split('\t')
                    else:
                        filename = line
                        vin = extract_vin_from_filename(filename)
                    
                    img_path = str(data_path / filename)
                    if Path(img_path).exists():
                        image_paths.append(img_path)
                        ground_truths[img_path] = vin
            
            splits[split_name] = DatasetSplit(
                name=split_name,
                image_paths=image_paths,
                ground_truths=ground_truths
            )
            print(f"Loaded {split_name} split: {len(image_paths)} samples")
    
    return splits


# =============================================================================
# METRICS CALCULATION
# =============================================================================

# Note: levenshtein_distance is imported from vin_utils (Single Source of Truth)


def calculate_cer(predictions: List[str], references: List[str]) -> float:
    """
    Calculate Character Error Rate (CER).
    
    CER = (S + D + I) / N
    where S=substitutions, D=deletions, I=insertions, N=total reference chars
    """
    total_errors = 0
    total_chars = 0
    
    for pred, ref in zip(predictions, references):
        total_errors += levenshtein_distance(pred, ref)
        total_chars += len(ref)
    
    return total_errors / total_chars if total_chars > 0 else 0.0


def calculate_character_metrics(
    predictions: List[str],
    references: List[str]
) -> Tuple[float, float, float]:
    """
    Calculate character-level precision, recall, and F1 using sequence ALIGNMENT.

    Counts are derived from the edit alignment between prediction and
    reference, not from a positional zip:

        equal   -> TP for each character in the block
        replace -> FP for predicted chars, FN for reference chars
        insert  -> FP (predicted characters with no reference counterpart)
        delete  -> FN (reference characters the prediction missed)

    Why this matters: the previous implementation compared pred[i] to ref[i]
    positionally, so a SINGLE inserted character at the front shifted every
    subsequent position and drove F1 to ~0.12 for a string that is 88%
    correct by edit distance — while a substitution of the same edit distance
    scored 0.94. That is ~8x more punitive for insertions than substitutions.

    Leading-artifact insertions ("*", "2E", "X" prefixes) are the documented
    dominant failure mode for these engraved plates, so the old metric
    systematically understated character-level accuracy and disagreed with
    calculate_cer() computed over the same data.

    Returns:
        Tuple of (precision, recall, f1)
    """
    # Single implementation: core.char_metrics. The opcode walk used to be
    # inlined here; three other files carried drifted copies of it, and the
    # drifted copies are what produced four different F1 values for
    # identical input.
    from ..core.char_metrics import AlignmentCounts, alignment_counts, micro_prf

    totals = AlignmentCounts()
    for pred, ref in zip(predictions, references):
        totals.update(alignment_counts(pred, ref))

    return micro_prf(totals)


def calculate_position_accuracy(
    predictions: List[str],
    references: List[str],
    vin_length: int = 17
) -> List[float]:
    """
    Calculate accuracy at each VIN position (1-17).
    
    Returns:
        List of 17 accuracy values
    """
    position_correct = [0] * vin_length
    position_total = [0] * vin_length
    
    for pred, ref in zip(predictions, references):
        # Only count if reference is proper length
        if len(ref) != vin_length:
            continue
            
        for i in range(vin_length):
            position_total[i] += 1
            if i < len(pred) and pred[i] == ref[i]:
                position_correct[i] += 1
    
    return [
        position_correct[i] / position_total[i] if position_total[i] > 0 else 0.0
        for i in range(vin_length)
    ]


# =============================================================================
# EVALUATION ENGINE
# =============================================================================

class VINEvaluator:
    """Evaluates VIN OCR pipeline on a dataset."""
    
    def __init__(
        self,
        pipeline: Optional[VINOCRPipeline] = None,
        preprocess_mode: str = 'engraved',
        verbose: bool = False
    ):
        """
        Initialize evaluator.
        
        Args:
            pipeline: Existing pipeline or None to create new
            preprocess_mode: Preprocessing mode for new pipeline
            verbose: Print verbose output
        """
        self.verbose = verbose
        
        if pipeline is None:
            print("Initializing VIN OCR Pipeline...")
            self.pipeline = VINOCRPipeline(
                preprocess_mode=preprocess_mode,
                enable_postprocess=True,
                verbose=False
            )
        else:
            self.pipeline = pipeline
    
    def evaluate(
        self,
        split: DatasetSplit,
        show_progress: bool = True,
        max_samples: Optional[int] = None
    ) -> Tuple[EvaluationMetrics, List[PredictionResult]]:
        """
        Evaluate pipeline on a dataset split.
        
        Args:
            split: DatasetSplit to evaluate
            show_progress: Show progress bar
            max_samples: Limit number of samples (for testing)
            
        Returns:
            Tuple of (metrics, list of prediction results)
        """
        image_paths = split.image_paths
        if max_samples:
            image_paths = image_paths[:max_samples]
        
        results = []
        predictions = []
        references = []
        confidences = []
        correct_confidences = []
        wrong_confidences = []
        
        total = len(image_paths)
        start_time = time.time()
        
        for i, img_path in enumerate(image_paths):
            if show_progress and (i + 1) % 10 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                eta = (total - i - 1) / rate
                print(f"Processing {i+1}/{total} ({rate:.1f} img/s, ETA: {eta:.0f}s)")
            
            ground_truth = split.ground_truths[img_path]
            
            try:
                # Run prediction
                pred_start = time.time()
                result = self.pipeline.recognize(img_path)
                pred_time = (time.time() - pred_start) * 1000
                
                prediction = result.get('vin', '')
                confidence = result.get('confidence', 0.0)
                raw_ocr = result.get('raw_ocr', '')
                error = result.get('error')
                
            except Exception as e:
                prediction = ''
                confidence = 0.0
                raw_ocr = ''
                error = str(e)
                pred_time = 0.0
            
            # Calculate metrics for this sample
            exact_match = prediction == ground_truth
            edit_dist = levenshtein_distance(prediction, ground_truth)
            
            # Store result
            pred_result = PredictionResult(
                image_path=img_path,
                ground_truth=ground_truth,
                prediction=prediction,
                confidence=confidence,
                raw_ocr=raw_ocr,
                exact_match=exact_match,
                edit_distance=edit_dist,
                processing_time_ms=pred_time,
                error=error
            )
            results.append(pred_result)
            
            # Collect for aggregate metrics
            if not error:
                predictions.append(prediction)
                references.append(ground_truth)
                confidences.append(confidence)
                
                if exact_match:
                    correct_confidences.append(confidence)
                else:
                    wrong_confidences.append(confidence)
        
        total_time = time.time() - start_time
        
        # Calculate aggregate metrics
        metrics = EvaluationMetrics(
            total_samples=len(results),
            split_name=split.name
        )
        
        # Exact match
        metrics.exact_match_count = sum(1 for r in results if r.exact_match)
        metrics.exact_match_rate = metrics.exact_match_count / len(results) if results else 0.0
        
        # Character-level metrics
        if predictions:
            precision, recall, f1 = calculate_character_metrics(predictions, references)
            metrics.character_precision = precision
            metrics.character_recall = recall
            metrics.character_f1 = f1
            
            # Error metrics
            metrics.character_error_rate = calculate_cer(predictions, references)
            
            edit_distances = [levenshtein_distance(p, r) for p, r in zip(predictions, references)]
            metrics.mean_edit_distance = sum(edit_distances) / len(edit_distances)
            
            ned_values = [
                ed / max(len(p), len(r)) if max(len(p), len(r)) > 0 else 0.0
                for ed, p, r in zip(edit_distances, predictions, references)
            ]
            metrics.normalized_edit_distance = sum(ned_values) / len(ned_values)
            
            # Position accuracy
            metrics.position_accuracy = calculate_position_accuracy(predictions, references)
            
            # Confidence stats
            metrics.mean_confidence = sum(confidences) / len(confidences)
            metrics.confidence_when_correct = (
                sum(correct_confidences) / len(correct_confidences) 
                if correct_confidences else 0.0
            )
            metrics.confidence_when_wrong = (
                sum(wrong_confidences) / len(wrong_confidences)
                if wrong_confidences else 0.0
            )
        
        # Processing stats
        metrics.total_processing_time_ms = total_time * 1000
        metrics.mean_processing_time_ms = (total_time * 1000) / len(results) if results else 0.0
        metrics.failed_count = sum(1 for r in results if r.error)
        
        return metrics, results


# =============================================================================
# REPORTING
# =============================================================================

def print_metrics_report(metrics: EvaluationMetrics):
    """Print formatted metrics report."""
    print("\n" + "=" * 70)
    print(f"EVALUATION RESULTS - {metrics.split_name.upper()} SPLIT")
    print("=" * 70)
    
    print(f"\nDataset: {metrics.total_samples} samples")
    print(f"Failed:  {metrics.failed_count} samples")
    
    print("\n--- PRIMARY METRICS ---")
    print(f"Exact Match Rate:      {metrics.exact_match_rate:.1%} ({metrics.exact_match_count}/{metrics.total_samples})")
    print(f"Character-Level F1:    {metrics.character_f1:.1%}")
    print(f"Character Precision:   {metrics.character_precision:.1%}")
    print(f"Character Recall:      {metrics.character_recall:.1%}")
    
    print("\n--- ERROR METRICS ---")
    print(f"Character Error Rate (CER):    {metrics.character_error_rate:.3f}")
    print(f"Normalized Edit Distance:      {metrics.normalized_edit_distance:.3f}")
    print(f"Mean Edit Distance:            {metrics.mean_edit_distance:.2f} chars")
    
    print("\n--- CONFIDENCE ANALYSIS ---")
    print(f"Mean Confidence:               {metrics.mean_confidence:.1%}")
    print(f"Confidence (correct preds):    {metrics.confidence_when_correct:.1%}")
    print(f"Confidence (wrong preds):      {metrics.confidence_when_wrong:.1%}")
    
    print("\n--- PROCESSING PERFORMANCE ---")
    print(f"Total Time:                    {metrics.total_processing_time_ms/1000:.1f}s")
    print(f"Mean Time per Image:           {metrics.mean_processing_time_ms:.0f}ms")
    
    if metrics.position_accuracy:
        print("\n--- PER-POSITION ACCURACY ---")
        print("Position: ", end="")
        for i in range(17):
            print(f"{i+1:>4}", end="")
        print()
        print("Accuracy: ", end="")
        for acc in metrics.position_accuracy:
            print(f"{acc:>4.0%}"[:-1], end="")
        print()
    
    print("\n" + "=" * 70)


def export_results(
    metrics: EvaluationMetrics,
    results: List[PredictionResult],
    output_path: str
):
    """Export evaluation results to JSON."""
    output = {
        'metrics': metrics.to_dict(),
        'predictions': [
            {
                'image': Path(r.image_path).name,
                'ground_truth': r.ground_truth,
                'prediction': r.prediction,
                'confidence': r.confidence,
                'exact_match': r.exact_match,
                'edit_distance': r.edit_distance,
                'error': r.error
            }
            for r in results
        ]
    }
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults exported to: {output_path}")


def export_csv(results: List[PredictionResult], output_path: str):
    """Export predictions to CSV."""
    with open(output_path, 'w') as f:
        f.write("image,ground_truth,prediction,confidence,exact_match,edit_distance,error\n")
        for r in results:
            f.write(f"{Path(r.image_path).name},{r.ground_truth},{r.prediction},"
                    f"{r.confidence:.4f},{r.exact_match},{r.edit_distance},"
                    f"{r.error or ''}\n")
    
    print(f"CSV exported to: {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate VIN OCR Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate on all images
    python -m src.vin_ocr.evaluation.evaluate --data-dir ./images
  
  # Create splits and evaluate test set
    python -m src.vin_ocr.evaluation.evaluate --data-dir ./images --create-splits --split test
  
  # Use existing splits
    python -m src.vin_ocr.evaluation.evaluate --data-dir ./images --split-dir ./splits --split val
  
  # Limit samples for quick test
    python -m src.vin_ocr.evaluation.evaluate --data-dir ./images --max-samples 50
        """
    )
    
    parser.add_argument(
        '--data-dir', '-d',
        required=True,
        help='Directory containing images'
    )
    parser.add_argument(
        '--split', '-s',
        choices=['train', 'val', 'test', 'all'],
        default='all',
        help='Which split to evaluate (default: all)'
    )
    parser.add_argument(
        '--create-splits',
        action='store_true',
        help='Create train/val/test splits'
    )
    parser.add_argument(
        '--split-dir',
        help='Directory containing split files'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.7,
        help='Training set ratio (default: 0.7)'
    )
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.15,
        help='Validation set ratio (default: 0.15)'
    )
    parser.add_argument(
        '--test-ratio',
        type=float,
        default=0.15,
        help='Test set ratio (default: 0.15)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for splits (default: 42)'
    )
    parser.add_argument(
        '--output', '-o',
        help='Output JSON file for results'
    )
    parser.add_argument(
        '--csv',
        help='Output CSV file for predictions'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        help='Maximum samples to evaluate (for testing)'
    )
    parser.add_argument(
        '--preprocess-mode',
        choices=['none', 'fast', 'balanced', 'engraved'],
        default='engraved',
        help='Preprocessing mode (default: engraved)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    
    args = parser.parse_args()
    
    # Load ground truth
    print(f"Loading ground truth from: {args.data_dir}")
    ground_truth = load_ground_truth(args.data_dir)
    print(f"Found {len(ground_truth)} images with ground truth")
    
    if len(ground_truth) == 0:
        print("Error: No images with valid ground truth found!")
        sys.exit(1)
    
    # Create or load splits
    if args.create_splits:
        print("\nCreating dataset splits...")
        split_output = args.split_dir or str(Path(args.data_dir).parent / 'splits')
        splits = create_splits(
            ground_truth,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
            output_dir=split_output
        )
    elif args.split_dir:
        print(f"\nLoading splits from: {args.split_dir}")
        splits = load_splits(args.split_dir, args.data_dir)
        # Add 'all' split
        splits['all'] = DatasetSplit(
            name='all',
            image_paths=list(ground_truth.keys()),
            ground_truths=ground_truth
        )
    else:
        # No splits, use all data
        splits = {
            'all': DatasetSplit(
                name='all',
                image_paths=list(ground_truth.keys()),
                ground_truths=ground_truth
            )
        }
    
    # Get the split to evaluate
    if args.split not in splits:
        print(f"Error: Split '{args.split}' not found!")
        print(f"Available splits: {list(splits.keys())}")
        sys.exit(1)
    
    split = splits[args.split]
    print(f"\nEvaluating {args.split} split: {len(split.image_paths)} samples")
    
    # Initialize evaluator
    evaluator = VINEvaluator(
        preprocess_mode=args.preprocess_mode,
        verbose=args.verbose
    )
    
    # Run evaluation
    metrics, results = evaluator.evaluate(
        split,
        show_progress=True,
        max_samples=args.max_samples
    )
    
    # Print report
    print_metrics_report(metrics)
    
    # Export results
    if args.output:
        export_results(metrics, results, args.output)
    
    if args.csv:
        export_csv(results, args.csv)
    
    # Print summary for CI/CD
    print("\n--- SUMMARY (for CI/CD) ---")
    print(f"EXACT_MATCH_RATE={metrics.exact_match_rate:.4f}")
    print(f"CHARACTER_F1={metrics.character_f1:.4f}")
    print(f"CER={metrics.character_error_rate:.4f}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
