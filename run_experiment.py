#!/usr/bin/env python3
"""
Complete Experiment Pipeline for VIN Recognition
=================================================

End-to-end pipeline that:
1. Prepares dataset (train/val/test splits)
2. Runs baseline evaluation (pretrained model)
3. Optionally fine-tunes the model
4. Evaluates on all splits
5. Generates comprehensive metrics report

This provides industry-standard metrics including:
- F1 Score (Precision, Recall)
- Exact Match Accuracy
- Character Error Rate (CER)
- Normalized Edit Distance (NED)
- Per-position accuracy

Usage:
    python run_experiment.py --data-dir data
    python run_experiment.py --data-dir data --output-dir experiments/exp1

Author: JRL-VIN Project
Date: January 2026
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Import from shared utilities (Single Source of Truth)
from src.vin_ocr.core.vin_utils import (
    extract_vin_from_filename,
    VIN_LENGTH,
    VIN_VALID_CHARS,
    levenshtein_distance,
)
from config import get_config

logger = logging.getLogger(__name__)

# VIN Character set reference (from shared utils)
VIN_CHARSET = VIN_VALID_CHARS


def calculate_metrics(predictions: List[Dict]) -> Dict:
    """
    Calculate comprehensive metrics for VIN recognition.
    
    Args:
        predictions: List of dicts with 'ground_truth', 'prediction', 'confidence'
    
    Returns:
        Dict with all metrics
    """
    if not predictions:
        return {"error": "No predictions to evaluate"}
    
    n = len(predictions)
    
    # Exact match
    exact_matches = sum(1 for p in predictions if p['ground_truth'] == p['prediction'])
    exact_match_accuracy = exact_matches / n
    
    # Character-level metrics
    total_chars = 0
    correct_chars = 0
    char_insertions = 0
    char_deletions = 0
    char_substitutions = 0
    
    # Per-position accuracy (VIN has 17 positions)
    position_correct = [0] * 17
    position_total = [0] * 17
    
    # F1 components
    true_positives = 0  # Correctly predicted characters
    false_positives = 0  # Extra predicted characters
    false_negatives = 0  # Missing characters
    
    for pred in predictions:
        gt = pred['ground_truth']
        pr = pred['prediction']
        
        total_chars += len(gt)
        
        # Character-level accuracy
        for i, (g, p) in enumerate(zip(gt, pr)):
            if i < 17:
                position_total[i] += 1
                if g == p:
                    position_correct[i] += 1
                    correct_chars += 1
                    true_positives += 1
                else:
                    char_substitutions += 1
                    false_negatives += 1  # Missed the correct char
                    false_positives += 1  # Predicted wrong char
        
        # Handle length differences
        if len(pr) > len(gt):
            extra = len(pr) - len(gt)
            char_insertions += extra
            false_positives += extra
        elif len(gt) > len(pr):
            missing = len(gt) - len(pr)
            char_deletions += missing
            false_negatives += missing
    
    # Calculate metrics
    cer = (char_substitutions + char_insertions + char_deletions) / total_chars if total_chars > 0 else 1.0
    char_accuracy = correct_chars / total_chars if total_chars > 0 else 0.0
    
    # F1 Score
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Per-position accuracy
    position_accuracy = {}
    for i in range(17):
        if position_total[i] > 0:
            position_accuracy[f"position_{i+1}"] = position_correct[i] / position_total[i]
    
    # Normalized Edit Distance (NED)
    total_ned = 0
    for pred in predictions:
        gt = pred['ground_truth']
        pr = pred['prediction']
        edit_dist = levenshtein_distance(gt, pr)
        max_len = max(len(gt), len(pr), 1)
        total_ned += edit_dist / max_len
    ned = total_ned / n
    
    # Confidence statistics
    confidences = [p['confidence'] for p in predictions if p['confidence'] is not None]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    
    return {
        "sample_count": n,
        "exact_match_accuracy": exact_match_accuracy,
        "exact_matches": exact_matches,
        "f1_score": f1_score,
        "precision": precision,
        "recall": recall,
        "character_error_rate": cer,
        "character_accuracy": char_accuracy,
        "normalized_edit_distance": ned,
        "total_characters": total_chars,
        "correct_characters": correct_chars,
        "substitutions": char_substitutions,
        "insertions": char_insertions,
        "deletions": char_deletions,
        "position_accuracy": position_accuracy,
        "average_confidence": avg_confidence
    }


# Note: levenshtein_distance is imported from vin_utils (Single Source of Truth)


def run_ocr_on_split(
    split_dir: Path,
    label_file: Optional[Path] = None,
    verbose: bool = True
) -> List[Dict]:
    """
    Run OCR on all images in a split directory.
    
    Returns list of predictions with ground truth.
    """
    # Import VIN pipeline
    sys.path.insert(0, str(Path(__file__).parent))
    from src.vin_ocr.pipeline.vin_pipeline import VINOCRPipeline
    
    # Initialize pipeline
    pipeline = VINOCRPipeline()
    
    # Get images
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    # Load ground truth from label file if provided
    ground_truth_map = {}
    if label_file and label_file.exists():
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    img_path = parts[0]
                    vin = parts[1]
                    # Extract just filename
                    filename = Path(img_path).name
                    ground_truth_map[filename] = vin
    
    predictions = []
    images = list(split_dir.iterdir()) if split_dir.exists() else []
    images = [f for f in images if f.suffix.lower() in image_extensions]
    
    if verbose:
        print(f"  Processing {len(images)} images...")
    
    for i, img_path in enumerate(images):
        # Get ground truth
        gt = ground_truth_map.get(img_path.name) or extract_vin_from_filename(img_path.name)
        
        if not gt:
            if verbose:
                print(f"    Skipping {img_path.name}: no ground truth")
            continue
        
        # Run OCR
        try:
            result = pipeline.recognize(str(img_path))
            pred_vin = result.get('vin', '')
            confidence = result.get('confidence', 0.0)
        except Exception as e:
            if verbose:
                print(f"    Error processing {img_path.name}: {e}")
            pred_vin = ''
            confidence = 0.0
        
        predictions.append({
            'image': img_path.name,
            'ground_truth': gt,
            'prediction': pred_vin,
            'confidence': confidence,
            'correct': gt == pred_vin
        })
        
        if verbose and (i + 1) % 10 == 0:
            print(f"    Processed {i + 1}/{len(images)} images")
    
    return predictions


class ExperimentRunner:
    """
    Runs complete training/evaluation experiments.
    """
    
    def __init__(
        self,
        data_dir: str,
        output_dir: str = "experiments",
        experiment_name: Optional[str] = None
    ):
        self.data_dir = Path(data_dir)
        self.base_output_dir = Path(output_dir)
        
        # Create experiment name with timestamp if not provided
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"exp_{timestamp}"
        
        self.experiment_name = experiment_name
        self.output_dir = self.base_output_dir / experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Experiment: {experiment_name}")
        print(f"Output: {self.output_dir}")
    
    def prepare_dataset(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42
    ) -> Dict:
        """Prepare dataset with train/val/test splits."""
        from prepare_dataset import prepare_dataset
        
        dataset_dir = self.output_dir / "dataset"
        
        stats = prepare_dataset(
            data_dir=str(self.data_dir),
            output_dir=str(dataset_dir),
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
            copy_images=True
        )
        
        self.dataset_dir = dataset_dir
        return stats
    
    def evaluate_split(self, split_name: str, verbose: bool = True) -> Dict:
        """Evaluate model on a specific split."""
        print(f"\n{'='*60}")
        print(f"Evaluating: {split_name.upper()} split")
        print(f"{'='*60}")
        
        split_dir = self.dataset_dir / split_name
        label_file = self.dataset_dir / f"{split_name}_labels.txt"
        
        # Run OCR
        predictions = run_ocr_on_split(split_dir, label_file, verbose)
        
        if not predictions:
            print(f"  No predictions generated for {split_name}")
            return {"error": f"No predictions for {split_name}"}
        
        # Calculate metrics
        metrics = calculate_metrics(predictions)
        
        # Save detailed results
        results_dir = self.output_dir / "results"
        results_dir.mkdir(exist_ok=True)
        
        # Save predictions
        predictions_file = results_dir / f"{split_name}_predictions.json"
        with open(predictions_file, 'w') as f:
            json.dump(predictions, f, indent=2)
        
        # Save metrics
        metrics_file = results_dir / f"{split_name}_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Print summary
        print(f"\n  Results for {split_name}:")
        print(f"  {'─'*40}")
        print(f"  Samples:              {metrics['sample_count']}")
        print(f"  Exact Match:          {metrics['exact_match_accuracy']*100:.2f}%")
        print(f"  F1 Score:             {metrics['f1_score']*100:.2f}%")
        print(f"  Precision:            {metrics['precision']*100:.2f}%")
        print(f"  Recall:               {metrics['recall']*100:.2f}%")
        print(f"  Character Accuracy:   {metrics['character_accuracy']*100:.2f}%")
        print(f"  CER:                  {metrics['character_error_rate']*100:.2f}%")
        print(f"  NED:                  {metrics['normalized_edit_distance']:.4f}")
        print(f"  Avg Confidence:       {metrics['average_confidence']*100:.2f}%")
        
        return {
            "split": split_name,
            "metrics": metrics,
            "predictions_file": str(predictions_file),
            "metrics_file": str(metrics_file)
        }
    
    def run_full_experiment(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
        verbose: bool = True
    ) -> Dict:
        """
        Run complete experiment pipeline.
        
        1. Prepare dataset
        2. Evaluate on train split (baseline check)
        3. Evaluate on validation split
        4. Evaluate on test split
        5. Generate summary report
        """
        print("\n" + "=" * 60)
        print("Starting Full Experiment Pipeline")
        print("=" * 60)
        print(f"Experiment: {self.experiment_name}")
        print(f"Data directory: {self.data_dir}")
        print(f"Splits: train={train_ratio:.0%}, val={val_ratio:.0%}, test={test_ratio:.0%}")
        
        experiment_start = datetime.now()
        
        # Step 1: Prepare dataset
        print("\n" + "─" * 60)
        print("STEP 1: Dataset Preparation")
        print("─" * 60)
        dataset_stats = self.prepare_dataset(train_ratio, val_ratio, test_ratio, seed)
        
        # Check if we have enough data
        if dataset_stats['total_images'] == 0:
            return {"error": "No images found in data directory"}
        
        # Step 2-4: Evaluate splits
        results = {}
        
        # Only evaluate splits that have data
        for split in ['train', 'val', 'test']:
            split_count = dataset_stats.get(f'{split}_images', 0)
            if split_count > 0:
                results[split] = self.evaluate_split(split, verbose)
            else:
                print(f"\n  Skipping {split} split (no images)")
        
        # Step 5: Generate summary
        print("\n" + "=" * 60)
        print("EXPERIMENT SUMMARY")
        print("=" * 60)
        
        experiment_end = datetime.now()
        duration = (experiment_end - experiment_start).total_seconds()
        
        summary = {
            "experiment_name": self.experiment_name,
            "timestamp": experiment_start.isoformat(),
            "duration_seconds": duration,
            "data_dir": str(self.data_dir),
            "output_dir": str(self.output_dir),
            "dataset_stats": dataset_stats,
            "split_results": {}
        }
        
        # Compile results table
        print(f"\n{'Split':<10} {'Samples':>8} {'Exact%':>10} {'F1%':>10} {'CER%':>10} {'Conf%':>10}")
        print("─" * 60)
        
        for split in ['train', 'val', 'test']:
            if split in results and 'metrics' in results[split]:
                m = results[split]['metrics']
                summary['split_results'][split] = m
                
                print(f"{split:<10} {m['sample_count']:>8} {m['exact_match_accuracy']*100:>9.2f}% "
                      f"{m['f1_score']*100:>9.2f}% {m['character_error_rate']*100:>9.2f}% "
                      f"{m['average_confidence']*100:>9.2f}%")
        
        print("─" * 60)
        
        # Overall metrics (weighted average across splits)
        total_samples = sum(
            results[s]['metrics']['sample_count'] 
            for s in results if 'metrics' in results.get(s, {})
        )
        
        if total_samples > 0:
            overall_exact = sum(
                results[s]['metrics']['exact_match_accuracy'] * results[s]['metrics']['sample_count']
                for s in results if 'metrics' in results.get(s, {})
            ) / total_samples
            
            overall_f1 = sum(
                results[s]['metrics']['f1_score'] * results[s]['metrics']['sample_count']
                for s in results if 'metrics' in results.get(s, {})
            ) / total_samples
            
            summary['overall_metrics'] = {
                'total_samples': total_samples,
                'weighted_exact_match': overall_exact,
                'weighted_f1_score': overall_f1
            }
            
            print(f"\nOverall Weighted Metrics:")
            print(f"  Total Samples:        {total_samples}")
            print(f"  Exact Match:          {overall_exact*100:.2f}%")
            print(f"  F1 Score:             {overall_f1*100:.2f}%")
        
        # Save summary
        summary_file = self.output_dir / "experiment_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nExperiment duration: {duration:.1f} seconds")
        print(f"Results saved to: {self.output_dir}")
        
        # Generate markdown report
        self._generate_markdown_report(summary, results)
        
        return summary
    
    def _generate_markdown_report(self, summary: Dict, results: Dict):
        """Generate a markdown report of the experiment."""
        report_file = self.output_dir / "EXPERIMENT_REPORT.md"
        
        with open(report_file, 'w') as f:
            f.write(f"# VIN Recognition Experiment Report\n\n")
            f.write(f"**Experiment:** {summary['experiment_name']}\n")
            f.write(f"**Date:** {summary['timestamp'][:10]}\n")
            f.write(f"**Duration:** {summary['duration_seconds']:.1f} seconds\n\n")
            
            f.write("## Dataset\n\n")
            ds = summary['dataset_stats']
            f.write(f"- **Total Images:** {ds['total_images']}\n")
            f.write(f"- **Train:** {ds['train_images']} ({ds['train_ratio']*100:.0f}%)\n")
            f.write(f"- **Validation:** {ds['val_images']} ({ds['val_ratio']*100:.0f}%)\n")
            f.write(f"- **Test:** {ds['test_images']} ({ds['test_ratio']*100:.0f}%)\n\n")
            
            f.write("## Results\n\n")
            f.write("| Split | Samples | Exact Match | F1 Score | CER | Confidence |\n")
            f.write("|-------|---------|-------------|----------|-----|------------|\n")
            
            for split in ['train', 'val', 'test']:
                if split in summary.get('split_results', {}):
                    m = summary['split_results'][split]
                    f.write(f"| {split} | {m['sample_count']} | "
                           f"{m['exact_match_accuracy']*100:.2f}% | "
                           f"{m['f1_score']*100:.2f}% | "
                           f"{m['character_error_rate']*100:.2f}% | "
                           f"{m['average_confidence']*100:.2f}% |\n")
            
            f.write("\n## Key Metrics Explained\n\n")
            f.write("- **Exact Match:** Percentage of VINs predicted completely correctly\n")
            f.write("- **F1 Score:** Harmonic mean of precision and recall (character-level)\n")
            f.write("- **CER:** Character Error Rate - lower is better\n")
            f.write("- **Confidence:** Average model confidence score\n\n")
            
            if 'overall_metrics' in summary:
                f.write("## Overall Performance\n\n")
                om = summary['overall_metrics']
                f.write(f"- **Weighted Exact Match:** {om['weighted_exact_match']*100:.2f}%\n")
                f.write(f"- **Weighted F1 Score:** {om['weighted_f1_score']*100:.2f}%\n")
            
            # Position accuracy if available
            if 'test' in summary.get('split_results', {}):
                m = summary['split_results']['test']
                if 'position_accuracy' in m:
                    f.write("\n## Per-Position Accuracy (Test Set)\n\n")
                    f.write("| Position | Accuracy |\n")
                    f.write("|----------|----------|\n")
                    for pos, acc in sorted(m['position_accuracy'].items()):
                        pos_num = pos.split('_')[1]
                        f.write(f"| {pos_num} | {acc*100:.2f}% |\n")
        
        print(f"Report saved to: {report_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Run complete VIN recognition experiment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_experiment.py --data-dir data
    python run_experiment.py --data-dir data --output-dir experiments
    python run_experiment.py --data-dir data --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1
        """
    )
    parser.add_argument('--data-dir', required=True, help='Directory containing VIN images')
    parser.add_argument('--output-dir', default='experiments', help='Base output directory')
    parser.add_argument('--experiment-name', help='Name for this experiment (auto-generated if not provided)')
    parser.add_argument('--train-ratio', type=float, default=0.7, help='Training set ratio')
    parser.add_argument('--val-ratio', type=float, default=0.15, help='Validation set ratio')
    parser.add_argument('--test-ratio', type=float, default=0.15, help='Test set ratio')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--quiet', action='store_true', help='Reduce output verbosity')
    
    args = parser.parse_args()
    
    runner = ExperimentRunner(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name
    )
    
    runner.run_full_experiment(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        verbose=not args.quiet
    )


if __name__ == '__main__':
    main()
