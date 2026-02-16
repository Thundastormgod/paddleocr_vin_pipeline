#!/usr/bin/env python3
"""
Comprehensive Training Metrics for VIN OCR
==========================================

Provides image-level and character-level metrics for training evaluation.

Image-Level Metrics:
- correct_images: Number of images with 100% correct VIN
- failed_images: Number of images with incorrect VIN
- image_accuracy: Percentage of correctly recognized images

Character-Level Metrics:
- total_characters: Total number of characters in ground truth
- correct_characters: Number of correctly predicted characters
- char_accuracy: Character-level accuracy
- char_error_rate (CER): Character error rate
- f1_micro: Micro-averaged F1 score
- f1_macro: Macro-averaged F1 score
- precision: Overall precision
- recall: Overall recall

Industry Standard Metrics:
- word_accuracy: Full VIN match accuracy (same as image_accuracy for VINs)
- normalized_edit_distance (NED): Normalized Levenshtein distance
- sequence_accuracy: Exact sequence match rate
"""

from typing import Dict, List, Tuple, Any, Optional
from collections import Counter
import json
import os
from pathlib import Path
from datetime import datetime


class VINMetricsCalculator:
    """
    Calculate comprehensive metrics for VIN OCR training.
    
    Usage:
        calc = VINMetricsCalculator()
        predictions = ["1HGBH41JXMN109186", "WBA3A5C55DF123456"]
        ground_truths = ["1HGBH41JXMN109186", "WBA3A5C55DF123457"]
        
        metrics = calc.calculate_all_metrics(predictions, ground_truths)
        print(metrics)
    """
    
    # VIN character set (no I, O, Q)
    VIN_CHARS = set("0123456789ABCDEFGHJKLMNPRSTUVWXYZ")
    VIN_LENGTH = 17
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset accumulated metrics."""
        self.predictions = []
        self.ground_truths = []
        self.char_predictions = []
        self.char_labels = []
    
    def add_batch(self, predictions: List[str], ground_truths: List[str]):
        """Add a batch of predictions and ground truths."""
        self.predictions.extend(predictions)
        self.ground_truths.extend(ground_truths)
        
        # Also track character-level
        for pred, gt in zip(predictions, ground_truths):
            pred_clean = pred.strip().upper()
            gt_clean = gt.strip().upper()
            
            # Pad/truncate to same length for character comparison
            max_len = max(len(pred_clean), len(gt_clean))
            pred_padded = pred_clean.ljust(max_len, ' ')
            gt_padded = gt_clean.ljust(max_len, ' ')
            
            self.char_predictions.extend(list(pred_padded))
            self.char_labels.extend(list(gt_padded))
    
    def calculate_all_metrics(
        self, 
        predictions: Optional[List[str]] = None, 
        ground_truths: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Calculate all metrics.
        
        Returns dict with:
        - image_level: Dict of image-level metrics
        - character_level: Dict of character-level metrics
        - industry: Dict of industry-standard metrics
        - summary: Human-readable summary
        """
        if predictions is not None and ground_truths is not None:
            self.reset()
            self.add_batch(predictions, ground_truths)
        
        if not self.predictions:
            return self._empty_metrics()
        
        image_metrics = self._calculate_image_level_metrics()
        char_metrics = self._calculate_character_level_metrics()
        industry_metrics = self._calculate_industry_metrics()
        
        return {
            'image_level': image_metrics,
            'character_level': char_metrics,
            'industry': industry_metrics,
            'summary': self._generate_summary(image_metrics, char_metrics, industry_metrics),
            'timestamp': datetime.now().isoformat(),
            'total_samples': len(self.predictions)
        }
    
    def _calculate_image_level_metrics(self) -> Dict[str, Any]:
        """Calculate image-level metrics."""
        correct = 0
        failed = 0
        
        for pred, gt in zip(self.predictions, self.ground_truths):
            pred_clean = pred.strip().upper()
            gt_clean = gt.strip().upper()
            
            if pred_clean == gt_clean:
                correct += 1
            else:
                failed += 1
        
        total = correct + failed
        accuracy = correct / total if total > 0 else 0.0
        
        return {
            'correct_images': correct,
            'failed_images': failed,
            'total_images': total,
            'image_accuracy': accuracy,
            'image_accuracy_pct': f"{accuracy * 100:.2f}%"
        }
    
    def _calculate_character_level_metrics(self) -> Dict[str, Any]:
        """Calculate character-level metrics including F1 scores."""
        if not self.char_predictions:
            return self._empty_char_metrics()
        
        # Basic character accuracy
        correct_chars = sum(
            1 for p, g in zip(self.char_predictions, self.char_labels) 
            if p == g
        )
        total_chars = len(self.char_labels)
        char_accuracy = correct_chars / total_chars if total_chars > 0 else 0.0
        
        # Calculate per-character metrics for F1
        # Build confusion data
        unique_chars = sorted(set(self.char_labels) | set(self.char_predictions))
        
        # True positives, false positives, false negatives per character
        tp = Counter()
        fp = Counter()
        fn = Counter()
        
        for pred, label in zip(self.char_predictions, self.char_labels):
            if pred == label:
                tp[label] += 1
            else:
                fp[pred] += 1
                fn[label] += 1
        
        # Micro F1 (global TP, FP, FN)
        total_tp = sum(tp.values())
        total_fp = sum(fp.values())
        total_fn = sum(fn.values())
        
        micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1_micro = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0
        
        # Macro F1 (average per-class F1)
        f1_scores = []
        for char in unique_chars:
            char_tp = tp[char]
            char_fp = fp[char]
            char_fn = fn[char]
            
            precision = char_tp / (char_tp + char_fp) if (char_tp + char_fp) > 0 else 0.0
            recall = char_tp / (char_tp + char_fn) if (char_tp + char_fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            f1_scores.append(f1)
        
        f1_macro = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
        
        return {
            'total_characters': total_chars,
            'correct_characters': correct_chars,
            'char_accuracy': char_accuracy,
            'char_accuracy_pct': f"{char_accuracy * 100:.2f}%",
            'f1_micro': f1_micro,
            'f1_macro': f1_macro,
            'precision': micro_precision,
            'recall': micro_recall,
            'unique_chars_seen': len(unique_chars)
        }
    
    def _calculate_industry_metrics(self) -> Dict[str, Any]:
        """Calculate industry-standard OCR metrics."""
        # Character Error Rate (CER) using Levenshtein
        total_gt_chars = sum(len(gt.strip()) for gt in self.ground_truths)
        total_edit_distance = sum(
            self._levenshtein(pred.strip().upper(), gt.strip().upper())
            for pred, gt in zip(self.predictions, self.ground_truths)
        )
        cer = total_edit_distance / total_gt_chars if total_gt_chars > 0 else 1.0
        
        # Normalized Edit Distance (NED)
        ned_scores = []
        for pred, gt in zip(self.predictions, self.ground_truths):
            pred_clean = pred.strip().upper()
            gt_clean = gt.strip().upper()
            max_len = max(len(pred_clean), len(gt_clean), 1)
            edit_dist = self._levenshtein(pred_clean, gt_clean)
            ned = edit_dist / max_len
            ned_scores.append(ned)
        avg_ned = sum(ned_scores) / len(ned_scores) if ned_scores else 1.0
        
        # Word/Sequence accuracy (exact match)
        exact_matches = sum(
            1 for p, g in zip(self.predictions, self.ground_truths)
            if p.strip().upper() == g.strip().upper()
        )
        word_accuracy = exact_matches / len(self.predictions) if self.predictions else 0.0
        
        # VIN-specific: Valid VIN rate
        valid_vin_count = sum(
            1 for p in self.predictions
            if len(p.strip()) == self.VIN_LENGTH and 
            all(c in self.VIN_CHARS for c in p.strip().upper())
        )
        valid_vin_rate = valid_vin_count / len(self.predictions) if self.predictions else 0.0
        
        return {
            'char_error_rate': cer,
            'cer_pct': f"{cer * 100:.2f}%",
            'normalized_edit_distance': avg_ned,
            'word_accuracy': word_accuracy,
            'sequence_accuracy': word_accuracy,
            'valid_vin_rate': valid_vin_rate,
            'valid_vin_rate_pct': f"{valid_vin_rate * 100:.2f}%"
        }
    
    @staticmethod
    def _levenshtein(s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings."""
        if len(s1) < len(s2):
            return VINMetricsCalculator._levenshtein(s2, s1)
        if len(s2) == 0:
            return len(s1)
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def _generate_summary(
        self, 
        image_metrics: Dict, 
        char_metrics: Dict, 
        industry_metrics: Dict
    ) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 60,
            "VIN OCR TRAINING METRICS SUMMARY",
            "=" * 60,
            "",
            "📊 IMAGE-LEVEL METRICS",
            "-" * 40,
            f"  Correct Images:    {image_metrics['correct_images']}",
            f"  Failed Images:     {image_metrics['failed_images']}",
            f"  Total Images:      {image_metrics['total_images']}",
            f"  Image Accuracy:    {image_metrics['image_accuracy_pct']}",
            "",
            "📝 CHARACTER-LEVEL METRICS",
            "-" * 40,
            f"  Total Characters:  {char_metrics['total_characters']}",
            f"  Correct Chars:     {char_metrics['correct_characters']}",
            f"  Char Accuracy:     {char_metrics['char_accuracy_pct']}",
            f"  F1 Micro:          {char_metrics['f1_micro']:.4f}",
            f"  F1 Macro:          {char_metrics['f1_macro']:.4f}",
            f"  Precision:         {char_metrics['precision']:.4f}",
            f"  Recall:            {char_metrics['recall']:.4f}",
            "",
            "🏭 INDUSTRY METRICS",
            "-" * 40,
            f"  Char Error Rate:   {industry_metrics['cer_pct']}",
            f"  Normalized ED:     {industry_metrics['normalized_edit_distance']:.4f}",
            f"  Word Accuracy:     {industry_metrics['word_accuracy']:.4f}",
            f"  Valid VIN Rate:    {industry_metrics['valid_vin_rate_pct']}",
            "",
            "=" * 60,
        ]
        return "\n".join(lines)
    
    def _empty_metrics(self) -> Dict[str, Any]:
        """Return empty metrics structure."""
        return {
            'image_level': self._empty_image_metrics(),
            'character_level': self._empty_char_metrics(),
            'industry': self._empty_industry_metrics(),
            'summary': "No data available",
            'timestamp': datetime.now().isoformat(),
            'total_samples': 0
        }
    
    def _empty_image_metrics(self) -> Dict[str, Any]:
        return {
            'correct_images': 0,
            'failed_images': 0,
            'total_images': 0,
            'image_accuracy': 0.0,
            'image_accuracy_pct': "0.00%"
        }
    
    def _empty_char_metrics(self) -> Dict[str, Any]:
        return {
            'total_characters': 0,
            'correct_characters': 0,
            'char_accuracy': 0.0,
            'char_accuracy_pct': "0.00%",
            'f1_micro': 0.0,
            'f1_macro': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'unique_chars_seen': 0
        }
    
    def _empty_industry_metrics(self) -> Dict[str, Any]:
        return {
            'char_error_rate': 1.0,
            'cer_pct': "100.00%",
            'normalized_edit_distance': 1.0,
            'word_accuracy': 0.0,
            'sequence_accuracy': 0.0,
            'valid_vin_rate': 0.0,
            'valid_vin_rate_pct': "0.00%"
        }
    
    def save_metrics(self, filepath: str, metrics: Optional[Dict] = None):
        """Save metrics to JSON file."""
        if metrics is None:
            metrics = self.calculate_all_metrics()
        
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {filepath}")
    
    @staticmethod
    def load_metrics(filepath: str) -> Dict[str, Any]:
        """Load metrics from JSON file."""
        with open(filepath, 'r') as f:
            return json.load(f)


def compute_training_metrics(
    predictions: List[str], 
    ground_truths: List[str],
    save_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function to compute all training metrics.
    
    Args:
        predictions: List of predicted VINs
        ground_truths: List of ground truth VINs
        save_path: Optional path to save metrics JSON
    
    Returns:
        Dict with all metrics
    """
    calc = VINMetricsCalculator()
    metrics = calc.calculate_all_metrics(predictions, ground_truths)
    
    if save_path:
        calc.save_metrics(save_path, metrics)
    
    return metrics


# Convenience aliases for backward compatibility
def calculate_image_level_metrics(predictions: List[str], ground_truths: List[str]) -> Dict:
    """Calculate only image-level metrics."""
    calc = VINMetricsCalculator()
    calc.add_batch(predictions, ground_truths)
    return calc._calculate_image_level_metrics()


def calculate_char_level_metrics(predictions: List[str], ground_truths: List[str]) -> Dict:
    """Calculate only character-level metrics."""
    calc = VINMetricsCalculator()
    calc.add_batch(predictions, ground_truths)
    return calc._calculate_character_level_metrics()


if __name__ == "__main__":
    # Demo usage
    print("VIN Metrics Calculator Demo")
    print("=" * 60)
    
    # Sample data
    predictions = [
        "1HGBH41JXMN109186",  # Correct
        "WBA3A5C55DF123456",  # Correct
        "JTEBU5JR5D5123457",  # 1 char wrong (7 vs 6)
        "5YJSA1E26HF00000",   # Missing last char
        "1G1YY22G965109876",  # Correct
    ]
    ground_truths = [
        "1HGBH41JXMN109186",
        "WBA3A5C55DF123456", 
        "JTEBU5JR5D5123456",
        "5YJSA1E26HF000001",
        "1G1YY22G965109876",
    ]
    
    calc = VINMetricsCalculator()
    metrics = calc.calculate_all_metrics(predictions, ground_truths)
    
    print(metrics['summary'])
    print("\nFull metrics JSON:")
    print(json.dumps(metrics, indent=2, default=str))
