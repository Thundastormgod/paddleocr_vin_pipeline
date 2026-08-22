#!/usr/bin/env python3
"""
Comprehensive Metrics Module for VIN OCR Training and Evaluation
=================================================================

This module provides industry-standard metrics for:
1. TRAINING METRICS - Tracked during model training
2. EVALUATION METRICS - Computed after training for model assessment

Training Metrics:
    - Training Loss (per step, per epoch, smoothed)
    - Validation Loss
    - Learning Rate (current, scheduled)
    - Gradient Norm (for monitoring training stability)
    - Throughput (samples/sec, images/sec)
    - GPU Memory Usage
    - Training Time (per epoch, total)
    - Convergence Rate

Evaluation Metrics (Image-Level):
    - Correct Images Predicted (exact match count)
    - Failed Images Predicted (mismatch count)  
    - Image Accuracy (exact match rate)
    - Word Error Rate (WER)

Evaluation Metrics (Character-Level):
    - Total Characters
    - Correct Characters
    - Character Accuracy
    - Character Error Rate (CER)
    - F1 Micro (character-level)
    - F1 Macro (character-level)
    - Precision (character-level)
    - Recall (character-level)
    - Levenshtein Distance (edit distance)
    - Normalized Edit Distance (NED)

Industry Standard Metrics:
    - BLEU Score (for sequence comparison)
    - Sequence Accuracy
    - Per-class Accuracy (per character class)

Author: VIN OCR Pipeline
Date: January 2026
"""

import json
import time
from dataclasses import dataclass, field, asdict
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple, Any, Set
from collections import defaultdict
from pathlib import Path
import numpy as np

from ..core.char_metrics import AlignmentCounts, alignment_counts, macro_f1, micro_prf


# =============================================================================
# TRAINING METRICS TRACKER
# =============================================================================

@dataclass
class TrainingMetrics:
    """
    Metrics tracked during model training.
    
    These metrics are logged at each step/epoch to monitor training progress.
    """
    # Loss metrics
    train_loss: float = 0.0
    train_loss_smoothed: float = 0.0  # Exponential moving average
    val_loss: float = 0.0
    best_val_loss: float = float('inf')
    
    # Learning rate
    learning_rate: float = 0.0
    initial_lr: float = 0.0
    
    # Gradient metrics (for training stability)
    gradient_norm: float = 0.0
    gradient_norm_clipped: float = 0.0
    
    # Progress metrics
    epoch: int = 0
    global_step: int = 0
    total_epochs: int = 0
    total_steps: int = 0
    
    # Throughput metrics
    samples_per_second: float = 0.0
    images_per_second: float = 0.0
    tokens_per_second: float = 0.0  # For transformer models
    
    # Time metrics
    epoch_time_seconds: float = 0.0
    total_time_seconds: float = 0.0
    estimated_remaining_seconds: float = 0.0
    
    # Memory metrics (GPU)
    gpu_memory_used_mb: float = 0.0
    gpu_memory_allocated_mb: float = 0.0
    gpu_utilization_percent: float = 0.0
    
    # Validation metrics during training
    val_accuracy: float = 0.0
    val_cer: float = 0.0  # Character Error Rate
    best_val_accuracy: float = 0.0
    
    # Convergence tracking
    epochs_without_improvement: int = 0
    is_converged: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return asdict(self)
    
    def to_log_string(self) -> str:
        """Format metrics for console logging."""
        return (
            f"Epoch {self.epoch}/{self.total_epochs} | "
            f"Step {self.global_step} | "
            f"Loss: {self.train_loss:.4f} (smooth: {self.train_loss_smoothed:.4f}) | "
            f"Val Loss: {self.val_loss:.4f} | "
            f"Val Acc: {self.val_accuracy:.2%} | "
            f"LR: {self.learning_rate:.2e} | "
            f"Speed: {self.samples_per_second:.1f} samples/s | "
            f"Time: {self.epoch_time_seconds:.1f}s"
        )


class TrainingMetricsTracker:
    """
    Tracks and aggregates training metrics across epochs and steps.
    
    Usage:
        tracker = TrainingMetricsTracker(total_epochs=100, total_steps=10000)
        
        for epoch in range(epochs):
            for batch in dataloader:
                loss = train_step(batch)
                tracker.update_step(loss=loss, lr=optimizer.lr)
            
            val_metrics = validate()
            tracker.update_epoch(val_loss=val_metrics['loss'], val_acc=val_metrics['acc'])
            
            print(tracker.get_metrics().to_log_string())
        
        tracker.save_history('training_metrics.json')
    """
    
    def __init__(
        self,
        total_epochs: int,
        total_steps: int,
        smoothing_factor: float = 0.99,
        patience: int = 10,
    ):
        self.total_epochs = total_epochs
        self.total_steps = total_steps
        self.smoothing_factor = smoothing_factor
        self.patience = patience
        
        # Current metrics
        self.metrics = TrainingMetrics(
            total_epochs=total_epochs,
            total_steps=total_steps
        )
        
        # History for plotting
        self.history: Dict[str, List[float]] = defaultdict(list)
        
        # Timing
        self.epoch_start_time: Optional[float] = None
        self.training_start_time: float = time.time()
        self.step_times: List[float] = []
        
        # Loss smoothing
        self._smoothed_loss: Optional[float] = None
        
    def update_step(
        self,
        loss: float,
        lr: float,
        gradient_norm: Optional[float] = None,
        batch_size: int = 1,
        step_time: Optional[float] = None,
    ) -> None:
        """Update metrics after each training step."""
        self.metrics.global_step += 1
        self.metrics.train_loss = loss
        self.metrics.learning_rate = lr
        
        # Smooth loss (exponential moving average)
        if self._smoothed_loss is None:
            self._smoothed_loss = loss
        else:
            self._smoothed_loss = (
                self.smoothing_factor * self._smoothed_loss +
                (1 - self.smoothing_factor) * loss
            )
        self.metrics.train_loss_smoothed = self._smoothed_loss
        
        # Gradient norm
        if gradient_norm is not None:
            self.metrics.gradient_norm = gradient_norm
        
        # Throughput
        if step_time is not None:
            self.step_times.append(step_time)
            self.metrics.samples_per_second = batch_size / step_time
            self.metrics.images_per_second = batch_size / step_time
        
        # Log to history
        self.history['step'].append(self.metrics.global_step)
        self.history['train_loss'].append(loss)
        self.history['train_loss_smoothed'].append(self._smoothed_loss)
        self.history['learning_rate'].append(lr)
        
    def update_epoch(
        self,
        val_loss: float,
        val_accuracy: float,
        val_cer: Optional[float] = None,
    ) -> None:
        """Update metrics after each epoch."""
        self.metrics.epoch += 1
        self.metrics.val_loss = val_loss
        self.metrics.val_accuracy = val_accuracy
        
        if val_cer is not None:
            self.metrics.val_cer = val_cer
        
        # Track best metrics
        if val_loss < self.metrics.best_val_loss:
            self.metrics.best_val_loss = val_loss
            self.metrics.epochs_without_improvement = 0
        else:
            self.metrics.epochs_without_improvement += 1
        
        if val_accuracy > self.metrics.best_val_accuracy:
            self.metrics.best_val_accuracy = val_accuracy
        
        # Convergence check
        if self.metrics.epochs_without_improvement >= self.patience:
            self.metrics.is_converged = True
        
        # Timing
        if self.epoch_start_time is not None:
            self.metrics.epoch_time_seconds = time.time() - self.epoch_start_time
        self.metrics.total_time_seconds = time.time() - self.training_start_time
        
        # Estimate remaining time
        if self.metrics.epoch > 0:
            avg_epoch_time = self.metrics.total_time_seconds / self.metrics.epoch
            remaining_epochs = self.total_epochs - self.metrics.epoch
            self.metrics.estimated_remaining_seconds = avg_epoch_time * remaining_epochs
        
        # Log to history
        self.history['epoch'].append(self.metrics.epoch)
        self.history['val_loss'].append(val_loss)
        self.history['val_accuracy'].append(val_accuracy)
        
    def start_epoch(self) -> None:
        """Call at the start of each epoch."""
        self.epoch_start_time = time.time()
        
    def get_metrics(self) -> TrainingMetrics:
        """Get current metrics."""
        return self.metrics
    
    def get_history(self) -> Dict[str, List[float]]:
        """Get full training history."""
        return dict(self.history)
    
    def save_history(self, filepath: str) -> None:
        """Save training history to JSON file."""
        data = {
            'final_metrics': self.metrics.to_dict(),
            'history': self.get_history(),
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def print_summary(self) -> None:
        """Print training summary."""
        print("\n" + "=" * 70)
        print("TRAINING SUMMARY")
        print("=" * 70)
        print(f"Total Epochs:          {self.metrics.epoch}/{self.total_epochs}")
        print(f"Total Steps:           {self.metrics.global_step}")
        print(f"Total Time:            {self.metrics.total_time_seconds/60:.1f} minutes")
        print("-" * 70)
        print("LOSS METRICS:")
        print(f"  Final Train Loss:    {self.metrics.train_loss_smoothed:.4f}")
        print(f"  Best Val Loss:       {self.metrics.best_val_loss:.4f}")
        print("-" * 70)
        print("ACCURACY METRICS:")
        print(f"  Final Val Accuracy:  {self.metrics.val_accuracy:.2%}")
        print(f"  Best Val Accuracy:   {self.metrics.best_val_accuracy:.2%}")
        print("-" * 70)
        print(f"Converged: {'Yes' if self.metrics.is_converged else 'No'}")
        print("=" * 70)


# =============================================================================
# EVALUATION METRICS
# =============================================================================

@dataclass
class ImageLevelMetrics:
    """Image-level (exact match) evaluation metrics."""
    total_images: int = 0
    correct_images: int = 0
    failed_images: int = 0
    accuracy: float = 0.0
    
    # Additional metrics
    word_error_rate: float = 0.0  # WER
    sequence_accuracy: float = 0.0  # Same as accuracy for VIN
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CharacterLevelMetrics:
    """Character-level evaluation metrics."""
    total_characters: int = 0
    correct_characters: int = 0
    incorrect_characters: int = 0
    
    # Core metrics
    char_accuracy: float = 0.0
    # CER is CORPUS-level: sum(edit distances) / sum(reference lengths).
    # Samples with longer references weigh more; 0.0 when there are no
    # reference characters (empty set = no measured error).
    char_error_rate: float = 0.0
    
    # F1 scores
    f1_micro: float = 0.0
    f1_macro: float = 0.0
    
    # Precision/Recall
    precision: float = 0.0
    recall: float = 0.0
    
    # Edit distance metrics
    total_edit_distance: int = 0
    avg_edit_distance: float = 0.0
    # NED is the mean of PER-SAMPLE normalized edit distance:
    # mean_i(edit_distance_i / max(len(gt_i), 1)) - every sample weighs
    # equally (the house definition, shared with evaluate.py). 0.0 when
    # there are no samples. This deliberately differs from char_error_rate,
    # which is corpus-weighted by reference length.
    normalized_edit_distance: float = 0.0
    
    # Per-position accuracy (for VIN: positions 1-17)
    position_accuracy: Dict[int, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvaluationMetrics:
    """
    Complete evaluation metrics combining image-level and character-level.
    
    These metrics are computed after training to assess model quality.
    """
    image_level: ImageLevelMetrics = field(default_factory=ImageLevelMetrics)
    character_level: CharacterLevelMetrics = field(default_factory=CharacterLevelMetrics)
    
    # Timing
    evaluation_time_seconds: float = 0.0
    
    # Per-class metrics (accuracy per character class)
    per_class_accuracy: Dict[str, float] = field(default_factory=dict)
    per_class_count: Dict[str, int] = field(default_factory=dict)
    
    # Confusion info
    most_confused_pairs: List[Tuple[str, str, int]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'image_level': self.image_level.to_dict(),
            'character_level': self.character_level.to_dict(),
            'evaluation_time_seconds': self.evaluation_time_seconds,
            'per_class_accuracy': self.per_class_accuracy,
            'per_class_count': self.per_class_count,
            'most_confused_pairs': self.most_confused_pairs,
        }
    
    def print_summary(self) -> None:
        """Print formatted evaluation summary."""
        print("\n" + "=" * 70)
        print("EVALUATION METRICS SUMMARY")
        print("=" * 70)
        
        print("\n📊 IMAGE-LEVEL METRICS:")
        print("-" * 70)
        print(f"  Total Images:            {self.image_level.total_images}")
        print(f"  ✅ Correct Predictions:   {self.image_level.correct_images}")
        print(f"  ❌ Failed Predictions:    {self.image_level.failed_images}")
        print(f"  📈 Image Accuracy:        {self.image_level.accuracy:.2%}")
        print(f"  📉 Word Error Rate:       {self.image_level.word_error_rate:.2%}")
        
        print("\n🔤 CHARACTER-LEVEL METRICS:")
        print("-" * 70)
        print(f"  Total Characters:        {self.character_level.total_characters}")
        print(f"  Correct Characters:      {self.character_level.correct_characters}")
        print(f"  📈 Character Accuracy:   {self.character_level.char_accuracy:.2%}")
        print(f"  📉 Character Error Rate: {self.character_level.char_error_rate:.2%}")
        print(f"  🎯 F1 Micro:             {self.character_level.f1_micro:.4f}")
        print(f"  🎯 F1 Macro:             {self.character_level.f1_macro:.4f}")
        print(f"  Precision:               {self.character_level.precision:.4f}")
        print(f"  Recall:                  {self.character_level.recall:.4f}")
        print(f"  Avg Edit Distance:       {self.character_level.avg_edit_distance:.2f}")
        print(f"  Normalized Edit Dist:    {self.character_level.normalized_edit_distance:.4f}")
        
        if self.character_level.position_accuracy:
            print("\n📍 PER-POSITION ACCURACY (VIN positions 1-17):")
            print("-" * 70)
            for pos in sorted(self.character_level.position_accuracy.keys()):
                acc = self.character_level.position_accuracy[pos]
                bar = "█" * int(acc * 20) + "░" * (20 - int(acc * 20))
                print(f"  Position {pos:2d}: {bar} {acc:.1%}")
        
        if self.most_confused_pairs:
            print("\n⚠️  MOST CONFUSED CHARACTER PAIRS:")
            print("-" * 70)
            for pred, true, count in self.most_confused_pairs[:10]:
                print(f"  '{pred}' ← '{true}': {count} times")
        
        print("\n" + "=" * 70)
        print(f"Evaluation Time: {self.evaluation_time_seconds:.2f} seconds")
        print("=" * 70)


class EvaluationMetricsCalculator:
    """
    Calculates comprehensive evaluation metrics from predictions and ground truth.
    
    Usage:
        calculator = EvaluationMetricsCalculator()
        
        # Add predictions
        for pred, true in zip(predictions, ground_truth):
            calculator.add_sample(pred, true)
        
        # Get metrics
        metrics = calculator.compute()
        metrics.print_summary()
    """
    
    # VIN valid characters (no I, O, Q)
    VIN_CHARS = set("0123456789ABCDEFGHJKLMNPRSTUVWXYZ")
    
    def __init__(self):
        self.predictions: List[str] = []
        self.ground_truth: List[str] = []
        self.start_time: Optional[float] = None
        
    def reset(self) -> None:
        """Reset calculator for new evaluation."""
        self.predictions = []
        self.ground_truth = []
        self.start_time = None
        
    def add_sample(self, prediction: str, ground_truth: str) -> None:
        """Add a single prediction-ground_truth pair."""
        if self.start_time is None:
            self.start_time = time.time()
        self.predictions.append(prediction.upper().strip())
        self.ground_truth.append(ground_truth.upper().strip())
        
    def add_batch(
        self,
        predictions: List[str],
        ground_truth: List[str]
    ) -> None:
        """
        Add a batch of predictions.

        Raises:
            ValueError: If the two lists differ in length (checked BEFORE
                anything is added, so a mismatch cannot leave the calculator
                partially updated). The unchecked zip here previously
                truncated the longer list silently - the exact defect class
                evaluate.py guards against with strict zips everywhere.
        """
        if len(predictions) != len(ground_truth):
            raise ValueError(
                f"add_batch length mismatch: {len(predictions)} predictions "
                f"vs {len(ground_truth)} ground truths"
            )
        if self.start_time is None:
            self.start_time = time.time()
        for pred, true in zip(predictions, ground_truth, strict=True):
            self.add_sample(pred, true)
    
    def compute(self) -> EvaluationMetrics:
        """Compute all evaluation metrics."""
        metrics = EvaluationMetrics()
        
        if not self.predictions:
            return metrics
        
        # Timing
        if self.start_time:
            metrics.evaluation_time_seconds = time.time() - self.start_time
        
        # Image-level metrics
        metrics.image_level = self._compute_image_level()
        
        # Character-level metrics
        metrics.character_level = self._compute_character_level()
        
        # Per-class metrics
        metrics.per_class_accuracy, metrics.per_class_count = self._compute_per_class()
        
        # Confusion pairs
        metrics.most_confused_pairs = self._compute_confusion_pairs()
        
        return metrics
    
    def _compute_image_level(self) -> ImageLevelMetrics:
        """Compute image-level (exact match) metrics."""
        total = len(self.predictions)
        correct = sum(
            1 for p, t in zip(self.predictions, self.ground_truth, strict=True)
            if p == t
        )
        failed = total - correct
        
        return ImageLevelMetrics(
            total_images=total,
            correct_images=correct,
            failed_images=failed,
            accuracy=correct / total if total > 0 else 0.0,
            word_error_rate=failed / total if total > 0 else 1.0,
            sequence_accuracy=correct / total if total > 0 else 0.0,
        )
    
    def _compute_character_level(self) -> CharacterLevelMetrics:
        """
        Compute character-level metrics.

        Error-rate definitions (the house definitions, shared with
        evaluate.py - keep them in sync):

        - ``char_error_rate`` (CER) is CORPUS-level::

              sum_i(levenshtein(pred_i, gt_i)) / sum_i(len(gt_i))

          Samples with longer references weigh more; the value can exceed
          1.0 for grossly over-long predictions. Empty set (no reference
          characters) -> 0.0: no samples means no measured error.

        - ``normalized_edit_distance`` (NED) is the mean of PER-SAMPLE NED::

              mean_i(levenshtein(pred_i, gt_i) / max(len(gt_i), 1))

          Every sample weighs equally regardless of reference length,
          matching evaluate.py's per-sample semantics. Empty set -> 0.0.

        These two previously shared numerator AND denominator (one number
        under two names) with contradictory empty-set fallbacks (0.0 vs
        1.0), while evaluate.py published different math under the same
        field name.
        """
        total_chars = 0
        correct_chars = 0
        total_edit_dist = 0
        sample_edit_distances: List[int] = []
        position_correct: Dict[int, int] = defaultdict(int)
        position_total: Dict[int, int] = defaultdict(int)
        
        # Per-character TP, FP, FN for F1 calculation
        char_tp: Dict[str, int] = defaultdict(int)
        char_fp: Dict[str, int] = defaultdict(int)
        char_fn: Dict[str, int] = defaultdict(int)
        
        for pred, true in zip(self.predictions, self.ground_truth, strict=True):
            # Edit distance
            edit_dist = self._levenshtein(pred, true)
            total_edit_dist += edit_dist
            sample_edit_distances.append(edit_dist)

            # --- Positional comparison -------------------------------------
            # Used ONLY for per-position accuracy and char_accuracy, which are
            # inherently positional quantities. It is deliberately NOT used for
            # F1: a single insertion shifts every later position and would
            # collapse the score (see _compute_f1_scores docstring).
            #
            # Positions are counted over REFERENCE positions only. Iterating
            # max(len(pred), len(true)) fabricated positions past the
            # reference (a "position 18+" of a 17-character VIN) whenever a
            # prediction carried trailing insertions, and those phantom
            # positions inflated every denominator. Insertions still cost
            # CER/NED and precision - they just cannot invent positions the
            # reference does not have.
            for i, true_char in enumerate(true):
                position_total[i + 1] += 1
                total_chars += 1
                if i < len(pred) and pred[i] == true_char:
                    correct_chars += 1
                    position_correct[i + 1] += 1

            # --- Alignment-based TP/FP/FN for F1 ---------------------------
            # Single implementation: core.char_metrics.alignment_counts.
            # (The opcode walk was previously inlined here with a '_' filter
            # that silently patched over padded predictions; padding is now
            # a caller bug, not something the metric hides.)
            pair_counts = alignment_counts(pred, true)
            for ch, bucket in pair_counts.per_class.items():
                char_tp[ch] += bucket['tp']
                char_fp[ch] += bucket['fp']
                char_fn[ch] += bucket['fn']

        # Calculate metrics
        char_accuracy = correct_chars / total_chars if total_chars > 0 else 0.0

        # Character Error Rate is an EDIT-distance quantity: (S+D+I)/N,
        # CORPUS-level (summed distances over summed reference lengths).
        # It was previously defined as 1 - positional_accuracy, which is a
        # different measure entirely: for a single leading insertion that
        # formula reported CER=0.882 while the correct value was 0.118.
        # Empty set -> 0.0 (no reference characters = no measured error).
        total_true_len_for_cer = sum(len(t) for t in self.ground_truth)
        char_error_rate = (
            total_edit_dist / total_true_len_for_cer
            if total_true_len_for_cer > 0 else 0.0
        )

        # Position accuracy
        position_accuracy = {
            pos: position_correct[pos] / position_total[pos]
            for pos in position_total
        }
        
        # F1 calculations
        f1_micro, f1_macro, precision, recall = self._compute_f1_scores(
            char_tp, char_fp, char_fn
        )
        
        # Normalized edit distance: mean of PER-SAMPLE NED (the house
        # definition, matching evaluate.py) - each sample contributes
        # edit_distance / max(len(gt), 1) and weighs equally in the mean.
        # Empty set -> 0.0. This was previously a byte-for-byte duplicate
        # of char_error_rate above (same numerator, same denominator) with
        # a contradictory empty-set fallback of 1.0.
        per_sample_ned = [
            dist / max(len(true), 1)
            for dist, true in zip(
                sample_edit_distances, self.ground_truth, strict=True
            )
        ]
        ned = (
            sum(per_sample_ned) / len(per_sample_ned)
            if per_sample_ned else 0.0
        )
        
        return CharacterLevelMetrics(
            total_characters=total_chars,
            correct_characters=correct_chars,
            incorrect_characters=total_chars - correct_chars,
            char_accuracy=char_accuracy,
            char_error_rate=char_error_rate,
            f1_micro=f1_micro,
            f1_macro=f1_macro,
            precision=precision,
            recall=recall,
            total_edit_distance=total_edit_dist,
            avg_edit_distance=total_edit_dist / len(self.predictions) if self.predictions else 0.0,
            normalized_edit_distance=ned,
            position_accuracy=position_accuracy,
        )
    
    def _compute_f1_scores(
        self,
        tp: Dict[str, int],
        fp: Dict[str, int],
        fn: Dict[str, int],
    ) -> Tuple[float, float, float, float]:
        """
        Compute F1 micro and macro scores via the canonical implementation.

        Macro-F1 semantics follow core.char_metrics: the unweighted mean of
        per-class F1 over classes with support > 0 (present in the
        references). This previously averaged over the union including
        FP-only classes, so a single hallucinated character class dragged
        the macro mean with a guaranteed 0.0 entry.
        """
        counts = AlignmentCounts(
            tp=sum(tp.values()),
            fp=sum(fp.values()),
            fn=sum(fn.values()),
            per_class={
                char: {
                    'tp': tp.get(char, 0),
                    'fp': fp.get(char, 0),
                    'fn': fn.get(char, 0),
                }
                for char in set(tp) | set(fp) | set(fn)
            },
        )

        micro_precision, micro_recall, f1_micro = micro_prf(counts)
        return f1_micro, macro_f1(counts), micro_precision, micro_recall
    
    def _compute_per_class(self) -> Tuple[Dict[str, float], Dict[str, int]]:
        """
        Compute per-character class accuracy from ALIGNMENT, not position.

        A reference character counts as correct when the edit alignment
        (core.char_metrics.alignment_counts - the same opcode mapping used
        for F1) places it in an 'equal' block, i.e. per-class accuracy is
        tp / (tp + fn) = per-class recall. The count per class is its
        support in the references (tp + fn), exactly as before.

        The previous positional ``pred[i] == true[i]`` comparison meant a
        single leading-artifact insertion shifted every position and marked
        every subsequent reference character wrong, contradicting this
        module's own design comment on positional comparison.
        """
        totals = AlignmentCounts()
        for pred, true in zip(self.predictions, self.ground_truth, strict=True):
            totals.update(alignment_counts(pred, true))

        per_class_accuracy: Dict[str, float] = {}
        per_class_count: Dict[str, int] = {}
        for char, bucket in totals.per_class.items():
            support = bucket['tp'] + bucket['fn']
            if support > 0:  # classes present in the references only
                per_class_accuracy[char] = bucket['tp'] / support
                per_class_count[char] = support

        return per_class_accuracy, per_class_count
    
    def _compute_confusion_pairs(self) -> List[Tuple[str, str, int]]:
        """
        Find most commonly confused (predicted, true) SUBSTITUTION pairs.

        Pairs are read off the same SequenceMatcher opcodes that
        core.char_metrics.alignment_counts uses (same argument order, same
        autojunk=False): only 'replace' blocks yield confusion pairs, with
        characters paired positionally WITHIN the block. A length mismatch
        inside a replace block leaves the overhang as pure insertion or
        deletion - not a confusion between two characters - so that inner
        zip is intentionally non-strict.

        The previous positional zip fabricated pairs: one leading-artifact
        insertion shifted every position and reported a full column of
        spurious "confusions" between characters the model never confused.
        """
        confusion: Dict[Tuple[str, str], int] = defaultdict(int)
        
        for pred, true in zip(self.predictions, self.ground_truth, strict=True):
            matcher = SequenceMatcher(None, pred, true, autojunk=False)
            for tag, p0, p1, r0, r1 in matcher.get_opcodes():
                if tag != 'replace':
                    continue
                for pred_char, true_char in zip(pred[p0:p1], true[r0:r1]):
                    confusion[(pred_char, true_char)] += 1
        
        # Sort by frequency
        sorted_pairs = sorted(
            confusion.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [(p, t, c) for (p, t), c in sorted_pairs[:20]]
    
    @staticmethod
    def _levenshtein(s1: str, s2: str) -> int:
        """Calculate Levenshtein edit distance."""
        if len(s1) < len(s2):
            return EvaluationMetricsCalculator._levenshtein(s2, s1)
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


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def compute_metrics_from_results(
    predictions: List[str],
    ground_truth: List[str],
    print_summary: bool = True,
) -> EvaluationMetrics:
    """
    Convenience function to compute metrics from prediction lists.
    
    Args:
        predictions: List of predicted VIN strings
        ground_truth: List of ground truth VIN strings
        print_summary: Whether to print formatted summary
        
    Returns:
        EvaluationMetrics object with all computed metrics
    """
    calculator = EvaluationMetricsCalculator()
    calculator.add_batch(predictions, ground_truth)
    metrics = calculator.compute()
    
    if print_summary:
        metrics.print_summary()
    
    return metrics


def save_metrics_to_json(
    metrics: EvaluationMetrics,
    filepath: str,
) -> None:
    """Save evaluation metrics to JSON file."""
    with open(filepath, 'w') as f:
        json.dump(metrics.to_dict(), f, indent=2)


def load_metrics_from_json(filepath: str) -> EvaluationMetrics:
    """Load evaluation metrics from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    metrics = EvaluationMetrics()
    
    if 'image_level' in data:
        metrics.image_level = ImageLevelMetrics(**data['image_level'])
    if 'character_level' in data:
        # Handle position_accuracy dict
        char_data = data['character_level'].copy()
        char_data['position_accuracy'] = {
            int(k): v for k, v in char_data.get('position_accuracy', {}).items()
        }
        metrics.character_level = CharacterLevelMetrics(**char_data)
    
    metrics.evaluation_time_seconds = data.get('evaluation_time_seconds', 0.0)
    metrics.per_class_accuracy = data.get('per_class_accuracy', {})
    metrics.per_class_count = data.get('per_class_count', {})
    metrics.most_confused_pairs = [
        tuple(x) for x in data.get('most_confused_pairs', [])
    ]
    
    return metrics


# =============================================================================
# TEST / DEMO
# =============================================================================

if __name__ == "__main__":
    # Demo usage
    print("=" * 70)
    print("METRICS MODULE DEMO")
    print("=" * 70)
    
    # Test data
    predictions = [
        "1HGBH41JXMN109186",  # Correct
        "1HGBH41JXMN109187",  # 1 char wrong
        "1HGBH41JXMN10918",   # 1 char missing
        "WVWZZZ3CZWE123456",  # Correct
        "WVWZZZ3CZWE12345X",  # 1 char wrong
    ]
    
    ground_truth = [
        "1HGBH41JXMN109186",
        "1HGBH41JXMN109186",
        "1HGBH41JXMN109186",
        "WVWZZZ3CZWE123456",
        "WVWZZZ3CZWE123456",
    ]
    
    # Compute metrics
    metrics = compute_metrics_from_results(predictions, ground_truth)
    
    # Save to JSON
    save_metrics_to_json(metrics, '/tmp/test_metrics.json')
    print(f"\nMetrics saved to /tmp/test_metrics.json")
    
    # The training-tracker demo that used to live here has been removed:
    # it synthesised a decreasing loss curve and val_accuracy = 0.5 + 0.15
    # per epoch and printed them through TrainingMetricsTracker, producing
    # output indistinguishable from a real training log. This repository
    # has already shipped fabricated metrics once; a built-in generator of
    # realistic-looking fake training logs is a liability, not a demo.
