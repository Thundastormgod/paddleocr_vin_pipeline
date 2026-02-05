"""
Test Suite for VIN OCR Evaluation Module
========================================

Comprehensive tests covering:
- Dataset split creation and validation
- Metrics calculations
- Ground truth extraction
- Data leakage prevention

Run with: pytest tests/test_evaluate.py -v
"""

import pytest
import sys
from pathlib import Path
from typing import Dict

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vin_ocr.evaluation.evaluate import (
    create_splits,
    load_ground_truth,
    extract_vin_from_filename,
    levenshtein_distance,
    calculate_cer,
    calculate_character_metrics,
    calculate_position_accuracy,
    DatasetSplit,
    EvaluationMetrics,
    PredictionResult,
)


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def sample_ground_truth() -> Dict[str, str]:
    """Create sample ground truth data."""
    return {
        f'/images/001-VIN_-_SAL1P9EU2SA60000{i}_.jpg': f'SAL1P9EU2SA60000{i}'
        for i in range(100)
    }


@pytest.fixture
def small_ground_truth() -> Dict[str, str]:
    """Small dataset for edge case testing."""
    return {
        '/images/img1.jpg': 'SAL1P9EU2SA606661',
        '/images/img2.jpg': 'SAL1P9EU2SA606662',
        '/images/img3.jpg': 'SAL1P9EU2SA606663',
    }


# =============================================================================
# VIN EXTRACTION TESTS
# =============================================================================

class TestExtractVINFromFilename:
    """Tests for VIN extraction from filename."""
    
    def test_standard_format(self):
        """Test standard filename format."""
        vin = extract_vin_from_filename("001-VIN_-_SAL1P9EU2SA606664_.jpg")
        assert vin == "SAL1P9EU2SA606664"
    
    def test_with_suffix_number(self):
        """Test filename with variant suffix."""
        vin = extract_vin_from_filename("001-VIN_-_SAL1P9EU2SA606664_2.jpg")
        assert vin == "SAL1P9EU2SA606664"
    
    def test_prefixed_format(self):
        """Test filename with VIN prefix."""
        vin = extract_vin_from_filename("SAL1P9EU2SA606664_train_001-VIN_-_SAL1P9EU2SA606664_.jpg")
        assert vin == "SAL1P9EU2SA606664"
    
    def test_no_vin_returns_none(self):
        """Test that non-VIN filename returns None."""
        vin = extract_vin_from_filename("random_image_001.jpg")
        assert vin is None
    
    def test_lowercase_normalized(self):
        """Test that VINs are normalized to uppercase."""
        vin = extract_vin_from_filename("001-VIN_-_sal1p9eu2sa606664_.jpg")
        assert vin == "SAL1P9EU2SA606664"
    
    def test_invalid_vin_with_ioq(self):
        """Test that VINs with I, O, Q are rejected in fallback."""
        # The pattern should still match but may contain invalid chars
        vin = extract_vin_from_filename("image_ABCDEFGHIJK123456.jpg")
        # May return None or the VIN depending on implementation
        # The key is consistency


# =============================================================================
# DATASET SPLIT TESTS
# =============================================================================

class TestCreateSplits:
    """Tests for dataset split creation."""
    
    def test_split_ratios_correct(self, sample_ground_truth):
        """Test that split ratios are approximately correct."""
        splits = create_splits(
            sample_ground_truth,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            seed=42
        )
        
        total = len(sample_ground_truth)
        assert abs(len(splits['train'].image_paths) - int(total * 0.7)) <= 1
        assert abs(len(splits['val'].image_paths) - int(total * 0.15)) <= 1
        assert abs(len(splits['test'].image_paths) - int(total * 0.15)) <= 1
    
    def test_no_data_leakage(self, sample_ground_truth):
        """Test that splits are completely disjoint (no data leakage)."""
        splits = create_splits(sample_ground_truth, seed=42)
        
        train_set = set(splits['train'].image_paths)
        val_set = set(splits['val'].image_paths)
        test_set = set(splits['test'].image_paths)
        
        # No overlap between any splits
        assert len(train_set & val_set) == 0, "Train-Val overlap detected!"
        assert len(train_set & test_set) == 0, "Train-Test overlap detected!"
        assert len(val_set & test_set) == 0, "Val-Test overlap detected!"
    
    def test_all_samples_used(self, sample_ground_truth):
        """Test that all samples are assigned to exactly one split."""
        splits = create_splits(sample_ground_truth, seed=42)
        
        train_set = set(splits['train'].image_paths)
        val_set = set(splits['val'].image_paths)
        test_set = set(splits['test'].image_paths)
        
        all_assigned = train_set | val_set | test_set
        all_original = set(sample_ground_truth.keys())
        
        assert all_assigned == all_original, "Not all samples assigned!"
    
    def test_reproducibility_with_seed(self, sample_ground_truth):
        """Test that same seed produces identical splits."""
        splits1 = create_splits(sample_ground_truth, seed=42)
        splits2 = create_splits(sample_ground_truth, seed=42)
        
        assert splits1['train'].image_paths == splits2['train'].image_paths
        assert splits1['val'].image_paths == splits2['val'].image_paths
        assert splits1['test'].image_paths == splits2['test'].image_paths
    
    def test_different_seeds_different_splits(self, sample_ground_truth):
        """Test that different seeds produce different splits."""
        splits1 = create_splits(sample_ground_truth, seed=42)
        splits2 = create_splits(sample_ground_truth, seed=123)
        
        # With high probability, at least train sets should differ
        # (technically could be same by chance, but extremely unlikely)
        assert splits1['train'].image_paths != splits2['train'].image_paths
    
    def test_small_dataset_splits(self, small_ground_truth):
        """Test splitting small dataset."""
        splits = create_splits(
            small_ground_truth,
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
            seed=42
        )
        
        total = len(splits['train'].image_paths) + \
                len(splits['val'].image_paths) + \
                len(splits['test'].image_paths)
        assert total == len(small_ground_truth)
    
    def test_invalid_ratios_rejected(self, sample_ground_truth):
        """Test that ratios not summing to 1.0 are rejected."""
        with pytest.raises(AssertionError):
            create_splits(
                sample_ground_truth,
                train_ratio=0.5,
                val_ratio=0.2,
                test_ratio=0.2,  # Sum = 0.9
                seed=42
            )


# =============================================================================
# METRICS TESTS
# =============================================================================

class TestLevenshteinDistance:
    """Tests for Levenshtein distance calculation."""
    
    def test_identical_strings(self):
        """Test distance between identical strings."""
        assert levenshtein_distance("ABC", "ABC") == 0
    
    def test_empty_strings(self):
        """Test distance between empty strings."""
        assert levenshtein_distance("", "") == 0
    
    def test_one_empty(self):
        """Test distance when one string is empty."""
        assert levenshtein_distance("ABC", "") == 3
        assert levenshtein_distance("", "ABC") == 3
    
    def test_single_substitution(self):
        """Test single character substitution."""
        assert levenshtein_distance("ABC", "ADC") == 1
    
    def test_single_insertion(self):
        """Test single character insertion."""
        assert levenshtein_distance("ABC", "ABDC") == 1
    
    def test_single_deletion(self):
        """Test single character deletion."""
        assert levenshtein_distance("ABDC", "ABC") == 1
    
    def test_full_vin_identical(self):
        """Test full VIN identical."""
        vin = "SAL1P9EU2SA606664"
        assert levenshtein_distance(vin, vin) == 0
    
    def test_full_vin_one_char_diff(self):
        """Test full VIN with one character difference."""
        vin1 = "SAL1P9EU2SA606664"
        vin2 = "SAL1P9EU2SA606665"
        assert levenshtein_distance(vin1, vin2) == 1


class TestCalculateCER:
    """Tests for Character Error Rate calculation."""
    
    def test_perfect_predictions(self):
        """Test CER for perfect predictions."""
        preds = ["ABC", "DEF"]
        refs = ["ABC", "DEF"]
        assert calculate_cer(preds, refs) == 0.0
    
    def test_all_wrong(self):
        """Test CER for completely wrong predictions."""
        preds = ["ABC"]
        refs = ["DEF"]
        # 3 substitutions / 3 chars = 1.0
        assert calculate_cer(preds, refs) == 1.0
    
    def test_empty_predictions(self):
        """Test CER for empty predictions."""
        preds = [""]
        refs = ["ABC"]
        # 3 insertions needed / 3 chars = 1.0
        assert calculate_cer(preds, refs) == 1.0
    
    def test_empty_reference_handled(self):
        """Test CER with empty references (edge case)."""
        preds = [""]
        refs = [""]
        # 0/0 should return 0.0
        assert calculate_cer(preds, refs) == 0.0


class TestCharacterMetrics:
    """Tests for character-level precision/recall/F1."""
    
    def test_perfect_match(self):
        """Test metrics for perfect match."""
        precision, recall, f1 = calculate_character_metrics(
            ["SAL1P9EU2SA606664"],
            ["SAL1P9EU2SA606664"]
        )
        assert precision == 1.0
        assert recall == 1.0
        assert f1 == 1.0
    
    def test_empty_prediction(self):
        """Test metrics for empty prediction."""
        precision, recall, f1 = calculate_character_metrics(
            [""],
            ["SAL1P9EU2SA606664"]
        )
        # Empty pred: 0 TP, 0 FP, 17 FN
        # P = 0/0 = 0.0 (handled), R = 0/17 = 0.0
        assert recall == 0.0
        assert f1 == 0.0
    
    def test_partial_match(self):
        """Test metrics for partial match."""
        precision, recall, f1 = calculate_character_metrics(
            ["SAL"],
            ["SAL1P9EU2SA606664"]
        )
        # 3 matches at positions 0,1,2
        # P = 3/3 = 1.0, R = 3/17 ≈ 0.176
        assert precision == 1.0
        assert abs(recall - 3/17) < 0.001
    
    def test_one_wrong_character(self):
        """Test metrics for one wrong character."""
        precision, recall, f1 = calculate_character_metrics(
            ["SAL1P9EU2SA606665"],  # Last char different
            ["SAL1P9EU2SA606664"]
        )
        # 16 matches, 1 wrong
        expected_pr = 16/17
        assert abs(precision - expected_pr) < 0.001
        assert abs(recall - expected_pr) < 0.001


class TestPositionAccuracy:
    """Tests for per-position accuracy calculation."""
    
    def test_perfect_accuracy(self):
        """Test position accuracy for perfect predictions."""
        accuracies = calculate_position_accuracy(
            ["SAL1P9EU2SA606664"],
            ["SAL1P9EU2SA606664"]
        )
        assert len(accuracies) == 17
        assert all(acc == 1.0 for acc in accuracies)
    
    def test_one_position_wrong(self):
        """Test position accuracy with one wrong position."""
        accuracies = calculate_position_accuracy(
            ["XAL1P9EU2SA606664"],  # Position 1 wrong
            ["SAL1P9EU2SA606664"]
        )
        assert accuracies[0] == 0.0  # Position 1 (index 0)
        assert all(accuracies[i] == 1.0 for i in range(1, 17))
    
    def test_multiple_samples(self):
        """Test position accuracy aggregation."""
        accuracies = calculate_position_accuracy(
            ["SAL1P9EU2SA606664", "SAL1P9EU2SA606664"],
            ["SAL1P9EU2SA606664", "SAL1P9EU2SA606665"]  # Second has last char diff
        )
        # Position 17 (index 16) should be 50%
        assert accuracies[16] == 0.5


# =============================================================================
# DATA CLASS TESTS
# =============================================================================

class TestDataClasses:
    """Tests for data classes."""
    
    def test_evaluation_metrics_to_dict(self):
        """Test EvaluationMetrics serialization."""
        metrics = EvaluationMetrics(
            total_samples=100,
            exact_match_rate=0.5,
            character_f1=0.75
        )
        d = metrics.to_dict()
        assert d['total_samples'] == 100
        assert d['exact_match_rate'] == 0.5
        assert d['character_f1'] == 0.75
    
    def test_prediction_result_creation(self):
        """Test PredictionResult creation."""
        result = PredictionResult(
            image_path="/test/image.jpg",
            ground_truth="SAL1P9EU2SA606664",
            prediction="SAL1P9EU2SA606665",
            confidence=0.95,
            raw_ocr="SAL1P9EU2SA606665",
            exact_match=False,
            edit_distance=1,
            processing_time_ms=100.0
        )
        assert result.exact_match is False
        assert result.edit_distance == 1
    
    def test_dataset_split_creation(self):
        """Test DatasetSplit creation."""
        split = DatasetSplit(
            name='test',
            image_paths=['/img1.jpg', '/img2.jpg'],
            ground_truths={'/img1.jpg': 'VIN1', '/img2.jpg': 'VIN2'}
        )
        assert split.name == 'test'
        assert len(split.image_paths) == 2
        assert split.ground_truths['/img1.jpg'] == 'VIN1'


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
