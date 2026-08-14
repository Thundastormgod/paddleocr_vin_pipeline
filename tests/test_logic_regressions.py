"""
Regression tests for logic defects found during the correctness audit.

Each test pins behaviour that was previously WRONG. The docstrings record the
old behaviour so a future change that reintroduces it fails loudly rather than
silently degrading results.
"""

from unittest.mock import Mock

import pytest

from src.vin_ocr.core.vin_utils import (
    _score_vin_candidate,
    extract_vin_from_text,
    levenshtein_distance,
    validate_checksum,
)
from src.vin_ocr.evaluation.evaluate import (
    calculate_cer,
    calculate_character_metrics,
)
from src.vin_ocr.evaluation.metrics import EvaluationMetricsCalculator
from src.vin_ocr.providers.ocr_providers import (
    EnsembleOCRProvider,
    OCRProvider,
    OCRResult,
)
from src.vin_ocr.utils.prepare_dataset import assert_no_vin_leakage, create_splits

# A real, checksum-valid VIN used throughout.
VIN = "SAL1A2A40SA606662"


def test_reference_vin_is_checksum_valid():
    """Guard the fixture itself - the rest of the file depends on it."""
    assert validate_checksum(VIN)
    assert len(VIN) == 17


# ---------------------------------------------------------------------------
# Character metrics: alignment, not positional zip
# ---------------------------------------------------------------------------

class TestAlignmentBasedCharacterMetrics:
    """
    WAS BROKEN: precision/recall/F1 compared pred[i] to ref[i] positionally, so
    one leading insertion shifted every position and drove F1 to 0.118 for a
    string 88% correct by edit distance - while a substitution of comparable
    edit distance scored 0.941 (~8x discrepancy). Leading artifacts are the
    documented dominant failure mode for engraved plates.
    """

    def test_leading_insertion_does_not_collapse_f1(self):
        pred = "X" + VIN[:-1]  # one inserted char at the front
        _, _, f1 = calculate_character_metrics([pred], [VIN])
        assert f1 > 0.9, f"leading insertion collapsed F1 to {f1:.3f}"

    def test_insertion_and_substitution_score_comparably(self):
        """Equal-ish edit distance should give equal-ish F1."""
        insertion = "X" + VIN[:-1]
        substitution = VIN[:-1] + ("3" if VIN[-1] != "3" else "4")
        _, _, f1_ins = calculate_character_metrics([insertion], [VIN])
        _, _, f1_sub = calculate_character_metrics([substitution], [VIN])
        assert abs(f1_ins - f1_sub) < 0.1, (
            f"insertion F1={f1_ins:.3f} vs substitution F1={f1_sub:.3f}"
        )

    def test_f1_tracks_edit_distance_monotonically(self):
        preds = [VIN, "X" + VIN[:-1], "XY" + VIN[:-2], "XYZ" + VIN[:-3]]
        f1s = [calculate_character_metrics([p], [VIN])[2] for p in preds]
        assert f1s == sorted(f1s, reverse=True), f"F1 not monotonic: {f1s}"

    def test_perfect_and_empty_still_correct(self):
        assert calculate_character_metrics([VIN], [VIN]) == (1.0, 1.0, 1.0)
        p, r, f = calculate_character_metrics([""], [VIN])
        assert (r, f) == (0.0, 0.0)


class TestCerDefinition:
    """
    WAS BROKEN: char_error_rate was defined as 1 - positional_accuracy. For a
    single leading insertion it reported 0.882 while normalized_edit_distance,
    computed from the same edit distance on the same object, reported 0.118.
    CER is an edit-distance quantity: (S+D+I)/N.
    """

    @pytest.mark.parametrize("pred", [VIN, "X" + VIN[:-1], VIN[:-1] + "3", ""])
    def test_cer_equals_normalized_edit_distance(self, pred):
        calc = EvaluationMetricsCalculator()
        calc.add_sample(pred, VIN)
        cl = calc.compute().character_level
        assert cl.char_error_rate == pytest.approx(cl.normalized_edit_distance)

    def test_cer_matches_manual_edit_distance(self):
        pred = "X" + VIN[:-1]
        expected = levenshtein_distance(pred, VIN) / len(VIN)
        assert calculate_cer([pred], [VIN]) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# VIN extraction: checksum must be decisive
# ---------------------------------------------------------------------------

class TestVinExtractionScoring:
    """
    WAS BROKEN: extraction returned the FIRST window starting with any known
    WMI, behind a `score > 10` guard that 17 arbitrary letters already cleared
    (score 34). A spurious earlier "SAL" hijacked the result, and the check
    digit contributed nothing to scoring (valid and invalid both scored 62).
    """

    def test_checksum_valid_outranks_invalid(self):
        invalid = VIN[:8] + ("1" if VIN[8] != "1" else "2") + VIN[9:]
        assert not validate_checksum(invalid)
        assert _score_vin_candidate(VIN) > _score_vin_candidate(invalid)

    def test_checksum_bonus_exceeds_all_other_terms(self):
        """Max non-checksum score is 62; the bonus must dominate it."""
        invalid = VIN[:8] + ("1" if VIN[8] != "1" else "2") + VIN[9:]
        assert _score_vin_candidate(VIN) - _score_vin_candidate(invalid) > 62

    @pytest.mark.parametrize("noisy", [
        "SAL99" + VIN,          # spurious WMI before the real VIN
        "SALSALSAL" + VIN,      # several spurious WMIs
        VIN + "XXXXX",          # trailing noise
        "***" + VIN + "###",    # surrounding artifacts
    ])
    def test_extracts_checksum_valid_vin_from_noise(self, noisy):
        assert extract_vin_from_text(noisy) == VIN

    def test_exact_length_passthrough(self):
        assert extract_vin_from_text(VIN) == VIN

    def test_too_short_returned_unchanged(self):
        assert extract_vin_from_text("SAL123") == "SAL123"


# ---------------------------------------------------------------------------
# Dataset splitting: group by VIN
# ---------------------------------------------------------------------------

class TestSplitLeakage:
    """
    WAS BROKEN: src/vin_ocr/utils/prepare_dataset.create_splits shuffled image
    paths, so multiple images of the SAME physical plate could land in both
    train and test - the model memorises a plate and is then scored on it.
    """

    def test_all_images_of_a_vin_share_one_split(self):
        images = {}
        for i in range(4):
            images[f"/d/A_{i}.jpg"] = VIN
        for i in range(3):
            images[f"/d/B_{i}.jpg"] = "1HGBH41JXMN109186"
        for i in range(3):
            images[f"/d/C_{i}.jpg"] = "SAL1P9EU2SA606664"

        train, val, test = create_splits(images, 0.34, 0.33, 0.33, seed=1)

        assert set(train.values()) & set(val.values()) == set()
        assert set(train.values()) & set(test.values()) == set()
        assert set(val.values()) & set(test.values()) == set()
        # nothing lost
        assert len(train) + len(val) + len(test) == len(images)

    def test_leakage_detector_raises(self):
        with pytest.raises(ValueError, match="leakage"):
            assert_no_vin_leakage({"a": VIN}, {"b": VIN}, {"c": "OTHER"})

    def test_clean_split_passes(self):
        assert_no_vin_leakage({"a": VIN}, {"b": "1HGBH41JXMN109186"}, {"c": "X"})


# ---------------------------------------------------------------------------
# Ensemble: align before voting
# ---------------------------------------------------------------------------

def _ensemble():
    """
    Build a real EnsembleOCRProvider over a stub provider.

    Deliberately does NOT monkeypatch EnsembleOCRProvider.name: assigning to
    `type(ens).name` mutates the class for the whole session and leaks into
    unrelated tests (it broke test_ocr_providers.py::test_creation_with_providers).
    """
    stub = Mock(spec=OCRProvider)
    stub.name = "stub"
    stub.is_available = True
    stub.is_initialized = True
    return EnsembleOCRProvider(providers=[stub], strategy="weighted_char_vote")


class TestEnsembleCharVote:
    """
    WAS BROKEN: per-character voting indexed candidates positionally, so a
    candidate with a leading artifact voted a shifted character into every
    slot. Given one exactly-correct input and two shifted ones, the ensemble
    returned edit distance 2 - worse than its own best member. Confidence was
    a raw sum of weights clamped to 1.0, so any 3 providers with confidence
    >= ~0.34 reported 1.0 regardless of agreement.
    """

    def test_ensemble_not_worse_than_best_member(self):
        results = [
            OCRResult(text=VIN, confidence=0.80, provider="A"),
            OCRResult(text="*" + VIN[:-1], confidence=0.75, provider="B"),
            OCRResult(text="X" + VIN[:-1], confidence=0.70, provider="C"),
        ]
        out = _ensemble()._weighted_char_vote_strategy(results)
        best_member = min(levenshtein_distance(r.text, VIN) for r in results)
        assert levenshtein_distance(out.text, VIN) <= best_member
        assert out.text == VIN

    def test_confidence_reflects_provider_confidence(self):
        """Unanimous-but-unsure must not report 1.0."""
        low = [OCRResult(text=VIN, confidence=0.34, provider=p) for p in "ABC"]
        high = [OCRResult(text=VIN, confidence=0.90, provider=p) for p in "ABC"]
        c_low = _ensemble()._weighted_char_vote_strategy(low).confidence
        c_high = _ensemble()._weighted_char_vote_strategy(high).confidence
        assert c_low < c_high
        assert c_low < 0.5

    def test_confidence_reflects_disagreement(self):
        agree = [OCRResult(text=VIN, confidence=0.9, provider=p) for p in "AB"]
        disagree = [
            OCRResult(text=VIN, confidence=0.9, provider="A"),
            OCRResult(text="Z" * 17, confidence=0.9, provider="B"),
        ]
        c_agree = _ensemble()._weighted_char_vote_strategy(agree).confidence
        c_dis = _ensemble()._weighted_char_vote_strategy(disagree).confidence
        assert c_dis < c_agree

    def test_metadata_exposes_confidence_components(self):
        results = [OCRResult(text=VIN, confidence=0.8, provider="A")]
        meta = _ensemble()._weighted_char_vote_strategy(results).metadata
        assert "agreement" in meta and "mean_provider_confidence" in meta
