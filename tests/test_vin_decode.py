"""
Tests for checksum-constrained CTC decoding (core.vin_decode).

Grounded in the measured failure modes of the stage-3b model
(2026-08-19): position 9 (the deterministic ISO-3779 check digit) read at
8% accuracy, and confusable glyph pairs (6<->5, 8<->6/0, B->A) where the
correct character is the CLOSE second-best probability that greedy
decoding discards.
"""

import numpy as np
import pytest

from src.vin_ocr.core.charset import BLANK_INDEX, load_char_dict
from src.vin_ocr.core.vin_decode import (
    ConstrainedDecodeResult,
    constrained_vin_decode,
    ctc_prefix_beam_search,
)
from src.vin_ocr.core.vin_utils import calculate_check_digit, validate_vin

C2I, I2C = load_char_dict("configs/vin_dict.txt")
N_CLASSES = len(I2C)

# A checksum-valid VIN (check digit X at position 9) used across tests.
GOLDEN_VIN = "1M8GDM9AXKP042788"
assert validate_vin(GOLDEN_VIN).checksum_valid


def probs_for(text, *, confidence=0.98, overrides=None):
    """
    Build [T, C] probabilities encoding `text` blank-separated.

    overrides: {char_position: {char: prob, ...}} - remaining mass goes to
    the intended character; used to make a position ambiguous.
    """
    overrides = overrides or {}
    seq = []
    char_steps = {}  # char position -> timestep
    for i, ch in enumerate(text):
        seq.append(BLANK_INDEX)
        char_steps[i] = len(seq)
        seq.append(C2I[ch])
    seq.append(BLANK_INDEX)

    T = len(seq)
    probs = np.full((T, N_CLASSES), (1 - confidence) / (N_CLASSES - 1))
    for t, idx in enumerate(seq):
        probs[t, :] = (1 - confidence) / (N_CLASSES - 1)
        probs[t, idx] = confidence
    for pos, dist in overrides.items():
        t = char_steps[pos]
        probs[t, :] = 0.0
        assigned = 0.0
        for ch, p in dist.items():
            probs[t, C2I[ch]] = p
            assigned += p
        probs[t, C2I[text[pos]]] += 1.0 - assigned
    return probs / probs.sum(axis=1, keepdims=True)


class TestPrefixBeamSearch:
    def test_matches_greedy_on_unambiguous_input(self):
        probs = probs_for(GOLDEN_VIN)
        results = ctc_prefix_beam_search(probs, I2C, beam_width=8)
        assert results[0][0] == GOLDEN_VIN
        assert results[0][1] > results[-1][1] or len(results) == 1

    def test_blank_separated_repeat_survives(self):
        probs = probs_for("AA1")
        results = ctc_prefix_beam_search(probs, I2C, beam_width=8)
        assert results[0][0] == "AA1"

    def test_close_second_hypothesis_is_kept(self):
        """The measured 6<->5 confusion: greedy commits, beam keeps both."""
        probs = probs_for(GOLDEN_VIN, overrides={11: {"5": 0.55, GOLDEN_VIN[11]: 0.45}})
        texts = [t for t, _ in ctc_prefix_beam_search(probs, I2C, beam_width=8)]
        wrong = GOLDEN_VIN[:11] + "5" + GOLDEN_VIN[12:]
        assert texts[0] == wrong            # greedy's choice ranks first...
        assert GOLDEN_VIN in texts          # ...but the truth is in the beam

    def test_rejects_bad_inputs(self):
        with pytest.raises(ValueError):
            ctc_prefix_beam_search(np.zeros((0, N_CLASSES)), I2C)
        with pytest.raises(ValueError):
            ctc_prefix_beam_search(np.zeros((4, N_CLASSES)), I2C, beam_width=0)


class TestConstrainedDecode:
    def test_checksum_selects_truth_over_confusable_argmax(self):
        """
        THE payoff case: at a confusable position the wrong glyph has the
        higher probability. Greedy returns a checksum-INVALID reading;
        the constrained decoder selects the checksum-valid second
        hypothesis - unrepaired, as read.
        """
        probs = probs_for(GOLDEN_VIN, overrides={11: {"5": 0.55, GOLDEN_VIN[11]: 0.45}})
        result = constrained_vin_decode(probs, I2C, beam_width=8)
        assert result.vin == GOLDEN_VIN
        assert result.checksum_valid and not result.checksum_repaired
        assert result.beam_rank > 0  # it was NOT the argmax hypothesis

    def test_position9_is_computed_when_misread(self):
        """
        Measured: the model reads the check digit at 8% accuracy. When
        every hypothesis carries a misread position 9, the decoder
        computes it from the other 16 - flagged, raw preserved.
        """
        misread = GOLDEN_VIN[:8] + "3" + GOLDEN_VIN[9:]  # true digit is X
        assert not validate_vin(misread).checksum_valid
        probs = probs_for(misread)
        result = constrained_vin_decode(probs, I2C, beam_width=8)
        assert result.vin == GOLDEN_VIN
        assert result.checksum_repaired is True
        assert result.raw_reading == misread
        assert result.checksum_valid is True

    def test_repair_never_masks_other_errors(self):
        """
        A hypothesis wrong in a NON-check-digit position gets repaired to
        a checksum-valid string that differs from the truth - the repair
        flag plus raw reading keep that visible; validity is reported for
        the returned string, never asserted as ground truth.
        """
        wrong = GOLDEN_VIN[:5] + "Z" + GOLDEN_VIN[6:]  # corrupt position 6
        probs = probs_for(wrong)
        result = constrained_vin_decode(probs, I2C, beam_width=4)
        assert result.checksum_repaired is True
        assert result.raw_reading == wrong
        assert result.vin != GOLDEN_VIN  # repair cannot resurrect position 6

    def test_valid_raw_preferred_over_repaired(self):
        probs = probs_for(GOLDEN_VIN)
        result = constrained_vin_decode(probs, I2C, beam_width=8)
        assert result.vin == GOLDEN_VIN
        assert not result.checksum_repaired
        assert result.beam_rank == 0

    def test_empty_lattice_returns_none_not_empty_vin(self):
        probs = np.zeros((6, N_CLASSES))
        probs[:, BLANK_INDEX] = 1.0
        assert constrained_vin_decode(probs, I2C) is None

    def test_fallback_is_flagged_invalid(self):
        """Short garbage: returned raw, honestly marked invalid."""
        probs = probs_for("1M8")
        result = constrained_vin_decode(probs, I2C, beam_width=4)
        assert result.vin == "1M8"
        assert result.checksum_valid is False
        assert result.checksum_repaired is False
