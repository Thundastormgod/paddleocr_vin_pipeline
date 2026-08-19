"""
Checksum-constrained CTC decoding for VINs.

Motivation, measured on the stage-3b model (2026-08-19): greedy decode
reads position 9 - the ISO-3779 check digit - at 8% accuracy, because it
is a low-contrast stamped digit the model must SEE. But the check digit
is DETERMINISTIC: a mod-11 function of the other 16 characters
(core.vin_utils.calculate_check_digit). Greedy decoding also commits to
the argmax character at confusable glyphs (measured confusions: 6<->5,
8<->6/0, B->A/9) where the second-best probability is often close.

This module exploits both facts without retraining:

1. **CTC prefix beam search** (Hannun-style, log-space) keeps the top-K
   sequence hypotheses instead of the single argmax path, so close
   confusable alternatives survive to selection.
2. **Constrained selection** prefers hypotheses whose ISO-3779 checksum
   validates; for 17-char hypotheses whose only defect may be the check
   digit itself, a POSITION-9-REPAIRED twin (check digit computed from
   the other 16) is added and flagged.

Honesty invariants:
- Repair happens ONLY inside probability-space hypothesis selection and
  the result carries ``checksum_repaired=True`` plus the raw reading.
  A text-level blanket rewrite of position 9 would make every output
  checksum-valid by construction and destroy the checksum's diagnostic
  value; that is deliberately NOT offered.
- The decoder never fabricates length: hypotheses are what the lattice
  produced (trimmed to 17 only when longer, recorded via ``truncated``).

MEASURED APPLICABILITY LIMIT (2026-08-19, stage-3b model, mean ~5
errors/plate): constrained selection made results WORSE than greedy -
ungated: val char accuracy 0.6661 -> 0.6436; gated to edit distance 1
from the top hypothesis: 0.6661 -> 0.6471. Mechanism: the checksum
carries information about distance from the TRUTH, and at d~5 no beam
hypothesis is near the truth, while ~9% of random 17-char strings
checksum-validate - so every checksum-driven substitution is noise.
DO NOT enable this decoder as a default until the model's edit-distance
mass sits at d<=2 (where mod-11's ~91% single-error detection is
actually informative). It is deliberately wired into no default path.

Complexity: O(T * K * C) time, K beam width (default 16), T timesteps,
C classes (34). Pure numpy + stdlib.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .charset import BLANK_INDEX, BLANK_TOKEN
from .vin_utils import VIN_LENGTH, calculate_check_digit, validate_vin

NEG_INF = float("-inf")


def _logsumexp2(a: float, b: float) -> float:
    """log(exp(a) + exp(b)) for two scalars, tolerant of -inf."""
    if a == NEG_INF:
        return b
    if b == NEG_INF:
        return a
    hi, lo = (a, b) if a > b else (b, a)
    return hi + float(np.log1p(np.exp(lo - hi)))


def ctc_prefix_beam_search(
    probs: np.ndarray,
    idx_to_char: Dict[int, str],
    beam_width: int = 16,
    blank_index: int = BLANK_INDEX,
) -> List[Tuple[str, float]]:
    """
    Standard CTC prefix beam search in log space.

    Args:
        probs: [T, C] per-timestep class PROBABILITIES (softmax output).
        idx_to_char: Index -> character map (blank entry ignored).
        beam_width: Hypotheses kept per step. 1 degenerates to greedy-like
            behaviour; 16 is ample for 34 classes.
        blank_index: CTC blank class index.

    Returns:
        Up to beam_width (text, log_probability) tuples, best first. The
        log probability is the CTC-summed probability of the LABELLING
        (all alignments), which is what makes beam search strictly more
        faithful than best-path greedy.

    Raises:
        ValueError: On empty/ill-shaped input or beam_width < 1.
    """
    if beam_width < 1:
        raise ValueError(f"beam_width must be >= 1, got {beam_width}")
    if probs.ndim != 2 or probs.shape[0] == 0:
        raise ValueError(f"expected probs [T, C], got shape {probs.shape}")

    log_probs = np.log(np.clip(probs, 1e-30, 1.0))
    T, C = log_probs.shape

    # prefix -> (log P(prefix ending in blank), log P(ending in non-blank))
    beams: Dict[Tuple[int, ...], Tuple[float, float]] = {(): (0.0, NEG_INF)}

    for t in range(T):
        step = log_probs[t]
        new_beams: Dict[Tuple[int, ...], Tuple[float, float]] = {}

        def add(prefix: Tuple[int, ...], p_b: float, p_nb: float) -> None:
            old_b, old_nb = new_beams.get(prefix, (NEG_INF, NEG_INF))
            new_beams[prefix] = (
                _logsumexp2(old_b, p_b), _logsumexp2(old_nb, p_nb)
            )

        for prefix, (p_b, p_nb) in beams.items():
            total = _logsumexp2(p_b, p_nb)
            # extend with blank: prefix unchanged, now ends in blank
            add(prefix, total + step[blank_index], NEG_INF)

            for c in range(C):
                if c == blank_index:
                    continue
                p_c = step[c]
                if prefix and prefix[-1] == c:
                    # same char: without separator it merges (from p_nb),
                    # after a blank it extends (from p_b)
                    add(prefix, NEG_INF, p_nb + p_c)
                    add(prefix + (c,), NEG_INF, p_b + p_c)
                else:
                    add(prefix + (c,), NEG_INF, total + p_c)

        beams = dict(
            sorted(
                new_beams.items(),
                key=lambda kv: _logsumexp2(*kv[1]),
                reverse=True,
            )[:beam_width]
        )

    results = []
    for prefix, (p_b, p_nb) in beams.items():
        text = "".join(
            idx_to_char.get(i, "")
            for i in prefix
            if idx_to_char.get(i, "") not in ("", BLANK_TOKEN)
        )
        results.append((text, _logsumexp2(p_b, p_nb)))
    results.sort(key=lambda x: -x[1])
    return results


@dataclass(frozen=True)
class ConstrainedDecodeResult:
    """Outcome of checksum-constrained decoding, with full honesty trail."""

    vin: str                    #: the selected 17-char hypothesis
    log_prob: float             #: CTC log-probability of the SOURCE hypothesis
    checksum_valid: bool        #: ISO-3779 validity of `vin` as returned
    checksum_repaired: bool     #: True if position 9 was COMPUTED, not read
    raw_reading: str            #: the unrepaired source hypothesis
    beam_rank: int              #: rank of the source hypothesis in the beam
    truncated: bool             #: source was longer than 17 and trimmed


def constrained_vin_decode(
    probs: np.ndarray,
    idx_to_char: Dict[int, str],
    beam_width: int = 16,
    blank_index: int = BLANK_INDEX,
    max_edit_from_top: Optional[int] = 1,
) -> Optional[ConstrainedDecodeResult]:
    """
    Decode a VIN from CTC probabilities under the ISO-3779 constraint.

    Selection order over the beam hypotheses (all 17-char after trim):
      1. checksum-valid AS READ (highest beam probability first);
      2. checksum-valid after POSITION-9 REPAIR (check digit computed
         from the other 16; flagged, raw reading preserved);
      3. best raw hypothesis (checksum invalid - reported as such).

    Args:
        max_edit_from_top: A checksum-valid alternative (raw or repaired)
            is only eligible when its RAW reading is within this edit
            distance of the top hypothesis. Default 1 - MEASURED
            rationale: with the stage-3b model (mean ~5 errors/plate),
            ungated selection made results WORSE (val char accuracy
            0.6661 -> 0.6436), because at that error rate the true VIN is
            not in the beam while random wrong-but-checksummy variants
            are (~9% of strings validate). Within edit distance 1 the
            checksum's single-error detection (~91%) is actually
            informative. Pass None to disable the gate (appropriate only
            once the model operates in the d<=2 regime).

    Rationale for repair: position 9 carries no independent information
    (it is a function of the rest) and the model reads it at 8% accuracy;
    a hypothesis correct in the other 16 positions is strictly more
    likely to be the true VIN with its check digit computed than with a
    misread digit retained.

    Returns:
        ConstrainedDecodeResult, or None when the lattice produced no
        non-empty hypothesis (nothing was read - callers must treat this
        as an absent measurement, not an empty VIN).
    """
    from .vin_utils import levenshtein_distance
    candidates = ctc_prefix_beam_search(
        probs, idx_to_char, beam_width=beam_width, blank_index=blank_index
    )

    # Nothing read: when the TOP hypothesis is the empty labelling, the
    # lattice's most probable explanation is "no characters" - the junk
    # hypotheses ranked beneath it (astronomically less probable) must not
    # be promoted into a reading. Absent measurement, not an empty VIN.
    if not candidates or candidates[0][0] == "":
        return None

    top_reading = candidates[0][0][:VIN_LENGTH]

    valid_raw: List[ConstrainedDecodeResult] = []
    valid_repaired: List[ConstrainedDecodeResult] = []
    fallback: Optional[ConstrainedDecodeResult] = None

    for rank, (text, log_prob) in enumerate(candidates):
        if not text:
            continue
        truncated = len(text) > VIN_LENGTH
        vin = text[:VIN_LENGTH]
        if fallback is None:
            fallback = ConstrainedDecodeResult(
                vin=vin, log_prob=log_prob, checksum_valid=False,
                checksum_repaired=False, raw_reading=vin,
                beam_rank=rank, truncated=truncated,
            )
        if len(vin) != VIN_LENGTH:
            continue
        if (
            max_edit_from_top is not None
            and rank > 0
            and levenshtein_distance(vin, top_reading) > max_edit_from_top
        ):
            continue  # too far from the model's reading to adjudicate

        if validate_vin(vin).checksum_valid:
            valid_raw.append(ConstrainedDecodeResult(
                vin=vin, log_prob=log_prob, checksum_valid=True,
                checksum_repaired=False, raw_reading=vin,
                beam_rank=rank, truncated=truncated,
            ))
            continue

        expected = calculate_check_digit(vin)
        if expected is not None and expected != vin[8]:
            repaired = vin[:8] + expected + vin[9:]
            if validate_vin(repaired).checksum_valid:
                valid_repaired.append(ConstrainedDecodeResult(
                    vin=repaired, log_prob=log_prob, checksum_valid=True,
                    checksum_repaired=True, raw_reading=vin,
                    beam_rank=rank, truncated=truncated,
                ))

    if valid_raw:
        return valid_raw[0]
    if valid_repaired:
        return valid_repaired[0]
    return fallback
