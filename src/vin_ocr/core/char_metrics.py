"""
Canonical character-level metrics for VIN OCR.

This is the single implementation of character-level accuracy, precision,
recall and F1 in this repository. The audit of 2026-08-18 found the same
concept implemented in four places with three incompatible definitions:
identical input scored char-F1 0.67 / 0.95 / 0.4706 / 0.9412 depending on
which file computed it. Two of the copies compared ``pred[i] == ref[i]``
positionally, so a single leading artifact ("*SAL1..." for "SAL1...")
shifted every subsequent position and scored a 94%-correct prediction at
0.059 - and one of them could not count false positives for characters that
are invalid in a VIN, making precision >= recall structurally, with 100%
precision reported at 29% recall for truncated predictions.

Definitions used here (documented so every consumer means the same thing):

- Alignment counts come from difflib.SequenceMatcher opcodes between the
  prediction and the reference (the same convention as the previously fixed
  copy in evaluation/evaluate.py):

      equal    -> TP for each aligned character
      replace  -> FP for each predicted char, FN for each reference char
      insert   -> FN (reference characters the prediction missed)
      delete   -> FP (predicted characters with no reference counterpart)

  EVERY predicted character is either a TP or an FP - including characters
  that could never occur in a VIN and characters emitted beyond the
  reference length. A wrong emission always costs precision.

- CER is an edit-distance quantity: levenshtein(pred, ref) summed over the
  corpus, divided by total reference length. It can exceed 1.0 when
  predictions are much longer than references.

- char_accuracy = max(0.0, 1.0 - CER). It is NOT the positional
  "pred[i] == ref[i]" rate: that quantity collapses under a single leading
  insertion and is only meaningful as a per-position diagnostic.

- Micro P/R/F1 are computed from global TP/FP/FN. Macro F1 is the
  unweighted mean of per-class F1 over classes that appear in the
  references (support > 0); hallucinated classes (FP-only) affect micro
  scores and appear in the per-class table, but do not dilute the macro
  mean of classes the references actually contain.

Complexity: O(n * m) per pair for both SequenceMatcher and Levenshtein,
with n, m the string lengths (VINs: 17). Pure stdlib; no optional imports.
"""

from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Dict, Iterable, List, Tuple

from .vin_utils import levenshtein_distance


@dataclass
class AlignmentCounts:
    """Global and per-class TP/FP/FN derived from sequence alignment."""

    tp: int = 0
    fp: int = 0
    fn: int = 0
    #: char -> {'tp': int, 'fp': int, 'fn': int}
    per_class: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def _bucket(self, char: str) -> Dict[str, int]:
        bucket = self.per_class.get(char)
        if bucket is None:
            bucket = {'tp': 0, 'fp': 0, 'fn': 0}
            self.per_class[char] = bucket
        return bucket

    def add_tp(self, char: str) -> None:
        self.tp += 1
        self._bucket(char)['tp'] += 1

    def add_fp(self, char: str) -> None:
        self.fp += 1
        self._bucket(char)['fp'] += 1

    def add_fn(self, char: str) -> None:
        self.fn += 1
        self._bucket(char)['fn'] += 1

    def update(self, other: "AlignmentCounts") -> None:
        """Merge another count set into this one."""
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn
        for char, bucket in other.per_class.items():
            mine = self._bucket(char)
            mine['tp'] += bucket['tp']
            mine['fp'] += bucket['fp']
            mine['fn'] += bucket['fn']


def alignment_counts(prediction: str, reference: str) -> AlignmentCounts:
    """
    Count TP/FP/FN between one prediction and one reference by alignment.

    Args:
        prediction: OCR output, exactly as produced (no padding).
        reference: Ground-truth string.

    Returns:
        AlignmentCounts with global and per-class tallies. Every character
        of the prediction is counted exactly once (TP or FP), and every
        character of the reference is counted exactly once (TP or FN).
    """
    counts = AlignmentCounts()
    matcher = SequenceMatcher(None, prediction, reference, autojunk=False)
    for tag, p0, p1, r0, r1 in matcher.get_opcodes():
        if tag == 'equal':
            for ch in reference[r0:r1]:
                counts.add_tp(ch)
        elif tag == 'replace':
            for ch in prediction[p0:p1]:
                counts.add_fp(ch)
            for ch in reference[r0:r1]:
                counts.add_fn(ch)
        elif tag == 'insert':      # present in reference, absent in prediction
            for ch in reference[r0:r1]:
                counts.add_fn(ch)
        elif tag == 'delete':      # present in prediction, absent in reference
            for ch in prediction[p0:p1]:
                counts.add_fp(ch)

    # Conservation invariants: alignment must account for every character.
    # Explicit raises, not `assert`: python -O strips assert statements, and
    # this module's central guarantee must hold in optimized deployments too.
    if counts.tp + counts.fp != len(prediction):
        raise AssertionError(
            f"prediction chars lost by alignment: {prediction!r} vs {reference!r}"
        )
    if counts.tp + counts.fn != len(reference):
        raise AssertionError(
            f"reference chars lost by alignment: {prediction!r} vs {reference!r}"
        )
    return counts


def _prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    """
    Precision/recall/F1 from raw counts - the ONE place the harmonic-mean
    formula is written in this module. (It was previously inlined here twice,
    which is exactly the duplication pattern this module exists to end.)
    """
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )
    return precision, recall, f1


def micro_prf(counts: AlignmentCounts) -> Tuple[float, float, float]:
    """Micro precision, recall and F1 from global TP/FP/FN."""
    return _prf(counts.tp, counts.fp, counts.fn)


def per_class_prf(counts: AlignmentCounts) -> Dict[str, Dict[str, float]]:
    """
    Per-class precision/recall/F1/support for every class seen.

    Support is the number of occurrences in the references (tp + fn).
    Classes that appear only as false positives are included with
    support 0 so hallucinated characters remain visible.
    """
    table: Dict[str, Dict[str, float]] = {}
    for char in sorted(counts.per_class):
        bucket = counts.per_class[char]
        precision, recall, f1 = _prf(bucket['tp'], bucket['fp'], bucket['fn'])
        table[char] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': bucket['tp'] + bucket['fn'],
        }
    return table


def macro_f1(counts: AlignmentCounts) -> float:
    """Unweighted mean F1 over classes present in the references."""
    scores = [
        row['f1'] for row in per_class_prf(counts).values() if row['support'] > 0
    ]
    return sum(scores) / len(scores) if scores else 0.0


def corpus_cer(pairs: Iterable[Tuple[str, str]]) -> float:
    """
    Character Error Rate over a corpus: sum(editdist) / sum(len(reference)).

    Returns 0.0 for an empty corpus (nothing to be wrong about). May exceed
    1.0 when predictions are much longer than references.
    """
    total_edits = 0
    total_ref = 0
    for prediction, reference in pairs:
        total_edits += levenshtein_distance(prediction, reference)
        total_ref += len(reference)
    return total_edits / total_ref if total_ref > 0 else 0.0


@dataclass(frozen=True)
class CharLevelMetrics:
    """Corpus-level character metrics under the canonical definitions."""

    total_reference_chars: int
    total_predicted_chars: int
    true_positives: int
    false_positives: int
    false_negatives: int
    cer: float
    char_accuracy: float
    precision: float
    recall: float
    f1_micro: float
    f1_macro: float
    per_class: Dict[str, Dict[str, float]]


def char_level_metrics(pairs: List[Tuple[str, str]]) -> CharLevelMetrics:
    """
    Compute the full canonical character-level metric set.

    Args:
        pairs: (prediction, reference) tuples, predictions unpadded.

    Returns:
        CharLevelMetrics. For an empty corpus every rate is 0.0.
    """
    totals = AlignmentCounts()
    for prediction, reference in pairs:
        totals.update(alignment_counts(prediction, reference))

    cer = corpus_cer(pairs)
    precision, recall, f1 = micro_prf(totals)

    return CharLevelMetrics(
        total_reference_chars=totals.tp + totals.fn,
        total_predicted_chars=totals.tp + totals.fp,
        true_positives=totals.tp,
        false_positives=totals.fp,
        false_negatives=totals.fn,
        cer=cer,
        char_accuracy=max(0.0, 1.0 - cer),
        precision=precision,
        recall=recall,
        f1_micro=f1,
        f1_macro=macro_f1(totals),
        per_class=per_class_prf(totals),
    )
