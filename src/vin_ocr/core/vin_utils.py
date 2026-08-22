"""
VIN Utilities - Single Source of Truth
======================================

Shared utilities for VIN processing across all modules.
This module consolidates duplicate code and provides consistent behavior.

Author: JRL-VIN Project
Date: January 2026
"""

import re
import logging
from difflib import SequenceMatcher
from typing import Optional, Dict, List, Tuple, FrozenSet
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


# =============================================================================
# VIN CONSTANTS
# =============================================================================

class VINConstants:
    """Immutable VIN specification constants per ISO 3779 / NHTSA."""
    
    LENGTH: int = 17
    
    # Valid characters (I, O, Q excluded to avoid confusion with 1, 0)
    VALID_CHARS: FrozenSet[str] = frozenset("0123456789ABCDEFGHJKLMNPRSTUVWXYZ")
    INVALID_CHARS: FrozenSet[str] = frozenset("IOQ")
    
    # Position indices (1-based as per VIN spec)
    CHECK_DIGIT_POSITION: int = 9
    YEAR_POSITION: int = 10
    PLANT_POSITION: int = 11
    SEQUENTIAL_START: int = 12
    SEQUENTIAL_END: int = 17
    
    # Checksum weights by position (NHTSA standard)
    CHECKSUM_WEIGHTS: Tuple[int, ...] = (8, 7, 6, 5, 4, 3, 2, 10, 0, 9, 8, 7, 6, 5, 4, 3, 2)
    
    # Character to value mapping for checksum (ISO 3779)
    CHAR_VALUES: Dict[str, int] = {
        'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8,
        'J': 1, 'K': 2, 'L': 3, 'M': 4, 'N': 5, 'P': 7, 'R': 9,
        'S': 2, 'T': 3, 'U': 4, 'V': 5, 'W': 6, 'X': 7, 'Y': 8, 'Z': 9,
        '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9
    }
    
    # Common World Manufacturer Identifiers (first 3 chars)
    COMMON_WMIS: Tuple[str, ...] = (
        'SAL', 'WVW', 'WBA', 'WDB', 'WDD', 'WF0', 'WMW', 'WP0', 'WUA', 'WVG',
        '1G1', '1GC', '1GT', '1G6', '1FA', '1FM', '1FT', '1HG', '1J4', '1N4',
        '2G1', '2HG', '2HM', '2T1', '3FA', '3G1', '3VW', '4T1', '5FN', '5NP',
        'JN1', 'JT2', 'JTD', 'JTE', 'JTH', 'KM8', 'KNA', 'KND', 'VF1', 'VF3',
        'YV1', 'ZFF', 'ZFA',
    )


VIN_LENGTH = VINConstants.LENGTH
VIN_VALID_CHARS = VINConstants.VALID_CHARS
VIN_INVALID_CHARS = VINConstants.INVALID_CHARS

# Artifact characters that may be stripped from OCR output.
#
# An "artifact" is any character that cannot appear in a VIN at all - plate
# borders, stamps, scratches and reflections produce these. Deliberately does
# NOT include letters like X, Y, T, F or A: those are valid VIN characters and
# stripping them corrupts legitimate VINs. (RuleBasedCorrector previously used
# the patterns `^[*#XYT]+` and `^[IYTFA][*#]*`; the first removed the leading
# character of every Volvo "YV1..." VIN, and the second matched with zero
# trailing artifacts, so any VIN beginning with I/Y/T/F/A silently lost its
# first character.)
#
# This is the single definition. VINPostProcessor (pipeline) and
# RuleBasedCorrector (this module) both strip with NON_VIN_RUN below; a
# second copy of either is how the Y-eating bug survived its first fix.
ARTIFACT_CHARS: FrozenSet[str] = frozenset('*#@$%^&()[]{}<>/\\|!?,;:"\'`~+=_ .-')

# Matches any run of characters that cannot occur in a VIN. I/O/Q are
# deliberately allowed through so the invalid-character stage can map them to
# 1/0/0 instead of deleting the evidence.
NON_VIN_RUN: re.Pattern = re.compile(r'[^0-9A-Z]+')


# =============================================================================
# FILENAME VIN EXTRACTION
# =============================================================================

# Pre-compiled regex patterns for performance
_FILENAME_PATTERNS = [
    # Primary: "1-VIN -SAL1A2A40SA606662.jpg" or "1-VIN - SAL1A2A40SA606662.jpg"
    re.compile(r'^\d+-VIN\s*-\s*([A-Z0-9]{17})(?:\s|\.|_|$)', re.IGNORECASE),
    # Flexible: "VIN -VINCODE" or "VIN-VINCODE" or "VIN - VINCODE" anywhere
    re.compile(r'VIN\s*-\s*([A-Z0-9]{17})(?:\s|\.|_|$)', re.IGNORECASE),
    # Legacy: "42 -SAL1A2A40SA606662 2.jpg"
    re.compile(r'^\d+\s*-\s*([A-Z0-9]{17})(?:\s|\.)', re.IGNORECASE),
    # Underscore format: "VIN_-_SAL1A2A40SA606662_" or "VIN_-_SAL1A2A40SA606662.jpg"
    re.compile(r'VIN_-_([A-Z0-9]{17})(?:_|\.|$)', re.IGNORECASE),
    # Full underscore format: "7-VIN_-_SAL109F97TA467227.jpg" (number prefix with underscores)
    re.compile(r'^\d+-VIN_-_([A-Z0-9]{17})(?:_|\.|$)', re.IGNORECASE),
    # Mixed format: "VIN_VINCODE" or "VIN _ VINCODE"
    re.compile(r'VIN[_\s]+([A-Z0-9]{17})(?:\s|\.|_|$)', re.IGNORECASE),
]

# Fallback pattern for any 17-char alphanumeric (excluding I, O, Q)
_FALLBACK_PATTERN = re.compile(r'\b([A-HJ-NPR-Z0-9]{17})\b', re.IGNORECASE)


def extract_vin_from_filename(filename: str) -> Optional[str]:
    """
    Extract VIN from filename pattern.
    
    Supported formats (in priority order):
    1. NUMBER-VIN -VINCODE.ext  (e.g., "1-VIN -SAL1A2A40SA606662.jpg")
    2. NUMBER-VIN - VINCODE.ext (e.g., "7-VIN - SAL109F97TA467227.jpg")
    3. VIN -VINCODE.ext or VIN-VINCODE.ext or VIN - VINCODE.ext
    4. NUMBER -VINCODE rest.ext (legacy)
    5. NUMBER-VIN_-_VINCODE.ext (e.g., "7-VIN_-_SAL109F97TA467227.jpg")
    6. VIN_-_VINCODE_ or VIN_-_VINCODE.ext (underscore format)
    7. VIN_VINCODE or VIN _ VINCODE (mixed underscore/space)
    8. Any 17-char valid VIN sequence (fallback)
    
    Args:
        filename: Image filename (not full path)
        
    Returns:
        Extracted VIN (17 uppercase characters) or None if not found
        
    Examples:
        >>> extract_vin_from_filename("1-VIN -SAL1A2A40SA606662.jpg")
        'SAL1A2A40SA606662'
        >>> extract_vin_from_filename("7-VIN_-_SAL109F97TA467227.jpg")
        'SAL109F97TA467227'
        >>> extract_vin_from_filename("random_file.jpg")
        None
    """
    if not filename:
        return None
    
    # Try each pattern in priority order
    for pattern in _FILENAME_PATTERNS:
        match = pattern.search(filename)
        if match:
            vin = match.group(1).upper()
            if _is_valid_vin_chars(vin):
                return vin
    
    # Fallback: find any 17-char sequence that could be a VIN. The pattern's
    # character class already excludes I/O/Q; the guard below re-checks with
    # the canonical validator so a future regex edit cannot silently start
    # emitting impossible ground truths.
    match = _FALLBACK_PATTERN.search(filename)
    if match:
        vin = match.group(1).upper()
        if _is_valid_vin_chars(vin):
            return vin
    
    return None


def _is_valid_vin_chars(vin: str) -> bool:
    """
    Check that every character is a valid VIN character.

    I, O and Q are NOT valid. This previously tested membership in
    VALID_CHARS | INVALID_CHARS - the union of both sets is every character
    either set mentions - so I/O/Q passed and the filename patterns could
    return ground-truth "VINs" that cannot exist.
    """
    return all(c in VIN_VALID_CHARS for c in vin.upper())


# =============================================================================
# STRING METRICS (Edit Distance)
# =============================================================================

def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Calculate Levenshtein (edit) distance between two strings.
    
    The Levenshtein distance is the minimum number of single-character edits
    (insertions, deletions, or substitutions) required to transform s1 into s2.
    
    Time Complexity: O(len(s1) * len(s2))
    Space Complexity: O(min(len(s1), len(s2)))
    
    Args:
        s1: First string
        s2: Second string
        
    Returns:
        Integer edit distance between the strings
        
    Examples:
        >>> levenshtein_distance("kitten", "sitting")
        3
        >>> levenshtein_distance("ABC", "ABC")
        0
        >>> levenshtein_distance("", "test")
        4
    """
    # Ensure s1 is the longer string for space optimization
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    # Only keep two rows (current and previous) for space efficiency
    previous_row = range(len(s2) + 1)
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
# VIN VALIDATION
# =============================================================================

@dataclass
class VINValidationResult:
    """Result of VIN validation."""
    vin: str
    is_valid_length: bool
    has_valid_chars: bool
    invalid_chars: List[str]
    checksum_valid: bool
    expected_check_digit: Optional[str]
    is_fully_valid: bool
    
    def to_dict(self) -> Dict:
        return {
            'vin': self.vin,
            'is_valid_length': self.is_valid_length,
            'has_valid_chars': self.has_valid_chars,
            'invalid_chars': self.invalid_chars,
            'checksum_valid': self.checksum_valid,
            'expected_check_digit': self.expected_check_digit,
            'is_fully_valid': self.is_fully_valid,
        }


def validate_vin(vin: str) -> VINValidationResult:
    """
    Comprehensive VIN validation.
    
    Checks:
    1. Length (must be 17)
    2. Character validity (no I, O, Q)
    3. Checksum at position 9
    
    Args:
        vin: VIN string to validate
        
    Returns:
        VINValidationResult with all validation details
    """
    vin = vin.upper().strip()
    
    is_valid_length = len(vin) == VIN_LENGTH
    
    # Every character that is not a valid VIN character is reported: I/O/Q,
    # but also artifacts like '*' or '-'. Previously only I/O/Q were listed,
    # so a '*' could set has_valid_chars=False while invalid_chars said [].
    invalid_chars = [c for c in vin if c not in VIN_VALID_CHARS]
    has_valid_chars = len(invalid_chars) == 0
    
    checksum_valid = False
    expected_check_digit = None
    
    if is_valid_length and has_valid_chars:
        expected_check_digit = calculate_check_digit(vin)
        if expected_check_digit:
            checksum_valid = vin[8] == expected_check_digit
    
    is_fully_valid = is_valid_length and has_valid_chars and checksum_valid
    
    return VINValidationResult(
        vin=vin,
        is_valid_length=is_valid_length,
        has_valid_chars=has_valid_chars,
        invalid_chars=invalid_chars,
        checksum_valid=checksum_valid,
        expected_check_digit=expected_check_digit,
        is_fully_valid=is_fully_valid,
    )


def calculate_check_digit(vin: str) -> Optional[str]:
    """
    Calculate the expected check digit for a VIN.
    
    The check digit (position 9) is calculated by:
    1. Assigning numeric values to each character
    2. Multiplying by position weights
    3. Summing and taking mod 11
    4. Result 10 becomes 'X'
    
    Args:
        vin: 17-character VIN (check digit position will be ignored)
        
    Returns:
        Expected check digit ('0'-'9' or 'X'), or None if calculation fails
    """
    if len(vin) != VIN_LENGTH:
        return None
    
    vin = vin.upper()
    
    try:
        total = 0
        for i, char in enumerate(vin):
            if i == 8:  # Skip check digit position
                continue
            value = VINConstants.CHAR_VALUES.get(char)
            if value is None:
                return None
            weight = VINConstants.CHECKSUM_WEIGHTS[i]
            total += value * weight
        
        remainder = total % 11
        return 'X' if remainder == 10 else str(remainder)
        
    except (IndexError, TypeError):
        return None


def validate_checksum(vin: str) -> bool:
    """
    Validate VIN checksum at position 9.
    
    This is the Single Source of Truth for checksum validation.
    All other modules should call this function rather than
    implementing their own checksum logic.
    
    Args:
        vin: 17-character VIN to validate
        
    Returns:
        True if checksum is valid, False otherwise
    """
    if len(vin) != VIN_LENGTH:
        return False
    
    expected = calculate_check_digit(vin)
    if expected is None:
        return False
    
    return vin[8].upper() == expected


def extract_vin_from_text(text: str) -> str:
    """
    Extract 17-character VIN from longer text (Single Source of Truth).
    
    When OCR picks up extra characters before/after the VIN,
    this function attempts to find the valid VIN substring.
    
    Strategy:
    Score EVERY 17-character window and return the highest scoring one.
    The scorer already rewards a known WMI prefix and, decisively, a valid
    ISO 3779 check digit (see _score_vin_candidate).

    This replaces an earlier two-stage approach whose first stage returned the
    first window starting with any known WMI, guarded by `score > 10`. Two
    defects made that unsafe:

      * The guard was a no-op - 17 arbitrary letters already score 34, so it
        admitted essentially anything.
      * Returning the FIRST match meant a spurious WMI earlier in the string
        won outright. For input "SAL99SAL1A2A40SA606662" it returned
        "SAL99SAL1A2A40SA6" (checksum invalid) even though the genuine,
        checksum-valid "SAL1A2A40SA606662" was present later in the same text.
        Which WMI won also depended on the order of the COMMON_WMIS tuple
        rather than on the text.

    Args:
        text: Raw text that may contain a VIN plus extra characters
        
    Returns:
        Best 17-character VIN candidate, or original text if < 17 chars
    """
    text = text.upper().strip()
    
    if len(text) == VIN_LENGTH:
        return text
    
    if len(text) < VIN_LENGTH:
        return text  # Too short, can't extract

    # Score every window; first-best wins ties, preserving left-to-right
    # preference for otherwise equally plausible candidates.
    best_candidate = text[:VIN_LENGTH]
    best_score = _score_vin_candidate(best_candidate)

    for i in range(1, len(text) - VIN_LENGTH + 1):
        candidate = text[i:i + VIN_LENGTH]
        score = _score_vin_candidate(candidate)
        if score > best_score:
            best_score = score
            best_candidate = candidate

    return best_candidate


def _score_vin_candidate(candidate: str) -> int:
    """
    Score a VIN candidate (higher = more likely valid).
    
    Scoring:
    - +100 for a valid ISO 3779 check digit  (decisive)
    - +2 for each valid VIN character
    - +3 for each digit in sequential positions (12-17)
    - +10 for known WMI prefix
    - -5 for each invalid character (I, O, Q)

    The check-digit bonus dominates deliberately. A random 17-character string
    satisfies the check digit roughly 1 time in 11, so it is by far the
    strongest evidence available that a window is a real VIN - much stronger
    than the WMI heuristic, which only covers the 43 manufacturers hardcoded in
    VINConstants.COMMON_WMIS and is biased toward this dataset.

    It was previously unused: a checksum-valid and a checksum-invalid candidate
    scored identically (62 vs 62), so extraction could return a window that
    failed its own check digit while a valid one existed in the same text.

    The bonus (100) exceeds the maximum achievable from all other terms
    (34 valid chars + 18 digits + 10 WMI = 62), so a checksum-valid candidate
    always outranks a checksum-invalid one, while the other terms still break
    ties among candidates of equal checksum status.

    Args:
        candidate: 17-character string to score
        
    Returns:
        Integer score (higher = better candidate)
    """
    score = 0
    
    # Valid VIN characters
    score += sum(2 for c in candidate if c in VIN_VALID_CHARS)
    
    # Digits in sequential positions (12-17, indices 11-16)
    if len(candidate) >= VIN_LENGTH:
        score += sum(3 for c in candidate[11:17] if c.isdigit())
    
    # Starts with known WMI
    if candidate[:3] in VINConstants.COMMON_WMIS:
        score += 10
    
    # Penalty for invalid chars
    score -= sum(5 for c in candidate if c in VIN_INVALID_CHARS)

    # Valid check digit - decisive evidence this really is a VIN
    if len(candidate) == VIN_LENGTH and validate_checksum(candidate):
        score += 100

    return score


# =============================================================================
# RULE-BASED CHARACTER CORRECTION
# =============================================================================

class RuleBasedCorrector:
    """
    Rule-based character correction for VIN OCR errors.
    
    This corrector applies deterministic rules learned from common OCR
    confusion patterns on engraved metal VIN plates.
    
    Rules are organized by:
    1. Global substitutions (always apply)
    2. Position-specific rules (context-aware)
    3. Learned rules (from training data)
    
    Thread Safety: This class is thread-safe for concurrent use.
    """
    
    # Invalid VIN characters -> valid replacements
    INVALID_CHAR_RULES: Dict[str, str] = {
        'I': '1',  # I looks like 1
        'O': '0',  # O looks like 0
        'Q': '0',  # Q looks like 0 (round shape)
    }
    
    # Global (position-independent) confusion mappings.
    #
    # Empty by design, not by omission: correct() uppercases first (making
    # lowercase keys unreachable) and then strips every non-[0-9A-Z] run with
    # the canonical NON_VIN_RUN (removing punctuation before any mapping could
    # fire). The punctuation entries this dict used to carry ('|'->'1',
    # '('->'C', ...) were therefore dead code, and I/O/Q are handled by
    # INVALID_CHAR_RULES. The mechanism stays so add_learned_rules() can
    # inject mappings measured from data.
    GLOBAL_CONFUSION_RULES: Dict[str, str] = {}
    
    # Letter->digit confusions for the sequential-number positions.
    # Uppercase keys only: correct() uppercases at step 1, so lowercase keys
    # were unreachable dead entries. Application is checksum-gated in
    # _apply_position_rules - these are hypotheses, not unconditional edits.
    SEQUENTIAL_POSITION_RULES: Dict[str, str] = {
        'S': '5',
        'G': '6',
        'B': '8',
        'A': '4',
        'L': '1',
        'Z': '2',
        'E': '3',
        'T': '7',
        'D': '0',
        'O': '0',
        'I': '1',
        'C': '0',  # C can look like 0
    }
    
    # learn_from_errors acceptance thresholds: a mined rule a->b needs at
    # least MIN_RULE_OBSERVATIONS aligned observations AND purity
    # errors / (errors + correct reads of 'a') >= RULE_PURITY_THRESHOLD.
    MIN_RULE_OBSERVATIONS: int = 2
    RULE_PURITY_THRESHOLD: float = 0.8
    
    def __init__(self, learned_rules: Optional[Dict[str, str]] = None):
        """
        Initialize corrector with optional learned rules.
        
        Args:
            learned_rules: Additional char->char mappings learned from data.
                Copied on ingest: an empty dict and a non-empty dict get the
                same treatment (the old ``learned_rules or {}`` aliased
                non-empty dicts, so caller-side mutation silently changed
                this corrector - including the module-global singleton).
        """
        self.learned_rules: Dict[str, str] = (
            dict(learned_rules) if learned_rules is not None else {}
        )
    
    def add_learned_rules(self, rules: Dict[str, str]):
        """Add rules learned from training data (pairs copied in, not aliased)."""
        self.learned_rules.update(rules)
    
    def correct(self, raw_text: str, confidence: float = 0.0) -> Dict:
        """
        Apply rule-based corrections to raw OCR output.
        
        Processing steps:
        1. Normalize (uppercase, strip whitespace)
        2. Remove artifacts
        3. Map invalid characters (I/O/Q -> 1/0/0; never legal in any VIN)
        4. Extract best 17-char VIN candidate (canonical scoring path)
        5. Apply learned character substitutions, checksum-gated
        6. Apply position-specific corrections, checksum-gated
        7. Validate result
        
        Steps 5 and 6 never touch a VIN whose ISO 3779 check digit is
        already valid, and their rewrites are rolled back when they fail to
        produce a checksum-valid VIN (see _apply_learned_rules and
        _apply_position_rules). Ungated application corrupted correct reads:
        the audited corrector turned the checksum-valid 5FNRL6H09LBB00001
        into the invalid 5FNRL6H09LB800001.
        
        Args:
            raw_text: Raw OCR output
            confidence: OCR confidence score (passed through)
            
        Returns:
            Dict with corrected VIN and metadata
        """
        corrections = []
        
        # Step 1: Normalize
        text = raw_text.upper().strip()
        text = ''.join(text.split())  # Remove all whitespace
        original = text
        
        # Step 2: Remove artifacts
        text = self._remove_artifacts(text)
        if text != original:
            corrections.append(f"Removed artifacts: '{original}' -> '{text}'")
        
        # Step 3: Map I/O/Q to 1/0/0 (always safe: they occur in no VIN)
        text_before = text
        text = self._apply_invalid_char_rules(text)
        if text != text_before:
            corrections.append(f"Invalid-char corrections: '{text_before}' -> '{text}'")
        
        # Step 4: Extract 17-char VIN candidate
        text_before = text
        text = self._extract_vin_candidate(text)
        if text != text_before:
            corrections.append(f"Extracted VIN: '{text_before}' -> '{text}'")
        
        # Step 5: Learned substitutions (checksum-gated)
        text_before = text
        text = self._apply_learned_rules(text)
        if text != text_before:
            corrections.append(f"Global corrections: '{text_before}' -> '{text}'")
        
        # Step 6: Position-specific corrections (checksum-gated)
        text_before = text
        text = self._apply_position_rules(text)
        if text != text_before:
            corrections.append(f"Position corrections: '{text_before}' -> '{text}'")
        
        # Step 7: Validate
        validation = validate_vin(text)
        
        return {
            'vin': text,
            'raw_ocr': raw_text,
            'confidence': confidence,
            'is_valid_length': validation.is_valid_length,
            'checksum_valid': validation.checksum_valid,
            'corrections': corrections,
            'correction_count': len(corrections),
        }
    
    def _remove_artifacts(self, text: str) -> str:
        """
        Strip characters that cannot occur in a VIN.

        Uses the module-level NON_VIN_RUN - the same rule VINPostProcessor
        applies - so only non-alphanumeric noise (plate borders, stamps,
        scratches: ``* # / | \\ - . space`` etc.) is removed. Letters are
        never removed: this method previously applied ``^[*#XYT]+``, which
        deleted the leading character of every VIN starting with X, Y or T
        (e.g. Volvo "YV1...") and returned a 16-character result.

        I/O/Q are intentionally preserved here - they are plausible OCR
        output that _apply_invalid_char_rules maps to 1/0/0.
        """
        return NON_VIN_RUN.sub('', text)
    
    def _apply_invalid_char_rules(self, text: str) -> str:
        """
        Map I/O/Q to 1/0/0 (INVALID_CHAR_RULES).

        Unconditional by design: these characters occur in no VIN, so the
        mapping can never damage a correct read. Running it before the
        learned rules also preserves the old priority ordering - after this
        step no I/O/Q remain for a conflicting learned rule to rewrite.
        """
        return ''.join(self.INVALID_CHAR_RULES.get(c, c) for c in text)
    
    def _apply_learned_rules(self, text: str) -> str:
        """
        Apply learned/global confusion substitutions, checksum-gated.

        Learned rules are position-independent hypotheses mined from noisy
        data; applied blindly they corrupt correct reads (the audited H5c
        failure). Gating:

        * text that is not 17 chars long is returned unchanged (there is no
          check digit to verify a rewrite against);
        * a VIN whose check digit is already valid is returned unchanged -
          rules must never degrade a correct read;
        * otherwise the rewrite is kept only if it produces a checksum-valid
          VIN; anything less is rolled back.
        """
        rules = {**self.GLOBAL_CONFUSION_RULES, **self.learned_rules}
        if not rules or len(text) != VIN_LENGTH:
            return text
        if validate_checksum(text):
            return text  # never degrade an already-valid VIN
        candidate = ''.join(rules.get(c, c) for c in text)
        if candidate != text and validate_checksum(candidate):
            return candidate
        return text  # rollback on non-improvement
    
    def _extract_vin_candidate(self, text: str) -> str:
        """
        Extract the best 17-character VIN candidate from text.

        Delegates to the module's canonical extract_vin_from_text so the
        corrector and every other consumer share ONE extraction path. This
        method previously carried a stale copy of the pre-fix algorithm the
        canonical function's docstring documents as removed: first WMI match
        won outright behind a `> 10` score guard that any 17 valid
        characters already cleared (score >= 34), scored by a copy that
        lacked the decisive +100 checksum bonus - so a checksum-invalid
        window beat a checksum-valid VIN present in the same text.
        """
        return extract_vin_from_text(text)
    
    def _apply_position_rules(self, text: str) -> str:
        """
        Apply position-specific correction rules, checksum-gated.

        Positions 12-17 (indices 11-16) hold the sequential production
        number. High-volume manufacturers use digits there, but 49 CFR 565
        allows alphanumerics at positions 12-14 for small-volume
        manufacturers - a letter there is NOT proof of an OCR error.

        Gating:
        * A VIN whose ISO 3779 check digit is already valid is returned
          unchanged (this method previously rewrote 'B' at position 12 of
          the valid 5FNRL6H09LBB00001, emitting a checksum-invalid string).
        * Rules applied to an invalid VIN are kept when they make it
          checksum-valid. Otherwise each rewrite is rolled back if the
          original character was plausible where it stood: any valid VIN
          character at positions 12-14. Rewrites of characters that could
          not be correct (I/O/Q anywhere; letters at the digits-only
          positions 15-17) are kept - the original was certainly wrong and
          the rewrite at least restores the required format.
        """
        if len(text) != VIN_LENGTH:
            return text
        
        if validate_checksum(text):
            return text  # never degrade an already-valid VIN
        
        result = list(text)
        changed_indices: List[int] = []
        for idx in range(11, 17):
            char = result[idx]
            if char in self.SEQUENTIAL_POSITION_RULES:
                result[idx] = self.SEQUENTIAL_POSITION_RULES[char]
                changed_indices.append(idx)
        
        if not changed_indices:
            return text
        
        corrected = ''.join(result)
        if validate_checksum(corrected):
            return corrected  # the rules repaired the VIN
        
        # Rollback on non-improvement wherever the original was plausible.
        for idx in changed_indices:
            original_char = text[idx]
            if original_char in VIN_INVALID_CHARS:
                continue  # I/O/Q occur in no VIN: keep the rewrite
            if idx <= 13:  # indices 11-13 = positions 12-14: letters legal
                result[idx] = original_char
            # indices 14-16 (positions 15-17) must be digits: keep the rewrite
        return ''.join(result)
    
    def learn_from_errors(self, predictions: List[Dict]) -> Dict[str, str]:
        """
        Learn correction rules from prediction errors.
        
        Rules are mined by aligning each prediction to its ground truth with
        difflib.SequenceMatcher opcodes, NOT by positional zip: a single
        insertion or deletion misaligns every later position, and the
        zip-mined "rules" corrupted perfect reads (audit H5c - two samples
        with one leading artifact each turned a correct read into garbage).
        
        Mining discipline:
        * Only 'replace' opcodes of EQUAL length yield per-position character
          pairs; insertions, deletions and unequal-length replacements carry
          no per-character evidence.
        * Pairs harvested at the check digit (index 8 on either side) are
          discarded: the check digit is a function of the other 16
          characters, so confusions observed there do not generalise, and a
          context-free rule learned from them rewrites that character
          everywhere.
        * 'equal' opcodes tally how often each character was read correctly
          at aligned positions, feeding the purity test below.
        
        A rule ``a -> b`` is accepted only when:
        * it was observed at least MIN_RULE_OBSERVATIONS (2) times, AND
        * errors / (errors + correct reads of 'a') >= RULE_PURITY_THRESHOLD
          (0.8) - a frequent-but-usually-correct character (e.g. the 'S' of
          every 'SAL...' VIN) must not be globally rewritten because of a
          handful of misreads, AND
        * 'b' is a valid VIN character.
        
        Application of the learned rules stays checksum-gated in correct()
        (see _apply_learned_rules), so even a rule that passes these gates
        can never degrade a checksum-valid read.
        
        Args:
            predictions: List of dicts with 'ground_truth' and 'prediction' keys
            
        Returns:
            Dict of learned char->char mappings (also added to this instance)
        """
        error_counts: Dict[str, Dict[str, int]] = {}
        correct_counts: Dict[str, int] = {}
        
        for pred in predictions:
            gt = str(pred.get('ground_truth', '') or '').upper().strip()
            pr = str(pred.get('prediction', '') or '').upper().strip()
            
            if not gt or not pr:
                continue
            
            matcher = SequenceMatcher(None, pr, gt, autojunk=False)
            for tag, p0, p1, g0, g1 in matcher.get_opcodes():
                if tag == 'equal':
                    for offset in range(p1 - p0):
                        ch = pr[p0 + offset]
                        correct_counts[ch] = correct_counts.get(ch, 0) + 1
                elif tag == 'replace' and (p1 - p0) == (g1 - g0):
                    for offset in range(p1 - p0):
                        p_idx = p0 + offset
                        g_idx = g0 + offset
                        if p_idx == 8 or g_idx == 8:
                            continue  # never learn check-digit rewrites
                        p_char = pr[p_idx]
                        g_char = gt[g_idx]
                        if p_char == g_char:
                            continue
                        bucket = error_counts.setdefault(p_char, {})
                        bucket[g_char] = bucket.get(g_char, 0) + 1
        
        # Build rules from corrections that clear the count and purity gates
        new_rules: Dict[str, str] = {}
        for predicted_char, confusions in error_counts.items():
            best_correction = max(confusions, key=confusions.get)
            errors = confusions[best_correction]
            
            if errors < self.MIN_RULE_OBSERVATIONS:
                continue
            if best_correction not in VIN_VALID_CHARS:
                continue
            
            correct_reads = correct_counts.get(predicted_char, 0)
            purity = errors / (errors + correct_reads)
            if purity < self.RULE_PURITY_THRESHOLD:
                continue
            
            new_rules[predicted_char] = best_correction
            logger.info(
                f"Learned rule: '{predicted_char}' -> '{best_correction}' "
                f"(seen {errors} times, purity {purity:.2f})"
            )
        
        # Add to instance rules
        self.add_learned_rules(new_rules)
        
        return new_rules
    
    def get_all_rules(self) -> Dict[str, Dict[str, str]]:
        """Get all active correction rules by category."""
        return {
            'invalid_char_rules': self.INVALID_CHAR_RULES.copy(),
            'global_confusion_rules': self.GLOBAL_CONFUSION_RULES.copy(),
            'sequential_position_rules': self.SEQUENTIAL_POSITION_RULES.copy(),
            'learned_rules': self.learned_rules.copy(),
        }
    
    def export_rules(self) -> Dict:
        """
        Export rules for serialization.

        Returns a deep copy: keys and values are strings (immutable), so a
        fresh dict fully detaches the export. Mutating the returned mapping
        can no longer alter this corrector - the live dict was previously
        returned, and editing an export silently rewired the module-global
        singleton's rules.
        """
        return {
            'learned_rules': dict(self.learned_rules),
            'version': '1.0',
        }
    
    @classmethod
    def from_exported(cls, data: Dict) -> 'RuleBasedCorrector':
        """Create corrector from exported rules (data is copied, not aliased)."""
        return cls(learned_rules=data.get('learned_rules', {}))


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

# Module-level corrector instance for simple usage
_default_corrector = RuleBasedCorrector()


def correct_vin(raw_text: str, confidence: float = 0.0) -> Dict:
    """
    Apply rule-based corrections to raw OCR output.
    
    Convenience function using the default corrector.
    
    Args:
        raw_text: Raw OCR output
        confidence: OCR confidence score
        
    Returns:
        Dict with corrected VIN and metadata
    """
    return _default_corrector.correct(raw_text, confidence)


def get_corrector() -> RuleBasedCorrector:
    """Get the default corrector instance."""
    return _default_corrector


def validate_vin_format(vin: str) -> bool:
    """
    Quick check if VIN has valid format (length and characters).
    
    Does NOT check checksum. Use validate_vin() for full validation.
    
    Args:
        vin: VIN string to check
        
    Returns:
        True if format is valid (17 chars, no I/O/Q)
    """
    vin = vin.upper().strip()
    if len(vin) != VIN_LENGTH:
        return False
    return all(c in VIN_VALID_CHARS for c in vin)


def validate_vin_checksum(vin: str) -> bool:
    """
    Check if VIN checksum is valid.
    
    Args:
        vin: 17-character VIN string
        
    Returns:
        True if checksum at position 9 is correct
    """
    if len(vin) != VIN_LENGTH:
        return False
    
    expected = calculate_check_digit(vin)
    if expected is None:
        return False
    
    return vin[8].upper() == expected

