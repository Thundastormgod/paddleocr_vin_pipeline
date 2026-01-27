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


# =============================================================================
# FILENAME VIN EXTRACTION
# =============================================================================

# Pre-compiled regex patterns for performance
_FILENAME_PATTERNS = [
    # Primary: "1-VIN -SAL1A2A40SA606662.jpg"
    re.compile(r'^\d+-VIN\s+-([A-Z0-9]{17})\.', re.IGNORECASE),
    # Flexible: "VIN -VINCODE" or "VIN-VINCODE" anywhere
    re.compile(r'VIN\s*-\s*([A-Z0-9]{17})(?:\s|\.)', re.IGNORECASE),
    # Legacy: "42 -SAL1A2A40SA606662 2.jpg"
    re.compile(r'^\d+\s*-\s*([A-Z0-9]{17})(?:\s|\.)', re.IGNORECASE),
    # Underscore format: "VIN_-_SAL1A2A40SA606662_"
    re.compile(r'VIN_-_([A-Z0-9]{17})_', re.IGNORECASE),
]

# Fallback pattern for any 17-char alphanumeric (excluding I, O, Q)
_FALLBACK_PATTERN = re.compile(r'\b([A-HJ-NPR-Z0-9]{17})\b', re.IGNORECASE)


def extract_vin_from_filename(filename: str) -> Optional[str]:
    """
    Extract VIN from filename pattern.
    
    Supported formats (in priority order):
    1. NUMBER-VIN -VINCODE.ext  (e.g., "1-VIN -SAL1A2A40SA606662.jpg")
    2. VIN -VINCODE.ext or VIN-VINCODE.ext
    3. NUMBER -VINCODE rest.ext (legacy)
    4. VIN_-_VINCODE_ (underscore format)
    5. Any 17-char valid VIN sequence (fallback)
    
    Args:
        filename: Image filename (not full path)
        
    Returns:
        Extracted VIN (17 uppercase characters) or None if not found
        
    Examples:
        >>> extract_vin_from_filename("1-VIN -SAL1A2A40SA606662.jpg")
        'SAL1A2A40SA606662'
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
    
    # Fallback: find any 17-char sequence that could be a VIN
    match = _FALLBACK_PATTERN.search(filename)
    if match:
        vin = match.group(1).upper()
        # Verify no invalid characters
        if not any(c in vin for c in VIN_INVALID_CHARS):
            return vin
    
    return None


def _is_valid_vin_chars(vin: str) -> bool:
    """Check if all characters are valid VIN characters."""
    return all(c in VIN_VALID_CHARS or c in VIN_INVALID_CHARS for c in vin.upper())


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
    
    invalid_chars = [c for c in vin if c in VIN_INVALID_CHARS]
    has_valid_chars = len(invalid_chars) == 0 and all(c in VIN_VALID_CHARS for c in vin)
    
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
    1. If exactly 17 chars, return as-is
    2. Look for common WMI prefixes (SAL, WVW, 1G1, etc.)
    3. Find best 17-char substring with valid VIN characters
    
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
    
    # Strategy 1: Look for known WMI at any position
    for wmi in VINConstants.COMMON_WMIS:
        idx = text.find(wmi)
        if idx != -1 and idx + VIN_LENGTH <= len(text):
            candidate = text[idx:idx + VIN_LENGTH]
            # Verify it has mostly valid VIN characters
            if _score_vin_candidate(candidate) > 10:
                return candidate
    
    # Strategy 2: Try all 17-char substrings, find one with best score
    best_candidate = text[:VIN_LENGTH]  # Default: first 17 chars
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
    - +2 for each valid VIN character
    - +3 for each digit in sequential positions (12-17)
    - +10 for known WMI prefix
    - -5 for each invalid character (I, O, Q)
    
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
    
    # Common OCR confusions on engraved metal
    GLOBAL_CONFUSION_RULES: Dict[str, str] = {
        # Lowercase to uppercase (OCR sometimes outputs lowercase)
        'i': '1', 'l': '1', 'o': '0', 'q': '0',
        # Similar-looking characters
        '|': '1', '!': '1', '/': '1',
        '(': 'C', ')': 'J',
        '$': 'S', '§': 'S',
        '@': 'A', '&': '8',
    }
    
    # Position 12-17 (sequential number) should be digits
    # These rules only apply to those positions
    SEQUENTIAL_POSITION_RULES: Dict[str, str] = {
        'S': '5', 's': '5',
        'G': '6', 'g': '6',
        'B': '8', 'b': '8',
        'A': '4', 'a': '4',
        'L': '1', 'l': '1',
        'Z': '2', 'z': '2',
        'E': '3', 'e': '3',
        'T': '7', 't': '7',
        'D': '0', 'd': '0',
        'O': '0', 'o': '0',
        'I': '1', 'i': '1',
        'C': '0', 'c': '0',  # C can look like 0
    }
    
    # Artifact characters to remove
    ARTIFACT_CHARS: FrozenSet[str] = frozenset('*#@$%^&')
    
    # Artifact patterns at string boundaries
    ARTIFACT_PATTERNS: List[re.Pattern] = [
        re.compile(r'^[*#XYT]+'),      # Start artifacts
        re.compile(r'[*#]+$'),          # End artifacts
        re.compile(r'^[IYTFA][*#]+'),   # Common prefix + artifacts
    ]
    
    def __init__(self, learned_rules: Optional[Dict[str, str]] = None):
        """
        Initialize corrector with optional learned rules.
        
        Args:
            learned_rules: Additional char->char mappings learned from data
        """
        self.learned_rules = learned_rules or {}
        self._build_combined_rules()
    
    def _build_combined_rules(self):
        """Build combined rule set with priority ordering."""
        # Priority: Invalid chars > Learned > Global confusions
        self._global_rules = {}
        self._global_rules.update(self.GLOBAL_CONFUSION_RULES)
        self._global_rules.update(self.learned_rules)
        self._global_rules.update(self.INVALID_CHAR_RULES)
    
    def add_learned_rules(self, rules: Dict[str, str]):
        """Add rules learned from training data."""
        self.learned_rules.update(rules)
        self._build_combined_rules()
    
    def correct(self, raw_text: str, confidence: float = 0.0) -> Dict:
        """
        Apply rule-based corrections to raw OCR output.
        
        Processing steps:
        1. Normalize (uppercase, strip whitespace)
        2. Remove artifacts
        3. Apply global character substitutions
        4. Extract best 17-char VIN candidate
        5. Apply position-specific corrections
        6. Validate result
        
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
        
        # Step 3: Apply global substitutions
        text_before = text
        text = self._apply_global_rules(text)
        if text != text_before:
            corrections.append(f"Global corrections: '{text_before}' -> '{text}'")
        
        # Step 4: Extract 17-char VIN candidate
        text_before = text
        text = self._extract_vin_candidate(text)
        if text != text_before:
            corrections.append(f"Extracted VIN: '{text_before}' -> '{text}'")
        
        # Step 5: Position-specific corrections
        text_before = text
        text = self._apply_position_rules(text)
        if text != text_before:
            corrections.append(f"Position corrections: '{text_before}' -> '{text}'")
        
        # Step 6: Validate
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
        """Remove common artifact characters and patterns."""
        # Apply regex patterns
        for pattern in self.ARTIFACT_PATTERNS:
            text = pattern.sub('', text)
        
        # Remove individual artifact chars
        text = ''.join(c for c in text if c not in self.ARTIFACT_CHARS)
        
        return text
    
    def _apply_global_rules(self, text: str) -> str:
        """Apply global character substitution rules."""
        return ''.join(self._global_rules.get(c, c) for c in text)
    
    def _extract_vin_candidate(self, text: str) -> str:
        """Extract the best 17-character VIN candidate from text."""
        if len(text) == VIN_LENGTH:
            return text
        
        if len(text) < VIN_LENGTH:
            return text  # Too short, return as-is
        
        # Strategy 1: Look for known WMI (World Manufacturer Identifier)
        for wmi in VINConstants.COMMON_WMIS:
            idx = text.find(wmi)
            if idx != -1 and idx + VIN_LENGTH <= len(text):
                candidate = text[idx:idx + VIN_LENGTH]
                if self._score_candidate(candidate) > 10:
                    return candidate
        
        # Strategy 2: Score all 17-char substrings
        best_candidate = text[:VIN_LENGTH]
        best_score = self._score_candidate(best_candidate)
        
        for i in range(1, len(text) - VIN_LENGTH + 1):
            candidate = text[i:i + VIN_LENGTH]
            score = self._score_candidate(candidate)
            if score > best_score:
                best_score = score
                best_candidate = candidate
        
        return best_candidate
    
    def _score_candidate(self, candidate: str) -> int:
        """Score a VIN candidate (higher = more likely valid)."""
        score = 0
        
        # Valid VIN characters
        score += sum(2 for c in candidate if c in VIN_VALID_CHARS)
        
        # Digits in sequential positions (12-17)
        if len(candidate) >= VIN_LENGTH:
            score += sum(3 for c in candidate[11:17] if c.isdigit())
        
        # Starts with known WMI
        if candidate[:3] in VINConstants.COMMON_WMIS:
            score += 10
        
        # Penalty for invalid chars
        score -= sum(5 for c in candidate if c in VIN_INVALID_CHARS)
        
        return score
    
    def _apply_position_rules(self, text: str) -> str:
        """Apply position-specific correction rules."""
        if len(text) != VIN_LENGTH:
            return text
        
        result = list(text)
        
        # Positions 12-17 (indices 11-16) should be digits
        for idx in range(11, 17):
            char = result[idx]
            if char in self.SEQUENTIAL_POSITION_RULES:
                result[idx] = self.SEQUENTIAL_POSITION_RULES[char]
        
        return ''.join(result)
    
    def learn_from_errors(self, predictions: List[Dict]) -> Dict[str, str]:
        """
        Learn correction rules from prediction errors.
        
        Analyzes mismatches between predictions and ground truth
        to discover new character confusion patterns.
        
        Args:
            predictions: List of dicts with 'ground_truth' and 'prediction' keys
            
        Returns:
            Dict of learned char->char mappings
        """
        char_errors: Dict[str, Dict[str, int]] = {}
        
        for pred in predictions:
            gt = pred.get('ground_truth', '')
            pr = pred.get('prediction', '')
            
            if gt == pr:
                continue
            
            # Analyze character-level differences
            for i, (g, p) in enumerate(zip(gt, pr)):
                if g != p:
                    if p not in char_errors:
                        char_errors[p] = {}
                    char_errors[p][g] = char_errors[p].get(g, 0) + 1
        
        # Build rules from most common corrections
        new_rules = {}
        for predicted_char, corrections in char_errors.items():
            if corrections:
                # Find most frequent correction
                best_correction = max(corrections, key=corrections.get)
                count = corrections[best_correction]
                
                # Only add rule if seen multiple times and target is valid
                if count >= 2 and best_correction in VIN_VALID_CHARS:
                    new_rules[predicted_char] = best_correction
                    logger.info(
                        f"Learned rule: '{predicted_char}' -> '{best_correction}' "
                        f"(seen {count} times)"
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
        """Export rules for serialization."""
        return {
            'learned_rules': self.learned_rules,
            'version': '1.0',
        }
    
    @classmethod
    def from_exported(cls, data: Dict) -> 'RuleBasedCorrector':
        """Create corrector from exported rules."""
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

