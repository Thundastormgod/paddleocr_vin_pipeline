"""
Character Dictionary Loading - Single Source of Truth
=====================================================

Training and inference MUST agree on the character<->index mapping. When they
disagree, the model emits correct logits and the decoder turns them into
garbage.

This module exists because that exact bug shipped: training used
``enumerate(f)`` (blank at 0, '0' at 1, ... 'Z' at 33) while both inference
backends seeded ``{'<blank>': 0}`` and then used ``enumerate(f, start=1)``,
which re-mapped '<blank>' to 1 and shifted every character up by one. A
perfectly-trained model decoded "SAL1A2A40SA606662" as "R9K09193...", and
characters at low indices were dropped entirely, producing empty predictions.

Every component must call load_char_dict() from here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Canonical VIN charset (excludes I, O, Q per ISO 3779)
VIN_CHARSET: str = "0123456789ABCDEFGHJKLMNPRSTUVWXYZ"

# Tokens recognised as an explicit CTC blank at the top of a dict file
_BLANK_TOKENS = frozenset({"<blank>", "blank", "<BLANK>", "[blank]", "<pad>"})

# Canonical name used for the blank class in the returned mappings
BLANK_TOKEN: str = "<blank>"

# Index reserved for the CTC blank class
BLANK_INDEX: int = 0

# Conventional filename searched for inside model/config directories
DICT_FILENAME: str = "vin_dict.txt"


def _resolve_dict_path(dict_path: Optional[str], search_dirs: Optional[List[Path]] = None) -> Optional[Path]:
    """Resolve a character dict path, searching common locations if needed."""
    if dict_path:
        p = Path(dict_path)
        if p.is_file():
            return p
        # Caller passed a directory - look for the conventional filename inside
        if p.is_dir() and (p / DICT_FILENAME).is_file():
            return p / DICT_FILENAME
        # Try relative to the project root (src/vin_ocr/core/ -> up 3)
        project_root = Path(__file__).resolve().parents[3]
        candidate = project_root / dict_path
        if candidate.is_file():
            return candidate
        return None

    # search_dirs are DIRECTORIES; look for the dict file inside each.
    candidates: List[Path] = [
        Path(d) / DICT_FILENAME for d in (search_dirs or [])
    ]
    project_root = Path(__file__).resolve().parents[3]
    candidates += [
        project_root / "configs" / DICT_FILENAME,
        Path("configs") / DICT_FILENAME,
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


def load_char_dict(
    dict_path: Optional[str] = None,
    search_dirs: Optional[List[Path]] = None,
) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Load the character dictionary used by both training and inference.

    Handles the two dict-file conventions:

    1. **Explicit blank** — the file's first non-empty line is a blank token
       (``<blank>``). Indices are assigned from 0, so the blank lands at 0.
       This is the convention used by ``configs/vin_dict.txt``.

    2. **Implicit blank** — the file contains only real characters (standard
       PaddleOCR rec dicts). The blank is inserted at index 0 and characters
       are assigned from index 1.

    Either way the result satisfies the CTC contract: ``BLANK_INDEX == 0``.

    Args:
        dict_path: Path to the dictionary file. If None, common locations are
            searched, falling back to the built-in VIN charset.
        search_dirs: Extra directories to search when dict_path is None
            (e.g. the model directory).

    Returns:
        (char_to_idx, idx_to_char)

    Raises:
        ValueError: If the resolved dictionary is empty or maps the blank to a
            non-zero index.
    """
    resolved = _resolve_dict_path(dict_path, search_dirs)

    if resolved is None:
        # Fall back to the built-in charset with an implicit blank at 0.
        entries = [BLANK_TOKEN, *VIN_CHARSET]
    else:
        with open(resolved, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f]
        entries = [line for line in lines if line]

        if not entries:
            raise ValueError(f"Character dictionary is empty: {resolved}")

        # Convention 2: no explicit blank -> insert one at index 0
        if entries[0] not in _BLANK_TOKENS:
            entries = [BLANK_TOKEN, *entries]
        else:
            # Normalise whatever blank spelling was used to BLANK_TOKEN
            entries[0] = BLANK_TOKEN

    char_to_idx: Dict[str, int] = {}
    idx_to_char: Dict[int, str] = {}
    for idx, char in enumerate(entries):
        char_to_idx[char] = idx
        idx_to_char[idx] = char

    if char_to_idx[BLANK_TOKEN] != BLANK_INDEX:
        raise ValueError(
            f"CTC blank must map to index {BLANK_INDEX}, got "
            f"{char_to_idx[BLANK_TOKEN]} (dict: {resolved})"
        )

    return char_to_idx, idx_to_char


def num_classes(char_to_idx: Dict[str, int]) -> int:
    """Number of output classes, including the CTC blank."""
    return len(char_to_idx)
