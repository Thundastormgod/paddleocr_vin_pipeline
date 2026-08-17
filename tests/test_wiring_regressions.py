"""
Regression tests for wiring defects found during the correctness audit.

"Wiring" here means the joins BETWEEN components, rather than the logic inside
any one of them:

  * **Repository-root resolution** joins code to the data, configs and
    checkpoints it loads.
  * **The character<->index mapping** joins training to inference.

Both share a failure signature that makes them worth pinning separately from
tests/test_logic_regressions.py: when they are wrong, nothing raises. A wrong
root silently resolves to a directory containing no images; a shifted charset
silently decodes a correctly-trained model into garbage. In both cases the
pipeline reports a bad model instead of a broken join, which is why these
defects survived several rounds of review.

Each test pins behaviour that was previously WRONG. The docstrings record the
old behaviour so a change that reintroduces it fails loudly rather than
quietly degrading results.
"""

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator, List, Optional, Tuple

from src.vin_ocr.core.charset import (
    BLANK_INDEX,
    BLANK_TOKEN,
    VIN_CHARSET,
    load_char_dict,
)
from src.vin_ocr.core.vin_utils import (
    extract_vin_from_text as canonical_extract_vin,
    validate_checksum,
)
from src.vin_ocr.evaluation import multi_model_evaluation as mme
from src.vin_ocr.evaluation.multi_model_evaluation import VINCharValidator
from src.vin_ocr.training.train_from_scratch import PaddleOCRScratchTrainer

REPO_ROOT = Path(__file__).resolve().parents[1]
VIN_DICT = REPO_ROOT / "configs" / "vin_dict.txt"

# A real, checksum-valid VIN from the dataset. It contains 'L' at index 2 -
# the character the CHAR_MAP defect corrupted.
VIN = "SAL1A2A40SA606662"


def test_reference_fixtures_are_sound():
    """Guard the fixtures themselves - the rest of the file depends on them."""
    assert len(VIN) == 17
    assert validate_checksum(VIN)
    assert "L" in VIN, "the CHAR_MAP tests below are vacuous without an 'L'"
    assert VIN_DICT.is_file(), f"missing character dictionary: {VIN_DICT}"


# ---------------------------------------------------------------------------
# Repository-root resolution
# ---------------------------------------------------------------------------

_ROOT_NAMES = frozenset({"project_root", "PROJECT_ROOT", "_project_root"})


def _levels_up_from_file(node: ast.expr) -> Optional[int]:
    """
    Count the directory levels a ``Path(__file__)`` expression walks up.

    ``Path(__file__).parent`` is 1 level (the file's own directory),
    ``.parents[n]`` is ``n + 1``, and ``.resolve()`` is transparent.

    Args:
        node: Right-hand side of an assignment.

    Returns:
        Number of levels walked up, or None if the expression is not rooted at
        ``Path(__file__)`` or uses a non-literal ``parents`` index (which
        cannot be resolved statically).
    """
    levels = 0
    current: ast.expr = node

    while True:
        if (
            isinstance(current, ast.Subscript)
            and isinstance(current.value, ast.Attribute)
            and current.value.attr == "parents"
        ):
            index = current.slice
            if not isinstance(index, ast.Constant) or not isinstance(index.value, int):
                return None
            levels += index.value + 1
            current = current.value.value
        elif isinstance(current, ast.Attribute) and current.attr == "parent":
            levels += 1
            current = current.value
        elif (
            isinstance(current, ast.Call)
            and isinstance(current.func, ast.Attribute)
            and current.func.attr == "resolve"
        ):
            current = current.func.value
        elif (
            isinstance(current, ast.Call)
            and isinstance(current.func, ast.Name)
            and current.func.id == "Path"
        ):
            args = current.args
            is_dunder_file = (
                len(args) == 1
                and isinstance(args[0], ast.Name)
                and args[0].id == "__file__"
            )
            return levels if is_dunder_file else None
        else:
            return None


def _root_assignments(path: Path) -> Iterator[Tuple[int, int]]:
    """
    Yield ``(lineno, levels_up)`` for each project-root assignment in a file.

    Args:
        path: Python source file to parse.

    Yields:
        Line number and level count for every assignment whose target is named
        in ``_ROOT_NAMES`` and whose value is a ``Path(__file__)`` expression.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets: List[ast.expr] = list(node.targets)
            value = node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets = [node.target]
            value = node.value
        else:
            continue

        for target in targets:
            if isinstance(target, ast.Name) and target.id in _ROOT_NAMES:
                levels = _levels_up_from_file(value)
                if levels is not None:
                    yield node.lineno, levels


class TestRepositoryRootResolution:
    """
    WAS BROKEN in two places, both silently:

      * ``multi_model_evaluation.py`` used ``Path(__file__).parent`` for a file
        three levels below the root, resolving to
        ``<root>/src/vin_ocr/evaluation``. The fine-tuned checkpoint, the
        DeepSeek and ONNX search roots and all three default dataset roots were
        therefore wrong. ``load_dataset()`` found zero images,
        ``run_evaluation()`` returned None, and the CLI still exited 0.
      * ``finetune_paddleocr.py:62`` walked up three levels (to ``<root>/src``)
        while the three other root computations in the SAME file walked four.

    A third occurrence, ``PROJECT_ROOT`` in ``train_from_scratch.py``, was both
    dead and wrong; it was deleted rather than fixed.

    This checks the invariant across the whole package rather than the three
    known sites, so a new occurrence anywhere under src/ or scripts/ fails here.
    """

    def test_every_root_assignment_reaches_the_repo_root(self):
        sources = sorted(
            [*REPO_ROOT.glob("src/**/*.py"), *REPO_ROOT.glob("scripts/**/*.py")]
        )
        assert sources, "no sources discovered - the glob is wrong"

        wrong: List[str] = []
        checked = 0

        for path in sources:
            relative = path.relative_to(REPO_ROOT)
            # Levels from the file up to the repo root == its path depth.
            expected = len(relative.parts)
            for lineno, levels in _root_assignments(path):
                checked += 1
                if levels != expected:
                    wrong.append(
                        f"{relative}:{lineno} walks up {levels}, needs {expected}"
                    )

        # Negative control for the detector itself: if a refactor changes the
        # idiom so nothing matches, this test must fail rather than pass
        # vacuously. There were 19 sites when this was written.
        assert checked >= 15, (
            f"detector matched only {checked} root assignments - it has "
            f"stopped recognising the idiom it is meant to police"
        )
        assert not wrong, "project roots that do not reach the repo root:\n" + "\n".join(
            wrong
        )

    def test_multi_model_evaluation_root_is_the_repo_root_at_runtime(self):
        """The resolved value, not just the source text."""
        assert mme.project_root == REPO_ROOT
        assert (mme.project_root / "pyproject.toml").is_file()


# ---------------------------------------------------------------------------
# VINCharValidator: character substitution
# ---------------------------------------------------------------------------


class TestVinCharValidatorCharMap:
    """
    WAS BROKEN: CHAR_MAP contained ``'l': '1', 'L': '1'``. 'L' is a perfectly
    legal VIN character and a member of VIN_CHARS. Because ``clean_vin()``
    uppercases BEFORE substituting, every 'L' in every prediction was rewritten
    to '1' before scoring::

        SAL1A2A40SA606662  ->  SA11A2A40SA606662   (the dataset's own VIN)
        WBALL31069PY12345  ->  WBA1131069PY12345

    A model that read the plate perfectly was scored as wrong, so every accuracy
    figure this module produced for an L-bearing VIN was invalid. The lowercase
    keys were dead in any case - the text is uppercased before substitution.
    """

    def test_substitutes_only_characters_illegal_in_a_vin(self):
        overlap = set(VINCharValidator.CHAR_MAP) & VINCharValidator.VIN_CHARS
        assert not overlap, f"CHAR_MAP rewrites legal VIN characters: {sorted(overlap)}"

    def test_covers_exactly_the_iso_3779_exclusions(self):
        assert set(VINCharValidator.CHAR_MAP) == VINCharValidator.INVALID_CHARS

    def test_clean_vin_preserves_the_letter_L(self):
        assert VINCharValidator.clean_vin(VIN) == VIN

    def test_clean_vin_maps_the_three_illegal_characters(self):
        assert VINCharValidator.clean_vin("IOQ") == "100"

    def test_clean_vin_strips_separators_and_noise(self):
        assert VINCharValidator.clean_vin("  sal1a2a40sa606662  ") == VIN
        assert VINCharValidator.clean_vin("SAL1A2A-40SA*606662") == VIN

    def test_clean_vin_does_not_truncate(self):
        """
        WAS BROKEN: clean_vin() truncated to 17 characters, discarding the tail
        before the extractor could search it. Truncation is extraction's job.
        """
        cleaned = VINCharValidator.clean_vin("PLATE NO " + VIN)
        assert len(cleaned) > 17
        assert VIN in cleaned

    def test_clean_vin_handles_empty_input(self):
        assert VINCharValidator.clean_vin("") == ""


class TestVinCharValidatorExtraction:
    """
    WAS BROKEN: ``extract_vin_from_text()`` was a no-op dressed up as a search::

        # Try to find 17 consecutive valid chars
        if len(cleaned) >= 17:
            return cleaned[:17]

    There was no search - it returned the first 17 characters. ``run_paddleocr``
    joins ALL detected text regions before calling this, so any text preceding
    the VIN on the plate shifted the window and guaranteed a miss, which was
    then reported as a model error rather than a harness bug.
    """

    LEADING_TEXT = "PLATE NO " + VIN

    def test_finds_the_vin_after_leading_plate_text(self):
        assert VINCharValidator.extract_vin_from_text(self.LEADING_TEXT) == VIN

    def test_the_old_first_17_behaviour_would_have_failed_here(self):
        """Negative control: proves the input actually exercises the defect."""
        cleaned = VINCharValidator.clean_vin(self.LEADING_TEXT)
        assert cleaned[:17] != VIN

    def test_delegates_to_the_single_source_of_truth(self):
        cleaned = VINCharValidator.clean_vin(self.LEADING_TEXT)
        assert VINCharValidator.extract_vin_from_text(
            self.LEADING_TEXT
        ) == canonical_extract_vin(cleaned)

    def test_exact_length_input_is_returned_unchanged(self):
        assert VINCharValidator.extract_vin_from_text(VIN) == VIN

    def test_short_input_is_neither_padded_nor_mangled(self):
        assert VINCharValidator.extract_vin_from_text("SAL123") == "SAL123"

    def test_empty_input(self):
        assert VINCharValidator.extract_vin_from_text("") == ""


# ---------------------------------------------------------------------------
# Scratch trainer: char<->index mapping
# ---------------------------------------------------------------------------


class TestScratchTrainerCharsetIsCanonical:
    """
    WAS BROKEN: ``PaddleOCRScratchTrainer._load_char_dict`` reimplemented the
    mapping instead of delegating to ``core.charset``::

        char_dict = {'<blank>': 0}
        for idx, line in enumerate(f, start=1):
            char_dict[line.strip()] = idx

    ``configs/vin_dict.txt`` ships '<blank>' as its FIRST line, so '<blank>' was
    re-mapped to 1 and every character shifted up by one. Measured against the
    real dictionary: blank->1, '0'->2, 'A'->12, 'Z'->34, len == 34. Three
    failures followed:

      * ``num_classes = len(char_dict) = 34`` means valid indices 0..33, but
        'Z' encoded to 34 - every label containing 'Z' was an out-of-alphabet
        CTC target.
      * ``CTCLoss(blank=0)`` and the greedy decoder both treat 0 as blank,
        while the dict said class 0 was unused and class 1 was '<blank>'. The
        decoder could splice the literal string "<blank>" into a predicted VIN.
      * Inference already delegated to ``core.charset`` and mapped '0'->1,
        'A'->11, 'Z'->33 against training's 2/12/34, so a correctly-trained
        model decoded to garbage.

    It also depended on whether the dict file already existed: ``create_vin_dict()``
    writes the 33 characters WITHOUT a blank line, which loaded correctly - so
    identical code produced two different charsets.
    """

    @staticmethod
    def _load_via_trainer(path: Path):
        """
        Invoke the method against a stub instance.

        The real ``__init__`` requires PaddlePaddle, which is an optional
        dependency and absent from the CI test environment. ``_load_char_dict``
        touches only ``self.config.character_dict_path`` and assigns
        ``self.idx_to_char``, so a namespace stub exercises it faithfully.

        Args:
            path: Character dictionary to load.

        Returns:
            Tuple of the returned char->index mapping and the stub, so callers
            can inspect the cached reverse map.
        """
        stub = SimpleNamespace(
            config=SimpleNamespace(character_dict_path=str(path))
        )
        return PaddleOCRScratchTrainer._load_char_dict(stub), stub

    def test_matches_the_single_source_of_truth_exactly(self):
        char_to_idx, _ = self._load_via_trainer(VIN_DICT)
        canonical, _ = load_char_dict(str(VIN_DICT))
        assert char_to_idx == canonical

    def test_blank_maps_to_index_zero(self):
        char_to_idx, _ = self._load_via_trainer(VIN_DICT)
        assert char_to_idx[BLANK_TOKEN] == BLANK_INDEX

    def test_every_index_is_a_valid_output_class(self):
        """
        The invariant the old mapping violated. With
        ``num_classes == len(char_dict)`` the largest index must be
        ``len - 1``; the old mapping produced max index 34 with length 34, so
        'Z' addressed a class the final Linear layer could not emit.
        """
        char_to_idx, _ = self._load_via_trainer(VIN_DICT)
        assert max(char_to_idx.values()) == len(char_to_idx) - 1

    def test_characters_are_not_shifted(self):
        """Pins the exact indices that diverged between training and inference."""
        char_to_idx, _ = self._load_via_trainer(VIN_DICT)
        assert char_to_idx["0"] == 1
        assert char_to_idx["A"] == 11
        assert char_to_idx["Z"] == 33

    def test_caches_the_canonical_reverse_map(self):
        char_to_idx, stub = self._load_via_trainer(VIN_DICT)
        _, canonical_reverse = load_char_dict(str(VIN_DICT))
        assert stub.idx_to_char == canonical_reverse

    def test_reverse_map_still_contains_blank_so_decoding_must_filter_it(self):
        """
        The canonical reverse map intentionally holds ``BLANK_INDEX ->
        '<blank>'``. ``evaluate()`` must therefore strip it before decoding; if
        it does not, a stray blank prediction splices the literal 7-character
        string "<blank>" into a predicted VIN. This pins the precondition that
        makes that filtering necessary.
        """
        _, stub = self._load_via_trainer(VIN_DICT)
        assert stub.idx_to_char[BLANK_INDEX] == BLANK_TOKEN

    def test_implicit_blank_dictionary_is_normalised_to_the_same_mapping(self, tmp_path):
        """
        ``create_vin_dict()`` writes the 33 characters with no blank line. The
        old code produced a different charset depending on whether the file
        already existed; both conventions must now converge.
        """
        implicit = tmp_path / "vin_dict.txt"
        implicit.write_text("\n".join(VIN_CHARSET) + "\n", encoding="utf-8")

        char_to_idx, _ = self._load_via_trainer(implicit)
        explicit, _ = load_char_dict(str(VIN_DICT))

        assert char_to_idx[BLANK_TOKEN] == BLANK_INDEX
        assert max(char_to_idx.values()) == len(char_to_idx) - 1
        assert char_to_idx == explicit
