"""
Happy-Path Test Suite for the VIN OCR Application
==================================================

Explicit success-flow coverage for every core module. The rest of the suite
is dominated by error/edge-case and regression tests; this file guarantees
the PRIMARY success path of each component is exercised against real data:

- Core VIN utilities         (extraction, validation, checksum, distance)
- Post-processor             (real observed raw OCR -> corrected VIN)
- Preprocessor               (all 4 modes on the real bundled plate image)
- Evaluation metrics         (perfect-prediction invariants, ground truth)
- CLI                        (help / version success paths)
- Full OCR pipeline          (end-to-end recognize on the real image)
- Streamlit web application  (renders all 5 pages without exception)

Every assertion in this file was verified empirically against the actual
behaviour of THIS repository's pipeline on this machine before being
written down: the bundled image `data/1-VIN -SAL1A2A40SA606662.jpg` is
recognised as `SAL1A2A40SA606662` (raw OCR `SAL1A2A40SA606662*`,
confidence ~0.98, checksum valid) by the PP-OCRv3 models on CPU.

None of these tests are marked `e2e`: in this repository that marker means
"requires a live Streamlit server on :8501" and is deselected by default.
Everything here is self-contained (the OCR tests use the on-disk models and
the web tests run the app in-process via streamlit.testing), so the happy
paths execute on EVERY plain `pytest` run - they must never be silently
skipped.

Run with:  pytest tests/test_happy_paths.py -v
"""

import os
import sys
from pathlib import Path

# Must be set before paddleocr/paddlex are imported (skips network probe).
os.environ.setdefault("DISABLE_MODEL_SOURCE_CHECK", "True")

# Repo root on sys.path (conftest.py also does this; kept for direct runs).
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from src.vin_ocr.core.vin_utils import (
    calculate_check_digit,
    extract_vin_from_filename,
    extract_vin_from_text,
    levenshtein_distance,
    validate_checksum,
    validate_vin_format,
)
from src.vin_ocr.evaluation.evaluate import (
    calculate_cer,
    calculate_character_metrics,
    calculate_position_accuracy,
    load_ground_truth,
)
from src.vin_ocr.pipeline.vin_pipeline import (
    VIN_LENGTH,
    VINImagePreprocessor,
    VINPostProcessor,
)

# =============================================================================
# GROUND TRUTH (real bundled sample, verified by manual pipeline run)
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
REAL_IMAGE = PROJECT_ROOT / "data" / "1-VIN -SAL1A2A40SA606662.jpg"
GROUND_TRUTH_VIN = "SAL1A2A40SA606662"
# Raw OCR output observed for REAL_IMAGE (trailing '*' border artifact).
OBSERVED_RAW_OCR = "SAL1A2A40SA606662*"


def test_real_sample_image_is_present():
    """The bundled sample image is a test dependency; fail loudly if gone."""
    assert REAL_IMAGE.exists(), (
        f"Bundled sample image missing: {REAL_IMAGE}. "
        "Happy-path tests require the real VIN plate image."
    )


# =============================================================================
# CORE VIN UTILITIES - SUCCESS FLOWS
# =============================================================================

class TestCoreUtilsHappyPath:
    """Primary success flows of src/vin_ocr/core/vin_utils.py."""

    def test_extract_vin_from_real_filename(self):
        """The bundled image filename yields the exact ground-truth VIN."""
        assert extract_vin_from_filename(REAL_IMAGE.name) == GROUND_TRUTH_VIN

    def test_ground_truth_vin_has_valid_format(self):
        """The real VIN passes format validation (17 chars, no I/O/Q)."""
        assert validate_vin_format(GROUND_TRUTH_VIN) is True

    def test_ground_truth_vin_has_valid_checksum(self):
        """The real VIN passes ISO 3779 checksum validation."""
        assert validate_checksum(GROUND_TRUTH_VIN) is True

    def test_check_digit_matches_position_nine(self):
        """Computed check digit equals the VIN's actual 9th character."""
        assert calculate_check_digit(GROUND_TRUTH_VIN) == GROUND_TRUTH_VIN[8]

    def test_extract_vin_from_noisy_ocr_text(self):
        """Artifact-wrapped OCR text yields the clean 17-char VIN."""
        noisy = f"*{GROUND_TRUTH_VIN}#"
        assert extract_vin_from_text(noisy) == GROUND_TRUTH_VIN

    def test_levenshtein_identity(self):
        """A correct prediction is zero edits from the reference."""
        assert levenshtein_distance(GROUND_TRUTH_VIN, GROUND_TRUTH_VIN) == 0


# =============================================================================
# POST-PROCESSOR - SUCCESS FLOWS
# =============================================================================

class TestPostProcessorHappyPath:
    """Primary success flows of VINPostProcessor.process()."""

    @pytest.fixture
    def postprocessor(self):
        return VINPostProcessor(verbose=False)

    def test_real_raw_ocr_corrected_to_ground_truth(self, postprocessor):
        """The raw OCR actually observed for the bundled image is corrected
        to the exact ground-truth VIN with the artifact removal tracked."""
        result = postprocessor.process(OBSERVED_RAW_OCR, confidence=0.98)
        assert result["vin"] == GROUND_TRUTH_VIN
        assert result["raw_ocr"] == OBSERVED_RAW_OCR
        assert result["is_valid_length"] is True
        assert result["checksum_valid"] is True
        assert len(result["corrections"]) == 1
        assert "artifact" in result["corrections"][0].lower()

    def test_clean_vin_passes_through_unchanged(self, postprocessor):
        """An already-perfect VIN needs no corrections at all."""
        result = postprocessor.process(GROUND_TRUTH_VIN, confidence=0.98)
        assert result["vin"] == GROUND_TRUTH_VIN
        assert result["is_valid_length"] is True
        assert result["checksum_valid"] is True
        assert result["corrections"] == []


# =============================================================================
# PREPROCESSOR - SUCCESS FLOWS (REAL IMAGE)
# =============================================================================

class TestPreprocessorHappyPath:
    """Every preprocessing mode succeeds on the real plate image and
    returns a 3-channel uint8 BGR image PaddleOCR can consume."""

    @pytest.fixture(scope="class")
    def real_image(self):
        import cv2

        image = cv2.imread(str(REAL_IMAGE))
        assert image is not None, f"cv2 could not decode {REAL_IMAGE}"
        return image

    @pytest.mark.parametrize("mode", ["none", "fast", "balanced", "engraved"])
    def test_mode_returns_bgr_uint8(self, real_image, mode):
        processed = VINImagePreprocessor(mode=mode).preprocess(real_image)
        assert isinstance(processed, np.ndarray)
        assert processed.ndim == 3
        assert processed.shape[2] == 3
        assert processed.dtype == np.uint8
        assert processed.shape[0] > 0 and processed.shape[1] > 0

    @pytest.mark.parametrize("mode", ["fast", "balanced", "engraved"])
    def test_enhancing_modes_bound_image_size(self, real_image, mode):
        """Enhancing modes cap the 1600px-wide source to max_dimension."""
        processed = VINImagePreprocessor(mode=mode).preprocess(real_image)
        assert max(processed.shape[:2]) <= max(real_image.shape[:2])


# =============================================================================
# EVALUATION METRICS - SUCCESS FLOWS
# =============================================================================

class TestEvaluationHappyPath:
    """Perfect predictions produce perfect metrics; ground truth loads."""

    def test_load_ground_truth_finds_real_image(self):
        ground_truth = load_ground_truth(str(PROJECT_ROOT / "data"))
        assert str(REAL_IMAGE) in ground_truth
        assert ground_truth[str(REAL_IMAGE)] == GROUND_TRUTH_VIN

    def test_perfect_prediction_zero_cer(self):
        assert calculate_cer([GROUND_TRUTH_VIN], [GROUND_TRUTH_VIN]) == 0.0

    def test_perfect_prediction_perfect_f1(self):
        precision, recall, f1 = calculate_character_metrics(
            [GROUND_TRUTH_VIN], [GROUND_TRUTH_VIN]
        )
        assert (precision, recall, f1) == (1.0, 1.0, 1.0)

    def test_perfect_prediction_all_positions_accurate(self):
        accuracy = calculate_position_accuracy(
            [GROUND_TRUTH_VIN], [GROUND_TRUTH_VIN]
        )
        assert len(accuracy) == VIN_LENGTH
        assert all(value == 1.0 for value in accuracy)


# =============================================================================
# CLI - SUCCESS FLOWS
# =============================================================================

class TestCLIHappyPath:
    """The CLI entry point succeeds for its argument-level success paths."""

    def test_no_arguments_prints_help_and_succeeds(self, monkeypatch, capsys):
        from src.vin_ocr.cli import main

        monkeypatch.setattr(sys, "argv", ["vin-ocr"])
        exit_code = main()
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "recognize" in captured.out
        assert "batch" in captured.out

    def test_version_flag_exits_zero(self, monkeypatch, capsys):
        from src.vin_ocr.cli import main

        monkeypatch.setattr(sys, "argv", ["vin-ocr", "--version"])
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0
        assert "1.0.0" in capsys.readouterr().out


# =============================================================================
# FULL PIPELINE - END-TO-END SUCCESS FLOW (REAL OCR, REAL IMAGE)
# =============================================================================

RESULT_SCHEMA_KEYS = {
    "vin",
    "confidence",
    "raw_ocr",
    "is_valid_length",
    "checksum_valid",
    "corrections",
    "processing_time_ms",
    "error",
}


@pytest.fixture(scope="module")
def pipeline():
    """One real pipeline for all OCR tests (PaddleOCR init is expensive)."""
    from src.vin_ocr.pipeline.vin_pipeline import VINOCRPipeline

    return VINOCRPipeline(preprocess_mode="engraved", enable_postprocess=True)


class TestPipelineEndToEndHappyPath:
    """THE application happy path: real image in, exact VIN out.

    Deterministic on CPU with the cached PP-OCRv3 models; verified to
    produce SAL1A2A40SA606662 at ~0.98 confidence on this machine.
    """

    def test_recognize_real_image_exact_match(self, pipeline):
        result = pipeline.recognize(str(REAL_IMAGE))
        assert set(result.keys()) >= RESULT_SCHEMA_KEYS
        assert result["error"] is None
        assert result["vin"] == GROUND_TRUTH_VIN
        assert result["is_valid_length"] is True
        assert result["checksum_valid"] is True
        assert result["confidence"] >= 0.9
        assert isinstance(result["confidence"], float)
        assert isinstance(result["corrections"], list)
        assert GROUND_TRUTH_VIN in result["raw_ocr"]

    def test_recognize_numpy_array_input_exact_match(self, pipeline):
        """The documented ndarray input path yields the same exact VIN."""
        import cv2

        image = cv2.imread(str(REAL_IMAGE))
        assert image is not None
        result = pipeline.recognize(image)
        assert result["error"] is None
        assert result["vin"] == GROUND_TRUTH_VIN

    def test_recognize_batch_single_real_image(self, pipeline):
        """Batch API succeeds and annotates each result with its file."""
        results = pipeline.recognize_batch(
            [str(REAL_IMAGE)], show_progress=False
        )
        assert len(results) == 1
        assert results[0]["vin"] == GROUND_TRUTH_VIN
        assert results[0]["error"] is None
        assert results[0]["file"] == str(REAL_IMAGE)


# =============================================================================
# STREAMLIT WEB APP - SUCCESS FLOWS (in-process, no server required)
# =============================================================================

APP_PATH = PROJECT_ROOT / "src" / "vin_ocr" / "web" / "app.py"
EXPECTED_PAGES = [
    "📁 Data Management",
    "🎯 Training",
    "🔍 Inference",
    "📊 Results Dashboard",
    "🔧 System Health",
]


class TestWebAppHappyPath:
    """The Streamlit application renders successfully (headless AppTest)."""

    def test_app_renders_without_exception(self):
        from streamlit.testing.v1 import AppTest

        app_test = AppTest.from_file(str(APP_PATH), default_timeout=300)
        app_test.run()
        assert len(app_test.exception) == 0, [
            str(e.value) for e in app_test.exception
        ]
        assert len(app_test.sidebar.radio) >= 1
        assert list(app_test.sidebar.radio[0].options) == EXPECTED_PAGES

    def test_every_page_renders_without_exception(self):
        from streamlit.testing.v1 import AppTest

        app_test = AppTest.from_file(str(APP_PATH), default_timeout=300)
        app_test.run()
        for page in EXPECTED_PAGES:
            app_test.sidebar.radio[0].set_value(page)
            app_test.run()
            assert len(app_test.exception) == 0, (
                page,
                [str(e.value) for e in app_test.exception],
            )
