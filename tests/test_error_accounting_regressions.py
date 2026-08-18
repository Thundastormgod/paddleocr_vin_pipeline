"""
Regression tests completing the crash-scoring cleanup outside
multi_model_evaluation: the single-pipeline evaluator's denominators, the
pipeline error contract, the torch availability probe, and the removal of
the fake-training-log generator.

Residue found by the 2026-08-18 AST sweep after the main evaluation fixes:

- evaluate.py excluded errored samples from character metrics (the
  ``if not error`` filter) but still divided exact_match_count by
  len(results) - every crashed image silently counted as a miss.
- VINOCRPipeline.recognize converts its internal exceptions into a result
  dict carrying an 'error' key; multi_model's run_vin_pipeline read only
  'vin' from it, re-introducing crash-as-empty-prediction one layer down.
- hardware_utils caught ``except Exception`` around ``import torch``,
  converting arbitrary failures into "torch unavailable" silently.
- evaluation/metrics.py's __main__ demo synthesised a decreasing loss and
  val_accuracy = 0.5 + 0.15/epoch through TrainingMetricsTracker,
  printing output indistinguishable from a real training log.
- web/app.py contained 15 bare ``except:`` clauses (catching SystemExit
  and KeyboardInterrupt along with everything else).
"""

import ast
from pathlib import Path

import pytest

from src.vin_ocr.evaluation.errors import ModelExecutionError
from src.vin_ocr.evaluation.evaluate import DatasetSplit, VINEvaluator
from src.vin_ocr.evaluation.multi_model_evaluation import MultiModelEvaluator

REPO_ROOT = Path(__file__).resolve().parents[1]
GT = "1M8GDM9AXKP042788"


class _StubPipeline:
    """Recognize() driven by a per-path outcome table."""

    def __init__(self, outcomes):
        self.outcomes = outcomes

    def recognize(self, image_path):
        return self.outcomes[image_path]


def _ok(vin, confidence=0.9):
    return {
        'vin': vin, 'confidence': confidence, 'raw_ocr': vin,
        'is_valid_length': len(vin) == 17, 'checksum_valid': False,
        'corrections': [], 'processing_time_ms': 5.0,
    }


def _err(message):
    return {
        'vin': '', 'confidence': 0.0, 'raw_ocr': '',
        'is_valid_length': False, 'checksum_valid': False,
        'corrections': [], 'processing_time_ms': 0.0, 'error': message,
    }


class TestEvaluatorDenominators:
    """WAS BROKEN: exact_match_rate divided by attempts, not measurements."""

    def _run(self, outcomes):
        split = DatasetSplit(
            name='test',
            image_paths=list(outcomes),
            ground_truths={p: GT for p in outcomes},
        )
        evaluator = VINEvaluator(pipeline=_StubPipeline(outcomes))
        return evaluator.evaluate(split, show_progress=False)

    def test_errored_samples_leave_the_exact_match_denominator(self):
        metrics, results = self._run({
            'a.jpg': _ok(GT),                    # correct
            'b.jpg': _ok("WRONGWRONGWRONG17"),   # wrong
            'c.jpg': _err("OCR backend crashed"),  # error, not a miss
        })
        assert metrics.failed_count == 1
        assert metrics.exact_match_count == 1
        assert metrics.exact_match_rate == pytest.approx(1 / 2), (
            "crashed image counted as a miss in the denominator"
        )
        assert metrics.total_samples == 3  # attempts stay visible

    def test_all_errored_reports_zero_not_crash(self):
        metrics, _ = self._run({'a.jpg': _err("x"), 'b.jpg': _err("y")})
        assert metrics.failed_count == 2
        assert metrics.exact_match_rate == 0.0
        assert metrics.character_f1 == 0.0  # nothing measured

    def test_error_free_run_is_unchanged(self):
        metrics, _ = self._run({'a.jpg': _ok(GT), 'b.jpg': _ok(GT)})
        assert metrics.failed_count == 0
        assert metrics.exact_match_rate == 1.0


class TestPipelineErrorContractIsHonoured:
    """
    WAS BROKEN: run_vin_pipeline read result['vin'] and ignored
    result['error'], so pipeline-internal crashes became empty predictions.
    """

    def test_errored_result_raises_model_execution_error(self):
        ev = MultiModelEvaluator.__new__(MultiModelEvaluator)
        engine = _StubPipeline({'img.jpg': _err("cv2.imread exploded")})
        with pytest.raises(ModelExecutionError) as excinfo:
            ev.run_vin_pipeline(engine, 'img.jpg')
        assert "cv2.imread exploded" in str(excinfo.value)

    def test_clean_result_passes_through(self):
        ev = MultiModelEvaluator.__new__(MultiModelEvaluator)
        engine = _StubPipeline({'img.jpg': _ok(GT, confidence=0.77)})
        vin, conf = ev.run_vin_pipeline(engine, 'img.jpg')
        assert vin == GT
        assert conf == 0.77


class TestNarrowExceptionHandling:
    """Broad catches that converted real failures into silence."""

    def test_torch_probe_catches_only_import_and_os_errors(self):
        path = REPO_ROOT / "src" / "vin_ocr" / "utils" / "hardware_utils.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        torch_handlers = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Try):
                continue
            imports_torch = any(
                isinstance(stmt, ast.Import)
                and any(alias.name == 'torch' for alias in stmt.names)
                for stmt in node.body
            )
            if imports_torch:
                torch_handlers.extend(node.handlers)
        assert torch_handlers, "torch import probe not found"
        for handler in torch_handlers:
            assert isinstance(handler.type, ast.Tuple), (
                f"line {handler.lineno}: torch import must catch exactly "
                f"(ImportError, OSError)"
            )
            names = {
                elt.id for elt in handler.type.elts if isinstance(elt, ast.Name)
            }
            assert names == {'ImportError', 'OSError'}

    def test_app_py_has_no_bare_excepts(self):
        path = REPO_ROOT / "src" / "vin_ocr" / "web" / "app.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        bare = [
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.ExceptHandler) and node.type is None
        ]
        assert bare == [], (
            f"bare except (catches SystemExit/KeyboardInterrupt) at {bare}"
        )


class TestNoFakeTrainingLogGenerator:
    """
    WAS PRESENT: evaluation/metrics.py __main__ fabricated an improving
    training run through TrainingMetricsTracker - realistic-looking fake
    logs from a repo that has already shipped fabricated metrics once.
    """

    def test_main_block_does_not_drive_the_training_tracker(self):
        path = REPO_ROOT / "src" / "vin_ocr" / "evaluation" / "metrics.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        main_blocks = [
            node for node in tree.body
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name)
            and node.test.left.id == '__name__'
        ]
        assert main_blocks, "__main__ block not found"
        for block in main_blocks:
            for node in ast.walk(block):
                if isinstance(node, ast.Call):
                    func = node.func
                    name = (
                        func.id if isinstance(func, ast.Name)
                        else getattr(func, 'attr', '')
                    )
                    assert name != 'TrainingMetricsTracker', (
                        "the fake-training-log demo is back"
                    )
