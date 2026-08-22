"""
Regression tests for evaluation-layer honesty: the canonical character
metrics, the canonical CTC decode, the multi-model dispatch table, and
per-image error accounting.

The defects these pin, all confirmed live by direct execution before the
fix (audit of 2026-08-18):

- H4: the loader registered type 'finetuned_deepseek_onnx' while dispatch
  tested 'deepseek_finetuned_onnx', so every fine-tuned DeepSeek ONNX model
  fell through to an else-branch scoring ("", 0.0) per image - tabulated as
  a model that legitimately scored 0%.
- H5: character metrics compared pred[i] == gt[i] positionally and could
  not count FP for missing/invalid/extra characters: a 5-char correct
  prefix scored precision 1.000 at recall 0.294, and '*' + 16 correct
  chars scored char_accuracy 0.059. Predictions were also padded with '_'
  before scoring AND before being written into the results JSON.
- H6: a third CTC decoder used blank_idx = len(charset) = 33 against
  models trained with blank 0: canonically-encoded "1M8" decoded to
  "020N090". That decoder sat in run_onnx - the live path for every
  exported PaddleOCR model - so every ONNX number this file ever recorded
  measured the decoder, not the model.
- Crash-scoring: nine runner handlers converted exceptions into ("", 0.0)
  "predictions" inside the accuracy denominator; a model whose
  initialisation failed was scored 0% across the whole dataset.
"""

import ast
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from src.vin_ocr.core.char_metrics import (
    AlignmentCounts,
    alignment_counts,
    char_level_metrics,
    corpus_cer,
    micro_prf,
)
from src.vin_ocr.core.charset import (
    BLANK_INDEX,
    ctc_greedy_decode,
    load_char_dict,
)
from src.vin_ocr.evaluation.errors import (
    ModelExecutionError,
    ModelUnavailableError,
)
from src.vin_ocr.evaluation.multi_model_evaluation import (
    ModelMetrics,
    MultiModelEvaluator,
)

GT = "1M8GDM9AXKP042788"

REPO_ROOT = Path(__file__).resolve().parents[1]
MULTI_MODEL_PATH = (
    REPO_ROOT / "src" / "vin_ocr" / "evaluation" / "multi_model_evaluation.py"
)


def _evaluator() -> MultiModelEvaluator:
    """An evaluator instance without loading any real models."""
    ev = MultiModelEvaluator.__new__(MultiModelEvaluator)
    return ev


class TestCanonicalCharMetricsGoldens:
    """Hand-computed values for the canonical definitions."""

    def test_truncated_prefix(self):
        m = char_level_metrics([(GT[:5], GT)])
        assert m.precision == pytest.approx(1.0)         # all 5 emitted are right
        assert m.recall == pytest.approx(5 / 17)
        assert m.f1_micro == pytest.approx(2 * (5 / 17) / (1 + 5 / 17))
        assert m.char_accuracy == pytest.approx(5 / 17)  # 1 - 12/17

    def test_single_leading_artifact(self):
        """One '*' plus 16 correct chars is a 94%-correct prediction."""
        m = char_level_metrics([("*" + GT[:-1], GT)])
        assert m.precision == pytest.approx(16 / 17)
        assert m.recall == pytest.approx(16 / 17)
        assert m.f1_micro == pytest.approx(16 / 17)
        assert m.char_accuracy == pytest.approx(15 / 17)  # editdist 2
        # The positional value this replaced was 0.059.
        assert m.f1_micro > 0.9

    def test_wrong_invalid_char_costs_precision(self):
        """'*' in place of a char is FP + FN - it can no longer hide."""
        pred = GT[:8] + "*" + GT[9:]
        counts = alignment_counts(pred, GT)
        assert counts.fp == 1
        assert counts.fn == 1
        p, r, _ = micro_prf(counts)
        assert p == pytest.approx(16 / 17)
        assert r == pytest.approx(16 / 17)

    def test_extra_trailing_chars_cost_precision(self):
        pred = GT + "ZZZ"
        counts = alignment_counts(pred, GT)
        assert counts.fp == 3
        p, r, _ = micro_prf(counts)
        assert p == pytest.approx(17 / 20)
        assert r == pytest.approx(1.0)

    def test_exact_match_is_perfect(self):
        m = char_level_metrics([(GT, GT)])
        assert (m.precision, m.recall, m.f1_micro) == (1.0, 1.0, 1.0)
        assert m.char_accuracy == 1.0
        assert m.cer == 0.0

    def test_empty_prediction_is_all_misses(self):
        m = char_level_metrics([("", GT)])
        assert (m.precision, m.recall, m.f1_micro) == (0.0, 0.0, 0.0)
        assert m.cer == pytest.approx(1.0)
        assert m.char_accuracy == 0.0

    def test_conservation_invariants(self):
        """Every char of both strings is accounted for exactly once."""
        for pred in ["", GT, GT[:5], "*" + GT, GT + "XX", "ZZZZZ"]:
            c = alignment_counts(pred, GT)
            assert c.tp + c.fp == len(pred)
            assert c.tp + c.fn == len(GT)

    def test_cer_can_exceed_one_and_accuracy_floors_at_zero(self):
        pairs = [(GT + GT + GT, GT)]
        assert corpus_cer(pairs) > 1.0
        assert char_level_metrics(pairs).char_accuracy == 0.0


class TestEveryScorerAgreesOnIdenticalInput:
    """
    The audit measured F1 = 0.67 / 0.95 / 0.4706 / 0.9412 for identical
    input across the four copies. All four now delegate to core.char_metrics
    and must produce the same number.
    """

    PAIRS = [
        (GT, GT),                    # exact
        (GT[:5], GT),                # truncation
        ("*" + GT[:-1], GT),         # leading artifact shift
        ("ZZZZZZZZZZZZZZZZZ", GT),   # total mismatch
    ]

    def _canonical_f1(self):
        preds = [p for p, _ in self.PAIRS]
        refs = [r for _, r in self.PAIRS]
        return char_level_metrics(list(zip(preds, refs))).f1_micro

    def test_evaluate_py_agrees(self):
        from src.vin_ocr.evaluation.evaluate import calculate_character_metrics

        preds = [p for p, _ in self.PAIRS]
        refs = [r for _, r in self.PAIRS]
        _, _, f1 = calculate_character_metrics(preds, refs)
        assert f1 == pytest.approx(self._canonical_f1())

    def test_training_metrics_agrees(self):
        from src.vin_ocr.training.metrics import calculate_char_level_metrics

        preds = [p for p, _ in self.PAIRS]
        refs = [r for _, r in self.PAIRS]
        out = calculate_char_level_metrics(preds, refs)
        assert out['f1_micro'] == pytest.approx(self._canonical_f1())

    def test_evaluation_metrics_calculator_agrees(self):
        from src.vin_ocr.evaluation.metrics import EvaluationMetricsCalculator

        calc = EvaluationMetricsCalculator()
        for pred, ref in self.PAIRS:
            calc.add_sample(prediction=pred, ground_truth=ref)
        result = calc.compute()
        assert result.character_level.f1_micro == pytest.approx(
            self._canonical_f1()
        )

    def test_multi_model_evaluator_agrees(self):
        ev = _evaluator()
        preds = [p for p, _ in self.PAIRS]
        refs = [r for _, r in self.PAIRS]
        m = MultiModelEvaluator._calculate_metrics(
            ev, "m", preds, refs, [0.9] * 4, [1.0] * 4, [{}] * 4
        )
        assert m.f1_micro == pytest.approx(self._canonical_f1())

    def test_h5_probes_are_dead(self):
        """The two execution probes that proved H5, asserted fixed."""
        ev = _evaluator()
        m = MultiModelEvaluator._calculate_metrics(
            ev, "m", [GT[:5]], [GT], [0.9], [1.0], [{}]
        )
        # precision is genuinely 1.0 for a correct prefix - but recall now
        # exposes it and char_accuracy is edit-based, not positional.
        assert m.micro_recall == pytest.approx(5 / 17)
        assert m.character_accuracy == pytest.approx(5 / 17)

        m2 = MultiModelEvaluator._calculate_metrics(
            ev, "m", ["*" + GT[:-1]], [GT], [0.9], [1.0], [{}]
        )
        assert m2.character_accuracy == pytest.approx(15 / 17)
        assert m2.f1_micro == pytest.approx(16 / 17)


class TestCanonicalCTCDecode:
    """One decoder, blank at index 0, collapse-then-strip."""

    @pytest.fixture(scope="class")
    def dicts(self):
        return load_char_dict("configs/vin_dict.txt")

    def test_decodes_canonical_encoding(self, dicts):
        c2i, i2c = dicts
        seq = [BLANK_INDEX, c2i['1'], BLANK_INDEX, c2i['M'], BLANK_INDEX, c2i['8']]
        text, kept = ctc_greedy_decode(seq, i2c)
        assert text == "1M8"
        assert kept == [1, 3, 5]

    def test_blank_separated_repeat_survives(self, dicts):
        c2i, i2c = dicts
        text, _ = ctc_greedy_decode(
            [c2i['A'], c2i['A'], BLANK_INDEX, c2i['A']], i2c
        )
        assert text == "AA"

    def test_h6_regression_the_shifted_decoder_is_gone(self):
        """
        No decoder in multi_model_evaluation may define its own blank as
        len(charset). The defective pattern decoded '1M8' as '020N090'.
        Checked in the AST, not the raw text: docstrings legitimately
        describe the old defect.
        """
        source = MULTI_MODEL_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            # any assignment of a name containing 'blank' to a Call(len)
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                func = node.value.func
                if isinstance(func, ast.Name) and func.id == 'len':
                    for target in node.targets:
                        if isinstance(target, ast.Name) and 'blank' in target.id:
                            pytest.fail(
                                f"local blank definition at line {node.lineno}"
                            )

    def test_onnx_inference_delegates_to_canonical_decode(self):
        from src.vin_ocr.inference.onnx_inference import ONNXVINRecognizer

        source = inspect.getsource(ONNXVINRecognizer.decode_ctc)
        assert "ctc_greedy_decode" in source

    def test_scratch_trainer_delegates_to_canonical_decode(self):
        source = (
            REPO_ROOT / "src" / "vin_ocr" / "training" / "train_from_scratch.py"
        ).read_text(encoding="utf-8")
        assert "ctc_greedy_decode" in source


class _StubSession:
    """Minimal onnxruntime.InferenceSession stand-in for run_onnx."""

    class _Tensor:
        def __init__(self, name, shape):
            self.name = name
            self.shape = shape

    def __init__(self, logits):
        self._logits = logits

    def get_inputs(self):
        return [self._Tensor("x", [1, 1, 32, 320])]

    def get_outputs(self):
        return [self._Tensor("y", list(self._logits.shape))]

    def run(self, _output_names, _feed):
        return [self._logits]


class TestRunOnnxDecodesWithCanonicalDict:
    """
    End-to-end H6 kill: run_onnx on a synthetic image and a stub session
    whose logits encode a VIN under the CANONICAL mapping must decode that
    VIN. Under the old blank=33 decoder this exact input produced garbage.
    """

    def _logits_for(self, text, c2i, n_classes):
        seq = []
        for ch in text:
            seq.extend([BLANK_INDEX, c2i[ch]])
        seq.append(BLANK_INDEX)
        logits = np.full((1, len(seq), n_classes), -10.0, dtype=np.float32)
        for t, idx in enumerate(seq):
            logits[0, t, idx] = 10.0
        return logits

    def test_decodes_canonically_encoded_vin(self, tmp_path):
        import cv2

        c2i, i2c = load_char_dict("configs/vin_dict.txt")
        logits = self._logits_for(GT, c2i, n_classes=len(i2c))

        img_path = tmp_path / "plate.png"
        cv2.imwrite(str(img_path), np.full((32, 320), 128, dtype=np.uint8))

        ev = _evaluator()
        model_info = {
            'engine': _StubSession(logits),
            'input_name': 'x',
            'output_name': 'y',
        }
        vin, confidence = ev.run_onnx(model_info, str(img_path))

        assert vin == GT
        assert 0.0 < confidence <= 1.0

    def test_unreadable_image_is_an_error_not_a_prediction(self, tmp_path):
        ev = _evaluator()
        model_info = {'engine': _StubSession(np.zeros((1, 2, 34), np.float32)),
                      'input_name': 'x', 'output_name': 'y'}
        with pytest.raises(ModelExecutionError):
            ev.run_onnx(model_info, str(tmp_path / "does_not_exist.png"))


class TestDispatchTable:
    """H4: registration and dispatch draw from one table."""

    def test_every_registered_type_literal_has_a_runner(self):
        """
        AST check over the loader methods: every string assigned to a
        'type' key must be a key of MODEL_RUNNERS. This is the exact shape
        of the H4 bug - a transposed word in ONE registration string.
        """
        tree = ast.parse(MULTI_MODEL_PATH.read_text(encoding="utf-8"))
        registered = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Dict):
                for key, value in zip(node.keys, node.values):
                    if (
                        isinstance(key, ast.Constant) and key.value == 'type'
                        and isinstance(value, ast.Constant)
                        and isinstance(value.value, str)
                    ):
                        registered.add(value.value)
            # model_type = '<literal>' assignments feeding 'type': model_type
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
                if (
                    isinstance(node.value.value, str)
                    and any(
                        isinstance(t, ast.Name) and t.id == 'model_type'
                        for t in node.targets
                    )
                ):
                    registered.add(node.value.value)

        assert registered, "no registered type literals found - test is broken"
        unknown = registered - set(MultiModelEvaluator.MODEL_RUNNERS)
        assert unknown == set(), (
            f"registered model types with no dispatch entry: {unknown}"
        )

    def test_descriptions_and_prefixes_use_registered_types(self):
        runners = set(MultiModelEvaluator.MODEL_RUNNERS)
        assert set(MultiModelEvaluator.MODEL_TYPE_DESCRIPTIONS) <= runners
        assert set(MultiModelEvaluator.ONNX_MODEL_PREFIXES.values()) <= runners

    def test_every_runner_in_the_table_exists(self):
        for type_name, method_name in MultiModelEvaluator.MODEL_RUNNERS.items():
            assert callable(getattr(MultiModelEvaluator, method_name, None)), (
                f"type '{type_name}' maps to missing runner '{method_name}'"
            )

    def test_unknown_type_raises_instead_of_scoring_zero(self):
        ev = _evaluator()
        with pytest.raises(ModelUnavailableError) as excinfo:
            ev.evaluate_model(
                'k', {'name': 'M', 'type': 'no_such_type', 'engine': None}, []
            )
        assert 'no_such_type' in str(excinfo.value)

    def test_deepseek_finetuned_onnx_reaches_its_runner(self, monkeypatch):
        """The exact H4 case: this type used to fall into the else-branch."""
        calls = []

        def fake_run(self, model_info, img):
            calls.append(img)
            return GT, 0.9

        monkeypatch.setattr(MultiModelEvaluator, 'run_deepseek_onnx', fake_run)
        ev = _evaluator()
        metrics = ev.evaluate_model(
            'k',
            {'name': 'DS', 'type': 'deepseek_finetuned_onnx', 'engine': object()},
            [("img1.png", GT)],
        )
        assert calls == ["img1.png"]
        assert metrics.exact_match_accuracy == 1.0


class TestCrashesAreErrorsNotMeasurements:
    """A per-image crash must never enter the accuracy denominator."""

    def test_crash_on_one_image_is_excluded_from_metrics(self, monkeypatch):
        outcomes = {
            "a.png": (GT, 0.9),
            "b.png": RuntimeError("OCR backend crashed"),
            "c.png": ("WRONGWRONGWRONG17", 0.4),
        }

        def fake_run(self, engine, img):
            result = outcomes[img]
            if isinstance(result, Exception):
                raise result
            return result

        monkeypatch.setattr(MultiModelEvaluator, 'run_paddleocr', fake_run)
        ev = _evaluator()
        metrics = ev.evaluate_model(
            'k',
            {'name': 'P', 'type': 'paddleocr', 'engine': object()},
            [(name, GT) for name in ("a.png", "b.png", "c.png")],
        )

        assert metrics.evaluation_errors == 1
        assert metrics.total_images == 2, "crashed image entered the denominator"
        assert metrics.exact_match_accuracy == pytest.approx(1 / 2)
        error_rows = [s for s in metrics.sample_results if s.get('status') == 'error']
        assert len(error_rows) == 1
        assert 'OCR backend crashed' in error_rows[0]['error']
        # H10: an error row carries the FULL result schema with NEUTRAL
        # values (so _print_comparison and the CSV writers cannot KeyError
        # mid-report), while status='error' remains the discriminator and
        # the row stays out of every metric denominator (asserted above).
        # This previously asserted the key was absent - which is exactly
        # the reduced schema that crashed both consumers.
        assert error_rows[0]['prediction'] == '', (
            "an error row must carry a neutral empty prediction, never a value"
        )
        assert error_rows[0]['exact_match'] is False
        assert error_rows[0]['chars_correct'] == 0
        assert error_rows[0]['char_accuracy'] == 0.0
        assert error_rows[0]['match_pattern'] == ''
        assert error_rows[0]['confidence'] == 0.0
        assert error_rows[0]['processing_time'] == 0.0

    def test_predictions_are_never_padded(self, monkeypatch):
        """The '_' padding wrote falsified predictions into the JSON."""
        monkeypatch.setattr(
            MultiModelEvaluator, 'run_paddleocr',
            lambda self, engine, img: (GT[:5], 0.9),
        )
        ev = _evaluator()
        metrics = ev.evaluate_model(
            'k', {'name': 'P', 'type': 'paddleocr', 'engine': object()},
            [("a.png", GT)],
        )
        measured = [s for s in metrics.sample_results if s['status'] == 'measured']
        assert measured[0]['prediction'] == GT[:5]
        assert '_' not in measured[0]['prediction']
        assert metrics.micro_recall == pytest.approx(5 / 17)

    def test_init_failure_makes_the_model_unavailable(self):
        class BrokenEngine:
            _initialized = False

            def initialize(self):
                raise RuntimeError("weights missing")

        ev = _evaluator()
        with pytest.raises(ModelUnavailableError) as excinfo:
            ev.run_deepseek(BrokenEngine(), "img.png")
        assert "weights missing" in str(excinfo.value)

    def test_unavailable_model_is_recorded_not_scored(self, monkeypatch, tmp_path):
        """run_evaluation reports NOT EVALUATED instead of a 0% row."""
        ev = _evaluator()
        ev.output_dir = tmp_path
        ev.models = {
            'broken': {'name': 'B', 'type': 'deepseek', 'engine': None},
        }
        monkeypatch.setattr(
            MultiModelEvaluator, 'load_models', lambda self: None
        )
        monkeypatch.setattr(
            MultiModelEvaluator, 'load_dataset',
            lambda self, custom_folder=None, labels_file=None: [("a.png", GT)],
        )

        def unavailable(self, engine, img):
            raise ModelUnavailableError("no runtime")

        monkeypatch.setattr(MultiModelEvaluator, 'run_deepseek', unavailable)

        all_metrics = ev.run_evaluation()

        assert all_metrics == {}
        saved = json.loads(
            (tmp_path / 'multi_model_evaluation.json').read_text(encoding='utf-8')
        )
        assert saved['not_evaluated'] == {'broken': 'no runtime'}
        assert saved['models'] == {}

    def test_no_runner_returns_a_constant_on_exception(self):
        """
        AST pin over every run_* method: no except handler may return a
        constant tuple - the ('', 0.0) crash-scoring pattern.
        """
        tree = ast.parse(MULTI_MODEL_PATH.read_text(encoding="utf-8"))
        offenders = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name.startswith('run_'):
                for handler in [n for n in ast.walk(node)
                                if isinstance(n, ast.ExceptHandler)]:
                    for ret in [n for n in ast.walk(handler)
                                if isinstance(n, ast.Return)]:
                        if isinstance(ret.value, (ast.Constant, ast.Tuple)):
                            offenders.append(f"{node.name}:{ret.lineno}")
        assert offenders == [], (
            f"except handlers returning constants (crash scored as "
            f"prediction): {offenders}"
        )


class TestModelMetricsSchema:
    """The results JSON must distinguish measured from failed."""

    def test_evaluation_errors_field_exists_and_defaults_to_zero(self):
        fields = {f.name for f in ModelMetrics.__dataclass_fields__.values()}
        assert 'evaluation_errors' in fields
        ev = _evaluator()
        m = MultiModelEvaluator._calculate_metrics(
            ev, "m", [GT], [GT], [0.9], [1.0], [{}]
        )
        assert m.evaluation_errors == 0
