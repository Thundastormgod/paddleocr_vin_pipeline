"""
Regression tests for the in-process tuner
(src.vin_ocr.training.hyperparameter_tuning.optuna_tuning).

The defect these pin is the same fabrication mechanism fixed in the repo-root
tuner, in a worse form: the objectives called
``trainer.train(epoch_callback=...)`` - a keyword no trainer in this
repository accepts - so every single trial raised TypeError, which
``except Exception: return 0.0`` then converted into a measured zero. Every
value this tuner ever reported was therefore fabricated, and its crashes were
additionally mislabelled as pruned trials. The duplication map for this
repository is explicit that a fix applied to one copy is not a fix; these
tests pin the propagation to this copy.
"""

import ast
import inspect
import json
import os
import time
from pathlib import Path

import pytest

optuna = pytest.importorskip("optuna", reason="optuna is an optional dependency")

from optuna.trial import TrialState

import src.vin_ocr.training.finetune_deepseek as finetune_deepseek
import src.vin_ocr.training.train_from_scratch as train_from_scratch
from optuna_tuning import REPO_ROOT
from src.vin_ocr.training.hyperparameter_tuning import TrialExecutionError
from src.vin_ocr.training.hyperparameter_tuning.optuna_tuning import (
    DeepSeekObjective,
    DeepSeekSearchSpace,
    OptunaHyperparameterTuner,
    PaddleOCRObjective,
    PaddleOCRSearchSpace,
    TuningConfig,
)

PACKAGE_TUNER_PATH = (
    REPO_ROOT / "src" / "vin_ocr" / "training" / "hyperparameter_tuning" / "optuna_tuning.py"
)
ROOT_TUNER_PATH = REPO_ROOT / "optuna_tuning.py"


def _fresh_trial() -> optuna.Trial:
    """A real trial, so suggest_* calls behave exactly as in production."""
    return optuna.create_study(direction="maximize").ask()


def _stub_paddle_trainer(monkeypatch, *, result=None, raises=None) -> dict:
    """
    Replace PaddleOCRScratchTrainer with a stub.

    The stub's ``train()`` accepts NO keyword arguments, exactly like the real
    trainer (train_from_scratch.py defines ``def train(self)``). If the
    objective regresses to passing ``epoch_callback``, the call raises
    TypeError and the tests below fail.
    """
    created = {}

    class StubTrainer:
        def __init__(self, config):
            created["config"] = config

        def train(self):
            if raises is not None:
                raise raises
            return result

    monkeypatch.setattr(train_from_scratch, "PaddleOCRScratchTrainer", StubTrainer)
    return created


def _stub_deepseek_trainer(monkeypatch, *, progress=None, raises=None) -> dict:
    """
    Replace DeepSeekVINTrainer with a stub mirroring the real contract:
    ``train()`` returns None; the measurement, if any, is the
    ``training_progress.json`` its ProgressCallback writes to the output dir.
    """
    created = {}

    class StubTuner:
        def __init__(self, config):
            created["config"] = config

        def train(self):
            if raises is not None:
                raise raises
            if progress is not None:
                out = Path(created["config"].output_dir)
                out.mkdir(parents=True, exist_ok=True)
                text = progress if isinstance(progress, str) else json.dumps(progress)
                (out / "training_progress.json").write_text(text, encoding="utf-8")

    monkeypatch.setattr(finetune_deepseek, "DeepSeekVINTrainer", StubTuner)
    return created


class TestPaddleObjectiveNeverFabricates:
    """
    WAS BROKEN: ``trainer.train(epoch_callback=...)`` raised TypeError on
    every call (no trainer accepts that keyword) and
    ``except Exception: return 0.0`` scored the crash as a measured zero.
    """

    def test_trial_scores_the_trainers_returned_accuracy(self, tmp_path, monkeypatch):
        created = _stub_paddle_trainer(monkeypatch, result=0.4186)
        objective = PaddleOCRObjective(
            PaddleOCRSearchSpace(), TuningConfig(output_dir=str(tmp_path)), device="cpu"
        )

        value = objective(_fresh_trial())

        assert value == 0.4186
        assert "trial_0" in created["config"].output_dir
        assert created["config"].use_gpu is False

    def test_training_crash_raises_rather_than_scoring_zero(self, tmp_path, monkeypatch):
        _stub_paddle_trainer(monkeypatch, raises=RuntimeError("CUDA out of memory"))
        objective = PaddleOCRObjective(
            PaddleOCRSearchSpace(), TuningConfig(output_dir=str(tmp_path)), device="cpu"
        )

        with pytest.raises(TrialExecutionError) as excinfo:
            objective(_fresh_trial())

        assert "training crashed" in str(excinfo.value)

    def test_a_crash_is_not_mislabelled_as_pruned(self, tmp_path, monkeypatch):
        """
        Pruned means "stopped by the pruner's policy"; some analyses include
        pruned trials as legitimate outcomes. A crash must surface as FAILED,
        which requires the raised type not to be TrialPruned.
        """
        _stub_paddle_trainer(monkeypatch, raises=RuntimeError("boom"))
        objective = PaddleOCRObjective(
            PaddleOCRSearchSpace(), TuningConfig(output_dir=str(tmp_path)), device="cpu"
        )

        with pytest.raises(TrialExecutionError):
            objective(_fresh_trial())
        assert not issubclass(TrialExecutionError, optuna.TrialPruned)

    def test_trainer_returning_none_is_no_measurement(self, tmp_path, monkeypatch):
        _stub_paddle_trainer(monkeypatch, result=None)
        objective = PaddleOCRObjective(
            PaddleOCRSearchSpace(), TuningConfig(output_dir=str(tmp_path)), device="cpu"
        )

        with pytest.raises(TrialExecutionError) as excinfo:
            objective(_fresh_trial())

        assert "nothing was measured" in str(excinfo.value)


class TestDeepSeekObjectiveNeverFabricates:
    """
    DeepSeekFineTuner.train() returns None; its best accuracy exists only in
    the training_progress.json its ProgressCallback writes - inside a silent
    try/except, so the file's absence or staleness means nothing was measured.
    """

    def _objective(self, tmp_path) -> DeepSeekObjective:
        return DeepSeekObjective(
            DeepSeekSearchSpace(), TuningConfig(output_dir=str(tmp_path)), device="cpu"
        )

    def test_best_accuracy_is_read_from_the_progress_record(self, tmp_path, monkeypatch):
        _stub_deepseek_trainer(monkeypatch, progress={"best_accuracy": 0.31})

        value = self._objective(tmp_path)(_fresh_trial())

        assert value == 0.31

    def test_missing_progress_record_raises(self, tmp_path, monkeypatch):
        _stub_deepseek_trainer(monkeypatch, progress=None)

        with pytest.raises(TrialExecutionError) as excinfo:
            self._objective(tmp_path)(_fresh_trial())

        assert "wrote no progress record" in str(excinfo.value)

    def test_stale_progress_record_raises(self, tmp_path, monkeypatch):
        """A record predating the trial is another trial's data, not this one's."""
        trial = _fresh_trial()
        stale = tmp_path / f"trial_{trial.number}" / "training_progress.json"
        stale.parent.mkdir(parents=True)
        stale.write_text(json.dumps({"best_accuracy": 0.9999}), encoding="utf-8")
        old = time.time() - 3600
        os.utime(stale, (old, old))
        _stub_deepseek_trainer(monkeypatch, progress=None)

        # Guard against the stub racing the mtime check.
        assert stale.is_file()

        with pytest.raises(TrialExecutionError) as excinfo:
            self._objective(tmp_path)(trial)

        assert "predates this trial" in str(excinfo.value)

    def test_corrupt_progress_record_raises(self, tmp_path, monkeypatch):
        _stub_deepseek_trainer(monkeypatch, progress="{not json")

        with pytest.raises(TrialExecutionError) as excinfo:
            self._objective(tmp_path)(_fresh_trial())

        assert "could not read best_accuracy" in str(excinfo.value)

    def test_non_numeric_best_accuracy_raises(self, tmp_path, monkeypatch):
        _stub_deepseek_trainer(monkeypatch, progress={"best_accuracy": None})

        with pytest.raises(TrialExecutionError) as excinfo:
            self._objective(tmp_path)(_fresh_trial())

        assert "not a number" in str(excinfo.value)


class TestStudyAccounting:
    """Failed trials are FAILED - visible, uncounted, and non-fatal to the study."""

    def test_failed_trials_are_recorded_failed_and_the_study_continues(
        self, tmp_path, monkeypatch
    ):
        tuner = OptunaHyperparameterTuner(
            model_type="paddleocr",
            config=TuningConfig(output_dir=str(tmp_path), enable_pruning=False),
        )

        def crash_on_even_trials(self, params, trial):
            if trial.number % 2 == 0:
                raise TrialExecutionError(f"trial {trial.number}: training crashed")
            return 0.1 * trial.number

        monkeypatch.setattr(
            PaddleOCRObjective, "_train_and_evaluate", crash_on_even_trials
        )

        tuner.optimize(n_trials=4)

        states = [t.state for t in tuner.study.trials]
        assert states.count(TrialState.FAIL) == 2
        assert states.count(TrialState.COMPLETE) == 2
        assert tuner.get_best_value() == pytest.approx(0.3)

        results = json.loads(
            (tmp_path / "optimization_results.json").read_text(encoding="utf-8")
        )
        assert results["n_trials"] == 4
        assert results["completed_trials"] == 2
        assert results["best_value"] == pytest.approx(0.3)

    def test_a_study_with_no_measurements_reports_no_best(self, tmp_path, monkeypatch):
        tuner = OptunaHyperparameterTuner(
            model_type="paddleocr",
            config=TuningConfig(output_dir=str(tmp_path), enable_pruning=False),
        )

        def always_fails(self, params, trial):
            raise TrialExecutionError("nothing measured")

        monkeypatch.setattr(PaddleOCRObjective, "_train_and_evaluate", always_fails)

        tuner.optimize(n_trials=2)

        assert tuner.get_best_value() is None, "a best value was fabricated"
        assert tuner.get_best_params() == {}

        results = json.loads(
            (tmp_path / "optimization_results.json").read_text(encoding="utf-8")
        )
        assert results["completed_trials"] == 0
        assert results["best_value"] is None
        assert results["error"] == "No trials completed successfully"


class TestSingleSourceOfTruth:
    """
    The audit's rule: a fix applied to one copy is not a fix. These pin the
    structural properties across BOTH tuners so the fabrication pattern
    cannot silently return to either.
    """

    def test_trial_execution_error_has_exactly_one_definition(self):
        import optuna_tuning as root_tuner

        assert root_tuner.TrialExecutionError is TrialExecutionError, (
            "two same-named exception classes: an except clause written "
            "against one silently fails to catch the other"
        )

    def test_no_except_handler_in_either_tuner_returns_a_numeric_literal(self):
        """The fabrication pattern itself: `except ...: return 0.0`."""
        for path in (ROOT_TUNER_PATH, PACKAGE_TUNER_PATH):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.ExceptHandler):
                    continue
                for inner in ast.walk(node):
                    if (
                        isinstance(inner, ast.Return)
                        and isinstance(inner.value, ast.Constant)
                        and isinstance(inner.value.value, (int, float))
                        and not isinstance(inner.value.value, bool)
                    ):
                        pytest.fail(
                            f"{path.name}:{inner.lineno} returns a numeric "
                            f"literal from an except handler - a crash scored "
                            f"as a measurement"
                        )

    def test_objectives_do_not_pass_the_nonexistent_epoch_callback(self):
        """
        No trainer in this repository accepts an ``epoch_callback`` keyword
        (train() signatures: ``train(self)`` and ``train(self, resume_from)``).
        Passing it made every trial crash before this fix. Checked in the
        AST, not the raw text, because the tuner's docstrings legitimately
        mention the old keyword when documenting the defect.
        """
        tree = ast.parse(PACKAGE_TUNER_PATH.read_text(encoding="utf-8"))
        offending = [
            f"{PACKAGE_TUNER_PATH.name}:{node.lineno}"
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            for kw in node.keywords
            if kw.arg == "epoch_callback"
        ]
        assert not offending, (
            f"epoch_callback passed at {offending}; no trainer accepts it"
        )

    def test_optimize_records_trial_failures_without_aborting_the_study(self):
        source = inspect.getsource(OptunaHyperparameterTuner.optimize)
        assert "catch=(TrialExecutionError,)" in source


class TestWebAppWiring:
    """
    WAS BROKEN: app.py imported ``HyperparameterTuner``, a name the package
    has never exported, so the ImportError branch always ran and
    HYPERPARAMETER_TUNING_AVAILABLE could never be True.
    """

    APP_PATH = REPO_ROOT / "src" / "vin_ocr" / "web" / "app.py"

    def test_app_imports_a_name_the_package_actually_exports(self):
        app_source = self.APP_PATH.read_text(encoding="utf-8")
        assert "import HyperparameterTuner" not in app_source
        assert (
            "from src.vin_ocr.training.hyperparameter_tuning "
            "import OptunaHyperparameterTuner" in app_source
        )

    def test_the_package_exports_what_app_py_imports(self):
        import src.vin_ocr.training.hyperparameter_tuning as pkg

        assert hasattr(pkg, "OptunaHyperparameterTuner")
        assert "OptunaHyperparameterTuner" in pkg.__all__
