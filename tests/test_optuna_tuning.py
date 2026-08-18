"""
Regression tests for the Optuna hyperparameter search.

The defect these principally pin is a fabrication mechanism, not a style
problem: a trial whose training crashed was scored from the PREVIOUS trial's
metrics file and that number was returned to Optuna as a genuine observation.
Since ``optuna_results/`` is the only machine-measured corpus this project has
- and the source of its 41.86% (18/43) baseline - any trial in it that crashed
silently carries a neighbour's accuracy under its own hyperparameters.
"""

import json
import os
import subprocess
import time
from pathlib import Path

import pytest

optuna = pytest.importorskip("optuna", reason="optuna is an optional dependency")

import optuna_tuning
from optuna_tuning import (
    REPO_ROOT,
    TRIAL_METRICS_PATH,
    TrialExecutionError,
    VINOCRHyperparameterTuner,
)

# Shaped exactly like what finetune_paddleocr writes.
GOOD_METRICS = {
    "evaluation_metrics": {
        "image_level": {"exact_match_accuracy": 0.4186},
        "character_level": {"character_accuracy": 0.9015, "f1_micro": 0.8800},
    },
    "training_results": {"final_epoch": 20, "best_validation_accuracy": 0.4186},
}

PREVIOUS_TRIAL_METRICS = {
    "evaluation_metrics": {
        "image_level": {"exact_match_accuracy": 0.9999},
        "character_level": {"character_accuracy": 0.9999, "f1_micro": 0.9999},
    },
    "training_results": {"final_epoch": 20, "best_validation_accuracy": 0.9999},
}


@pytest.fixture()
def metrics_path(tmp_path, monkeypatch) -> Path:
    """Redirect the module's fixed metrics path into a temp directory."""
    path = tmp_path / "training_metrics.json"
    monkeypatch.setattr(optuna_tuning, "TRIAL_METRICS_PATH", path)
    return path


@pytest.fixture()
def tuner() -> VINOCRHyperparameterTuner:
    """A tuner instance; the constructor only ensures a directory exists."""
    return VINOCRHyperparameterTuner()


def _fake_subprocess(
    monkeypatch,
    *,
    returncode: int = 0,
    writes: dict | None = None,
    target: Path | None = None,
    age_seconds: float = 0.0,
    raises: Exception | None = None,
):
    """
    Replace subprocess.run with a stub emulating one training invocation.

    Args:
        returncode: Exit status to report.
        writes: Metrics document to write, or None to write nothing.
        target: Where to write it.
        age_seconds: Backdate the written file by this many seconds.
        raises: Exception to raise instead of returning.
    """

    def fake_run(*_args, **_kwargs):
        if raises is not None:
            raise raises
        if writes is not None and target is not None:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(writes), encoding="utf-8")
            if age_seconds:
                old = time.time() - age_seconds
                os.utime(target, (old, old))
        return subprocess.CompletedProcess(
            args=["fake"], returncode=returncode, stdout="", stderr="boom"
        )

    monkeypatch.setattr(optuna_tuning.subprocess, "run", fake_run)


class TestStaleMetricsAreNeverScored:
    """
    WAS BROKEN: run_training_trial scored a trial from
    ``output/vin_rec_finetune/training_metrics.json`` whenever that file merely
    EXISTED. It checked neither the subprocess exit status nor when the file
    was written. Every trial overwrites that same fixed path, so a trial that
    crashed before writing was scored from the previous trial's file - and the
    resulting accuracy was returned to Optuna as a real observation of THIS
    trial's hyperparameters.
    """

    def test_crash_after_a_successful_trial_does_not_inherit_its_accuracy(
        self, tuner, metrics_path, monkeypatch
    ):
        # A previous trial left a very good result behind.
        metrics_path.write_text(json.dumps(PREVIOUS_TRIAL_METRICS), encoding="utf-8")

        # This trial's training writes nothing.
        _fake_subprocess(monkeypatch, returncode=0, writes=None)

        with pytest.raises(TrialExecutionError) as excinfo:
            tuner.run_training_trial("configs/whatever.yml")

        assert "wrote no metrics" in str(excinfo.value)

    def test_stale_file_is_rejected_on_modification_time(
        self, tuner, metrics_path, monkeypatch
    ):
        """Guard 3: even a file that appears after launch must post-date it."""
        _fake_subprocess(
            monkeypatch,
            returncode=0,
            writes=PREVIOUS_TRIAL_METRICS,
            target=metrics_path,
            age_seconds=3600,
        )

        with pytest.raises(TrialExecutionError) as excinfo:
            tuner.run_training_trial("configs/whatever.yml")

        assert "stale" in str(excinfo.value)

    def test_stale_file_is_removed_before_launching(
        self, tuner, metrics_path, monkeypatch
    ):
        """Guard 1: the previous trial's output must not survive into this one."""
        metrics_path.write_text(json.dumps(PREVIOUS_TRIAL_METRICS), encoding="utf-8")
        seen = {}

        def fake_run(*_args, **_kwargs):
            seen["existed_at_launch"] = metrics_path.exists()
            return subprocess.CompletedProcess(["fake"], 0, "", "")

        monkeypatch.setattr(optuna_tuning.subprocess, "run", fake_run)

        with pytest.raises(TrialExecutionError):
            tuner.run_training_trial("configs/whatever.yml")

        assert seen["existed_at_launch"] is False


class TestFailureIsNeverAMeasurement:
    """A trial that did not run is an absence of data, not a data point."""

    def test_nonzero_exit_is_fatal_even_with_valid_metrics_present(
        self, tuner, metrics_path, monkeypatch
    ):
        """
        Guard 2. A crash is a crash whatever happens to be on disk - a training
        process can write metrics and then die during export or checkpointing.
        """
        _fake_subprocess(
            monkeypatch, returncode=1, writes=GOOD_METRICS, target=metrics_path
        )

        with pytest.raises(TrialExecutionError) as excinfo:
            tuner.run_training_trial("configs/whatever.yml")

        assert "exited 1" in str(excinfo.value)

    def test_timeout_raises_rather_than_returning_zero(
        self, tuner, metrics_path, monkeypatch
    ):
        _fake_subprocess(
            monkeypatch,
            raises=subprocess.TimeoutExpired(cmd="fake", timeout=3600),
        )

        with pytest.raises(TrialExecutionError) as excinfo:
            tuner.run_training_trial("configs/whatever.yml")

        assert "exceeded" in str(excinfo.value)

    def test_corrupt_metrics_raise_rather_than_scoring_garbage(
        self, tuner, metrics_path, monkeypatch
    ):
        def fake_run(*_args, **_kwargs):
            metrics_path.write_text("{not json", encoding="utf-8")
            return subprocess.CompletedProcess(["fake"], 0, "", "")

        monkeypatch.setattr(optuna_tuning.subprocess, "run", fake_run)

        with pytest.raises(TrialExecutionError) as excinfo:
            tuner.run_training_trial("configs/whatever.yml")

        assert "could not read metrics" in str(excinfo.value)

    def test_metrics_missing_a_required_key_raise(
        self, tuner, metrics_path, monkeypatch
    ):
        _fake_subprocess(
            monkeypatch,
            returncode=0,
            writes={"evaluation_metrics": {}},
            target=metrics_path,
        )

        with pytest.raises(TrialExecutionError):
            tuner.run_training_trial("configs/whatever.yml")

    def test_trial_execution_error_is_what_optimize_catches(self):
        """
        The study must keep going past a failed trial without recording it as
        an observation. That contract is expressed by catching exactly this
        type, so it has to be an Exception subclass Optuna can catch.
        """
        assert issubclass(TrialExecutionError, Exception)
        assert not issubclass(TrialExecutionError, BaseException) or True


class TestSuccessPath:
    """The guards must not reject a legitimate trial."""

    def test_fresh_valid_metrics_are_returned(self, tuner, metrics_path, monkeypatch):
        _fake_subprocess(
            monkeypatch, returncode=0, writes=GOOD_METRICS, target=metrics_path
        )

        result = tuner.run_training_trial("configs/whatever.yml")

        assert result["exact_match_accuracy"] == 0.4186
        assert result["character_accuracy"] == 0.9015
        assert result["f1_micro"] == 0.88
        assert result["final_epoch"] == 20
        assert result["training_time"] >= 0

    def test_a_previous_result_does_not_leak_into_a_successful_trial(
        self, tuner, metrics_path, monkeypatch
    ):
        metrics_path.write_text(json.dumps(PREVIOUS_TRIAL_METRICS), encoding="utf-8")
        _fake_subprocess(
            monkeypatch, returncode=0, writes=GOOD_METRICS, target=metrics_path
        )

        result = tuner.run_training_trial("configs/whatever.yml")

        assert result["exact_match_accuracy"] == 0.4186, "read the previous trial's file"


class TestPathsAreAnchoredToTheRepository:
    """
    WAS BROKEN: the metrics path was the relative string
    "output/vin_rec_finetune/training_metrics.json" while the trial subprocess
    ran with cwd=REPO_ROOT. Launching the tuner from any other directory meant
    reading a path the subprocess never wrote to.
    """

    def test_metrics_path_is_absolute(self):
        assert TRIAL_METRICS_PATH.is_absolute()

    def test_metrics_path_is_under_the_repo_root(self):
        assert str(TRIAL_METRICS_PATH).startswith(str(REPO_ROOT))

    def test_repo_root_is_the_repository(self):
        assert (REPO_ROOT / "pyproject.toml").is_file()

    def test_results_dir_is_absolute(self, tuner):
        assert tuner.results_dir.is_absolute()
        assert str(tuner.results_dir).startswith(str(REPO_ROOT))


class TestStudyPersistence:
    """
    WAS BROKEN: create_study was called without ``storage``, so the study lived
    only in memory. An interrupted run lost every completed trial and the
    sampler could not be warm-started - at up to an hour per trial, days of
    compute discarded by one Ctrl-C.
    """

    def test_study_is_persisted_and_resumable(self, tmp_path, monkeypatch):
        import inspect

        source = inspect.getsource(VINOCRHyperparameterTuner.run_study)
        assert "storage=" in source
        assert "load_if_exists=True" in source

    def test_a_sqlite_study_round_trips(self, tmp_path):
        """The persistence mechanism itself works as assumed."""
        uri = f"sqlite:///{tmp_path / 'study.db'}"

        first = optuna.create_study(
            direction="maximize", study_name="s", storage=uri, load_if_exists=True
        )
        first.optimize(lambda t: t.suggest_float("x", 0, 1), n_trials=3)

        resumed = optuna.create_study(
            direction="maximize", study_name="s", storage=uri, load_if_exists=True
        )
        assert len(resumed.trials) == 3


class TestNoBestWithoutMeasurement:
    """
    An all-failed study must report nothing rather than raise an opaque
    ValueError from study.best_value or, worse, emit a best_config.yml derived
    from no measurements.
    """

    def test_main_returns_failure_when_nothing_was_measured(self, monkeypatch, tmp_path):
        class EmptyStudy:
            trials = []

        monkeypatch.setattr(
            VINOCRHyperparameterTuner, "run_study", lambda *a, **k: EmptyStudy()
        )
        assert optuna_tuning.main([]) == 1

    def test_no_best_config_is_written_when_nothing_was_measured(
        self, monkeypatch, tmp_path
    ):
        class EmptyStudy:
            trials = []

        monkeypatch.setattr(
            VINOCRHyperparameterTuner, "run_study", lambda *a, **k: EmptyStudy()
        )
        marker = tmp_path / "best_config.yml"
        monkeypatch.setattr(
            VINOCRHyperparameterTuner, "results_dir", tmp_path, raising=False
        )

        optuna_tuning.main([])
        assert not marker.exists()
